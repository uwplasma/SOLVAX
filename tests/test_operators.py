"""Tests for solvax.operators: matvec/materialize/adjoint consistency and composition."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    BlockTridiagonalOperator,
    BorderedOperator,
    KroneckerOperator,
    MatrixFreeOperator,
    SumOperator,
    block_thomas,
    block_thomas_factor,
    block_thomas_solve,
    gmres,
    schur_projected_precond,
)

jax.config.update("jax_enable_x64", True)

TOL = dict(rtol=1e-13, atol=1e-13)


def make_block_system(n_blocks, m, seed=0, dominance=4.0):
    """Random well-conditioned block-tridiagonal bands + their dense form.

    Same assembly as tests/test_direct.py::make_system.
    """
    rng = np.random.default_rng(seed)
    lower = rng.standard_normal((n_blocks, m, m))
    diag = rng.standard_normal((n_blocks, m, m)) + dominance * m * np.eye(m)
    upper = rng.standard_normal((n_blocks, m, m))

    dense = np.zeros((n_blocks * m, n_blocks * m))
    for k in range(n_blocks):
        s = slice(k * m, (k + 1) * m)
        dense[s, s] = diag[k]
        if k > 0:
            dense[s, slice((k - 1) * m, k * m)] = lower[k]
        if k < n_blocks - 1:
            dense[s, slice((k + 1) * m, (k + 2) * m)] = upper[k]
    return map(jnp.asarray, (lower, diag, upper)), dense


def check_matvec_materialize_adjoint(op, seed=0):
    """The two universal invariants: op(v) == materialize() @ v and
    <A x, y> == <x, A^T y>."""
    n_out, n_in = op.shape
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(n_in))
    w = jnp.asarray(rng.standard_normal(n_out))

    dense = np.asarray(op.materialize())
    assert dense.shape == (n_out, n_in)
    assert np.allclose(np.asarray(op.matvec(v)), dense @ np.asarray(v), **TOL)
    assert np.allclose(np.asarray(op(v)), dense @ np.asarray(v), **TOL)

    op_t = op.T
    assert op_t.shape == (n_in, n_out)
    lhs = float(op.matvec(v) @ w)
    rhs = float(v @ op_t.matvec(w))
    assert np.isclose(lhs, rhs, **TOL)
    assert np.allclose(np.asarray(op_t.materialize()), dense.T, **TOL)


# ---------------------------------------------------------------- MatrixFree


@pytest.mark.parametrize("shape", [(6, 6), (4, 7)])
def test_matrix_free_linear_transpose(shape):
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal(shape))
    op = MatrixFreeOperator(lambda v: a @ v, shape=shape)
    assert np.allclose(np.asarray(op.materialize()), np.asarray(a), **TOL)
    check_matvec_materialize_adjoint(op, seed=1)


def test_matrix_free_explicit_transpose():
    rng = np.random.default_rng(2)
    a = jnp.asarray(rng.standard_normal((5, 8)))
    op = MatrixFreeOperator(lambda v: a @ v, lambda w: a.T @ w, shape=(5, 8))
    check_matvec_materialize_adjoint(op, seed=2)
    # With an explicit transpose_apply, the double transpose is the original.
    v = jnp.asarray(rng.standard_normal(8))
    assert np.allclose(np.asarray(op.T.T(v)), np.asarray(op(v)), **TOL)
    assert op.T.T.apply is op.apply


# ----------------------------------------------------------------------- Sum


def test_sum_operator_mixed_terms():
    rng = np.random.default_rng(3)
    n = 7
    a = jnp.asarray(rng.standard_normal((n, n)))
    b = jnp.asarray(rng.standard_normal((n, n)))
    c = jnp.asarray(rng.standard_normal((n, n)))
    op = SumOperator(
        [
            MatrixFreeOperator(lambda v: a @ v, shape=(n, n)),  # operator
            lambda v: b @ v,  # plain callable
            c,  # dense matrix
        ]
    )
    assert op.shape == (n, n)
    dense = np.asarray(a + b + c)
    v = jnp.asarray(rng.standard_normal(n))
    assert np.allclose(np.asarray(op(v)), dense @ np.asarray(v), **TOL)
    check_matvec_materialize_adjoint(op, seed=3)


def test_sum_operator_shape_requires_a_shaped_term():
    op = SumOperator([lambda v: v, lambda v: 2.0 * v])
    with pytest.raises(ValueError, match="shape"):
        _ = op.shape


# ----------------------------------------------------------------- Kronecker


@pytest.mark.parametrize(
    "shape_a,shape_b", [((3, 3), (4, 4)), ((3, 5), (2, 4)), ((6, 2), (3, 7))]
)
def test_kronecker_matches_dense_kron(shape_a, shape_b):
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.standard_normal(shape_a))
    b = jnp.asarray(rng.standard_normal(shape_b))
    op = KroneckerOperator(a, b)
    kron = np.kron(np.asarray(a), np.asarray(b))
    assert op.shape == kron.shape
    v = jnp.asarray(rng.standard_normal(kron.shape[1]))
    assert np.allclose(np.asarray(op(v)), kron @ np.asarray(v), **TOL)
    assert np.allclose(np.asarray(op.materialize()), kron, **TOL)
    check_matvec_materialize_adjoint(op, seed=4)


def test_kronecker_of_operators():
    rng = np.random.default_rng(5)
    a = jnp.asarray(rng.standard_normal((4, 3)))
    b = jnp.asarray(rng.standard_normal((2, 5)))
    op = KroneckerOperator(
        MatrixFreeOperator(lambda v: a @ v, lambda w: a.T @ w, shape=(4, 3)), b
    )
    kron = np.kron(np.asarray(a), np.asarray(b))
    v = jnp.asarray(rng.standard_normal(kron.shape[1]))
    assert np.allclose(np.asarray(op(v)), kron @ np.asarray(v), **TOL)
    check_matvec_materialize_adjoint(op, seed=5)


def test_kronecker_rejects_bare_callables():
    with pytest.raises(ValueError, match="MatrixFreeOperator"):
        _ = KroneckerOperator(lambda v: v, jnp.eye(2)).shape
    with pytest.raises(ValueError, match="MatrixFreeOperator"):
        _ = KroneckerOperator(lambda v: v, jnp.eye(2)).T


# ----------------------------------------------------------- BlockTridiagonal


@pytest.mark.parametrize("n_blocks,m", [(1, 3), (4, 3), (12, 5)])
def test_block_tridiagonal_matches_dense(n_blocks, m):
    (lower, diag, upper), dense = make_block_system(n_blocks, m, seed=6)
    op = BlockTridiagonalOperator(lower, diag, upper)
    assert op.shape == dense.shape
    rng = np.random.default_rng(6)
    v = jnp.asarray(rng.standard_normal(n_blocks * m))
    assert np.allclose(np.asarray(op(v)), dense @ np.asarray(v), **TOL)
    assert np.allclose(np.asarray(op.materialize()), dense, **TOL)
    check_matvec_materialize_adjoint(op, seed=6)


def test_block_tridiagonal_to_blocks_feeds_block_thomas():
    n_blocks, m = 9, 4
    (lower, diag, upper), dense = make_block_system(n_blocks, m, seed=7)
    op = BlockTridiagonalOperator(lower, diag, upper)
    rng = np.random.default_rng(7)
    b = rng.standard_normal(n_blocks * m)

    x = block_thomas(*op.to_blocks(), jnp.asarray(b.reshape(n_blocks, m)))
    x_flat = np.asarray(x).reshape(-1)
    assert np.allclose(x_flat, np.linalg.solve(dense, b), atol=1e-12)
    # And the operator maps the solution back to the right-hand side.
    assert np.allclose(np.asarray(op(jnp.asarray(x_flat))), b, atol=1e-11)


def test_block_tridiagonal_shape_validation():
    with pytest.raises(ValueError, match="n_blocks, m, m"):
        BlockTridiagonalOperator(
            jnp.zeros((3, 2, 2)), jnp.zeros((3, 2, 2)), jnp.zeros((4, 2, 2))
        )


# ------------------------------------------------------------------ Bordered


def make_bordered(n=12, p=3, seed=8, dominance=3.0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + dominance * n ** 0.5 * np.eye(n)
    b_cols = rng.standard_normal((n, p))
    c_rows = rng.standard_normal((p, n))
    return jnp.asarray(a), jnp.asarray(b_cols), jnp.asarray(c_rows)


def test_bordered_matches_dense():
    a, b_cols, c_rows = make_bordered()
    n, p = b_cols.shape
    op = BorderedOperator(a, b_cols, c_rows)
    assert op.shape == (n + p, n + p)
    dense = np.block(
        [[np.asarray(a), np.asarray(b_cols)], [np.asarray(c_rows), np.zeros((p, p))]]
    )
    rng = np.random.default_rng(8)
    v = jnp.asarray(rng.standard_normal(n + p))
    assert np.allclose(np.asarray(op(v)), dense @ np.asarray(v), **TOL)
    assert np.allclose(np.asarray(op.materialize()), dense, **TOL)
    check_matvec_materialize_adjoint(op, seed=8)


def test_bordered_with_operator_block_and_rectangular_border():
    a, b_cols, c_rows = make_bordered(n=10, p=2, seed=9)
    op = BorderedOperator(
        MatrixFreeOperator(lambda v: a @ v, shape=(10, 10)),
        b_cols,
        c_rows[:1],  # q != p: rectangular zero block
    )
    assert op.shape == (11, 12)
    check_matvec_materialize_adjoint(op, seed=9)


def test_bordered_border_shape_validation():
    a, b_cols, c_rows = make_bordered()
    with pytest.raises(ValueError, match="b_cols"):
        BorderedOperator(a, b_cols.T, c_rows)
    with pytest.raises(ValueError, match="c_rows"):
        BorderedOperator(a, b_cols, c_rows.T)


def test_schur_projected_precond_exact_inverse():
    a, b_cols, c_rows = make_bordered(n=40, p=3, seed=10)
    n, p = b_cols.shape
    op = BorderedOperator(a, b_cols, c_rows)
    dense = np.asarray(op.materialize())
    rng = np.random.default_rng(10)
    rhs = rng.standard_normal(n + p)
    x_ref = np.linalg.solve(dense, rhs)

    a_inv_mat = jnp.asarray(np.linalg.inv(np.asarray(a)))
    precond = schur_projected_precond(lambda v: a_inv_mat @ v, b_cols, c_rows)
    sol = gmres(op, jnp.asarray(rhs), precond=precond, rtol=1e-12, restart=10)
    assert bool(sol.converged)
    assert int(sol.iterations) <= 3
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-10


def test_schur_projected_precond_approximate_inverse():
    a, b_cols, c_rows = make_bordered(n=40, p=3, seed=11, dominance=2.0)
    n, p = b_cols.shape
    op = BorderedOperator(a, b_cols, c_rows)
    rng = np.random.default_rng(11)
    rhs = jnp.asarray(rng.standard_normal(n + p))
    x_ref = np.linalg.solve(np.asarray(op.materialize()), np.asarray(rhs))

    plain = gmres(op, rhs, rtol=1e-10, restart=50, max_restarts=40)
    assert bool(plain.converged)

    # Block-Jacobi-quality approximation of A^{-1}: invert 4x4 diagonal blocks.
    m = 4
    blocks = np.stack(
        [np.asarray(a)[k : k + m, k : k + m] for k in range(0, n, m)]
    )
    inv_blocks = jnp.asarray(np.linalg.inv(blocks))

    def a_inv(v):
        return jnp.einsum("kij,kj->ki", inv_blocks, v.reshape(-1, m)).reshape(-1)

    precond = schur_projected_precond(a_inv, b_cols, c_rows)
    pre = gmres(op, rhs, precond=precond, rtol=1e-10, restart=50, max_restarts=40)
    assert bool(pre.converged)
    assert int(pre.iterations) < int(plain.iterations)
    err = np.linalg.norm(np.asarray(pre.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-8


def test_schur_precond_requires_square_schur():
    a, b_cols, c_rows = make_bordered()
    a_inv_mat = jnp.asarray(np.linalg.inv(np.asarray(a)))
    with pytest.raises(ValueError, match="square"):
        schur_projected_precond(lambda v: a_inv_mat @ v, b_cols, c_rows[:1])


# --------------------------------------------------------------- Composition


def test_sum_of_tridiagonal_and_perturbation_preconditioned_gmres():
    n_blocks, m = 8, 4
    n = n_blocks * m
    (lower, diag, upper), dense = make_block_system(n_blocks, m, seed=12)
    tri = BlockTridiagonalOperator(lower, diag, upper)

    rng = np.random.default_rng(12)
    p_mat = jnp.asarray(0.1 * rng.standard_normal((n, n)))
    perturbation = MatrixFreeOperator(lambda v: p_mat @ v, shape=(n, n))
    op = SumOperator([tri, perturbation])

    factors = block_thomas_factor(*tri.to_blocks())

    def precond(v):
        return block_thomas_solve(factors, v.reshape(n_blocks, m)).reshape(-1)

    b = jnp.asarray(rng.standard_normal(n))
    sol = gmres(op, b, precond=precond, rtol=1e-10, restart=20)
    assert bool(sol.converged)
    assert int(sol.iterations) < 20

    x_ref = np.linalg.solve(dense + np.asarray(p_mat), np.asarray(b))
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-8
