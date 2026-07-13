"""Tests for solvax.direct: block-tridiagonal elimination vs dense reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    block_thomas,
    block_thomas_factor,
    block_thomas_factor_fn,
    block_thomas_solve,
    block_thomas_truncated,
    block_thomas_truncated_fn,
    block_thomas_truncated_fn_with_residual,
    block_tridiag_matvec,
    block_tridiag_relative_residual,
)

jax.config.update("jax_enable_x64", True)


def make_system(n_blocks, m, n_rhs=None, seed=0, dominance=4.0):
    """Random well-conditioned block-tridiagonal system + its dense form."""
    rng = np.random.default_rng(seed)
    lower = rng.standard_normal((n_blocks, m, m))
    diag = rng.standard_normal((n_blocks, m, m)) + dominance * m * np.eye(m)
    upper = rng.standard_normal((n_blocks, m, m))
    shape = (n_blocks, m) if n_rhs is None else (n_blocks, m, n_rhs)
    rhs = rng.standard_normal(shape)

    dense = np.zeros((n_blocks * m, n_blocks * m))
    for k in range(n_blocks):
        s = slice(k * m, (k + 1) * m)
        dense[s, s] = diag[k]
        if k > 0:
            dense[s, slice((k - 1) * m, k * m)] = lower[k]
        if k < n_blocks - 1:
            dense[s, slice((k + 1) * m, (k + 2) * m)] = upper[k]
    return map(jnp.asarray, (lower, diag, upper, rhs)), dense


@pytest.mark.parametrize("n_rhs", [None, 3])
@pytest.mark.parametrize("n_blocks,m", [(4, 3), (12, 5), (40, 2)])
def test_block_thomas_matches_dense(n_blocks, m, n_rhs):
    (lower, diag, upper, rhs), dense = make_system(n_blocks, m, n_rhs)
    x = block_thomas(lower, diag, upper, rhs)
    x_dense = np.linalg.solve(dense, np.asarray(rhs).reshape(n_blocks * m, -1))
    assert np.allclose(np.asarray(x).reshape(n_blocks * m, -1), x_dense, atol=1e-12)


def test_factor_solve_reuse():
    (lower, diag, upper, rhs), dense = make_system(8, 4, seed=1)
    factors = block_thomas_factor(lower, diag, upper)
    x1 = block_thomas_solve(factors, rhs)
    x2 = block_thomas_solve(factors, 2.0 * rhs)
    assert np.allclose(np.asarray(x2), 2.0 * np.asarray(x1), atol=1e-12)
    x_dense = np.linalg.solve(dense, np.asarray(rhs).reshape(-1))
    assert np.allclose(np.asarray(x1).reshape(-1), x_dense, atol=1e-12)


@pytest.mark.parametrize("n_rhs", [None, 2])
@pytest.mark.parametrize("n_blocks", [1, 8])
def test_generated_factor_matches_materialized_primal_and_transpose(n_blocks, n_rhs):
    (lower, diag, upper, rhs), dense = make_system(n_blocks, 4, n_rhs, seed=14)
    generated = block_thomas_factor_fn(_fn_from_arrays(lower, diag, upper), n_blocks)
    materialized = block_thomas_factor(lower, diag, upper)
    for transpose in (False, True):
        actual = block_thomas_solve(generated, rhs, transpose=transpose)
        expected = block_thomas_solve(materialized, rhs, transpose=transpose)
        dense_expected = np.linalg.solve(
            dense.T if transpose else dense,
            np.asarray(rhs).reshape(n_blocks * 4, -1),
        )
        assert np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)
        assert np.allclose(np.asarray(actual).reshape(n_blocks * 4, -1), dense_expected, atol=1e-12)


def test_generated_factor_jit_grad_and_float32():
    n_blocks = 6
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 3, seed=15)
    lower, diag, upper, rhs = (value.astype(jnp.float32) for value in (lower, diag, upper, rhs))

    def loss(shift):
        def block_fn(index):
            diagonal = diag[index] + shift * jnp.eye(3, dtype=diag.dtype)
            return lower[index], diagonal, upper[index]

        factors = block_thomas_factor_fn(block_fn, n_blocks)
        return jnp.sum(block_thomas_solve(factors, rhs) ** 2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.float32(0.1))
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))


def test_generated_factor_assembles_each_index_once():
    n_blocks = 5
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 2, seed=16)
    seen = []

    def record(index):
        seen.append(int(index))

    def solve():
        def block_fn(index):
            jax.debug.callback(record, index, ordered=True)
            return lower[index], diag[index], upper[index]

        return block_thomas_solve(block_thomas_factor_fn(block_fn, n_blocks), rhs)

    jax.jit(solve)().block_until_ready()
    assert sorted(seen) == list(range(n_blocks))


def test_generated_factor_rejects_empty_system():
    with pytest.raises(ValueError, match="positive"):
        block_thomas_factor_fn(lambda _: (None, None, None), 0)


@pytest.mark.parametrize("n_rhs", [None, 3])
def test_block_operator_action_and_residual_match_dense(n_rhs):
    n_blocks, m = 7, 4
    (lower, diag, upper, rhs), dense = make_system(n_blocks, m, n_rhs, seed=11)
    x = block_thomas(lower, diag, upper, rhs)
    action = block_tridiag_matvec(lower, diag, upper, x)
    dense_action = dense @ np.asarray(x).reshape(n_blocks * m, -1)
    assert np.allclose(np.asarray(action).reshape(n_blocks * m, -1), dense_action, atol=1e-12)
    residual = block_tridiag_relative_residual(lower, diag, upper, x, rhs)
    expected_shape = () if n_rhs is None else (n_rhs,)
    assert residual.shape == expected_shape
    assert np.all(np.asarray(residual) < 1.0e-14)


@pytest.mark.parametrize("keep", [1, 3])
def test_truncated_matches_full(keep):
    n_blocks, m = 16, 4
    (lower, diag, upper, rhs), _ = make_system(n_blocks, m, seed=2)
    # Zero the rhs above the kept blocks, as the truncated solve assumes.
    rhs = rhs.at[keep:].set(0.0)
    x_full = block_thomas(lower, diag, upper, rhs)
    x_trunc = block_thomas_truncated(lower, diag, upper, rhs[:keep], keep)
    assert np.allclose(np.asarray(x_trunc), np.asarray(x_full[:keep]), atol=1e-12)


def test_vmap_over_batch():
    def solve_one(seed):
        (lower, diag, upper, rhs), _ = make_system(6, 3, seed=seed)
        return lower, diag, upper, rhs

    systems = [solve_one(s) for s in range(4)]
    stacked = [jnp.stack(arrs) for arrs in zip(*systems, strict=False)]
    x_batch = jax.vmap(block_thomas)(*stacked)
    for i, (lower, diag, upper, rhs) in enumerate(systems):
        x_i = block_thomas(lower, diag, upper, rhs)
        assert np.allclose(np.asarray(x_batch[i]), np.asarray(x_i), atol=1e-12)


def test_gradient_through_solve():
    (lower, diag, upper, rhs), _ = make_system(5, 3, seed=3)

    def loss(d):
        return jnp.sum(block_thomas(lower, d, upper, rhs) ** 2)

    g = jax.grad(loss)(diag)
    # Central finite difference on one entry.
    eps = 1e-6
    e = jnp.zeros_like(diag).at[2, 1, 1].set(eps)
    fd = (loss(diag + e) - loss(diag - e)) / (2 * eps)
    assert np.isclose(float(g[2, 1, 1]), float(fd), rtol=1e-5)


@pytest.mark.parametrize("n_rhs", [None, 2])
def test_transpose_solve_matches_dense(n_rhs):
    (lower, diag, upper, rhs), dense = make_system(10, 4, n_rhs, seed=5)
    factors = block_thomas_factor(lower, diag, upper)
    x = block_thomas_solve(factors, rhs, transpose=True)
    x_dense = np.linalg.solve(dense.T, np.asarray(rhs).reshape(dense.shape[0], -1))
    assert np.allclose(np.asarray(x).reshape(dense.shape[0], -1), x_dense, atol=1e-12)


def test_transpose_solve_matches_linear_transpose():
    (lower, diag, upper, rhs), _ = make_system(6, 3, seed=6)
    factors = block_thomas_factor(lower, diag, upper)

    def fwd(v):
        return block_thomas_solve(factors, v)

    (via_lt,) = jax.linear_transpose(fwd, rhs)(rhs)
    via_flag = block_thomas_solve(factors, rhs, transpose=True)
    assert np.allclose(np.asarray(via_lt), np.asarray(via_flag), atol=1e-11)


def test_transpose_solve_gradient():
    (lower, diag, upper, rhs), _ = make_system(5, 3, seed=7)

    def loss(d):
        f = block_thomas_factor(lower, d, upper)
        return jnp.sum(block_thomas_solve(f, rhs, transpose=True) ** 2)

    g = jax.grad(loss)(diag)
    eps = 1e-6
    e = jnp.zeros_like(diag).at[1, 2, 0].set(eps)
    fd = (loss(diag + e) - loss(diag - e)) / (2 * eps)
    assert np.isclose(float(g[1, 2, 0]), float(fd), rtol=1e-5)


def _fn_from_arrays(lower, diag, upper):
    def block_fn(k):
        return lower[k], diag[k], upper[k]

    return block_fn


@pytest.mark.parametrize("n_blocks,keep", [(16, 1), (16, 3), (16, 15), (16, 16), (2, 1), (2, 2)])
def test_truncated_fn_matches_materialized(n_blocks, keep):
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 4, seed=8)
    rhs = rhs.at[keep:].set(0.0)
    x_ref = block_thomas(lower, diag, upper, rhs)
    x_fn = block_thomas_truncated_fn(
        _fn_from_arrays(lower, diag, upper), n_blocks, rhs[:keep], keep
    )
    assert np.allclose(np.asarray(x_fn), np.asarray(x_ref[:keep]), atol=1e-12)


def test_truncated_materialized_keep_equals_n():
    n_blocks, keep = 8, 8
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 3, seed=9)
    x_ref = block_thomas(lower, diag, upper, rhs)
    x_tr = block_thomas_truncated(lower, diag, upper, rhs, keep)
    assert np.allclose(np.asarray(x_tr), np.asarray(x_ref), atol=1e-12)


def test_truncated_fn_under_jit():
    n_blocks, keep = 12, 3
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 4, seed=10)
    rhs = rhs.at[keep:].set(0.0)
    fn = _fn_from_arrays(lower, diag, upper)

    @jax.jit
    def run(r):
        return block_thomas_truncated_fn(fn, n_blocks, r, keep)

    x_jit = run(rhs[:keep])
    x_ref = block_thomas(lower, diag, upper, rhs)[:keep]
    assert np.allclose(np.asarray(x_jit), np.asarray(x_ref), atol=1e-12)


@pytest.mark.parametrize("n_rhs", [None, 2])
def test_truncated_fn_residual_includes_eliminated_tail(n_rhs):
    n_blocks, keep = 16, 3
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 4, n_rhs, seed=17)
    rhs = rhs.at[keep:].set(0.0)
    block_fn = _fn_from_arrays(lower, diag, upper)

    @jax.jit
    def solve(r):
        return block_thomas_truncated_fn_with_residual(block_fn, n_blocks, r, keep)

    solution, residual = solve(rhs[:keep])
    reference = block_thomas(lower, diag, upper, rhs)[:keep]
    assert np.allclose(np.asarray(solution), np.asarray(reference), atol=1e-12)
    assert float(residual) < 1.0e-13


def test_truncated_fn_residual_can_select_one_rhs():
    n_blocks, keep = 8, 3
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 3, 2, seed=18)
    rhs = rhs.at[keep:].set(0.0)
    solution, residual = block_thomas_truncated_fn_with_residual(
        _fn_from_arrays(lower, diag, upper),
        n_blocks,
        rhs[:keep],
        keep,
        residual_rhs_index=0,
    )
    assert solution.shape == rhs[:keep].shape
    assert float(residual) < 1.0e-13

    with pytest.raises(ValueError, match="multiple"):
        block_thomas_truncated_fn_with_residual(
            _fn_from_arrays(lower, diag, upper),
            n_blocks,
            rhs[:keep, :, 0],
            keep,
            residual_rhs_index=0,
        )
    with pytest.raises(ValueError, match="range"):
        block_thomas_truncated_fn_with_residual(
            _fn_from_arrays(lower, diag, upper),
            n_blocks,
            rhs[:keep],
            keep,
            residual_rhs_index=2,
        )


def test_truncated_fn_multiple_rhs_jit_vmap_and_grad():
    n_blocks, keep, n_rhs = 9, 3, 2
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 4, n_rhs, seed=12)
    rhs = rhs.at[keep:].set(0.0)
    fn = _fn_from_arrays(lower, diag, upper)

    def solve(r):
        return block_thomas_truncated_fn(fn, n_blocks, r, keep)

    x = jax.jit(solve)(rhs[:keep])
    x_full = block_thomas(lower, diag, upper, rhs)
    assert np.allclose(np.asarray(x), np.asarray(x_full[:keep]), atol=1e-12)

    batch = jnp.stack([rhs[:keep], 2.0 * rhs[:keep]])
    x_batch = jax.jit(jax.vmap(solve))(batch)
    assert np.allclose(np.asarray(x_batch[1]), 2.0 * np.asarray(x_batch[0]), atol=1e-12)

    gradient = jax.grad(lambda r: jnp.sum(solve(r) ** 2))(rhs[:keep])
    assert gradient.shape == rhs[:keep].shape
    assert np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 2.0e-5), (jnp.float64, 1.0e-12)])
def test_truncated_fn_low_order_recovery_by_precision(dtype, atol):
    """Three blocks represent the N_xi=2 boundary used by kinetic callers."""
    n_blocks, keep = 3, 3
    (lower, diag, upper, rhs), _ = make_system(n_blocks, 5, 2, seed=13)
    lower, diag, upper, rhs = (value.astype(dtype) for value in (lower, diag, upper, rhs))
    x_fn = block_thomas_truncated_fn(_fn_from_arrays(lower, diag, upper), n_blocks, rhs, keep)
    x_full = block_thomas(lower, diag, upper, rhs)
    assert np.allclose(np.asarray(x_fn), np.asarray(x_full), atol=atol, rtol=atol)
