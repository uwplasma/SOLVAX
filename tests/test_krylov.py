"""Tests for solvax.krylov: restarted FGMRES and GCROT recycling vs dense reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from solvax import gcrot, gmres

jax.config.update("jax_enable_x64", True)


def random_system(n, seed=0, spread=0.5):
    """Random well-conditioned nonsymmetric system: eigenvalues in a disk
    of radius ~``spread`` around 1 (circular law)."""
    rng = np.random.default_rng(seed)
    a = np.eye(n) + spread * rng.standard_normal((n, n)) / np.sqrt(n)
    b = rng.standard_normal(n)
    return jnp.asarray(a), jnp.asarray(b)


def random_complex_system(n, seed=0, spread=0.25):
    """Well-conditioned non-Hermitian complex system."""
    rng = np.random.default_rng(seed)
    perturbation = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    a = (2.5 + 0.2j) * np.eye(n) + spread * perturbation / np.sqrt(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return jnp.asarray(a), jnp.asarray(b)


def advection_diffusion(n=256, peclet=1e3):
    """1-D periodic advection-diffusion, central differences, unit shift.

    ``u + a u' - nu u''`` with grid Peclet ``a h / nu = peclet``; strongly
    advection-dominated, so the spectrum hugs the imaginary axis and
    unpreconditioned GMRES crawls.
    """
    h = 1.0 / n
    a_coef = 1.0
    nu = a_coef * h / peclet
    diag = 1.0 + 2.0 * nu / h**2
    up = a_coef / (2.0 * h) - nu / h**2
    lo = -a_coef / (2.0 * h) - nu / h**2
    dense = np.zeros((n, n))
    idx = np.arange(n)
    dense[idx, idx] = diag
    dense[idx, (idx + 1) % n] = up
    dense[idx, (idx - 1) % n] = lo
    return dense


@pytest.mark.parametrize("n", [50, 200])
def test_gmres_matches_dense(n):
    a, b = random_system(n, seed=n)
    sol = gmres(lambda v: a @ v, b, rtol=1e-10, restart=30)
    assert bool(sol.converged)
    assert float(sol.residual_norm) <= 1e-10 * np.linalg.norm(np.asarray(b))
    x_ref = np.linalg.solve(np.asarray(a), np.asarray(b))
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-8


@pytest.mark.parametrize("solver_name", ["gmres", "gcrot"])
@pytest.mark.parametrize(
    "dtype,solve_rtol,error_tolerance",
    [(jnp.complex64, 1.0e-6, 2.0e-5), (jnp.complex128, 1.0e-11, 1.0e-9)],
)
def test_complex_krylov_matches_dense_under_jit(
    solver_name, dtype, solve_rtol, error_tolerance
):
    a, b = random_complex_system(48, seed=21)
    a, b = a.astype(dtype), b.astype(dtype)

    @jax.jit
    def solve(rhs):
        if solver_name == "gmres":
            return gmres(lambda v: a @ v, rhs, restart=18, rtol=solve_rtol)
        return gcrot(lambda v: a @ v, rhs, m=18, k=5, rtol=solve_rtol)

    solution = solve(b)
    reference = np.linalg.solve(np.asarray(a), np.asarray(b))
    relative_error = np.linalg.norm(np.asarray(solution.x) - reference) / np.linalg.norm(
        reference
    )
    assert bool(solution.converged)
    assert relative_error <= error_tolerance
    assert float(solution.residual_norm) <= 2.0 * solve_rtol * float(jnp.linalg.norm(b))


def test_complex_gcrot_recycle_preserves_accuracy():
    a0, b = random_complex_system(50, seed=22)
    perturbation, _ = random_complex_system(50, seed=23, spread=0.01)
    a1 = a0 + 0.01 * perturbation
    first = gcrot(lambda v: a0 @ v, b, m=15, k=5, rtol=1.0e-11)
    second = gcrot(
        lambda v: a1 @ v,
        b,
        m=15,
        k=5,
        rtol=1.0e-11,
        recycle=first.recycle,
    )
    reference = np.linalg.solve(np.asarray(a1), np.asarray(b))
    assert bool(first.converged) and bool(second.converged)
    assert np.asarray(second.x) == pytest.approx(reference, rel=1.0e-9, abs=1.0e-9)


def test_complex_pytree_gmres_matches_dense_under_jit():
    matrix, rhs = random_complex_system(7, seed=24)
    blocks = ((matrix[:4, :4], matrix[:4, 4:]),
              (matrix[4:, :4], matrix[4:, 4:]))
    tree_rhs = (rhs[:4].reshape(2, 2), {"field": rhs[4:]})

    def matvec(value):
        distribution, fields = value
        x = distribution.reshape(-1)
        field = fields["field"]
        return (
            (blocks[0][0] @ x + blocks[0][1] @ field).reshape(2, 2),
            {"field": blocks[1][0] @ x + blocks[1][1] @ field},
        )

    solution = jax.jit(
        lambda value: gmres(matvec, value, restart=5, max_restarts=4, rtol=1e-11)
    )(tree_rhs)
    flat_solution = np.concatenate(
        [np.asarray(solution.x[0]).reshape(-1), np.asarray(solution.x[1]["field"])]
    )
    reference = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    assert bool(solution.converged)
    assert flat_solution == pytest.approx(reference, rel=1e-9, abs=1e-9)
    assert float(solution.residual_norm) <= 1e-11 * float(jnp.linalg.norm(rhs))


def test_pytree_gmres_validates_tree_structure_and_dtype():
    rhs = (jnp.ones(2), jnp.ones(1))
    with pytest.raises(ValueError, match="identical pytree structure"):
        gmres(lambda x: x, rhs, x0={"different": jnp.ones(3)})
    with pytest.raises(ValueError, match="common inexact dtype"):
        gmres(lambda x: x, (jnp.ones(2), jnp.ones(1, dtype=jnp.complex64)))


def test_scalar_gmres():
    solution = gmres(lambda x: 2 * x, jnp.asarray(4.0), rtol=1.0e-12)
    assert np.asarray(solution.x) == pytest.approx(2.0)


def test_exact_inverse_preconditioner():
    a, b = random_system(80, seed=3)
    a_inv = jnp.asarray(np.linalg.inv(np.asarray(a)))
    sol = gmres(
        lambda v: a @ v, b, precond=lambda v: a_inv @ v, rtol=1e-10, restart=30
    )
    assert bool(sol.converged)
    assert int(sol.iterations) <= 2


def test_preconditioning_iteration_counts():
    n = 256
    dense = advection_diffusion(n)
    rng = np.random.default_rng(4)
    b = jnp.asarray(rng.standard_normal(n))
    a = jnp.asarray(dense)
    matvec = lambda v: a @ v  # noqa: E731

    plain = gmres(matvec, b, restart=30, rtol=1e-8, max_restarts=10)
    assert (not bool(plain.converged)) or int(plain.iterations) > 100

    # Exact inverse of the tridiagonal part (drops only the two periodic
    # corner entries, so the preconditioned operator is I + rank-2).
    tridiag = np.triu(np.tril(dense, 1), -1)
    m_inv = jnp.asarray(scipy.linalg.inv(tridiag))
    pre = gmres(
        matvec, b, precond=lambda v: m_inv @ v, restart=30, rtol=1e-8
    )
    assert bool(pre.converged)
    assert int(pre.iterations) < 30

    x_ref = np.linalg.solve(dense, np.asarray(b))
    err = np.linalg.norm(np.asarray(pre.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-7


def test_gcrot_recycling_saves_iterations():
    n = 120
    rng = np.random.default_rng(7)
    # A handful of small eigenvalues limits restarted GMRES; recycling the
    # corresponding directions across the sequence should pay off.
    d = np.concatenate([np.full(5, 0.02), rng.uniform(1.0, 2.0, n - 5)])
    a0 = np.diag(d) + 0.05 * rng.standard_normal((n, n)) / np.sqrt(n)
    b_mat = rng.standard_normal((n, n)) / np.sqrt(n)
    mats = [jnp.asarray(a0 + i * 0.01 * b_mat) for i in range(5)]
    rhs = jnp.asarray(rng.standard_normal(n))

    def solve(a, recycle):
        return gcrot(
            lambda v: a @ v, rhs, m=20, k=10, rtol=1e-10, recycle=recycle
        )

    cold_iters = []
    for a in mats:
        sol = solve(a, None)
        assert bool(sol.converged)
        cold_iters.append(int(sol.iterations))

    warm_iters = []
    recycle = None
    for a in mats:
        sol = solve(a, recycle)
        assert bool(sol.converged)
        recycle = sol.recycle
        warm_iters.append(int(sol.iterations))

    # Solve 1 is identical (no recycle yet); solves 2..5 must win in total.
    assert warm_iters[0] == cold_iters[0]
    assert sum(warm_iters[1:]) < sum(cold_iters[1:])

    # And the warm-started solutions are still correct.
    x_ref = np.linalg.solve(np.asarray(mats[-1]), np.asarray(rhs))
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-8


def test_gcrot_matches_dense_without_recycle():
    a, b = random_system(90, seed=9)
    sol = gcrot(lambda v: a @ v, b, m=15, k=5, rtol=1e-10)
    assert bool(sol.converged)
    c, u = sol.recycle
    assert c.shape == (90, 5) and u.shape == (90, 5)
    x_ref = np.linalg.solve(np.asarray(a), np.asarray(b))
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-8


def test_gmres_under_jit():
    a, b = random_system(60, seed=11)
    matvec = lambda v: a @ v  # noqa: E731

    @jax.jit
    def solve(rhs):
        return gmres(matvec, rhs, rtol=1e-10, restart=25)

    sol_jit = solve(b)
    sol_ref = gmres(matvec, b, rtol=1e-10, restart=25)
    assert bool(sol_jit.converged)
    assert int(sol_jit.iterations) == int(sol_ref.iterations)
    assert np.allclose(np.asarray(sol_jit.x), np.asarray(sol_ref.x), atol=1e-12)


def test_gcrot_under_jit():
    a, b = random_system(60, seed=12)
    matvec = lambda v: a @ v  # noqa: E731

    @jax.jit
    def solve(rhs, recycle):
        return gcrot(matvec, rhs, m=12, k=4, rtol=1e-10, recycle=recycle)

    sol = solve(b, (jnp.zeros((60, 4)), jnp.zeros((60, 4))))
    sol_ref = gcrot(matvec, b, m=12, k=4, rtol=1e-10)
    assert bool(sol.converged)
    assert np.allclose(np.asarray(sol.x), np.asarray(sol_ref.x), atol=1e-10)


def test_multi_restart_convergence():
    a, b = random_system(100, seed=5, spread=0.7)
    sol = gmres(lambda v: a @ v, b, restart=10, rtol=1e-10, max_restarts=100)
    assert bool(sol.converged)
    assert int(sol.iterations) > 10  # needed more than one cycle
    x_ref = np.linalg.solve(np.asarray(a), np.asarray(b))
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-8
