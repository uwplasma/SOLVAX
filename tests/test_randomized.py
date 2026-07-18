"""Tests for solvax.randomized: Nystrom preconditioning for SPD systems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import nystrom_preconditioner, pcg, pcg_linear_solve

jax.config.update("jax_enable_x64", True)


def decay_spectrum_system(n=300, head=30, seed=0):
    """SPD operator with a decaying head and a flat tail: Nystrom's regime."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.concatenate([100.0 * 0.5 ** np.arange(head), 1e-2 * np.ones(n - head)])
    matrix = jnp.asarray((q * lam) @ q.T)
    rhs = jnp.asarray(rng.standard_normal(n))
    return matrix, rhs


def test_nystrom_cuts_pcg_iterations():
    matrix, rhs = decay_spectrum_system()
    n, mu = rhs.shape[0], 1e-2
    system = lambda v: matrix @ v + mu * v  # noqa: E731
    plain = pcg(system, rhs, rtol=1e-10, max_steps=1000)
    precond = nystrom_preconditioner(
        lambda v: matrix @ v, n, 50, jax.random.PRNGKey(0), mu=mu
    )
    accelerated = pcg(system, rhs, precond=precond, rtol=1e-10, max_steps=1000)
    assert bool(plain.converged) and bool(accelerated.converged)
    assert int(accelerated.iterations) <= int(plain.iterations) // 2


def test_nystrom_action_is_symmetric_positive_definite():
    matrix, _ = decay_spectrum_system(n=120, head=15, seed=1)
    precond = nystrom_preconditioner(
        lambda v: matrix @ v, 120, 25, jax.random.PRNGKey(1), mu=1e-3
    )
    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.standard_normal(120))
    v = jnp.asarray(rng.standard_normal(120))
    assert np.isclose(float(u @ precond(v)), float(v @ precond(u)), rtol=1e-12)
    assert float(v @ precond(v)) > 0.0


def test_nystrom_is_deterministic_under_fixed_key():
    matrix, rhs = decay_spectrum_system(n=100, head=10, seed=3)
    build = lambda: nystrom_preconditioner(  # noqa: E731
        lambda v: matrix @ v, 100, 20, jax.random.PRNGKey(7), mu=1e-3
    )
    assert np.allclose(np.asarray(build()(rhs)), np.asarray(build()(rhs)), atol=0.0)


def test_nystrom_gradient_matches_finite_differences():
    matrix, rhs = decay_spectrum_system(n=150, head=15, seed=4)
    n, mu = 150, 1e-2

    def loss(scale):
        precond = nystrom_preconditioner(
            lambda v: scale * (matrix @ v), n, 30, jax.random.PRNGKey(0), mu=mu
        )
        solution = pcg_linear_solve(
            lambda v: scale * (matrix @ v) + mu * v, rhs,
            precond=precond, rtol=1e-11, max_steps=600,
        )
        return jnp.sum(solution.x ** 2)

    gradient = jax.grad(loss)(1.0)
    eps = 1e-6
    finite = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    assert np.isclose(float(gradient), float(finite), rtol=1e-4)


def test_nystrom_rejects_bad_rank():
    with pytest.raises(ValueError, match="rank"):
        nystrom_preconditioner(lambda v: v, 10, 0, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="rank"):
        nystrom_preconditioner(lambda v: v, 10, 11, jax.random.PRNGKey(0))
