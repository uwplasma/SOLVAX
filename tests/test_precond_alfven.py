"""Gates for the exact Alfvén wave-block preconditioner."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import solvax

jax.config.update("jax_enable_x64", True)


def make_system(theta_scale: float, seed: int = 0):
    """Build the linearized implicit-MHD block system for random modes."""
    key_a, key_b, key_r = jax.random.split(jax.random.PRNGKey(seed), 3)
    modes = (24,)
    k_squared = jax.random.uniform(key_a, modes, minval=0.0, maxval=64.0)
    alpha_v = 1.0 + 1.0e-3 * k_squared
    alpha_b = 1.0 + 2.0e-3 * k_squared
    theta = theta_scale * jax.random.uniform(key_b, modes, minval=0.1, maxval=1.0)

    def matvec(x):
        v, b = x
        return (
            alpha_v * v - 1j * theta * b,
            -1j * theta * v + alpha_b * b,
        )

    rhs = (
        jax.random.normal(key_r, modes) + 1j * jax.random.normal(key_a, modes),
        jax.random.normal(key_b, modes) + 1j * jax.random.normal(key_r, modes),
    )
    precond = solvax.alfven_block(alpha_v, alpha_b, theta)
    return matvec, rhs, precond


def test_alfven_block_is_the_exact_inverse() -> None:
    matvec, rhs, precond = make_system(theta_scale=10.0)
    solved = precond(rhs)
    reconstructed = matvec(solved)
    error = max(
        float(jnp.max(jnp.abs(reconstructed[0] - rhs[0]))),
        float(jnp.max(jnp.abs(reconstructed[1] - rhs[1]))),
    )
    assert error < 1.0e-12


def test_preconditioned_gmres_converges_in_one_iteration() -> None:
    matvec, rhs, precond = make_system(theta_scale=10.0)
    solution = solvax.gmres(matvec, rhs, precond=precond, rtol=1.0e-12)
    assert bool(solution.converged)
    assert int(solution.iterations) <= 1


def test_iteration_count_plateaus_across_wave_strength() -> None:
    """The preconditioned count must not grow with B0 dt k_parallel."""
    counts = []
    for theta_scale in (1.0, 10.0, 100.0):
        matvec, rhs, precond = make_system(theta_scale=theta_scale)
        solution = solvax.gmres(matvec, rhs, precond=precond, rtol=1.0e-10)
        assert bool(solution.converged)
        counts.append(int(solution.iterations))
    assert max(counts) - min(counts) <= 2


def test_unpreconditioned_baseline_degrades_with_wave_strength() -> None:
    """Documents why the block preconditioner exists."""
    matvec_weak, rhs, _ = make_system(theta_scale=1.0)
    matvec_strong, _, _ = make_system(theta_scale=100.0)
    weak = solvax.gmres(matvec_weak, rhs, rtol=1.0e-10)
    strong = solvax.gmres(matvec_strong, rhs, rtol=1.0e-10)
    assert int(strong.iterations) > int(weak.iterations)
