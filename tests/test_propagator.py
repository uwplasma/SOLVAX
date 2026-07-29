"""Tests for stability-limited, residual-driven propagator policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import adaptive_eigenpair, estimate_rk4_timestep

jax.config.update("jax_enable_x64", True)


def test_rk4_timestep_keeps_the_full_known_spectrum_stable() -> None:
    """A conservative Arnoldi sketch must not amplify omitted peripheral modes."""

    frequencies = np.linspace(-24.0, 24.0, 18)
    eigenvalues = jnp.asarray(-0.2 + 1j * frequencies)
    apply = jax.jit(lambda vector: eigenvalues * vector)
    estimate = estimate_rk4_timestep(
        apply,
        jnp.ones_like(eigenvalues),
        dimension=12,
        safety=0.7,
    )

    z = estimate.dt * np.asarray(eigenvalues)
    amplification = np.abs(1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    assert estimate.dt > 0.0
    assert estimate.probe_count == 2
    assert estimate.operator_applications == 24
    assert np.max(amplification) <= 1.0


def test_rk4_timestep_broadband_probe_catches_invariant_seed_blindness() -> None:
    """A recycled eigenmode must not hide a stability-limiting peripheral mode."""

    frequencies = np.linspace(-80.0, 80.0, 18)
    eigenvalues = jnp.asarray(-0.2 + 1j * frequencies)
    apply = jax.jit(lambda vector: eigenvalues * vector)
    invariant_seed = jnp.zeros_like(eigenvalues).at[8].set(1.0)
    estimate = estimate_rk4_timestep(
        apply,
        invariant_seed,
        dimension=12,
        safety=0.7,
    )

    z = estimate.dt * np.asarray(eigenvalues)
    amplification = np.abs(1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    assert estimate.spectral_radius > 70.0
    assert np.max(amplification) <= 1.0


def test_adaptive_eigenpair_stops_on_the_original_residual() -> None:
    """The horizon must stop early once the continuous eigenpair is certified."""

    eigenvalues = jnp.asarray([0.3 + 0.2j, 0.1 - 0.4j, -0.5 + 2.0j])
    weights = jnp.exp(20.0 * eigenvalues)
    apply = jax.jit(lambda vector: eigenvalues * vector)

    def restart_once(vector):
        filtered = weights * vector
        filtered = filtered / jnp.linalg.norm(filtered)
        return jnp.vdot(filtered, apply(filtered)), filtered

    solution = adaptive_eigenpair(
        apply,
        restart_once,
        jnp.ones_like(eigenvalues),
        tol=1.0e-6,
        max_restarts=6,
        filter_dt=0.05,
        filter_steps=400,
        applications_per_restart=1,
    )

    assert solution.converged
    assert solution.stable
    assert solution.restarts < 6
    assert complex(np.asarray(solution.eigenvalue)) == pytest.approx(
        np.asarray(eigenvalues[0]).item()
    )
    assert float(np.asarray(solution.residual)) < 1.0e-6


def test_adaptive_eigenpair_rejects_numerical_rk4_growth() -> None:
    """A small residual may not certify a mode made dominant by unstable RK4."""

    eigenvalues = jnp.asarray([0.2 + 0.1j, -0.1 + 3.0j])
    apply = jax.jit(lambda vector: eigenvalues * vector)

    def restart_once(_vector):
        selected = jnp.asarray([0.0, 1.0], dtype=eigenvalues.dtype)
        return eigenvalues[1], selected

    solution = adaptive_eigenpair(
        apply,
        restart_once,
        jnp.ones_like(eigenvalues),
        tol=1.0e-12,
        max_restarts=4,
        filter_dt=1.0,
        filter_steps=1,
        applications_per_restart=1,
    )

    assert not solution.converged
    assert not solution.stable
    assert solution.restarts == 1
