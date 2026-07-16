"""Tests for solvax.elliptic: the spectral Fourier--Helmholtz elliptic solve.

Pins the per-mode tridiagonal solve against a dense reference built by hand,
checks a manufactured-solution round trip, and verifies the solve is
differentiable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    build_fourier_helmholtz_operator,
    solve_fourier_helmholtz,
)

jax.config.update("jax_enable_x64", True)


def _geometry(nx, nz, seed=0):
    rng = np.random.default_rng(seed)
    dx = jnp.full((nx,), 0.1, dtype=jnp.float64)
    dz = jnp.full((nx,), 0.05, dtype=jnp.float64)
    g11 = jnp.asarray(1.0 + 0.3 * rng.random(nx), dtype=jnp.float64)
    g33 = jnp.asarray(0.7 + 0.2 * rng.random(nx), dtype=jnp.float64)
    rhs_scale = jnp.ones((nx,), dtype=jnp.float64)
    return dx, dz, g11, g33, rhs_scale


def test_matches_dense_per_mode_reference():
    nx, nz = 8, 16
    dx, dz, g11, g33, rhs_scale = _geometry(nx, nz)
    operator = build_fourier_helmholtz_operator(
        dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=nz
    )
    rng = np.random.default_rng(1)
    rhs = jnp.asarray(rng.standard_normal((nx, nz)), dtype=jnp.float64)

    solution = np.asarray(solve_fourier_helmholtz(rhs, operator=operator))

    # Dense reference: FFT the rhs, solve each mode's tridiagonal system densely.
    rhs_hat = np.fft.rfft(np.asarray(rhs), axis=-1)
    lower = np.asarray(operator.lower_diagonals)
    diag = np.asarray(operator.diagonals)
    upper = np.asarray(operator.upper_diagonals)
    ref_hat = np.zeros_like(rhs_hat)
    for mode in range(rhs_hat.shape[1]):
        matrix = np.diag(diag[mode]) + np.diag(lower[mode][1:], -1) + np.diag(upper[mode][:-1], 1)
        ref_hat[:, mode] = np.linalg.solve(matrix, rhs_hat[:, mode])
    reference = np.fft.irfft(ref_hat, n=nz, axis=-1)

    assert np.allclose(solution, reference, rtol=1e-10, atol=1e-12)


def test_operator_round_trip_reproduces_rhs():
    # Apply the assembled operator to the computed solution and recover the rhs
    # (in Fourier space, mode by mode).
    nx, nz = 6, 12
    dx, dz, g11, g33, rhs_scale = _geometry(nx, nz, seed=2)
    operator = build_fourier_helmholtz_operator(
        dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=nz
    )
    rng = np.random.default_rng(3)
    rhs = jnp.asarray(rng.standard_normal((nx, nz)), dtype=jnp.float64)
    solution = solve_fourier_helmholtz(rhs, operator=operator)

    sol_hat = np.fft.rfft(np.asarray(solution), axis=-1)
    lower = np.asarray(operator.lower_diagonals)
    diag = np.asarray(operator.diagonals)
    upper = np.asarray(operator.upper_diagonals)
    applied = np.zeros_like(sol_hat)
    for mode in range(sol_hat.shape[1]):
        matrix = np.diag(diag[mode]) + np.diag(lower[mode][1:], -1) + np.diag(upper[mode][:-1], 1)
        applied[:, mode] = matrix @ sol_hat[:, mode]
    recovered = np.fft.irfft(applied, n=nz, axis=-1)
    expected = np.asarray(rhs) * np.asarray(rhs_scale)[:, None]
    assert np.allclose(recovered, expected, rtol=1e-9, atol=1e-11)


def test_solve_is_jit_and_grad_transparent():
    nx, nz = 6, 12
    dx, dz, g11, g33, rhs_scale = _geometry(nx, nz, seed=4)
    operator = build_fourier_helmholtz_operator(
        dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=nz
    )
    rng = np.random.default_rng(5)
    rhs = jnp.asarray(rng.standard_normal((nx, nz)), dtype=jnp.float64)

    jitted = jax.jit(lambda r: solve_fourier_helmholtz(r, operator=operator))
    assert np.allclose(
        np.asarray(jitted(rhs)), np.asarray(solve_fourier_helmholtz(rhs, operator=operator))
    )

    def objective(r):
        return jnp.sum(jnp.square(solve_fourier_helmholtz(r, operator=operator)))

    grad = jax.grad(objective)(rhs)
    step = 1e-4
    idx = (2, 3)
    perturbed = rhs.at[idx].add(step)
    fd = (float(objective(perturbed)) - float(objective(rhs.at[idx].add(-step)))) / (2 * step)
    assert float(grad[idx]) == pytest.approx(fd, rel=1e-5, abs=1e-8)
