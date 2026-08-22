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
    periodic_poisson_eigenvalues,
    solve_fourier_helmholtz,
    solve_periodic_poisson,
    solve_periodic_poisson_spectral,
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


def test_manufactured_cell_centered_dirichlet_mode():
    """Recover an independent eigenmode of the documented endpoint closure."""
    nx, nz = 12, 18
    length_x = 1.7
    length_z = 2.3
    dx_value = length_x / nx
    dz_value = length_z / nz
    dx = jnp.full((nx,), dx_value, dtype=jnp.float64)
    dz = jnp.full((nx,), dz_value, dtype=jnp.float64)
    coefficients = jnp.ones((nx,), dtype=jnp.float64)
    operator = build_fourier_helmholtz_operator(
        dx=dx,
        dz=dz,
        g11=coefficients,
        g33=coefficients,
        rhs_scale=coefficients,
        nz=nz,
    )

    x = (jnp.arange(nx, dtype=jnp.float64) + 0.5) * dx_value
    z = jnp.arange(nz, dtype=jnp.float64) * dz_value
    bounded_mode = 2
    periodic_mode = 3
    solution = jnp.sin(bounded_mode * jnp.pi * x[:, None] / length_x) * jnp.cos(
        2.0 * jnp.pi * periodic_mode * z[None, :] / length_z
    )
    bounded_eigenvalue = -4.0 * jnp.sin(bounded_mode * jnp.pi / (2.0 * nx)) ** 2 / dx_value**2
    periodic_eigenvalue = -((2.0 * jnp.pi * periodic_mode / length_z) ** 2)
    rhs = (bounded_eigenvalue + periodic_eigenvalue) * solution

    recovered = solve_fourier_helmholtz(rhs, operator=operator)

    assert np.allclose(recovered, solution, rtol=1e-10, atol=1e-11)


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


@pytest.mark.parametrize(
    "shape, spacing, mode", [((18,), 0.2, (3,)), ((12, 16), (0.3, 0.2), (2, 3))]
)
def test_periodic_poisson_recovers_fourier_mode(shape, spacing, mode):
    spacings = (spacing,) * len(shape) if isinstance(spacing, float) else spacing
    coordinates = [jnp.arange(size) * step for size, step in zip(shape, spacings, strict=True)]
    grids = jnp.meshgrid(*coordinates, indexing="ij")
    solution = jnp.ones(shape)
    eigenvalue = 0.0
    for grid, size, step, number in zip(grids, shape, spacings, mode, strict=True):
        wave_number = 2.0 * jnp.pi * number / (size * step)
        solution = solution * jnp.cos(wave_number * grid)
        eigenvalue += wave_number**2

    recovered = solve_periodic_poisson(eigenvalue * solution, spacing=spacing)

    assert recovered == pytest.approx(solution, rel=1.0e-11, abs=1.0e-11)


def test_periodic_poisson_projects_rhs_mean_and_sets_solution_mean():
    rhs = jnp.arange(35.0).reshape(5, 7)
    solution = solve_periodic_poisson(rhs, spacing=(0.2, 0.4), mean=2.5)
    zero_mean_solution = solution - jnp.mean(solution)
    symbol = periodic_poisson_eigenvalues(rhs.shape, (0.2, 0.4))
    applied = jnp.fft.ifftn(symbol * jnp.fft.fftn(zero_mean_solution)).real

    assert jnp.mean(solution) == pytest.approx(2.5)
    assert applied == pytest.approx(rhs - jnp.mean(rhs), rel=1.0e-10, abs=1.0e-10)


def test_periodic_poisson_spectral_api_is_complex_jit_and_grad_transparent():
    rhs = jnp.sin(jnp.arange(24.0).reshape(4, 6)) + 1j * jnp.cos(jnp.arange(24.0).reshape(4, 6))
    symbol = periodic_poisson_eigenvalues(rhs.shape, (0.5, 0.25))
    solve_hat = jax.jit(
        lambda values: solve_periodic_poisson_spectral(
            jnp.fft.fftn(values), eigenvalues=symbol, mean=1.0 - 0.5j
        )
    )
    spectrum = solve_hat(rhs)
    physical = jnp.fft.ifftn(spectrum)

    assert jnp.mean(physical) == pytest.approx(1.0 - 0.5j)
    assert physical == pytest.approx(
        solve_periodic_poisson(rhs, spacing=(0.5, 0.25), mean=1.0 - 0.5j)
    )
    gradient = jax.grad(lambda values: jnp.sum(solve_periodic_poisson(values) ** 2))(rhs.real)
    assert jnp.all(jnp.isfinite(gradient))


def test_periodic_poisson_spacing_is_differentiable_under_jit():
    size, mode = 24, 3
    rhs = jnp.sin(2.0 * jnp.pi * mode * jnp.arange(size) / size)

    def objective(spacing):
        return jnp.sum(solve_periodic_poisson(rhs, spacing=spacing) ** 2)

    spacing = jnp.asarray(0.2)
    value, derivative = jax.jit(jax.value_and_grad(objective))(spacing)
    assert derivative == pytest.approx(4.0 * value / spacing, rel=2.0e-12)
    invalid = jax.jit(periodic_poisson_eigenvalues, static_argnums=0)((size,), -spacing)
    assert jnp.isnan(invalid).all()


@pytest.mark.parametrize(
    "action, message",
    [
        (lambda: periodic_poisson_eigenvalues((), 1.0), "at least two"),
        (lambda: periodic_poisson_eigenvalues((4, 1), 1.0), "at least two"),
        (lambda: periodic_poisson_eigenvalues((4, 5), (1.0,)), "one positive"),
        (lambda: periodic_poisson_eigenvalues((4,), 0.0), "one positive"),
        (
            lambda: solve_periodic_poisson_spectral(
                jnp.zeros((3, 4)), eigenvalues=jnp.zeros((4, 3))
            ),
            "identical shapes",
        ),
    ],
)
def test_periodic_poisson_rejects_invalid_geometry(action, message):
    with pytest.raises(ValueError, match=message):
        action()
