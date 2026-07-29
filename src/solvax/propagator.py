"""Adaptive policies for extremal eigenmodes exposed by time propagators."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


class RK4Timestep(NamedTuple):
    """A stability-limited RK4 step inferred from an Arnoldi spectral sketch."""

    dt: float
    stability_boundary: float
    spectral_radius: float
    projected_dimension: int
    operator_applications: int


class AdaptiveEigenSolution(NamedTuple):
    """A residual-certified eigenpair and the work used to isolate it."""

    eigenvalue: jax.Array
    eigenvector: jax.Array
    residual: jax.Array
    converged: bool
    stable: bool
    restarts: int
    operator_applications: int
    filter_dt: float
    filter_steps: int
    filter_horizon: float
    filter_growth_defect: float


def _flatten(vector: jax.Array) -> jax.Array:
    return jnp.reshape(vector, (-1,))


def _arnoldi_spectrum(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    dimension: int,
) -> np.ndarray:
    """Estimate the peripheral spectrum with a two-pass Arnoldi sketch."""

    shape = v0.shape
    size = v0.size
    if not 1 < dimension < size:
        raise ValueError(
            f"dimension must lie between one and the operator size, got {dimension}"
        )
    dtype = jnp.result_type(v0, jnp.complex64)
    vector = _flatten(jnp.asarray(v0, dtype=dtype))
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("v0 must be finite and nonzero")
    basis = jnp.zeros((dimension + 1, size), dtype=dtype).at[0].set(vector / norm)
    projected = jnp.zeros((dimension + 1, dimension), dtype=dtype)
    for column in range(dimension):
        work = _flatten(apply(jnp.reshape(basis[column], shape)))
        coefficients = basis[: column + 1].conj() @ work
        work = work - coefficients @ basis[: column + 1]
        correction = basis[: column + 1].conj() @ work
        work = work - correction @ basis[: column + 1]
        coefficients = coefficients + correction
        projected = projected.at[: column + 1, column].set(coefficients)
        norm = jnp.linalg.norm(work)
        if column + 1 < dimension:
            projected = projected.at[column + 1, column].set(norm)
        basis = basis.at[column + 1].set(
            jnp.where(norm > 0.0, work / jnp.where(norm > 0.0, norm, 1.0), 0.0)
        )
    return scipy.linalg.eigvals(np.asarray(projected[:dimension, :dimension]))


def _rk4_amplification(z: np.ndarray | complex) -> np.ndarray:
    z = np.asarray(z)
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def estimate_rk4_timestep(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    dimension: int = 12,
    safety: float = 0.9,
    max_dt: float = np.inf,
    bisection_iterations: int = 48,
) -> RK4Timestep:
    """Choose an RK4 step that does not numerically amplify sketched modes.

    Arnoldi is used only to find the inexpensive peripheral spectral sketch.
    The boundary is then evaluated against the RK4 polynomial itself, including
    each Ritz value's complex angle, rather than assuming a purely imaginary
    spectrum. Positive physical growth is allowed; artificial growth beyond
    ``exp(dt * max(Re(lambda), 0))`` is not.
    """

    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie in (0, 1)")
    if max_dt <= 0.0:
        raise ValueError("max_dt must be positive")
    values = _arnoldi_spectrum(apply, v0, dimension)
    radius = float(np.max(np.abs(values)))
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("Arnoldi sketch did not produce a finite spectral radius")
    lower = 0.0
    upper = min(float(max_dt), 4.0 / radius)
    for _ in range(max(int(bisection_iterations), 1)):
        trial = 0.5 * (lower + upper)
        amplification = np.abs(_rk4_amplification(trial * values))
        physical = np.exp(trial * np.maximum(values.real, 0.0))
        if np.all(amplification <= physical * (1.0 + 1.0e-7)):
            lower = trial
        else:
            upper = trial
    return RK4Timestep(
        dt=safety * lower,
        stability_boundary=lower,
        spectral_radius=radius,
        projected_dimension=dimension,
        operator_applications=dimension,
    )


def adaptive_eigenpair(
    apply: Callable[[jax.Array], jax.Array],
    restart_once: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    v0: jax.Array,
    *,
    tol: float,
    max_restarts: int,
    filter_dt: float,
    filter_steps: int,
    applications_per_restart: int,
    base_operator_applications: int = 0,
    stability_atol: float = 1.0e-7,
    stability_rtol: float = 1.0e-6,
) -> AdaptiveEigenSolution:
    """Restart until the original residual passes or RK4 stability is suspect."""

    if tol <= 0.0 or max_restarts < 1:
        raise ValueError("tol must be positive and max_restarts must be positive")
    if filter_dt <= 0.0 or filter_steps < 1:
        raise ValueError("filter_dt and filter_steps must be positive")
    vector = v0
    value = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=v0.dtype)
    residual = jnp.asarray(jnp.inf, dtype=jnp.real(v0).dtype)
    defect = np.inf
    stable = False
    restarts = 0
    for restart_index in range(1, max_restarts + 1):
        restarts = restart_index
        _projected_value, vector = restart_once(vector)
        image = apply(vector)
        denominator = jnp.vdot(vector, vector)
        value = jnp.vdot(vector, image) / denominator
        residual = jnp.linalg.norm(image - value * vector)
        residual = residual / jnp.maximum(
            jnp.abs(value) * jnp.linalg.norm(vector),
            jnp.finfo(jnp.real(vector).dtype).tiny,
        )
        scalar = complex(np.asarray(value))
        amplification = abs(complex(_rk4_amplification(filter_dt * scalar)))
        filter_growth = np.log(max(amplification, np.finfo(float).tiny)) / filter_dt
        defect = filter_growth - scalar.real
        stability_limit = stability_atol + stability_rtol * max(abs(scalar.real), 1.0)
        stable = bool(np.isfinite(defect) and defect <= stability_limit)
        if not stable or float(np.asarray(residual)) < tol:
            break
    converged = stable and float(np.asarray(residual)) < tol
    operator_applications = (
        base_operator_applications
        + restarts * (4 * applications_per_restart * filter_steps + 2)
    )
    return AdaptiveEigenSolution(
        eigenvalue=value,
        eigenvector=vector,
        residual=residual,
        converged=converged,
        stable=stable,
        restarts=restarts,
        operator_applications=operator_applications,
        filter_dt=filter_dt,
        filter_steps=filter_steps,
        filter_horizon=filter_dt * filter_steps,
        filter_growth_defect=defect,
    )


__all__ = [
    "AdaptiveEigenSolution",
    "RK4Timestep",
    "adaptive_eigenpair",
    "estimate_rk4_timestep",
]
