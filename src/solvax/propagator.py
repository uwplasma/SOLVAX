"""Adaptive policies for extremal eigenmodes exposed by time propagators."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class RK4Timestep(NamedTuple):
    """A stability-limited RK4 step inferred from an Arnoldi spectral sketch."""

    dt: float
    stability_boundary: float
    spectral_radius: float
    projected_dimension: int
    probe_count: int
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


class PropagatorEigenSolution(NamedTuple):
    """Continuous eigenpairs extracted from one full-operator RK4 subspace."""

    eigenvalues: jax.Array
    eigenvectors: jax.Array
    residuals: jax.Array
    converged: jax.Array
    operator_applications: int


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
        raise ValueError(f"dimension must lie between one and the operator size, got {dimension}")
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
    return np.linalg.eigvals(np.asarray(projected[:dimension, :dimension]))


def _rk4_amplification(z: np.ndarray | complex) -> np.ndarray:
    z = np.asarray(z)
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def propagator_eigenpairs(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    dt: float,
    steps: int,
    krylov_dim: int = 24,
    candidates: int = 2,
    tol: float = 1.0e-9,
) -> PropagatorEigenSolution:
    """Return leading-growth continuous pairs from one compiled RK4 subspace.

    The full Arnoldi construction is compiled as one operation, including the
    RK4 loops, so this is suitable for an application-supplied adjoint action.
    Candidate ordering uses propagator magnitude, but values and convergence
    always come from the continuous operator.
    """

    size = int(v0.size)
    if not 1 <= candidates <= krylov_dim < size:
        raise ValueError("require 1 <= candidates <= krylov_dim < operator size")
    if dt <= 0.0 or steps < 1 or tol <= 0.0:
        raise ValueError("dt, steps, and tol must be positive")
    shape = v0.shape
    dtype = jnp.result_type(v0, jnp.complex64)
    initial = jnp.asarray(v0, dtype=dtype)
    dt_value = jnp.asarray(dt, dtype=jnp.real(initial).dtype)

    def rk4_step(state):
        first = apply(state)
        second = apply(state + 0.5 * dt_value * first)
        third = apply(state + 0.5 * dt_value * second)
        fourth = apply(state + dt_value * third)
        return state + (dt_value / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)

    def filtered(state):
        return jax.lax.fori_loop(
            0,
            steps,
            lambda _index, current: rk4_step(current),
            state,
        )

    @jax.jit
    def arnoldi(start):
        norm = jnp.linalg.norm(start)
        basis = jnp.zeros((krylov_dim + 1, *shape), dtype=dtype)
        basis = basis.at[0].set(start / jnp.where(norm > 0.0, norm, 1.0))
        projected = jnp.zeros((krylov_dim + 1, krylov_dim), dtype=dtype)

        def extend(column, carry):
            vectors, quotient = carry
            work = filtered(vectors[column])
            operator_scale = jnp.linalg.norm(work)

            def orthogonalize(index, inner):
                candidate, matrix = inner
                coefficient = jnp.vdot(vectors[index], candidate)
                candidate = candidate - coefficient * vectors[index]
                matrix = matrix.at[index, column].add(coefficient)
                return candidate, matrix

            work, quotient = jax.lax.fori_loop(
                0,
                column + 1,
                orthogonalize,
                (work, quotient),
            )
            work, quotient = jax.lax.fori_loop(
                0,
                column + 1,
                orthogonalize,
                (work, quotient),
            )
            next_norm = jnp.linalg.norm(work)
            real_dtype = jnp.real(jnp.empty((), dtype=dtype)).dtype
            resolved = next_norm > 10.0 * jnp.finfo(real_dtype).eps * jnp.maximum(
                operator_scale, 1.0
            )
            quotient = quotient.at[column + 1, column].set(jnp.where(resolved, next_norm, 0.0))
            next_vector = jnp.where(
                resolved,
                work / jnp.where(resolved, next_norm, 1.0),
                jnp.zeros_like(work),
            )
            vectors = vectors.at[column + 1].set(next_vector)
            return vectors, quotient

        return jax.lax.fori_loop(
            0,
            krylov_dim,
            extend,
            (basis, projected),
        )

    basis, projected = arnoldi(initial)
    propagator_values, coefficients = jnp.linalg.eig(projected[:krylov_dim, :krylov_dim])
    indices = jnp.argsort(jnp.abs(propagator_values))[-candidates:][::-1]
    lifted = jnp.tensordot(
        coefficients[:, indices].T,
        basis[:krylov_dim],
        axes=1,
    )
    flattened = lifted.reshape((candidates, -1))
    vector_norms = jnp.linalg.norm(flattened, axis=1)
    vectors = (flattened / jnp.where(vector_norms > 0.0, vector_norms, 1.0)[:, None]).reshape(
        (candidates, *shape)
    )

    @jax.jit
    def certify(vector):
        image = apply(vector)
        denominator = jnp.vdot(vector, vector)
        value = jnp.vdot(vector, image) / jnp.where(
            denominator != 0.0,
            denominator,
            1.0 + 0.0j,
        )
        residual = jnp.linalg.norm(image - value * vector)
        residual = residual / jnp.maximum(
            jnp.abs(value) * jnp.linalg.norm(vector),
            jnp.finfo(jnp.real(vector).dtype).tiny,
        )
        return value, residual

    values, residuals = jax.vmap(certify)(vectors)
    return PropagatorEigenSolution(
        eigenvalues=values,
        eigenvectors=vectors,
        residuals=residuals,
        converged=residuals < tol,
        operator_applications=4 * steps * krylov_dim + candidates,
    )


def estimate_rk4_timestep(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    dimension: int = 12,
    probe_count: int = 2,
    safety: float = 0.9,
    max_dt: float = np.inf,
    bisection_iterations: int = 48,
) -> RK4Timestep:
    """Choose an RK4 step that does not numerically amplify sketched modes.

    Arnoldi is used only to find inexpensive peripheral spectral sketches. The
    caller's seed is supplemented by deterministic broadband probes because a
    recycled eigenvector can be nearly invariant and blind to stability-limiting
    modes. The boundary is evaluated against the RK4 polynomial itself,
    including each Ritz value's complex angle, rather than assuming a purely
    imaginary spectrum. Positive physical growth is allowed; artificial growth
    beyond ``exp(dt * max(Re(lambda), 0))`` is not.
    """

    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie in (0, 1)")
    if probe_count < 1:
        raise ValueError("probe_count must be positive")
    if max_dt <= 0.0:
        raise ValueError("max_dt must be positive")
    seeds = [v0]
    real_dtype = jnp.real(v0).dtype
    for probe_index in range(1, probe_count):
        key_real = jax.random.PRNGKey(2 * probe_index - 1)
        probe = jax.random.normal(key_real, v0.shape, dtype=real_dtype)
        if jnp.iscomplexobj(v0):
            key_imag = jax.random.PRNGKey(2 * probe_index)
            probe = probe + 1j * jax.random.normal(
                key_imag,
                v0.shape,
                dtype=real_dtype,
            )
        seeds.append(jnp.asarray(probe, dtype=v0.dtype))
    values = np.concatenate([_arnoldi_spectrum(apply, seed, dimension) for seed in seeds])
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
        probe_count=probe_count,
        operator_applications=dimension * probe_count,
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
    operator_applications = base_operator_applications + restarts * (
        4 * applications_per_restart * filter_steps + 2
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
    "PropagatorEigenSolution",
    "RK4Timestep",
    "adaptive_eigenpair",
    "estimate_rk4_timestep",
    "propagator_eigenpairs",
]
