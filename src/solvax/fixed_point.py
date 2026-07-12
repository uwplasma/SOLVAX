"""JAX-native acceleration for expensive contractive fixed-point maps."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class FixedPointSolution(NamedTuple):
    """Result of a fixed-point solve."""

    x: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    relaxation: jax.Array


def aitken_relaxation(
    previous_residual: jax.Array,
    residual: jax.Array,
    previous_relaxation: jax.Array | float = 1.0,
    *,
    min_relaxation: float = 0.05,
    max_relaxation: float = 100.0,
) -> jax.Array:
    """Return one safeguarded vector Aitken relaxation update."""

    if min_relaxation <= 0.0 or max_relaxation < min_relaxation:
        raise ValueError("relaxation bounds must satisfy 0 < min <= max")
    previous_residual = jnp.asarray(previous_residual)
    residual = jnp.asarray(residual)
    if previous_residual.shape != residual.shape:
        raise ValueError("successive residuals must have identical shapes")
    real_dtype = jnp.real(residual).dtype
    omega = jnp.asarray(previous_relaxation, dtype=real_dtype)
    difference = residual - previous_residual
    denominator = jnp.vdot(difference, difference).real
    numerator = jnp.vdot(previous_residual, difference).real
    candidate = -omega * numerator / jnp.maximum(
        denominator, jnp.finfo(real_dtype).tiny
    )
    candidate = jnp.where(
        jnp.isfinite(candidate) & (denominator > jnp.finfo(real_dtype).eps),
        candidate,
        omega,
    )
    return jnp.clip(candidate, min_relaxation, max_relaxation)


def anderson_mixing(
    iterates: jax.Array,
    residuals: jax.Array,
    *,
    regularization: float = 1.0e-8,
    damping: float = 1.0,
) -> jax.Array:
    """Return a regularized Anderson update from a bounded fixed-point history.

    ``residuals[i]`` must equal ``mapping(iterates[i]) - iterates[i]``. The
    result is a residual-minimizing affine combination of the mapped points.
    Keeping map evaluation and stopping outside this primitive lets applications
    retain their own expensive subsystem solves and physical convergence gates.
    """

    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must lie in [0, 1]")
    iterates = jnp.asarray(iterates)
    residuals = jnp.asarray(residuals)
    if iterates.shape != residuals.shape:
        raise ValueError("iterate and residual histories must have identical shapes")
    if iterates.ndim < 1 or iterates.shape[0] < 1:
        raise ValueError("Anderson history must contain at least one entry")

    history_size = iterates.shape[0]
    flat_residuals = residuals.reshape((history_size, -1))
    gram = jnp.conj(flat_residuals) @ flat_residuals.T
    real_dtype = jnp.real(residuals).dtype
    scale = jnp.maximum(
        jnp.trace(gram).real / history_size,
        jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
    )
    stabilization = (regularization + jnp.finfo(real_dtype).eps) * scale
    system = gram + stabilization * jnp.eye(history_size, dtype=residuals.dtype)
    ones = jnp.ones((history_size,), dtype=residuals.dtype)
    weights = jnp.linalg.solve(system, ones)
    denominator = jnp.sum(weights)
    weights = weights / jnp.where(
        jnp.abs(denominator) > jnp.finfo(real_dtype).tiny,
        denominator,
        jnp.asarray(1.0, dtype=residuals.dtype),
    )
    fallback = jax.nn.one_hot(history_size - 1, history_size, dtype=residuals.dtype)
    weights = jnp.where(jnp.all(jnp.isfinite(weights)), weights, fallback)
    mapped = iterates + residuals
    accelerated = jnp.tensordot(weights, mapped, axes=(0, 0))
    return (1.0 - damping) * mapped[-1] + damping * accelerated


def aitken_fixed_point(
    mapping: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    *,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    max_steps: int = 100,
    min_relaxation: float = 0.05,
    max_relaxation: float = 100.0,
) -> FixedPointSolution:
    """Solve ``mapping(x) = x`` with safeguarded vector Aitken acceleration.

    The scalar relaxation is updated from successive fixed-point residuals and
    clipped to a caller-declared interval. The implementation uses
    :func:`jax.lax.while_loop`, so it is compatible with ``jit`` and ``vmap``.
    For derivatives of a converged root, wrap this primal solver with
    :func:`solvax.root_solve` instead of differentiating through its stopping
    iterations.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    if min_relaxation <= 0.0 or max_relaxation < min_relaxation:
        raise ValueError("relaxation bounds must satisfy 0 < min <= max")

    x0 = jnp.asarray(x0)
    residual0 = mapping(x0) - x0
    norm0 = jnp.linalg.norm(residual0)
    real_dtype = jnp.real(x0).dtype
    scale = jnp.maximum(jnp.linalg.norm(x0), jnp.asarray(1.0, dtype=real_dtype))
    tolerance = jnp.maximum(
        jnp.asarray(atol, dtype=real_dtype),
        jnp.asarray(rtol, dtype=real_dtype) * scale,
    )
    omega0 = jnp.asarray(1.0, dtype=real_dtype)
    previous0 = jnp.zeros_like(residual0)

    def condition(state):
        _, _, _, _, residual_norm, iterations = state
        return (iterations < max_steps) & (residual_norm > tolerance)

    def body(state):
        x, residual, previous, omega, _, iterations = state
        candidate = aitken_relaxation(
            previous,
            residual,
            omega,
            min_relaxation=min_relaxation,
            max_relaxation=max_relaxation,
        )
        next_omega = jnp.where(iterations > 0, candidate, omega)
        next_x = x + next_omega * residual
        next_residual = mapping(next_x) - next_x
        next_norm = jnp.linalg.norm(next_residual)
        return next_x, next_residual, residual, next_omega, next_norm, iterations + 1

    initial = (x0, residual0, previous0, omega0, norm0, jnp.int32(0))
    x, _, _, omega, _, iterations = lax.while_loop(condition, body, initial)
    residual_norm = jnp.linalg.norm(mapping(x) - x)
    return FixedPointSolution(
        x=x,
        residual_norm=residual_norm,
        iterations=iterations,
        converged=residual_norm <= tolerance,
        relaxation=omega,
    )
