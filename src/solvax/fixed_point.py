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
    scale = jnp.maximum(jnp.linalg.norm(x0), jnp.asarray(1.0, dtype=x0.dtype))
    tolerance = jnp.maximum(
        jnp.asarray(atol, dtype=x0.dtype), jnp.asarray(rtol, dtype=x0.dtype) * scale
    )
    omega0 = jnp.asarray(1.0, dtype=x0.dtype)
    previous0 = jnp.zeros_like(residual0)

    def condition(state):
        _, _, _, _, residual_norm, iterations = state
        return (iterations < max_steps) & (residual_norm > tolerance)

    def body(state):
        x, residual, previous, omega, _, iterations = state
        difference = residual - previous
        denominator = jnp.vdot(difference, difference).real
        numerator = jnp.vdot(previous, difference).real
        candidate = -omega * numerator / jnp.maximum(
            denominator, jnp.finfo(x.dtype).tiny
        )
        use_candidate = (iterations > 0) & jnp.isfinite(candidate) & (
            denominator > jnp.finfo(x.dtype).eps
        )
        next_omega = jnp.where(use_candidate, candidate, omega)
        next_omega = jnp.clip(next_omega, min_relaxation, max_relaxation)
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
