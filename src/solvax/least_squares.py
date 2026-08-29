"""Matrix-free nonlinear least squares with implicit stationarity derivatives."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from solvax.implicit import root_solve
from solvax.pcg import pcg

Array = jax.Array
Residual = Callable[[Array], Array]
Admissible = Callable[[Array], Array]
NormalPreconditioner = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class LeastSquaresConfig:
    """Static controls for :func:`gauss_newton_least_squares`."""

    rtol: float = 1.0e-6
    atol: float = 0.0
    max_steps: int = 40
    initial_damping: float = 1.0e-3
    minimum_damping: float = 1.0e-12
    maximum_damping: float = 1.0e12
    damping_decrease: float = 0.25
    damping_increase: float = 4.0
    acceptance_ratio: float = 1.0e-4
    good_ratio: float = 0.75
    poor_ratio: float = 0.25
    linear_rtol: float = 1.0e-3
    linear_atol: float = 0.0
    linear_max_steps: int = 200

    def __post_init__(self) -> None:
        finite = (
            self.rtol,
            self.atol,
            self.initial_damping,
            self.minimum_damping,
            self.maximum_damping,
            self.damping_decrease,
            self.damping_increase,
            self.acceptance_ratio,
            self.good_ratio,
            self.poor_ratio,
            self.linear_rtol,
            self.linear_atol,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("least-squares controls must be finite")
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("least-squares tolerances must be nonnegative")
        if self.max_steps < 0 or self.linear_max_steps < 1:
            raise ValueError("least-squares iteration limits are invalid")
        if not (
            0.0
            < self.minimum_damping
            <= self.initial_damping
            <= self.maximum_damping
        ):
            raise ValueError(
                "require 0 < minimum_damping <= initial_damping <= maximum_damping"
            )
        if not 0.0 < self.damping_decrease < 1.0:
            raise ValueError("damping_decrease must lie in (0, 1)")
        if self.damping_increase <= 1.0:
            raise ValueError("damping_increase must be greater than 1")
        if not 0.0 <= self.acceptance_ratio < self.poor_ratio:
            raise ValueError("acceptance_ratio must lie in [0, poor_ratio)")
        if not self.poor_ratio < self.good_ratio < 1.0:
            raise ValueError("require poor_ratio < good_ratio < 1")
        if self.linear_rtol < 0.0 or self.linear_atol < 0.0:
            raise ValueError("linear tolerances must be nonnegative")


class LeastSquaresHistory(NamedTuple):
    """Fixed-shape accepted/rejected trust-region history."""

    cost: Array
    gradient_norm: Array
    damping: Array
    ratio: Array
    accepted: Array
    linear_iterations: Array


class LeastSquaresSolution(NamedTuple):
    """Result and diagnostics of a matrix-free least-squares solve."""

    x: Array
    residual_norm: Array
    cost: Array
    gradient_norm: Array
    steps: Array
    accepted_steps: Array
    rejected_steps: Array
    linear_iterations: Array
    converged: Array
    damping: Array
    history: LeastSquaresHistory


def _dot(left: Array, right: Array) -> Array:
    return jnp.real(jnp.vdot(left, right))


def least_squares_stationarity(residual: Residual, x: Array) -> Array:
    """Return ``J(x).T @ residual(x)`` without materializing ``J``."""

    value, pullback = jax.vjp(residual, x)
    return pullback(value)[0]


def gauss_newton_least_squares(
    residual: Residual,
    x0: Array,
    *,
    config: LeastSquaresConfig | None = None,
    precond: NormalPreconditioner | None = None,
    admissible: Admissible | None = None,
) -> LeastSquaresSolution:
    """Minimize a rectangular residual with matrix-free damped Gauss--Newton.

    Every linear step applies ``J`` and ``J.T`` through JAX transforms and
    solves ``(J.T J + damping I) step = -J.T residual`` with PCG. A trust ratio
    accepts or rejects the trial state and adapts the Levenberg damping. The
    routine is compatible with :func:`jax.jit`; its fixed-size history avoids
    host callbacks or dynamic allocation.

    This is a primal solver. Use :func:`implicit_least_squares` when derivatives
    of the converged stationary point are required.
    """

    config = LeastSquaresConfig() if config is None else config
    x0 = jnp.asarray(x0)
    if x0.ndim != 1:
        raise ValueError("least-squares coordinates must be a one-dimensional array")
    admissible = (
        (lambda value: jnp.all(jnp.isfinite(value)))
        if admissible is None
        else admissible
    )

    residual0 = jnp.asarray(residual(x0))
    if residual0.ndim != 1:
        raise ValueError("least-squares residual must be a one-dimensional array")
    gradient0 = least_squares_stationarity(residual, x0)
    gradient_norm0 = jnp.linalg.norm(gradient0)
    cost0 = 0.5 * _dot(residual0, residual0)
    scalar_dtype = jnp.real(jnp.zeros((), dtype=x0.dtype)).dtype
    threshold = jnp.maximum(
        jnp.asarray(config.atol, dtype=scalar_dtype),
        jnp.asarray(config.rtol, dtype=scalar_dtype)
        * jnp.maximum(gradient_norm0, jnp.asarray(1.0, dtype=scalar_dtype)),
    )
    converged0 = (
        jnp.isfinite(cost0)
        & jnp.isfinite(gradient_norm0)
        & (gradient_norm0 <= threshold)
    )
    history = LeastSquaresHistory(
        cost=jnp.full((config.max_steps + 1,), cost0),
        gradient_norm=jnp.full((config.max_steps + 1,), gradient_norm0),
        damping=jnp.full(
            (config.max_steps + 1,),
            jnp.asarray(config.initial_damping, dtype=scalar_dtype),
        ),
        ratio=jnp.full((config.max_steps,), jnp.nan, dtype=scalar_dtype),
        accepted=jnp.zeros((config.max_steps,), dtype=bool),
        linear_iterations=jnp.zeros((config.max_steps,), dtype=jnp.int32),
    )
    initial = (
        jnp.int32(0),
        x0,
        residual0,
        cost0,
        gradient_norm0,
        jnp.asarray(config.initial_damping, dtype=scalar_dtype),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        converged0,
        history,
    )

    def cond_fun(state):
        step, _, _, _, _, _, _, _, _, converged, _ = state
        return (step < config.max_steps) & ~converged

    def body_fun(state):
        (
            step,
            x,
            value,
            cost,
            _,
            damping,
            accepted_steps,
            rejected_steps,
            linear_iterations,
            _,
            history,
        ) = state
        value, jvp = jax.linearize(residual, x)
        transpose = jax.linear_transpose(jvp, x)
        gradient = transpose(value)[0]

        def normal_matvec(direction):
            return transpose(jvp(direction))[0] + damping * direction

        normal_precond = (
            None
            if precond is None
            else lambda rhs: precond(x, rhs, damping)
        )
        linear = pcg(
            normal_matvec,
            -gradient,
            precond=normal_precond,
            rtol=config.linear_rtol,
            atol=config.linear_atol,
            max_steps=config.linear_max_steps,
        )
        direction = linear.x
        applied = jvp(direction)
        predicted = -_dot(gradient, direction) - 0.5 * _dot(applied, applied)
        trial = x + direction
        trial_value = residual(trial)
        trial_cost = 0.5 * _dot(trial_value, trial_value)
        actual = cost - trial_cost
        safe_predicted = jnp.where(predicted > 0.0, predicted, 1.0)
        ratio = actual / safe_predicted
        finite = (
            jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(trial_cost)
            & jnp.isfinite(ratio)
        )
        accept = (
            finite
            & (predicted > 0.0)
            & (ratio >= config.acceptance_ratio)
            & admissible(trial)
        )
        x_next = jnp.where(accept, trial, x)
        value_next = jnp.where(accept, trial_value, value)
        cost_next = jnp.where(accept, trial_cost, cost)
        gradient_next = least_squares_stationarity(residual, x_next)
        gradient_norm = jnp.linalg.norm(gradient_next)
        converged = (
            jnp.isfinite(cost_next)
            & jnp.isfinite(gradient_norm)
            & (gradient_norm <= threshold)
        )
        damping_next = jnp.where(
            accept & (ratio >= config.good_ratio),
            damping * config.damping_decrease,
            jnp.where(
                ~accept | (ratio < config.poor_ratio),
                damping * config.damping_increase,
                damping,
            ),
        )
        damping_next = jnp.clip(
            damping_next,
            config.minimum_damping,
            config.maximum_damping,
        )
        next_step = step + 1
        history = LeastSquaresHistory(
            cost=history.cost.at[next_step].set(cost_next),
            gradient_norm=history.gradient_norm.at[next_step].set(gradient_norm),
            damping=history.damping.at[next_step].set(damping_next),
            ratio=history.ratio.at[step].set(ratio),
            accepted=history.accepted.at[step].set(accept),
            linear_iterations=history.linear_iterations.at[step].set(
                linear.iterations
            ),
        )
        return (
            next_step,
            x_next,
            value_next,
            cost_next,
            gradient_norm,
            damping_next,
            accepted_steps + accept.astype(jnp.int32),
            rejected_steps + (~accept).astype(jnp.int32),
            linear_iterations + linear.iterations,
            converged,
            history,
        )

    (
        steps,
        x,
        value,
        cost,
        gradient_norm,
        damping,
        accepted_steps,
        rejected_steps,
        linear_iterations,
        converged,
        history,
    ) = lax.while_loop(cond_fun, body_fun, initial)
    return LeastSquaresSolution(
        x=x,
        residual_norm=jnp.linalg.norm(value),
        cost=cost,
        gradient_norm=gradient_norm,
        steps=steps,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        linear_iterations=linear_iterations,
        converged=converged,
        damping=damping,
        history=history,
    )


def implicit_least_squares(
    residual: Residual,
    x0: Array,
    *,
    config: LeastSquaresConfig | None = None,
    precond: NormalPreconditioner | None = None,
    admissible: Admissible | None = None,
    tangent_solve: Callable | None = None,
) -> Array:
    """Solve least-squares stationarity with implicit tangent/adjoint rules."""

    config = LeastSquaresConfig() if config is None else config
    def stationarity(value):
        return least_squares_stationarity(residual, value)

    def solver(_, initial):
        return gauss_newton_least_squares(
            residual,
            initial,
            config=config,
            precond=precond,
            admissible=admissible,
        ).x

    return root_solve(
        stationarity,
        jnp.asarray(x0),
        solver,
        tangent_solve=tangent_solve,
    )


__all__ = [
    "LeastSquaresConfig",
    "LeastSquaresHistory",
    "LeastSquaresSolution",
    "gauss_newton_least_squares",
    "implicit_least_squares",
    "least_squares_stationarity",
]
