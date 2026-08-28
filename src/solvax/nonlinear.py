"""Globalized matrix-free nonlinear solves and branch continuation.

The routines here complement :func:`solvax.implicit.newton_krylov` when a
full Newton step is not globally safe.  Pseudo-transient continuation solves

``(M / dtau + J) delta = -F``

with JAX JVPs, Eisenstat--Walker inexact forcing, switched-evolution-
relaxation pseudo-time updates, and bounded physical backtracking.  An
adaptive parameter driver records accepted and rejected homotopy stages, and
the pseudo-arclength helper supplies a square bordered residual near folds.

References
----------
- C. T. Kelley & D. E. Keyes, SIAM J. Numer. Anal. 35, 508 (1998).
- S. C. Eisenstat & H. F. Walker, SIAM J. Sci. Comput. 17, 16 (1996).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from solvax.krylov import gmres

PyTree = Any
InnerProduct = Callable[[PyTree, PyTree], jax.Array]
MassOperator = Callable[[PyTree, PyTree], PyTree]
ShiftedPreconditioner = Callable[[PyTree, PyTree, jax.Array], PyTree]


def _require_finite(name: str, *values: float) -> None:
    if not all(map(math.isfinite, values)):
        raise ValueError(f"{name} controls must be finite")


@dataclass(frozen=True)
class PseudoTransientConfig:
    """Static controls for :func:`pseudo_transient_continuation`."""

    rtol: float = 1.0e-8
    atol: float = 0.0
    max_steps: int = 40
    initial_dt: float = 1.0
    min_dt: float = 1.0e-12
    max_dt: float = 1.0e12
    dt_growth: float = 5.0
    dt_shrink: float = 0.25
    newton_switch: float = 1.0e-2
    armijo: float = 1.0e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 8
    eta_initial: float = 0.5
    eta_min: float = 1.0e-4
    eta_max: float = 0.9
    eta_gamma: float = 0.9
    eta_power: float = 2.0
    linear_restart: int = 30
    linear_max_restarts: int = 10

    def __post_init__(self) -> None:
        _require_finite(
            "solver",
            self.rtol,
            self.atol,
            self.initial_dt,
            self.min_dt,
            self.max_dt,
            self.dt_growth,
            self.dt_shrink,
            self.newton_switch,
            self.armijo,
            self.backtrack_factor,
            self.eta_initial,
            self.eta_min,
            self.eta_max,
            self.eta_gamma,
            self.eta_power,
        )
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("nonlinear tolerances must be nonnegative")
        if self.max_steps < 0 or self.max_backtracks < 0:
            raise ValueError("iteration limits must be nonnegative")
        if not 0.0 < self.min_dt <= self.initial_dt <= self.max_dt:
            raise ValueError("require 0 < min_dt <= initial_dt <= max_dt")
        if self.dt_growth < 1.0 or not 0.0 < self.dt_shrink < 1.0:
            raise ValueError("dt_growth must be >= 1 and dt_shrink must lie in (0, 1)")
        if not 0.0 <= self.newton_switch <= 1.0:
            raise ValueError("newton_switch must lie in [0, 1]")
        if not 0.0 <= self.armijo < 1.0:
            raise ValueError("armijo must lie in [0, 1)")
        if not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie in (0, 1)")
        if not 0.0 < self.eta_min <= self.eta_initial <= self.eta_max < 1.0:
            raise ValueError("require 0 < eta_min <= eta_initial <= eta_max < 1")
        if not 0.0 < self.eta_gamma <= 1.0 or self.eta_power <= 0.0:
            raise ValueError("eta_gamma must lie in (0, 1] and eta_power must be positive")
        if self.linear_restart < 1 or self.linear_max_restarts < 1:
            raise ValueError("linear iteration limits must be positive")


class PseudoTransientHistory(NamedTuple):
    """Fixed-size iteration record; entries after ``steps`` are NaN/zero."""

    residual_norm: jax.Array
    pseudo_dt: jax.Array
    eta: jax.Array
    step_length: jax.Array
    accepted: jax.Array
    linear_iterations: jax.Array


class PseudoTransientSolution(NamedTuple):
    """Result and diagnostics of pseudo-transient continuation."""

    x: PyTree
    residual_norm: jax.Array
    steps: jax.Array
    linear_iterations: jax.Array
    accepted_steps: jax.Array
    rejected_steps: jax.Array
    linear_failures: jax.Array
    residual_evaluations: jax.Array
    converged: jax.Array
    linear_converged: jax.Array
    pseudo_dt: jax.Array
    eta: jax.Array
    history: PseudoTransientHistory


@dataclass(frozen=True)
class ContinuationConfig:
    """Adaptive branch-parameter controls for :func:`adaptive_continuation`."""

    target: float = 1.0
    initial_step: float = 0.1
    min_step: float = 1.0e-4
    max_step: float = 0.5
    growth: float = 1.6
    shrink: float = 0.5
    fast_steps: int = 5
    max_stages: int = 100

    def __post_init__(self) -> None:
        _require_finite(
            "continuation",
            self.target,
            self.initial_step,
            self.min_step,
            self.max_step,
            self.growth,
            self.shrink,
        )
        if not 0.0 < self.min_step <= self.initial_step <= self.max_step:
            raise ValueError("require 0 < min_step <= initial_step <= max_step")
        if self.growth <= 1.0 or not 0.0 < self.shrink < 1.0:
            raise ValueError("growth must exceed 1 and shrink must lie in (0, 1)")
        if self.fast_steps < 0 or self.max_stages < 1:
            raise ValueError("stage limits are invalid")


@dataclass(frozen=True)
class ContinuationStep:
    """One attempted parameter step in an adaptive continuation."""

    alpha_from: float
    alpha_to: float
    step_size: float
    accepted: bool
    nonlinear_steps: int
    linear_iterations: int
    residual_evaluations: int
    residual_norm: float
    minimum_pseudo_dt: float


@dataclass(frozen=True)
class ContinuationSolution:
    """Final state and complete accepted/rejected continuation trajectory."""

    x: PyTree
    alpha: float
    converged: bool
    steps: tuple[ContinuationStep, ...]
    last_nonlinear: PseudoTransientSolution | None


def _tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    products = jax.tree.leaves(jax.tree.map(jnp.vdot, left, right))
    if not products:
        raise ValueError("a solver state must contain at least one array leaf")
    total = products[0]
    for product in products[1:]:
        total = total + product
    return total


def _tree_norm(value: PyTree, inner_product: InnerProduct) -> jax.Array:
    return jnp.sqrt(jnp.maximum(jnp.real(inner_product(value, value)), 0.0))


def _tree_finite(value: PyTree) -> jax.Array:
    leaves = jax.tree.leaves(value)
    if not leaves:
        raise ValueError("a solver state must contain at least one array leaf")
    flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.all(jnp.stack(flags))


def _tree_add_scaled(x: PyTree, direction: PyTree, scale: jax.Array) -> PyTree:
    return jax.tree.map(lambda value, step: value + scale * step, x, direction)


def _tree_select(mask: jax.Array, new: PyTree, old: PyTree) -> PyTree:
    return jax.tree.map(lambda a, b: jnp.where(mask, a, b), new, old)


def eisenstat_walker_forcing(
    residual_norm: jax.Array,
    previous_residual_norm: jax.Array,
    previous_eta: jax.Array,
    *,
    eta_min: float = 1.0e-4,
    eta_max: float = 0.9,
    gamma: float = 0.9,
    power: float = 2.0,
) -> jax.Array:
    """Safeguarded Eisenstat--Walker forcing term for an inexact Newton step.

    The ratio rule tightens the linear tolerance as nonlinear convergence
    accelerates.  Eisenstat--Walker choice 2 applies the published lower
    safeguard when ``gamma*eta_previous**power > 0.1``.  ``eta_min`` adds an
    application floor for finite-work pseudo-transient stages.
    """

    residual_norm = jnp.asarray(residual_norm)
    previous_residual_norm = jnp.asarray(previous_residual_norm)
    previous_eta = jnp.asarray(previous_eta)
    dtype = residual_norm.dtype
    safe_previous = jnp.maximum(previous_residual_norm, jnp.finfo(dtype).tiny)
    ratio_eta = float(gamma) * (residual_norm / safe_previous) ** float(power)
    safeguard = float(gamma) * previous_eta ** float(power)
    candidate = jnp.where(
        safeguard > 0.1,
        jnp.maximum(ratio_eta, safeguard),
        ratio_eta,
    )
    candidate = jnp.nan_to_num(
        candidate,
        nan=float(eta_max),
        posinf=float(eta_max),
        neginf=float(eta_min),
    )
    return jnp.clip(candidate, float(eta_min), float(eta_max))


def pseudo_transient_continuation(
    residual_fn: Callable[[PyTree], PyTree],
    x0: PyTree,
    *,
    mass: MassOperator | None = None,
    precond: ShiftedPreconditioner | None = None,
    admissible: Callable[[PyTree], jax.Array] | None = None,
    inner_product: InnerProduct | None = None,
    norm: Callable[[PyTree], jax.Array] | None = None,
    config: PseudoTransientConfig | None = None,
) -> PseudoTransientSolution:
    """Solve ``residual_fn(x)=0`` with globalized pseudo-transient JFNK.

    ``mass(x, vector)`` applies the positive pseudo-time metric ``M(x)``
    (identity by default). ``precond(x, rhs, dtau)`` may use the current state
    and pseudo-time shift to form a consistent right preconditioner.
    ``admissible(candidate)`` is a hard scalar predicate for physical
    conditions such as finite values, positive Jacobian, bounds, or nestedness;
    a rejected candidate is backtracked and never contaminates the iterate.

    All nonlinear and linear limits are static, so the iteration is compatible
    with ``jax.jit`` and arbitrary matching pytrees.
    """

    config = PseudoTransientConfig() if config is None else config
    x0 = jax.tree.map(jnp.asarray, x0)
    mass = (lambda _state, value: value) if mass is None else mass
    inner_product = _tree_dot if inner_product is None else inner_product
    norm = (lambda value: _tree_norm(value, inner_product)) if norm is None else norm
    admissible = (lambda value: jnp.asarray(True)) if admissible is None else admissible

    residual0 = residual_fn(x0)
    if jax.tree.structure(residual0) != jax.tree.structure(x0):  # type: ignore[operator]
        raise ValueError("residual_fn must preserve the state pytree structure")
    residual_norm0 = norm(residual0)
    tolerance = jnp.maximum(float(config.atol), float(config.rtol) * residual_norm0)
    history_size = int(config.max_steps) + 1
    real_dtype = jnp.real(residual_norm0).dtype
    nan_history = jnp.full((history_size,), jnp.nan, dtype=real_dtype)
    residual_history = nan_history.at[0].set(residual_norm0)
    dt_history = nan_history.at[0].set(float(config.initial_dt))
    eta_history = nan_history.at[0].set(float(config.eta_initial))
    length_history = nan_history
    accepted_history = jnp.zeros((history_size,), dtype=bool)
    linear_history = jnp.zeros((history_size,), dtype=jnp.int32)

    initial = (
        x0,
        residual_norm0,
        jnp.asarray(False),
        jnp.asarray(float(config.eta_initial), dtype=real_dtype),
        jnp.asarray(float(config.initial_dt), dtype=real_dtype),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(1),
        residual_history,
        dt_history,
        eta_history,
        length_history,
        accepted_history,
        linear_history,
    )

    def continue_iteration(state):
        residual_norm, terminal_linear_failure, steps = state[1], state[2], state[5]
        return (steps < config.max_steps) & (residual_norm > tolerance) & ~terminal_linear_failure

    def iteration(state):
        (
            x,
            residual_norm,
            _,
            eta,
            dt,
            steps,
            linear_iterations,
            accepted_steps,
            rejected_steps,
            linear_failures,
            residual_evaluations,
            residual_history,
            dt_history,
            eta_history,
            length_history,
            accepted_history,
            linear_history,
        ) = state
        linearized_residual, jvp = jax.linearize(residual_fn, x)

        def shifted_jacobian(value):
            return jax.tree.map(lambda m, j: m / dt + j, mass(x, value), jvp(value))

        right_precond = None if precond is None else lambda value: precond(x, value, dt)
        linear = gmres(
            shifted_jacobian,
            jax.tree.map(jnp.negative, linearized_residual),
            precond=right_precond,
            inner_product=inner_product,
            restart=config.linear_restart,
            rtol=eta,
            atol=0.0,
            max_restarts=config.linear_max_restarts,
        )

        def backtrack_condition(backtrack_state):
            iteration_index, _, _, accepted, _ = backtrack_state
            return (iteration_index <= config.max_backtracks) & ~accepted & linear.converged

        def backtrack(backtrack_state):
            iteration_index, best_x, best_norm, accepted, step_length = backtrack_state
            candidate = _tree_add_scaled(x, linear.x, step_length)
            candidate_residual = residual_fn(candidate)
            candidate_norm = norm(candidate_residual)
            sufficient_decrease = (
                candidate_norm <= (1.0 - float(config.armijo) * step_length) * residual_norm
            )
            valid = (
                linear.converged
                & _tree_finite(candidate)
                & _tree_finite(candidate_residual)
                & jnp.asarray(admissible(candidate), dtype=bool)
                & sufficient_decrease
            )
            take = ~accepted & valid
            return (
                iteration_index + 1,
                _tree_select(take, candidate, best_x),
                jnp.where(take, candidate_norm, best_norm),
                accepted | valid,
                step_length * float(config.backtrack_factor),
            )

        backtrack_initial = (
            jnp.int32(0),
            x,
            residual_norm,
            jnp.asarray(False),
            jnp.asarray(1.0, dtype=real_dtype),
        )
        backtrack_result = jax.lax.while_loop(backtrack_condition, backtrack, backtrack_initial)
        attempts, candidate_x, candidate_norm, accepted, next_trial_length = backtrack_result
        used_length = jnp.where(
            accepted,
            next_trial_length / float(config.backtrack_factor),
            jnp.asarray(0.0, dtype=real_dtype),
        )
        next_x = _tree_select(accepted, candidate_x, x)
        next_norm = jnp.where(accepted, candidate_norm, residual_norm)
        ratio = residual_norm / jnp.maximum(candidate_norm, jnp.finfo(real_dtype).tiny)
        ser_factor = jnp.clip(ratio, float(config.dt_shrink), float(config.dt_growth))
        accepted_dt = jnp.clip(dt * ser_factor, float(config.min_dt), float(config.max_dt))
        accepted_dt = jnp.where(
            candidate_norm <= float(config.newton_switch) * residual_norm0,
            jnp.asarray(float(config.max_dt), dtype=real_dtype),
            accepted_dt,
        )
        rejected_dt = jnp.maximum(float(config.min_dt), dt * float(config.dt_shrink))
        next_dt = jnp.where(accepted, accepted_dt, rejected_dt)
        next_eta = jnp.where(
            accepted,
            eisenstat_walker_forcing(
                candidate_norm,
                residual_norm,
                eta,
                eta_min=config.eta_min,
                eta_max=config.eta_max,
                gamma=config.eta_gamma,
                power=config.eta_power,
            ),
            jnp.minimum(float(config.eta_max), jnp.maximum(eta, float(config.eta_initial))),
        )
        next_step = steps + 1
        at_minimum_dt = dt <= float(config.min_dt) * (1.0 + 4.0 * jnp.finfo(real_dtype).eps)
        terminal_linear_failure = at_minimum_dt & ~linear.converged
        residual_history = residual_history.at[next_step].set(next_norm)
        dt_history = dt_history.at[next_step].set(next_dt)
        eta_history = eta_history.at[next_step].set(next_eta)
        length_history = length_history.at[next_step].set(used_length)
        accepted_history = accepted_history.at[next_step].set(accepted)
        linear_history = linear_history.at[next_step].set(linear.iterations)
        return (
            next_x,
            next_norm,
            terminal_linear_failure,
            next_eta,
            next_dt,
            next_step,
            linear_iterations + linear.iterations,
            accepted_steps + accepted.astype(jnp.int32),
            rejected_steps + (~accepted).astype(jnp.int32),
            linear_failures + (~linear.converged).astype(jnp.int32),
            residual_evaluations + attempts + 1,
            residual_history,
            dt_history,
            eta_history,
            length_history,
            accepted_history,
            linear_history,
        )

    final = jax.lax.while_loop(continue_iteration, iteration, initial)
    (
        x,
        residual_norm,
        terminal_linear_failure,
        eta,
        dt,
        steps,
        linear_iterations,
        accepted_steps,
        rejected_steps,
        linear_failures,
        residual_evaluations,
        residual_history,
        dt_history,
        eta_history,
        length_history,
        accepted_history,
        linear_history,
    ) = final
    converged = residual_norm <= tolerance
    return PseudoTransientSolution(
        x=x,
        residual_norm=residual_norm,
        steps=steps,
        linear_iterations=linear_iterations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        linear_failures=linear_failures,
        residual_evaluations=residual_evaluations,
        converged=converged,
        linear_converged=~terminal_linear_failure,
        pseudo_dt=dt,
        eta=eta,
        history=PseudoTransientHistory(
            residual_history,
            dt_history,
            eta_history,
            length_history,
            accepted_history,
            linear_history,
        ),
    )


@partial(
    jax.jit,
    static_argnames=(
        "residual_fn",
        "mass",
        "precond",
        "admissible",
        "inner_product",
        "norm",
        "config",
    ),
)
def _parameterized_pseudo_transient_continuation(
    residual_fn: Callable[[PyTree, jax.Array], PyTree],
    x0: PyTree,
    parameter: jax.Array,
    *,
    mass: MassOperator | None,
    precond: ShiftedPreconditioner | None,
    admissible: Callable[[PyTree, jax.Array], jax.Array] | None,
    inner_product: InnerProduct | None,
    norm: Callable[[PyTree], jax.Array] | None,
    config: PseudoTransientConfig,
) -> PseudoTransientSolution:
    """Compile one reusable nonlinear stage with a dynamic parameter."""

    stage_admissible = (
        None
        if admissible is None
        else lambda candidate: admissible(candidate, parameter)
    )
    return pseudo_transient_continuation(
        lambda candidate: residual_fn(candidate, parameter),
        x0,
        mass=mass,
        precond=precond,
        admissible=stage_admissible,
        inner_product=inner_product,
        norm=norm,
        config=config,
    )


def adaptive_continuation(
    residual_fn: Callable[[PyTree, float], PyTree],
    x0: PyTree,
    *,
    alpha0: float = 0.0,
    nonlinear_config: PseudoTransientConfig | None = None,
    continuation_config: ContinuationConfig | None = None,
    mass: MassOperator | None = None,
    precond: ShiftedPreconditioner | None = None,
    admissible: Callable[[PyTree, float], jax.Array] | None = None,
    accept_stage: Callable[[PyTree, float, PseudoTransientSolution], bool] | None = None,
    inner_product: InnerProduct | None = None,
    norm: Callable[[PyTree], jax.Array] | None = None,
) -> ContinuationSolution:
    """Follow a root branch with adaptive accepted/rejected parameter steps.

    The orchestration is intentionally host-side: each nonlinear stage remains
    fully JIT-able, while acceptance may inspect application-level validation
    certificates.  Rejected stages leave both the state and branch parameter
    unchanged and reduce the next step.
    """

    nonlinear_config = PseudoTransientConfig() if nonlinear_config is None else nonlinear_config
    continuation_config = (
        ContinuationConfig() if continuation_config is None else continuation_config
    )
    alpha = float(alpha0)
    if not math.isfinite(alpha):
        raise ValueError("alpha0 must be finite")
    direction = 1.0 if continuation_config.target >= alpha else -1.0
    x = x0
    step_size = float(continuation_config.initial_step)
    records: list[ContinuationStep] = []
    last_solution: PseudoTransientSolution | None = None

    for _ in range(continuation_config.max_stages):
        if direction * (continuation_config.target - alpha) <= 0.0:
            break
        trial_alpha = alpha + direction * min(abs(continuation_config.target - alpha), step_size)

        solution = _parameterized_pseudo_transient_continuation(
            residual_fn,
            x,
            jnp.asarray(trial_alpha),
            mass=mass,
            precond=precond,
            admissible=admissible,
            inner_product=inner_product,
            norm=norm,
            config=nonlinear_config,
        )
        last_solution = solution
        accepted = bool(solution.converged) and bool(solution.linear_converged)
        if accepted and accept_stage is not None:
            accepted = bool(accept_stage(solution.x, trial_alpha, solution))
        used_history = solution.history.pseudo_dt[: int(solution.steps) + 1]
        minimum_dt = float(jnp.nanmin(used_history))
        records.append(
            ContinuationStep(
                alpha_from=alpha,
                alpha_to=trial_alpha,
                step_size=trial_alpha - alpha,
                accepted=accepted,
                nonlinear_steps=int(solution.steps),
                linear_iterations=int(solution.linear_iterations),
                residual_evaluations=int(solution.residual_evaluations),
                residual_norm=float(solution.residual_norm),
                minimum_pseudo_dt=minimum_dt,
            )
        )
        if accepted:
            x = solution.x
            alpha = trial_alpha
            if int(solution.steps) <= continuation_config.fast_steps:
                step_size = min(
                    continuation_config.max_step,
                    step_size * continuation_config.growth,
                )
        else:
            step_size *= continuation_config.shrink
            if step_size < continuation_config.min_step:
                break
    return ContinuationSolution(
        x=x,
        alpha=alpha,
        converged=direction * (continuation_config.target - alpha) <= 0.0,
        steps=tuple(records),
        last_nonlinear=last_solution,
    )


def pseudo_arclength_residual(
    residual_fn: Callable[[PyTree, jax.Array], PyTree],
    state: tuple[PyTree, jax.Array],
    *,
    tangent: tuple[PyTree, jax.Array],
    predictor: tuple[PyTree, jax.Array],
    inner_product: InnerProduct | None = None,
) -> tuple[PyTree, jax.Array]:
    """Return the square bordered residual for a pseudo-arclength corrector."""

    x, alpha = state
    tangent_x, tangent_alpha = tangent
    predictor_x, predictor_alpha = predictor
    inner_product = _tree_dot if inner_product is None else inner_product
    displacement = jax.tree.map(lambda value, base: value - base, x, predictor_x)
    arclength = inner_product(tangent_x, displacement) + tangent_alpha * (alpha - predictor_alpha)
    return residual_fn(x, alpha), jnp.real(arclength)


def pseudo_arclength_corrector(
    residual_fn: Callable[[PyTree, jax.Array], PyTree],
    initial: tuple[PyTree, jax.Array],
    *,
    tangent: tuple[PyTree, jax.Array],
    predictor: tuple[PyTree, jax.Array],
    config: PseudoTransientConfig | None = None,
    admissible: Callable[[PyTree, jax.Array], jax.Array] | None = None,
    mass: Callable[
        [tuple[PyTree, jax.Array], tuple[PyTree, jax.Array]],
        tuple[PyTree, jax.Array],
    ]
    | None = None,
    precond: Callable[
        [
            tuple[PyTree, jax.Array],
            tuple[PyTree, jax.Array],
            jax.Array,
        ],
        tuple[PyTree, jax.Array],
    ]
    | None = None,
    inner_product: InnerProduct | None = None,
    norm: Callable[[tuple[PyTree, jax.Array]], jax.Array] | None = None,
) -> PseudoTransientSolution:
    """Correct one predictor on a fold-capable pseudo-arclength branch.

    ``mass(state, vector)`` and ``precond(state, rhs, dtau)`` act on the
    complete bordered ``(x, alpha)`` state.  This lets applications supply a
    Schur or block bordered preconditioner without reimplementing the
    pseudo-transient globalization.
    """

    bordered = lambda state: pseudo_arclength_residual(  # noqa: E731
        residual_fn, state, tangent=tangent, predictor=predictor
    )
    bordered_admissible = (
        None if admissible is None else lambda state: admissible(state[0], state[1])
    )
    return pseudo_transient_continuation(
        bordered,
        initial,
        mass=mass,
        precond=precond,
        admissible=bordered_admissible,
        inner_product=inner_product,
        norm=norm,
        config=config,
    )


__all__ = [
    "ContinuationConfig",
    "ContinuationSolution",
    "ContinuationStep",
    "PseudoTransientConfig",
    "PseudoTransientHistory",
    "PseudoTransientSolution",
    "adaptive_continuation",
    "eisenstat_walker_forcing",
    "pseudo_arclength_corrector",
    "pseudo_arclength_residual",
    "pseudo_transient_continuation",
]
