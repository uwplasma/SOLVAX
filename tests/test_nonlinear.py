"""Tests for pseudo-transient, forcing, homotopy, and arclength solves."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    ContinuationConfig,
    PseudoTransientConfig,
    adaptive_continuation,
    eisenstat_walker_forcing,
    pseudo_arclength_corrector,
    pseudo_arclength_residual,
    pseudo_transient_continuation,
)
from solvax import nonlinear as nonlinear_module

jax.config.update("jax_enable_x64", True)


def _scalar_config(**updates) -> PseudoTransientConfig:
    values = dict(
        rtol=1.0e-11,
        max_steps=50,
        initial_dt=1.0e-2,
        max_dt=1.0e10,
        linear_restart=4,
        linear_max_restarts=2,
    )
    values.update(updates)
    return PseudoTransientConfig(**values)


def test_eisenstat_walker_tightens_and_obeys_safeguards():
    loose = eisenstat_walker_forcing(0.8, 1.0, 0.7)
    tight = eisenstat_walker_forcing(0.05, 1.0, 0.2)
    floor = eisenstat_walker_forcing(0.0, 0.0, 0.0, eta_min=0.01, eta_max=0.8)
    ceiling = eisenstat_walker_forcing(2.0, 1.0, 0.8, eta_min=0.01, eta_max=0.6)
    assert float(tight) < float(loose)
    assert float(floor) == pytest.approx(0.01)
    assert float(ceiling) == pytest.approx(0.6)
    np.testing.assert_allclose(
        jax.jit(eisenstat_walker_forcing)(0.2, 1.0, 0.5),
        eisenstat_walker_forcing(0.2, 1.0, 0.5),
    )


def test_pseudo_transient_converges_under_jit_and_records_true_history():
    residual = lambda x: jnp.asarray([x[0] ** 2 - 2.0])  # noqa: E731
    solve = jax.jit(lambda x: pseudo_transient_continuation(residual, x, config=_scalar_config()))
    solution = solve(jnp.asarray([10.0]))
    np.testing.assert_allclose(solution.x, np.sqrt(2.0), rtol=3e-10, atol=3e-10)
    assert bool(solution.converged)
    assert bool(solution.linear_converged)
    assert int(solution.accepted_steps) == int(solution.steps)
    assert int(solution.rejected_steps) == 0
    assert int(solution.linear_iterations) > 0
    assert int(solution.residual_evaluations) == 1 + 2 * int(solution.steps)
    used = slice(0, int(solution.steps) + 1)
    history = np.asarray(solution.history.residual_norm[used])
    assert np.all(np.diff(history) < 0.0)
    assert history[-1] == pytest.approx(float(solution.residual_norm))
    assert float(solution.history.pseudo_dt[used][-1]) == pytest.approx(float(solution.pseudo_dt))
    assert np.all(np.isnan(np.asarray(solution.history.residual_norm[int(solution.steps) + 1 :])))


def test_backtracking_enforces_hard_admissibility_and_recovers():
    residual = lambda x: jnp.asarray([jnp.arctan(x[0])])  # noqa: E731
    solution = pseudo_transient_continuation(
        residual,
        jnp.asarray([2.0]),
        admissible=lambda x: x[0] >= 0.0,
        config=_scalar_config(
            initial_dt=1.0e8,
            min_dt=1.0e-12,
            max_dt=1.0e8,
            max_backtracks=8,
        ),
    )
    assert bool(solution.converged)
    assert float(solution.x[0]) >= 0.0
    lengths = np.asarray(solution.history.step_length[1 : int(solution.steps) + 1])
    assert np.any((lengths > 0.0) & (lengths < 1.0))
    assert int(solution.residual_evaluations) > 1 + 2 * int(solution.steps)


def test_pytree_mass_preconditioner_and_custom_norm():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    rhs = {"u": jnp.asarray([1.0, 2.0]), "v": jnp.asarray([-1.0])}

    def residual(state):
        return {
            "u": matrix @ state["u"] - rhs["u"],
            "v": 2.0 * state["v"] - rhs["v"],
        }

    def mass(state, value):
        del state
        return {"u": 2.0 * value["u"], "v": 0.5 * value["v"]}

    def precond(state, value, dt):
        del state
        return {
            "u": value["u"] / (4.0 + 2.0 / dt),
            "v": value["v"] / (2.0 + 0.5 / dt),
        }

    def maximum_norm(value):
        return jnp.maximum(jnp.max(jnp.abs(value["u"])), jnp.max(jnp.abs(value["v"])))

    solution = pseudo_transient_continuation(
        residual,
        jax.tree.map(jnp.zeros_like, rhs),
        mass=mass,
        precond=precond,
        norm=maximum_norm,
        config=_scalar_config(initial_dt=0.1, linear_restart=6),
    )
    assert bool(solution.converged)
    np.testing.assert_allclose(solution.x["u"], jnp.linalg.solve(matrix, rhs["u"]), rtol=2e-10)
    np.testing.assert_allclose(solution.x["v"], rhs["v"] / 2.0, rtol=2e-10)


def test_linear_failure_at_minimum_dt_is_reported_without_false_convergence():
    zero_mass = lambda state, value: jax.tree.map(jnp.zeros_like, value)  # noqa: E731
    solution = pseudo_transient_continuation(
        lambda x: jnp.ones_like(x),
        jnp.asarray([0.0, 0.0]),
        mass=zero_mass,
        config=_scalar_config(
            initial_dt=1.0e-4,
            min_dt=1.0e-4,
            max_dt=1.0,
            max_steps=5,
            linear_restart=2,
            linear_max_restarts=1,
        ),
    )
    assert not bool(solution.converged)
    assert not bool(solution.linear_converged)
    assert int(solution.linear_failures) == 1
    assert int(solution.rejected_steps) == 1
    assert int(solution.residual_evaluations) == 2


def test_recovered_linear_failure_does_not_poison_later_steps(monkeypatch):
    def staged_gmres(operator, rhs, **kwargs):
        del kwargs
        scale = operator(jnp.ones_like(rhs))[0]
        return SimpleNamespace(
            x=rhs / scale,
            converged=scale > 3.0,
            iterations=jnp.int32(1),
        )

    monkeypatch.setattr(nonlinear_module, "gmres", staged_gmres)
    solution = pseudo_transient_continuation(
        lambda x: x - 1.0,
        jnp.asarray([0.0]),
        config=_scalar_config(rtol=1.0e-2, initial_dt=1.0, min_dt=0.25, dt_shrink=0.25),
    )
    assert bool(solution.converged)
    assert bool(solution.linear_converged)
    assert int(solution.linear_failures) > 0
    assert int(solution.accepted_steps) > 0


def test_zero_residual_and_zero_step_budget_finish_without_linear_work():
    solved = pseudo_transient_continuation(lambda x: x, jnp.zeros(2), config=_scalar_config())
    budgeted = pseudo_transient_continuation(
        lambda x: x - 1.0,
        jnp.zeros(2),
        config=_scalar_config(max_steps=0),
    )
    assert bool(solved.converged)
    assert int(solved.steps) == 0
    assert not bool(budgeted.converged)
    assert int(budgeted.steps) == 0


def test_inexact_forcing_saves_krylov_work_against_fixed_tight_tolerance():
    size = 32
    diagonal = jnp.logspace(0.0, 5.0, size)
    matrix = jnp.diag(diagonal) + 0.2 * jnp.diag(diagonal[:-1], 1)
    residual = lambda x: matrix @ x + 0.2 * x**3 - 1.0  # noqa: E731
    common = dict(
        rtol=1.0e-9,
        max_steps=20,
        initial_dt=1.0e8,
        max_dt=1.0e8,
        linear_restart=20,
        linear_max_restarts=40,
        max_backtracks=4,
    )
    adaptive = pseudo_transient_continuation(
        residual,
        jnp.zeros(size),
        config=PseudoTransientConfig(**common),
    )
    fixed = pseudo_transient_continuation(
        residual,
        jnp.zeros(size),
        config=PseudoTransientConfig(
            **common,
            eta_initial=1.0e-6,
            eta_min=1.0e-6,
            eta_max=1.0e-6,
        ),
    )
    assert bool(adaptive.converged) and bool(fixed.converged)
    assert int(adaptive.linear_iterations) < int(fixed.linear_iterations)
    assert int(adaptive.residual_evaluations) == 13
    assert int(fixed.residual_evaluations) == 9


def test_adaptive_continuation_records_rejection_then_reaches_target():
    callback_count = [0]

    def residual(x, alpha):
        return jnp.asarray([x[0] ** 2 - (1.0 + alpha)])

    def reject_first_stage(x, alpha, solution):
        del x, alpha, solution
        callback_count[0] += 1
        return callback_count[0] > 1

    solution = adaptive_continuation(
        residual,
        jnp.asarray([1.0]),
        nonlinear_config=_scalar_config(initial_dt=0.1),
        continuation_config=ContinuationConfig(
            target=1.0,
            initial_step=0.4,
            min_step=1.0e-3,
            max_step=0.6,
            growth=1.5,
            shrink=0.5,
            fast_steps=20,
            max_stages=20,
        ),
        admissible=lambda x, alpha: (x[0] > 0.0) & (alpha <= 1.0),
        accept_stage=reject_first_stage,
    )
    assert solution.converged
    assert solution.alpha == pytest.approx(1.0)
    np.testing.assert_allclose(solution.x, np.sqrt(2.0), rtol=3e-10)
    assert not solution.steps[0].accepted
    assert any(step.accepted for step in solution.steps[1:])
    assert solution.steps[1].step_size < solution.steps[0].step_size
    assert all(
        step.residual_evaluations >= step.nonlinear_steps + 1
        for step in solution.steps
    )
    assert solution.last_nonlinear is not None


def test_continuation_stops_when_rejections_cross_minimum_step():
    solution = adaptive_continuation(
        lambda x, alpha: x - alpha,
        jnp.asarray([0.0]),
        nonlinear_config=_scalar_config(),
        continuation_config=ContinuationConfig(
            target=1.0,
            initial_step=0.2,
            min_step=0.05,
            max_step=0.2,
            max_stages=10,
        ),
        accept_stage=lambda x, alpha, result: False,
    )
    assert not solution.converged
    assert solution.alpha == 0.0
    assert all(not step.accepted for step in solution.steps)
    assert len(solution.steps) == 3


def test_continuation_supports_descending_parameter_branches():
    solution = adaptive_continuation(
        lambda x, alpha: x - alpha,
        jnp.asarray([1.0]),
        alpha0=1.0,
        continuation_config=ContinuationConfig(target=-1.0, initial_step=0.5),
    )
    assert solution.converged
    assert solution.alpha == pytest.approx(-1.0)
    np.testing.assert_allclose(solution.x, -1.0, atol=2e-10)
    assert all(step.step_size < 0.0 for step in solution.steps)


def test_pseudo_arclength_bordered_residual_and_corrector():
    residual = lambda x, alpha: jnp.asarray([x[0] ** 2 + alpha - 1.0])  # noqa: E731
    # The sign of a branch tangent is arbitrary; this orientation gives the
    # bordered Jacobian a positive-real spectrum for pseudo-time evolution.
    tangent = (jnp.asarray([-1.0]), jnp.asarray(0.2))
    predictor = (jnp.asarray([0.1]), jnp.asarray(0.98))
    initial = (jnp.asarray([0.12]), jnp.asarray(0.97))
    raw = pseudo_arclength_residual(residual, initial, tangent=tangent, predictor=predictor)
    assert jax.tree.structure(raw) == jax.tree.structure(initial)
    solution = pseudo_arclength_corrector(
        residual,
        initial,
        tangent=tangent,
        predictor=predictor,
        admissible=lambda x, alpha: (x[0] >= 0.0) & (alpha >= 0.0),
        mass=lambda state, vector: vector,
        precond=lambda state, rhs, dt: rhs,
        norm=lambda state: jnp.sqrt(
            jnp.vdot(state[0], state[0]).real + state[1] ** 2
        ),
        config=_scalar_config(initial_dt=1.0, linear_restart=4),
    )
    assert bool(solution.converged)
    corrected = pseudo_arclength_residual(
        residual, solution.x, tangent=tangent, predictor=predictor
    )
    np.testing.assert_allclose(corrected[0], 0.0, atol=2e-11)
    np.testing.assert_allclose(corrected[1], 0.0, atol=2e-11)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"rtol": float("nan")}, "finite"),
        ({"rtol": -1.0}, "tolerances"),
        ({"max_steps": -1}, "iteration limits"),
        ({"min_dt": 2.0}, "min_dt"),
        ({"dt_growth": 0.5}, "dt_growth"),
        ({"newton_switch": 2.0}, "newton_switch"),
        ({"armijo": 1.0}, "armijo"),
        ({"backtrack_factor": 1.0}, "backtrack_factor"),
        ({"eta_initial": 0.95}, "eta_min"),
        ({"eta_gamma": 0.0}, "eta_gamma"),
        ({"linear_restart": 0}, "linear iteration"),
    ],
)
def test_pseudo_transient_config_validation(updates, match):
    with pytest.raises(ValueError, match=match):
        PseudoTransientConfig(**updates)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"target": float("nan")}, "finite"),
        ({"min_step": 0.2}, "min_step"),
        ({"growth": 1.0}, "growth"),
        ({"max_stages": 0}, "stage limits"),
    ],
)
def test_continuation_config_validation(updates, match):
    with pytest.raises(ValueError, match=match):
        ContinuationConfig(**updates)


def test_invalid_structures_empty_trees_and_alpha_are_rejected():
    with pytest.raises(ValueError, match="preserve"):
        pseudo_transient_continuation(lambda x: {"bad": x}, jnp.ones(2))
    with pytest.raises(ValueError, match="at least one"):
        pseudo_transient_continuation(lambda x: x, {})
    with pytest.raises(ValueError, match="at least one"):
        nonlinear_module._tree_finite({})
    with pytest.raises(ValueError, match="finite"):
        adaptive_continuation(lambda x, alpha: x, jnp.ones(1), alpha0=float("nan"))
