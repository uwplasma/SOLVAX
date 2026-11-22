"""Tests for pseudo-transient, forcing, homotopy, and arclength solves.

Forcing terms are checked against Eisenstat and Walker (1996), SIAM J. Sci.
Comput. 17, 16, choice 2 with the safeguards of their section 2.1.
"""

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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_pseudo_transient_rejects_nonfinite_initial_residual(bad_value):
    result = jax.jit(lambda: pseudo_transient_continuation(
        lambda x: jnp.full_like(x, bad_value), jnp.zeros(1)
    ))()
    assert not bool(result.converged)
    assert not np.isfinite(float(result.residual_norm))
    assert int(result.steps) == 0
    assert int(result.linear_iterations) == 0


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


_TIGHT_FIXED_FORCING = dict(eta_initial=1.0e-6, eta_min=1.0e-6, eta_max=1.0e-6)


def _choice_2_forcing(norms, config):
    """Eisenstat--Walker (1996) choice 2, eq. (2.6), with the safeguard of their
    section 2.1, the cap eta_max of their section 3.1 and Kelley's floor
    0.5*tol/||F_k|| tied to the outer tolerance, written out in NumPy."""

    tolerance = max(config.atol, config.rtol * norms[0])
    eta = [config.eta_initial]
    for previous, current in zip(norms[:-1], norms[1:], strict=True):
        candidate = config.eta_gamma * (current / previous) ** config.eta_power
        safeguard = config.eta_gamma * eta[-1] ** config.eta_power
        if safeguard > 0.1:
            candidate = max(candidate, safeguard)
        candidate = max(candidate, 0.5 * tolerance / current)
        eta.append(min(config.eta_max, max(config.eta_min, candidate)))
    return np.asarray(eta)


def test_inexact_forcing_follows_choice_2_without_oversolving():
    # Graded diagonal (1 to 1e5) with GMRES(20) on 32 unknowns: Newton needs four
    # steps, and the inner cost depends on the right-hand side, not only on eta.
    # After a tight solve the next residual is the quadratic remainder, which
    # GMRES(20) reduces by 1e-4 in 26-217 iterations; after a loose solve it is
    # the GMRES residual, spread over the spectrum, which takes 330-391. The two
    # totals (1127 and 1119 with JAX 0.12.1, 1170 and 1352 with 0.12.0.1) are
    # therefore not ordered by the method, and only its guarantees are asserted.
    size = 32
    diagonal = jnp.logspace(0.0, 5.0, size)
    matrix = jnp.diag(diagonal) + 0.2 * jnp.diag(diagonal[:-1], 1)
    residual = lambda x: matrix @ x + 0.2 * x**3 - 1.0  # noqa: E731
    common = dict(
        rtol=1.0e-9,
        initial_dt=1.0e8,
        max_dt=1.0e8,
        linear_restart=20,
        linear_max_restarts=40,
        max_backtracks=4,
    )
    config = PseudoTransientConfig(max_steps=20, **common)
    adaptive = pseudo_transient_continuation(residual, jnp.zeros(size), config=config)
    fixed = pseudo_transient_continuation(
        residual,
        jnp.zeros(size),
        config=PseudoTransientConfig(max_steps=20, **common, **_TIGHT_FIXED_FORCING),
    )
    assert bool(adaptive.converged) and bool(fixed.converged)
    np.testing.assert_allclose(adaptive.x, fixed.x, rtol=1.0e-8, atol=0.0)
    steps = int(adaptive.steps)
    # Looser inner solves cost Newton steps; every full step is accepted here.
    assert steps > int(fixed.steps)
    for run in (adaptive, fixed):
        assert int(run.rejected_steps) == 0
        assert int(run.residual_evaluations) == 1 + 2 * int(run.steps)

    norms = np.asarray(adaptive.history.residual_norm[: steps + 1])
    eta = np.asarray(adaptive.history.eta[: steps + 1])
    np.testing.assert_allclose(eta, _choice_2_forcing(norms, config), rtol=1.0e-12)

    # GMRES stops at the first iterate with ||F_k + J_k s_k|| <= eta_k ||F_k||,
    # so the ratio is at most one; one GMRES(20) iteration on this spectrum
    # reduces the residual by less than a factor 10 (0.92-1.00 measured), so a
    # ratio below 0.1 would mean the inner solve ran past its forcing term.
    states = [
        pseudo_transient_continuation(
            residual, jnp.zeros(size), config=PseudoTransientConfig(max_steps=k, **common)
        ).x
        for k in range(steps + 1)
    ]
    for k in range(steps):
        value, jvp = jax.linearize(residual, states[k])
        step = states[k + 1] - states[k]
        inner = float(jnp.linalg.norm(value + jvp(step) + step / common["max_dt"]))
        assert 0.1 * eta[k] * norms[k] <= inner <= (1.0 + 1.0e-6) * eta[k] * norms[k]


@pytest.mark.parametrize("size", [32, 64, 128])
def test_inexact_forcing_saves_krylov_work_on_shifted_laplacian(size):
    # u'' discretized with eigenvalues 2 - 2 cos(k pi / (size + 1)), shifted by
    # sigma = 0.01, plus a cubic: the Jacobian condition number stays below
    # (4 + sigma) / sigma = 401 at every size, so the GMRES cost per requested
    # digit is bounded independently of the right-hand side. With the default
    # pseudo-time stages the fixed run asks for 6 digits in each of 7 steps
    # (42); choice 2 asks for about 11 over 8 steps. Measured iterations:
    # 28/87, 29/87, 29/84 for sizes 32, 64, 128 with JAX 0.12.1 (ratio 0.32-0.35).
    def residual(u):
        laplacian = 2.0 * u - jnp.pad(u[:-1], (1, 0)) - jnp.pad(u[1:], (0, 1))
        return laplacian + 0.01 * u + u**3 - 0.2

    common = dict(rtol=1.0e-9, max_steps=60)
    adaptive = pseudo_transient_continuation(
        residual, jnp.zeros(size), config=PseudoTransientConfig(**common)
    )
    fixed = pseudo_transient_continuation(
        residual,
        jnp.zeros(size),
        config=PseudoTransientConfig(**common, **_TIGHT_FIXED_FORCING),
    )
    assert bool(adaptive.converged) and bool(fixed.converged)
    np.testing.assert_allclose(adaptive.x, fixed.x, rtol=1.0e-8, atol=0.0)

    def requested_digits(run):
        return float(-np.sum(np.log10(np.asarray(run.history.eta[: int(run.steps)]))))

    assert requested_digits(adaptive) < 0.5 * requested_digits(fixed)
    assert int(adaptive.linear_iterations) < 0.5 * int(fixed.linear_iterations)


def test_forcing_floor_is_tied_to_the_outer_tolerance():
    # Choice 2 alone gives 0.9*(1e-8/1e-4)**2 = 9e-9; Kelley's floor
    # 0.5*tol/||F_k|| = 0.5*1e-10/1e-8 = 5e-3 keeps the inner solve from asking
    # for more than the outer stop needs.
    eta = eisenstat_walker_forcing(1.0e-8, 1.0e-4, 1.0e-3, tolerance=1.0e-10)
    assert float(eta) == pytest.approx(5.0e-3, rel=1.0e-12)
    assert float(eisenstat_walker_forcing(1.0e-8, 1.0e-4, 1.0e-3)) == pytest.approx(9.0e-9)
    assert PseudoTransientConfig().eta_min == 0.0


@pytest.mark.parametrize(
    "jacobi,rtol,newton_steps",
    [(False, 1.0e-13, 6), (True, 1.0e-10, 4)],
)
def test_tolerance_floor_keeps_the_superlinear_tail(jacobi, rtol, newton_steps):
    # The graded diagonal of the inexact-forcing test. A fixed floor eta >= 1e-4
    # makes the tail linear (contraction about 1e-4 per step) and costs one more
    # Newton step than choice 2 with the tolerance floor. Regression counts with
    # JAX 0.12.1: 7 and 5 steps with the former floor 1e-4, 6 and 4 without it.
    size = 32
    diagonal = jnp.logspace(0.0, 5.0, size)
    matrix = jnp.diag(diagonal) + 0.2 * jnp.diag(diagonal[:-1], 1)
    residual = lambda x: matrix @ x + 0.2 * x**3 - 1.0  # noqa: E731

    def precond(x, rhs, dt):
        return rhs / (diagonal + 0.6 * x**2 + 1.0 / dt)

    config = PseudoTransientConfig(
        rtol=rtol,
        max_steps=20,
        initial_dt=1.0e8,
        max_dt=1.0e8,
        linear_restart=20,
        linear_max_restarts=40,
        max_backtracks=4,
    )
    solution = pseudo_transient_continuation(
        residual, jnp.zeros(size), precond=precond if jacobi else None, config=config
    )
    assert bool(solution.converged)
    steps = int(solution.steps)
    norms = np.asarray(solution.history.residual_norm[: steps + 1])
    eta = np.asarray(solution.history.eta[:steps])
    tolerance = rtol * norms[0]
    # Every inner solve asks for an absolute residual of at least 0.5 tol.
    assert np.all(eta * norms[:steps] >= 0.5 * tolerance * (1.0 - 1.0e-12))
    assert steps == newton_steps


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


def test_continuation_accepts_parameter_dependent_preconditioner():
    solution = adaptive_continuation(
        lambda x, alpha: x**2 - (1.0 + alpha),
        jnp.asarray([1.0]),
        nonlinear_config=_scalar_config(initial_dt=0.1),
        continuation_config=ContinuationConfig(target=0.5, initial_step=0.25),
        parameterized_precond=lambda state, rhs, dt, alpha: rhs
        / (2.0 * state + 1.0 / dt + 0.0 * alpha),
    )
    assert solution.converged
    np.testing.assert_allclose(solution.x, np.sqrt(1.5), rtol=3e-10)


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


def test_pseudo_arclength_reuses_compile_for_dynamic_branch_data():
    traces = 0

    def residual(x, alpha):
        nonlocal traces
        traces += 1
        return jnp.asarray([x[0] ** 2 + alpha - 1.0])

    def precond(state, rhs, dt, tangent, predictor):
        del state, tangent, predictor
        return rhs[0] / (1.0 + 1.0 / dt), rhs[1] / (1.0 + 1.0 / dt)

    config = _scalar_config(initial_dt=1.0, linear_restart=4)
    common = dict(
        config=config,
        parameterized_precond=precond,
        norm=lambda state: jnp.sqrt(
            jnp.vdot(state[0], state[0]).real + state[1] ** 2
        ),
    )
    first = pseudo_arclength_corrector(
        residual,
        (jnp.asarray([0.12]), jnp.asarray(0.97)),
        tangent=(jnp.asarray([-1.0]), jnp.asarray(0.2)),
        predictor=(jnp.asarray([0.1]), jnp.asarray(0.98)),
        **common,
    )
    first_traces = traces
    second = pseudo_arclength_corrector(
        residual,
        (jnp.asarray([0.2]), jnp.asarray(0.95)),
        tangent=(jnp.asarray([-0.9]), jnp.asarray(0.3)),
        predictor=(jnp.asarray([0.18]), jnp.asarray(0.96)),
        **common,
    )
    assert first_traces > 0
    assert traces == first_traces
    assert bool(first.converged)
    assert bool(second.converged)


def test_pseudo_arclength_rejects_two_preconditioners():
    with pytest.raises(ValueError, match="precond or parameterized_precond"):
        pseudo_arclength_corrector(
            lambda x, alpha: x + alpha,
            (jnp.asarray([0.0]), jnp.asarray(0.0)),
            tangent=(jnp.asarray([1.0]), jnp.asarray(1.0)),
            predictor=(jnp.asarray([0.0]), jnp.asarray(0.0)),
            precond=lambda state, rhs, dt: rhs,
            parameterized_precond=lambda state, rhs, dt, tangent, predictor: rhs,
        )


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
    with pytest.raises(ValueError, match="precond or parameterized_precond"):
        adaptive_continuation(
            lambda x, alpha: x,
            jnp.ones(1),
            precond=lambda state, rhs, dt: rhs,
            parameterized_precond=lambda state, rhs, dt, alpha: rhs,
        )
