import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    BS3_2,
    DOPRI5_4,
    PAIRS,
    RK4_2,
    PIController,
    erk_integrate,
    erk_step,
    error_norm,
    step_factor,
)

jax.config.update("jax_enable_x64", True)


def oscillator(t, y):
    return jnp.stack([y[1], -y[0]])


def exact(t):
    return jnp.array([np.cos(t), -np.sin(t)])


Y0 = jnp.array([1.0, 0.0])


def advance(pair, n, t_end=2.0):
    h = t_end / n
    y = Y0
    for i in range(n):
        y, _, _ = erk_step(oscillator, i * h, y, h, pair)
    return float(jnp.abs(y - exact(t_end)).max())


@pytest.mark.parametrize("pair", list(PAIRS.values()), ids=list(PAIRS))
def test_observed_order_matches_tableau(pair):
    e1, e2 = advance(pair, 40), advance(pair, 80)
    assert np.log2(e1 / e2) == pytest.approx(pair.order, abs=0.15)


@pytest.mark.parametrize("pair", list(PAIRS.values()), ids=list(PAIRS))
def test_embedded_estimate_tracks_embedded_error(pair):
    # The estimate is the difference of two solutions, so it scales as h^(q+1).
    h1, h2 = 0.2, 0.1
    err = [float(jnp.abs(erk_step(oscillator, 0.0, Y0, h, pair)[1]).max()) for h in (h1, h2)]
    assert np.log2(err[0] / err[1]) == pytest.approx(pair.embedded_order + 1, abs=0.4)


@pytest.mark.parametrize("pair", [BS3_2, DOPRI5_4])
def test_fsal_stage_is_the_next_first_stage(pair):
    y_new, _, k_last = erk_step(oscillator, 0.0, Y0, 0.1, pair)
    np.testing.assert_allclose(k_last, oscillator(0.1, y_new), atol=1e-14)
    y2, _, _ = erk_step(oscillator, 0.1, y_new, 0.1, pair, k1=k_last)
    y2_ref, _, _ = erk_step(oscillator, 0.1, y_new, 0.1, pair)
    np.testing.assert_allclose(y2, y2_ref, atol=1e-14)


def test_project_applies_to_stages_only():
    calls = []

    def project(x):
        calls.append(1)
        return x

    erk_step(oscillator, 0.0, Y0, 0.1, RK4_2, project=project)
    assert len(calls) == 3


@pytest.mark.parametrize("pair", list(PAIRS.values()), ids=list(PAIRS))
def test_integrate_meets_tolerance_and_counts_work(pair):
    sol = erk_integrate(oscillator, Y0, 0.0, 10.0, pair=pair, rtol=1e-8, atol=1e-10)
    assert float(sol.t) == 10.0
    assert float(jnp.abs(sol.y - exact(10.0)).max()) < 1e-5
    assert int(sol.steps) > 0 and int(sol.evals) > int(sol.steps)


def test_integrate_rejects_and_recovers_from_a_large_first_step():
    sol = erk_integrate(oscillator, Y0, 0.0, 5.0, pair=DOPRI5_4, rtol=1e-9, atol=1e-12, h0=2.0)
    assert int(sol.rejected) > 0
    assert float(jnp.abs(sol.y - exact(5.0)).max()) < 1e-6


def test_error_norm_weights_pytree():
    y = {"a": jnp.ones(3), "b": jnp.ones(1)}
    err = {"a": jnp.full(3, 2e-3), "b": jnp.full(1, 2e-3)}
    assert float(error_norm(err, y, y, 1e-3, 0.0)) == pytest.approx(2.0)


def test_step_factor_shrinks_above_tolerance_and_clips():
    ctrl = PIController()
    assert float(step_factor(ctrl, jnp.asarray(4.0), jnp.asarray(1.0), 2)) < 1.0
    assert float(step_factor(ctrl, jnp.asarray(0.5), jnp.asarray(1.0), 2)) > 1.0
    assert float(step_factor(ctrl, jnp.asarray(0.0), jnp.asarray(1.0), 2)) == pytest.approx(
        ctrl.fac_max
    )
    assert float(step_factor(ctrl, jnp.asarray(1e30), jnp.asarray(1.0), 2)) == pytest.approx(
        ctrl.fac_min
    )


def test_scan_compatible():
    def run(y):
        def body(carry, _):
            return erk_step(oscillator, 0.0, carry, 0.01, RK4_2)[0], None

        return jax.lax.scan(body, y, None, length=5)[0]

    assert np.isfinite(np.asarray(jax.jit(run)(Y0))).all()
