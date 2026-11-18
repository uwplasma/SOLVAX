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


# --------------------------------------------------------------------------
# IMEX, Lawson, SSP steps, step control and the chunked driver
# --------------------------------------------------------------------------
from solvax import (  # noqa: E402
    IMEX_TABLEAUX,
    SMR_RK3_CN,
    SSPRK3,
    SSPRK43,
    SSPRK104,
    cfl_dt,
    controlled,
    imex_step,
    integrate,
    lawson_step,
    ssp_step,
)


def observed_order(advance_n, exact_value, ns=(20, 40, 80)):
    errs = [float(np.abs(np.asarray(advance_n(n)) - exact_value).max()) for n in ns]
    return np.log2(errs[-2] / errs[-1]), errs


# y' = (-2 + cos t / 2) y [implicit] + (cos t / 2) y [explicit]: y = exp(-2 t + sin t).
def f_imp(t, y):
    return (-2.0 + 0.5 * jnp.cos(t)) * y


def f_exp(t, y):
    return 0.5 * jnp.cos(t) * y


def solve_imp(t, r, gamma_h):
    return r / (1.0 - gamma_h * (-2.0 + 0.5 * jnp.cos(t)))


@pytest.mark.parametrize("name", list(IMEX_TABLEAUX))
def test_imex_tableaux_are_consistent_and_converge_at_their_order(name):
    tab = IMEX_TABLEAUX[name]
    assert np.allclose(sum(tab.b_exp), 1.0) and np.allclose(sum(tab.b_imp), 1.0)
    for i, row in enumerate(tab.a_exp):
        assert all(v == 0.0 for v in row[i:])
    t_end = 1.0

    def advance_n(n):
        h, y = t_end / n, jnp.array([1.0])
        for i in range(n):
            y, _ = imex_step(f_exp, f_imp, solve_imp, i * h, y, h, tab)
        return y

    p, _ = observed_order(advance_n, np.exp(-2 * t_end + np.sin(t_end)))
    assert p == pytest.approx(tab.order, abs=0.25), p


def test_imex_embedded_error_and_stiff_stability():
    tab = IMEX_TABLEAUX["ark3"]
    _, err = imex_step(f_exp, f_imp, solve_imp, 0.0, jnp.array([1.0]), 0.1, tab)
    assert 0 < float(jnp.abs(err[0])) < 1e-3
    # A stiff linear part (h |lambda| = 1e4) stays bounded with an L-stable implicit part.
    for name in ("ark3", "ars343", "smr_rk3_cn", "ssp2", "euler"):
        y = jnp.array([1.0])
        for _ in range(10):
            y, _ = imex_step(
                lambda t, y: 0.0 * y, lambda t, y: -1e5 * y,
                lambda t, r, g: r / (1 + 1e5 * g), 0.0, y, 0.1, IMEX_TABLEAUX[name],
            )
        assert float(jnp.abs(y[0])) <= 1.0, name


def test_smr_rk3_cn_matches_its_substep_form():
    # u_k = u_{k-1} + h [g_k N(u_{k-1}) + z_k N(u_{k-2}) + a_k (L u_{k-1} + L u_k)].
    lam, h = -3.0, 0.2
    n_fun = lambda y: jnp.sin(y)  # noqa: E731
    g, z, a = (8 / 15, 5 / 12, 3 / 4), (0.0, -17 / 60, -5 / 12), (4 / 15, 1 / 15, 1 / 6)
    u, prev = jnp.array([0.7]), None
    for k in range(3):
        rhs = u + h * (g[k] * n_fun(u) + (z[k] * n_fun(prev) if k else 0.0) + a[k] * lam * u)
        prev, u = u, rhs / (1 - h * a[k] * lam)
    got, _ = imex_step(
        lambda t, y: n_fun(y), lambda t, y: lam * y, lambda t, r, gh: r / (1 - gh * lam),
        0.0, jnp.array([0.7]), h, SMR_RK3_CN,
    )
    np.testing.assert_allclose(got, u, rtol=1e-14)


# y' = lam y + cos(t) y with complex lam: y = exp(lam t + sin t).
LAM = -0.5 + 3.0j


@pytest.mark.parametrize("pair", list(PAIRS.values()), ids=list(PAIRS))
def test_lawson_converges_at_pair_order_and_is_exact_for_the_linear_part(pair):
    prop = lambda x, tau: jnp.exp(LAM * tau) * x  # noqa: E731
    t_end = 1.0

    def advance_n(n, f=lambda t, y: jnp.cos(t) * y):
        h, y = t_end / n, jnp.array([1.0 + 0.0j])
        for i in range(n):
            y, err, _ = lawson_step(f, prop, i * h, y, h, pair)
        return y

    p, _ = observed_order(advance_n, np.exp(LAM * t_end + np.sin(t_end)), ns=(10, 20, 40))
    assert p == pytest.approx(pair.order, abs=0.3), p
    linear = advance_n(3, f=lambda t, y: 0.0 * y)
    np.testing.assert_allclose(linear, np.exp(LAM * t_end), rtol=1e-13)


def test_lawson_rk4_matches_the_hand_written_integrating_factor_step():
    decay, h, y = 0.3, 0.1, jnp.array([0.4])
    n_fun = lambda t, w: -(w**2)  # noqa: E731
    d, half = jnp.exp(-decay * h), jnp.exp(-decay * h / 2)
    k1 = n_fun(0, y)
    k2 = n_fun(0, half * (y + 0.5 * h * k1))
    k3 = n_fun(0, half * y + 0.5 * h * k2)
    k4 = n_fun(0, d * y + h * half * k3)
    ref = d * y + h / 6 * (d * k1 + 2 * half * (k2 + k3) + k4)
    got, _, _ = lawson_step(n_fun, lambda x, tau: jnp.exp(-decay * tau) * x, 0.0, y, h, RK4_2)
    np.testing.assert_allclose(got, ref, rtol=1e-14)


@pytest.mark.parametrize("scheme", [SSPRK3, SSPRK43, SSPRK104], ids=["3", "43", "104"])
def test_ssp_converges_at_its_order(scheme):
    t_end = 2.0

    def advance_n(n):
        h, y = t_end / n, {"y": jnp.array([1.0])}
        for i in range(n):
            y = ssp_step(lambda t, s: {"y": jnp.cos(t) * s["y"]}, i * h, y, h, scheme)
        return y["y"]

    p, _ = observed_order(advance_n, np.exp(np.sin(t_end)), ns=(10, 20, 40))
    assert p == pytest.approx(scheme.order, abs=0.3), p


def test_ssp104_is_stable_at_its_ssp_coefficient():
    # Upwind advection: forward Euler is TVD for h <= dx; SSPRK104 allows 6 dx.
    n = 64
    x = jnp.where(jnp.arange(n) < n // 2, 1.0, 0.0)
    rhs = lambda t, u: -(u - jnp.roll(u, 1)) * n  # noqa: E731
    for _ in range(20):
        x = ssp_step(rhs, 0.0, x, 6.0 / n, SSPRK104)
    assert float(x.max()) <= 1 + 1e-12 and float(x.min()) >= -1e-12


def test_cfl_dt_clips_and_handles_zero_rate():
    assert float(cfl_dt(4.0, 0.5)) == 0.125
    assert float(cfl_dt(0.0, 0.5, dt_max=0.3)) == 0.3
    assert float(cfl_dt(1e-6, 0.5, dt_max=0.3)) == 0.3
    assert float(cfl_dt(1e9, 0.5, dt_min=1e-3)) == 1e-3


def rk4_map(t, y, h):
    return erk_step(oscillator, t, y, h, RK4_2)[0]


def test_integrate_fixed_steps_match_a_python_loop_across_chunk_remainders():
    ref = Y0
    for i in range(23):
        ref = rk4_map(i * 0.05, ref, 0.05)
    for chunk in (1, 5, 23, 40):
        out = integrate(controlled(rk4_map), Y0, 0.0, 0.05, 23, chunk=chunk)
        np.testing.assert_allclose(out.y, ref, rtol=1e-13)
        assert int(out.steps) == 23 and bool(out.ok) and float(out.t) == pytest.approx(1.15)


def test_integrate_stops_at_t1_with_a_clipped_last_step():
    out = integrate(controlled(rk4_map), Y0, 0.0, 0.3, 100, t1=1.0, chunk=4)
    assert float(out.t) == 1.0 and int(out.steps) == 4
    np.testing.assert_allclose(out.y, exact(1.0), atol=1e-3)


def test_integrate_adaptive_with_pi_control_and_cfl_cap():
    step = controlled(
        lambda t, y, h: erk_step(oscillator, t, y, h, DOPRI5_4)[:2],
        embedded_order=4, rtol=1e-9, atol=1e-12,
    )
    out = integrate(step, Y0, 0.0, 1.0, 500, t1=10.0, chunk=50)
    assert bool(out.ok) and float(out.t) == 10.0 and int(out.rejected) > 0
    assert float(jnp.abs(out.y - exact(10.0)).max()) < 1e-6
    capped = controlled(rk4_map, dt_limit=lambda t, y: cfl_dt(2.0, 0.1))
    out = integrate(capped, Y0, 0.0, 0.01, 10, chunk=5)
    assert float(out.dt) == pytest.approx(0.05)


def blowup(t, y):
    return y**2  # y = 1 / (1 - t) from y(0) = 1


def test_integrate_flags_non_finite_and_keeps_the_last_good_state():
    euler = controlled(lambda t, y, h: y + h * blowup(t, y))
    out = integrate(euler, jnp.array([1.0]), 0.0, 1.0, 50, chunk=10, max_retries=0)
    assert not bool(out.ok) and bool(jnp.isfinite(out.y).all())
    assert int(out.rejected) == 1 and int(out.steps) > 0
    # Halving on a non-finite state lets a retried run go further.
    retried = integrate(euler, jnp.array([1.0]), 0.0, 1.0, 50, chunk=10, max_retries=8)
    assert float(retried.t) >= float(out.t)


def test_integrate_halves_the_step_after_a_non_finite_attempt():
    def step(t, y, h, aux):
        return jnp.where(h > 0.3, jnp.nan, y + h), h, True, aux

    out = integrate(step, jnp.array([0.0]), 0.0, 1.0, 6, t1=1.0, chunk=3, max_retries=2)
    assert bool(out.ok) and float(out.t) == 1.0 and int(out.rejected) == 2
    np.testing.assert_allclose(out.y, [1.0])


def test_integrate_gradient_with_and_without_checkpoint():
    def final(scale, checkpoint):
        f = lambda t, y: jnp.stack([y[1], -(scale**2) * y[0]])  # noqa: E731
        step = controlled(lambda t, y, h: erk_step(f, t, y, h, RK4_2)[0])
        return integrate(step, Y0, 0.0, 0.01, 200, chunk=14, checkpoint=checkpoint).y[0]

    g_ck = jax.grad(final)(1.3, True)
    g_plain = jax.grad(final)(1.3, False)
    assert float(g_ck) == pytest.approx(float(g_plain), rel=1e-12)
    assert float(g_ck) == pytest.approx(-2.0 * np.sin(2.6), rel=1e-7)  # d cos(2 s) / ds


def test_integrate_jits_vmaps_and_records(capfd):
    def run(y0):
        return integrate(
            controlled(rk4_map), y0, 0.0, 0.1, 30, chunk=10,
            observe=lambda t, y: y[0], progress=True,
        )

    out = jax.jit(run)(Y0)
    assert out.saved.shape == (3,)
    lines = [ln for ln in capfd.readouterr().out.splitlines() if ln.startswith("chunk")]
    assert len(lines) == 3 and "step 30" in lines[-1]
    batch = jax.vmap(lambda y: integrate(controlled(rk4_map), y, 0.0, 0.1, 30, chunk=10).y)(
        jnp.stack([Y0, 2 * Y0])
    )
    np.testing.assert_allclose(batch[1], 2 * batch[0], rtol=1e-13)
