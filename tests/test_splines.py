"""Cubic splines against closed-form results and SciPy.

Anchors: a not-a-knot spline reproduces any cubic exactly (de Boor 2001,
chapter IV); the error of a C2 cubic interpolant of a smooth function is
O(h^4) in value, O(h^3) in slope and O(h^2) in curvature (Hall & Meyer,
J. Approx. Theory 16 (1976) 105); a periodic spline is C2 across its seam;
Fritsch-Carlson slopes keep monotone data monotone (Fritsch & Carlson, SIAM
J. Numer. Anal. 17 (1980) 238). SciPy's ``CubicSpline`` and
``PchipInterpolator`` implement the same systems and slopes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    cubic_spline,
    cubic_spline_antiderivative,
    cubic_spline_evaluate,
    cubic_spline_integrate,
)

jax.config.update("jax_enable_x64", True)
interpolate = pytest.importorskip("scipy.interpolate")

RNG = np.random.default_rng(7)
X = np.sort(np.r_[0.0, RNG.uniform(0.0, 2.0, 9), 2.0])  # 11 non-uniform knots
P = np.linspace(-0.3, 2.3, 57)  # inside, at and beyond the ends


def _eval_all(spline, p):
    return [np.asarray(cubic_spline_evaluate(spline, p, derivative=k)) for k in range(3)]


def test_not_a_knot_reproduces_a_cubic_beyond_the_knots():
    c = np.array([0.3, -1.2, 0.7, 2.0])
    f = np.polynomial.Polynomial(c)
    values = _eval_all(cubic_spline(X, f(X)), P)
    hmin = np.diff(X).min()  # rounding of a k-th derivative grows as eps / hmin**k
    for k, v in enumerate(values):
        np.testing.assert_allclose(v, f.deriv(k)(P), rtol=0, atol=1e-12 / hmin**k)


@pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
def test_matches_scipy_with_batch_axes(bc):
    p = np.linspace(0.0, 2.0, 41)[:-1] if bc == "periodic" else P
    y = RNG.normal(size=(X.size, 2, 3))
    if bc == "periodic":
        y[-1] = y[0]
    ref = interpolate.CubicSpline(X, y, bc_type=bc, axis=0)
    spline = cubic_spline(X, y, bc=bc)
    for k, v in enumerate(_eval_all(spline, p)):
        assert v.shape == (p.size, 2, 3)
        np.testing.assert_allclose(v, ref(p, k), rtol=0, atol=1e-12 * (1 + np.abs(ref(p, k)).max()))


def test_clamped_takes_end_slopes():
    spline = cubic_spline(X, np.sin(X), bc="clamped", end_slopes=(1.0, np.cos(2.0)))
    ref = interpolate.CubicSpline(X, np.sin(X), bc_type=((1, 1.0), (1, np.cos(2.0))))
    np.testing.assert_allclose(cubic_spline_evaluate(spline, P), ref(P), atol=1e-13)


@pytest.mark.parametrize("bc", ["not-a-knot", "periodic"])
def test_convergence_orders(bc):
    """Errors in value, slope and curvature fall as h^4, h^3, h^2."""
    f = [np.sin, np.cos, lambda t: -np.sin(t)]
    errors = []
    for n in (128, 256, 512):
        t = np.linspace(0.0, 2 * np.pi, n + 1)
        t[1:-1] += 0.3 * (t[1] - t[0]) * np.sin(3 * t[1:-1])  # smooth non-uniform grid
        y = np.sin(t)
        y[-1] = y[0] if bc == "periodic" else y[-1]
        spline = cubic_spline(t, y, bc=bc)
        q = np.linspace(0.0, 2 * np.pi, 2001)
        errors.append([np.abs(v - f[k](q)).max() for k, v in enumerate(_eval_all(spline, q))])
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    for k, expected in enumerate((4, 3, 2)):
        assert np.all(np.abs(orders[:, k] - expected) < 0.35), (k, orders[:, k])


def test_periodic_is_c2_across_the_seam_and_repeats():
    t = np.sort(np.r_[0.0, RNG.uniform(0, 2 * np.pi, 12), 2 * np.pi])
    y = np.stack([np.sin(t), np.cos(t)], axis=-1)
    y[-1] = y[0]
    spline = cubic_spline(t, y, bc="periodic")
    eps = 1e-9
    for k in range(3):
        left = cubic_spline_evaluate(spline, 2 * np.pi - eps, derivative=k)
        right = cubic_spline_evaluate(spline, eps, derivative=k)
        np.testing.assert_allclose(left, right, atol=1e-6)
        shifted = cubic_spline_evaluate(spline, P + 4 * np.pi, derivative=k)
        np.testing.assert_allclose(
            shifted, cubic_spline_evaluate(spline, P, derivative=k), atol=1e-12
        )


def test_monotone_matches_pchip_and_does_not_overshoot():
    x = np.array([0.0, 0.5, 1.0, 1.2, 2.0, 3.0, 3.1, 4.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    spline = cubic_spline(x, y, monotone=True)
    ref = interpolate.PchipInterpolator(x, y)
    q = np.linspace(-0.2, 4.2, 801)
    for k in range(3):
        np.testing.assert_allclose(
            cubic_spline_evaluate(spline, q, derivative=k), ref(q, k), atol=1e-12
        )
    inside = np.asarray(cubic_spline_evaluate(spline, q[(q >= 0) & (q <= 4)]))
    assert np.all(np.diff(inside) >= -1e-15) and inside.min() >= 0 and inside.max() <= 2
    wavy = RNG.normal(size=(x.size, 3))
    np.testing.assert_allclose(
        cubic_spline_evaluate(cubic_spline(x, wavy, monotone=True), q),
        interpolate.PchipInterpolator(x, wavy, axis=0)(q),
        atol=1e-12,
    )


def test_values_at_knots_and_small_knot_counts():
    y = RNG.normal(size=X.size)
    for bc in ("not-a-knot", "natural", "clamped"):
        np.testing.assert_allclose(
            cubic_spline_evaluate(cubic_spline(X, y, bc=bc), X), y, atol=1e-13
        )
    for n in (2, 3, 4):
        x, y = X[:n], np.cos(X[:n])
        for bc in ("not-a-knot", "natural", "clamped"):
            ref = interpolate.CubicSpline(x, y, bc_type=bc)
            np.testing.assert_allclose(
                cubic_spline_evaluate(cubic_spline(x, y, bc=bc), P), ref(P), atol=1e-11
            )
        if n > 2:
            np.testing.assert_allclose(
                cubic_spline_evaluate(cubic_spline(x, y, monotone=True), P),
                interpolate.PchipInterpolator(x, y)(P),
                atol=1e-12,
            )
    with pytest.raises(ValueError, match="at least 4"):
        cubic_spline(X[:3], np.zeros(3), bc="periodic")
    with pytest.raises(ValueError, match="at least 2"):
        cubic_spline(X[:1], np.zeros(1))
    with pytest.raises(ValueError, match="unknown bc"):
        cubic_spline(X, np.zeros(X.size), bc="cyclic")
    with pytest.raises(ValueError, match="no boundary"):
        cubic_spline(X, np.zeros(X.size), bc="natural", monotone=True)
    with pytest.raises(ValueError, match="len"):
        cubic_spline(X, np.zeros(3))
    with pytest.raises(ValueError, match="derivative"):
        cubic_spline_evaluate(cubic_spline(X, np.zeros(X.size)), P, derivative=3)


@pytest.mark.parametrize(
    ("bc", "monotone"), [("not-a-knot", False), ("periodic", False), ("not-a-knot", True)]
)
def test_derivatives_against_finite_differences(bc, monotone):
    y = np.sin(X) + 0.1 * RNG.normal(size=X.size)
    if bc == "periodic":
        y[-1] = y[0]
    p = np.linspace(0.05, 1.95, 7)

    def f(x, y, p):
        return cubic_spline_evaluate(cubic_spline(x, y, bc=bc, monotone=monotone), p)

    f = jax.jit(f)
    loss = lambda x, y, p: jnp.sum(jnp.sin(f(x, y, p)))  # noqa: E731
    gx, gy, gp = jax.grad(loss, argnums=(0, 1, 2))(X, y, p)
    for arg, g in enumerate((gx, gy, gp)):
        args = [X, y, p]
        v = RNG.normal(size=np.shape(args[arg]))
        if arg == 0:
            v[[0, -1]] = 0.0 if bc == "periodic" else v[[0, -1]]
        if arg == 1 and bc == "periodic":
            v[-1] = v[0]
        eps = 1e-6
        plus, minus = list(args), list(args)
        plus[arg], minus[arg] = args[arg] + eps * v, args[arg] - eps * v
        fd = (loss(*plus) - loss(*minus)) / (2 * eps)
        np.testing.assert_allclose(np.vdot(g, v), fd, rtol=1e-7, atol=1e-9)
    spline = cubic_spline(X, y, bc=bc, monotone=monotone)
    for k in (0, 1):
        _, tangent = jax.jvp(
            lambda q, k=k: cubic_spline_evaluate(spline, q, derivative=k), (p,), (np.ones_like(p),)
        )
        np.testing.assert_allclose(
            tangent, cubic_spline_evaluate(spline, p, derivative=k + 1), atol=1e-12
        )


def test_vmap_and_float32():
    ys = RNG.normal(size=(4, X.size))
    batched = jax.vmap(lambda y: cubic_spline_evaluate(cubic_spline(X, y), P))(ys)
    together = cubic_spline_evaluate(cubic_spline(X, ys.T), P)
    np.testing.assert_allclose(batched, together.T, atol=1e-13)
    s32 = cubic_spline(X.astype(np.float32), ys[0].astype(np.float32), bc="natural")
    v32 = cubic_spline_evaluate(s32, P.astype(np.float32))
    assert v32.dtype == jnp.float32
    ref = interpolate.CubicSpline(X, ys[0], bc_type="natural")(P)
    np.testing.assert_allclose(v32, ref, atol=1e-4 * (1 + np.abs(ref).max()))


def test_complex_data_on_real_knots_matches_scipy_parts():
    y = np.exp(1j * X) + RNG.normal(size=X.size)
    spline = cubic_spline(X, y, bc="natural")
    assert spline.x.dtype == np.float64
    for k in range(3):
        v = np.asarray(cubic_spline_evaluate(spline, P, derivative=k))
        for part in (np.real, np.imag):
            ref = interpolate.CubicSpline(X, part(y), bc_type="natural")(P, k)
            np.testing.assert_allclose(part(v), ref, atol=1e-12)
    with pytest.raises(ValueError, match="real data"):
        cubic_spline(X, y, monotone=True)
    with pytest.raises(ValueError, match="real"):
        cubic_spline(X + 0j, np.real(y))


def test_periodic_ignores_last_value_and_knots_are_checked():
    # Fixed knots and a generator of its own: the seam check below must not
    # depend on which tests drew from the shared generator first.
    t = np.array([0.0, 0.07, 0.2, 0.33, 0.5, 0.61, 0.78, 0.9, 1.0])
    y = np.random.default_rng(11).normal(size=t.size)
    y[-1] = y[0] + 5.0  # ignored
    spline = cubic_spline(t, y, bc="periodic")
    y[-1] = y[0]
    ref = interpolate.CubicSpline(t, y, bc_type="periodic")
    for k in range(3):
        # 1e-12 from the seam: the jump allowed is 1e-12 times the next derivative.
        left = cubic_spline_evaluate(spline, 1 - 1e-12, derivative=k)
        np.testing.assert_allclose(
            left, cubic_spline_evaluate(spline, 0.0, derivative=k), atol=1e-6
        )
        np.testing.assert_allclose(
            cubic_spline_evaluate(spline, P, derivative=k), ref(P % 1, k), atol=1e-12
        )
    for bad in (X[::-1], np.r_[X[:3], X[2:]]):
        for bc in ("not-a-knot", "natural", "clamped", "periodic"):
            with pytest.raises(ValueError, match="increasing"):
                cubic_spline(bad, np.zeros(bad.size), bc=bc)
    traced = jax.jit(lambda x, y: cubic_spline_evaluate(cubic_spline(x, y), P))(X, np.sin(X))
    np.testing.assert_allclose(traced, interpolate.CubicSpline(X, np.sin(X))(P), atol=1e-12)


@pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
def test_integral_matches_scipy_and_its_derivative_is_the_value(bc):
    y = RNG.normal(size=(X.size, 2))
    if bc == "periodic":
        y[-1] = y[0]
    spline = cubic_spline(X, y, bc=bc)
    ref = interpolate.CubicSpline(X, y, bc_type=bc)
    a, b = np.array([-0.2, 0.3, 1.9]), np.array([2.4, 1.1, 0.4])  # last pair reversed
    if bc == "periodic":
        b = b + np.array([4.0, 0.0, -6.0])  # several periods both ways
    expected = np.stack([ref.integrate(lo, hi) for lo, hi in zip(a, b, strict=True)])
    integral = jax.jit(cubic_spline_integrate)
    np.testing.assert_allclose(integral(spline, a, b), expected, atol=1e-12)
    np.testing.assert_allclose(jax.vmap(integral, (None, 0, 0))(spline, a, b), expected, atol=1e-12)
    grad_b = jax.grad(lambda b: cubic_spline_integrate(spline, 0.1, b)[0])(1.3)
    np.testing.assert_allclose(grad_b, cubic_spline_evaluate(spline, 1.3)[0], atol=1e-13)
    grad_a = jax.grad(lambda a: cubic_spline_integrate(spline, a, 1.3)[1])(0.1)
    np.testing.assert_allclose(grad_a, -cubic_spline_evaluate(spline, 0.1)[1], atol=1e-13)


def test_integral_of_a_cubic_is_exact_and_differentiates_in_the_data():
    f = np.polynomial.Polynomial([0.3, -1.2, 0.7, 2.0])
    F = f.integ()
    spline = cubic_spline(X, f(X))
    np.testing.assert_allclose(cubic_spline_integrate(spline, -0.5, P), F(P) - F(-0.5), atol=1e-11)
    y = np.sin(X) + 1j * np.cos(X)
    complex_integral = cubic_spline_integrate(cubic_spline(X, y), 0.2, 1.7)
    for part in (np.real, np.imag):
        ref = interpolate.CubicSpline(X, part(y)).integrate(0.2, 1.7)
        np.testing.assert_allclose(part(complex_integral), ref, atol=1e-13)
    loss = lambda y: cubic_spline_integrate(cubic_spline(X, y, bc="natural"), 0.1, 1.9) ** 2  # noqa: E731
    y, v, eps = np.cos(X), RNG.normal(size=X.size), 1e-6
    fd = (loss(y + eps * v) - loss(y - eps * v)) / (2 * eps)
    np.testing.assert_allclose(np.vdot(jax.grad(loss)(y), v), fd, rtol=1e-8)


def test_cumulative_integral_equals_the_clamped_profile_formula():
    """VMEX's profile spline: second derivatives m of a clamped spline, and the
    integral of interval i from x_i to x_i + u written in m (Press et al.,
    Numerical Recipes, 3rd ed., eq. 3.3.3 integrated)."""
    x = np.array([0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 1.0])
    y, slopes = np.cos(3 * x) + x, (0.4, -1.1)
    spline = cubic_spline(x, y, bc="clamped", end_slopes=slopes)
    m = np.asarray(cubic_spline_evaluate(spline, x, derivative=2))
    h = np.diff(x)

    def piece(i, u):
        a, b = y[i] - m[i] * h[i] ** 2 / 6, y[i + 1] - m[i + 1] * h[i] ** 2 / 6
        return (
            m[i] * (h[i] ** 4 - (h[i] - u) ** 4) / (24 * h[i])
            + m[i + 1] * u**4 / (24 * h[i])
            + a * (u - u**2 / (2 * h[i]))
            + b * u**2 / (2 * h[i])
        )

    q = np.linspace(0.0, 1.0, 23)
    i = np.clip(np.searchsorted(x, q, side="right") - 1, 0, x.size - 2)
    prefix = np.r_[0.0, np.cumsum([piece(k, h[k]) for k in range(h.size)])]
    reference = prefix[i] + piece(i, q - x[i])
    np.testing.assert_allclose(cubic_spline_integrate(spline, 0.0, q), reference, atol=1e-15)


@pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
def test_antiderivative_is_the_integral_from_the_first_knot(bc):
    y = RNG.normal(size=(X.size, 2))
    if bc == "periodic":
        y[-1] = y[0]
    spline = cubic_spline(X, y, bc=bc)
    q = np.r_[P, P + 5.0, P - 7.0] if bc == "periodic" else P  # outside the knots
    F = jax.jit(cubic_spline_antiderivative)
    np.testing.assert_allclose(F(spline, q), cubic_spline_integrate(spline, X[0], q), atol=1e-13)
    np.testing.assert_allclose(F(spline, X[0]), 0.0, atol=0)
    scalar = jax.vmap(lambda p: cubic_spline_antiderivative(spline, p)[0])
    np.testing.assert_allclose(scalar(q), F(spline, q)[:, 0], atol=1e-14)
    slope = jax.vmap(jax.grad(lambda p: cubic_spline_antiderivative(spline, p)[1]))(q)
    np.testing.assert_allclose(slope, cubic_spline_evaluate(spline, q)[:, 1], atol=1e-13)
