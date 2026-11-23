"""Tests for solvax.special against closed forms, identities and SciPy.

Elliptic integrals: the arithmetic-geometric mean (Abramowitz and Stegun 17.6.1-17.6.4), the
logarithmic limit (A&S 17.3.26, 17.3.36) and the derivatives
dK/dm = (E - (1 - m) K) / (2 m (1 - m)),
dE/dm = (E - K) / (2 m) (A&S 17.3.43-44 in the parameter m) with their Maclaurin series at small m
(A&S 17.3.11-12). Bessel: SciPy, J0' = -J1, (x J1)' = x J0 (DLMF 10.6.2). Plasma dispersion
function (Fried and Conte 1961): SciPy's ``wofz``, Z(0) = i sqrt(pi), Z' = -2 (1 + zeta Z), the
large-argument series and the Landau continuation below the real axis.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special as ss

from solvax import special as sp

jax.config.update("jax_enable_x64", True)


def _agm_ke(m):
    """K and E by the arithmetic-geometric mean, A&S 17.6.1-17.6.4, in NumPy float64."""
    a, b, total, scale = np.ones_like(m), np.sqrt(1 - m), m / 2, 0.5
    for _ in range(40):
        c = (a - b) / 2
        a, b = (a + b) / 2, np.sqrt(a * b)
        scale *= 2
        total = total + scale * c * c
    k = np.pi / (2 * a)
    return k, k * (1 - total)


def test_elliptic_against_agm():
    """Gate of the step: K and E equal the AGM values to 1e-14 relative over m in [0, 1)."""
    m = np.r_[np.linspace(0, 0.999, 2000), 1 - np.logspace(-3, -12, 50), -np.logspace(-3, 2, 20)]
    k_ref, e_ref = _agm_ke(m)
    k, e = jax.jit(sp.ellipk)(m), jax.jit(sp.ellipe)(m)
    assert np.max(np.abs(k - k_ref) / k_ref) < 1e-14
    assert np.max(np.abs(e - e_ref) / e_ref) < 1e-14
    np.testing.assert_allclose(k, ss.ellipk(m), rtol=1e-14)
    # m -> 1: K = L + (k'^2/4)(L - 1), E = 1 + (k'^2/2)(L - 1/2), L = log(4/k'), error k'^4 L.
    m = 1 - np.logspace(-10, -15, 6)
    kp2 = 1 - m
    big_l = np.log(4 / np.sqrt(kp2))
    np.testing.assert_allclose(sp.ellipk(m), big_l + kp2 / 4 * (big_l - 1), rtol=1e-14)
    np.testing.assert_allclose(sp.ellipe(m), 1 + kp2 / 2 * (big_l - 0.5), rtol=1e-14)
    assert np.isinf(sp.ellipk(1.0)) and sp.ellipe(1.0) == pytest.approx(1.0, abs=1e-15)


def _series_derivatives(m, terms=400):
    """dK/dm, dE/dm from K = pi/2 sum c_n^2 m^n and E = -pi/2 sum c_n^2 m^n / (2n - 1)."""
    dk, de, c2, mp = 0.0 * m, 0.0 * m, 1.0, np.ones_like(m)
    for n in range(1, terms):
        c2 *= ((2 * n - 1) / (2 * n)) ** 2
        dk, de = dk + n * c2 * mp, de - n * c2 * mp / (2 * n - 1)
        mp = mp * m
    return np.pi / 2 * dk, np.pi / 2 * de


def test_elliptic_derivatives_closed_form():
    """dK/dm, dE/dm to 1e-12: closed forms (A&S 17.3.43-44), series below m = 0.5 (0/0 at 0)."""
    m = np.r_[0.0, 1e-300, 1e-12, 1e-6, 1e-3, 0.05, 0.3, 0.6, 0.9, 0.99, 0.999999]
    dk = jax.vmap(jax.grad(sp.ellipk))(m)
    de = jax.vmap(jax.grad(sp.ellipe))(m)
    small = m < 0.5
    dk_ref, de_ref = _series_derivatives(m[small])
    np.testing.assert_allclose(dk[small], dk_ref, rtol=1e-12)
    np.testing.assert_allclose(de[small], de_ref, rtol=1e-12)
    k, e = _agm_ke(m[~small])
    mb = m[~small]
    np.testing.assert_allclose(dk[~small], (e - (1 - mb) * k) / (2 * mb * (1 - mb)), rtol=1e-12)
    np.testing.assert_allclose(de[~small], (e - k) / (2 * mb), rtol=1e-12)
    dk2 = jax.grad(jax.grad(sp.ellipk))(0.3)  # d2K/dm2 through the rule, against differences
    assert float(dk2) == pytest.approx(
        float(jax.grad(sp.ellipk)(0.3 + 1e-6) - dk[6]) * 1e6, rel=1e-5
    )


def test_carlson_against_scipy_and_special_values():
    """R_F, R_D against SciPy's elliprf/elliprd; R_F(x, x, x) = x^-1/2, R_D(x, x, x) = x^-3/2."""
    rng = np.random.default_rng(0)
    x, y, z = (10.0 ** rng.uniform(-8, 8, 300) for _ in range(3))
    np.testing.assert_allclose(sp.carlson_rf(x, y, z), ss.elliprf(x, y, z), rtol=1e-14)
    np.testing.assert_allclose(sp.carlson_rd(x, y, z), ss.elliprd(x, y, z), rtol=1e-14)
    np.testing.assert_allclose([sp.carlson_rf(4.0, 4, 4), sp.carlson_rd(4.0, 4, 4)], [0.5, 0.125])


def test_bessel_against_scipy():
    """J0, J1 for |x| <= 1e3: 1e-14 relative, or 5e-16 (2 ulp of O(1)) absolute near the zeros.

    Against 30-digit mpmath, SciPy 1.15's j0/j1 err by 1.3e-15 beyond |x| = 70 and jv by 5e-16 at
    |x| = 10 to 22, so the reference is j0/j1 below 50 and jv above."""
    x = np.r_[np.linspace(-1000, 1000, 400001), 0.0, 1e-300, 1e-8, 1.999999, 2.0, 24.999999, 25.0]
    big = np.abs(x) > 50
    for fun, ref in (
        (sp.bessel_j0, np.where(big, ss.jv(0, x), ss.j0(x))),
        (sp.bessel_j1, np.where(big, ss.jv(1, x), ss.j1(x))),
    ):
        err = np.abs(np.asarray(jax.jit(fun)(x)) - ref)
        assert np.all(err <= np.maximum(1e-14 * np.abs(ref), 5e-16))


def test_bessel_derivative_identities():
    """J0' = -J1 and (x J1)' = x J0 (DLMF 10.6.2), and central differences of the primal."""
    x = np.r_[-700.3, -3.1, 0.0, 1e-5, 0.7, 2.0, 7.5, 25.0, 31.4, 999.9]
    dj0 = jax.vmap(jax.grad(sp.bessel_j0))(x)
    np.testing.assert_allclose(dj0, -ss.jv(1, x), rtol=1e-14, atol=1e-15)
    dxj1 = jax.vmap(jax.grad(lambda t: t * sp.bessel_j1(t)))(x)
    # atol: 5e-16 absolute in J0 times |x| = 1e3.
    np.testing.assert_allclose(dxj1, x * ss.jv(0, x), rtol=1e-13, atol=5e-13)
    h = 1e-5
    fd = (sp.bessel_j1(x + h) - sp.bessel_j1(x - h)) / (2 * h)
    np.testing.assert_allclose(jax.vmap(jax.grad(sp.bessel_j1))(x), fd, atol=1e-9)


def test_dispersion_against_scipy_wofz():
    """Z = i sqrt(pi) wofz, |Re|, |Im| <= 30, to 5e-14 (SciPy errs 2.5e-14 at -7.84 vs mpmath)."""
    re, im = np.meshgrid(
        np.linspace(-30, 30, 241), np.r_[np.linspace(-30, 30, 241), np.logspace(-12, -1, 12)]
    )
    z = (re + 1j * im).ravel()
    with np.errstate(over="ignore", invalid="ignore"):  # w ~ exp(-z**2) deep below the axis
        ref = 1j * np.sqrt(np.pi) * ss.wofz(z)
    keep = np.isfinite(ref) & (np.abs(ref) < 1e300)
    got = np.asarray(jax.jit(sp.plasma_dispersion)(z))
    assert np.max(np.abs(got[keep] - ref[keep]) / np.abs(ref[keep])) < 5e-14


def test_dispersion_identities():
    """Z(0) = i sqrt(pi); Z(-zeta*) = -Z(zeta)*; large-|zeta| series, next term below 1e-15."""
    assert complex(sp.plasma_dispersion(0.0)) == pytest.approx(1j * math.sqrt(math.pi), abs=1e-15)
    z = jnp.array([0.3 + 0.2j, -2.0 - 1.5j, 5.0 - 0.01j, 9.0 + 3.0j])
    np.testing.assert_allclose(
        sp.plasma_dispersion(-jnp.conj(z)), -jnp.conj(sp.plasma_dispersion(z)), rtol=1e-14
    )
    zl = np.array([100.0 + 50.0j, -80.0 + 200.0j, 300.0 + 1.0j])
    series = -1 / zl * (1 + 1 / (2 * zl**2) + 3 / (4 * zl**4) + 15 / (8 * zl**6))
    np.testing.assert_allclose(sp.plasma_dispersion(zl), series, rtol=1e-14)


def test_dispersion_is_the_landau_continuation():
    """Below the real axis Z = pi^-1/2 int exp(-t^2)/(t - zeta) dt + 2 i sqrt(pi) exp(-zeta^2).

    The integral is taken by the trapezoid rule on [-12, 12] with step 6e-3 (error
    exp(-2 pi d / h) < 1e-100 for a pole at distance d >= 0.3), summed with math.fsum; 1e-13
    relative covers the rounding of the sum."""
    t = np.linspace(-12, 12, 4001)
    for zeta in (1.0 - 0.5j, -0.7 - 1.2j, 2.5 - 0.3j):
        f = np.exp(-t * t) / (t - zeta)
        h = (t[1] - t[0]) / np.sqrt(np.pi)
        integral = complex(math.fsum(f.real), math.fsum(f.imag)) * h
        expected = integral + 2j * np.sqrt(np.pi) * np.exp(-zeta * zeta)
        assert abs(complex(sp.plasma_dispersion(zeta)) - expected) < 1e-13 * abs(expected)
        above = complex(sp.plasma_dispersion(np.real(zeta) + 1e-9j))  # continuous across the axis
        below = complex(sp.plasma_dispersion(np.real(zeta) - 1e-9j))
        assert above == pytest.approx(below, rel=1e-8)


def test_dispersion_derivative():
    """Gate: Z' = -2 (1 + zeta Z) to 1e-11 by holomorphic grad and jvp; differences to 1e-8."""
    z = jnp.array([0.0 + 0.0j, 0.4 + 0.1j, -3.0 - 2.0j, 6.0 + 0.001j, 20.0 - 0.5j, -50.0 + 40.0j])
    zf = sp.plasma_dispersion(z)
    expected = -2 * (1 + z * zf)
    grad = jax.vmap(jax.grad(sp.plasma_dispersion, holomorphic=True))(z)
    assert np.max(np.abs(grad - expected) / np.abs(expected)) < 1e-11
    _, tangent = jax.jvp(sp.plasma_dispersion, (z,), (jnp.ones_like(z),))
    assert np.max(np.abs(tangent - expected) / np.abs(expected)) < 1e-11
    h = 1e-6
    fd = (sp.plasma_dispersion(z + h) - sp.plasma_dispersion(z - h)) / (2 * h)
    assert np.max(np.abs(fd - expected) / np.abs(expected)) < 1e-8


def test_float32_precision_is_kept():
    """In float32 the dtype is kept and the values are within 1e-5 of the float64 reference."""
    cases = (
        (sp.ellipk, np.linspace(0, 0.9, 7), ss.ellipk, np.float32),
        (sp.bessel_j0, np.linspace(-40, 40, 81), ss.j0, np.float32),
        (sp.faddeeva, np.array([0.5 + 0.5j, -3 - 1j, 12 + 2j]), ss.wofz, np.complex64),
    )
    for fun, arg, ref, dtype in cases:
        got = fun(arg.astype(dtype))
        assert got.dtype == dtype
        np.testing.assert_allclose(
            got, ref(arg.astype(dtype).astype(arg.dtype)), rtol=1e-5, atol=1e-6
        )
