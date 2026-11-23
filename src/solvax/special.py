"""Special functions: Carlson elliptic integrals, Bessel J0 and J1, plasma dispersion function.

Every function is written in plain ``jax.numpy``, composes with ``jit``, ``vmap``
and ``grad``, and keeps the input precision. Accuracy is stated in float64; in
float32 expect about 1e-6 relative. The coefficients are Python scalars applied
by Horner's rule, never a captured table indexed inside the program.

Convention for the complete elliptic integrals: the argument is the parameter
``m = k**2``, as in SciPy's ``ellipk`` and ``ellipe``, not the modulus ``k``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "bessel_j0",
    "bessel_j1",
    "carlson_rd",
    "carlson_rf",
    "ellipe",
    "ellipk",
    "faddeeva",
    "plasma_dispersion",
]

_DUPLICATIONS = 16  # differences shrink 4x per step; 4**16 leaves a wide margin for float64


def _inexact(x):
    x = jnp.asarray(x)
    return x if jnp.issubdtype(x.dtype, jnp.inexact) else x.astype(jnp.result_type(float))


def carlson_rf(x, y, z):
    """Carlson's symmetric integral ``R_F(x, y, z)`` for non-negative arguments, at most one zero.

    Duplication theorem with the fifth-order series of Carlson (1995, Numer. Algorithms 10,
    eq. 2.2), at a fixed 16 steps so that reverse mode works.
    """
    x, y, z = jnp.broadcast_arrays(_inexact(x), _inexact(y), _inexact(z))

    def step(_, c):
        x, y, z = c
        sx, sy, sz = jnp.sqrt(x), jnp.sqrt(y), jnp.sqrt(z)
        lam = sx * sy + sx * sz + sy * sz
        return (x + lam) / 4, (y + lam) / 4, (z + lam) / 4

    x, y, z = jax.lax.fori_loop(0, _DUPLICATIONS, step, (x, y, z))
    a = (x + y + z) / 3
    dx, dy = 1 - x / a, 1 - y / a
    dz = -(dx + dy)
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    return (1 - e2 / 10 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44) / jnp.sqrt(a)


def carlson_rd(x, y, z):
    """Carlson's integral ``R_D(x, y, z) = R_J(x, y, z, z)``.

    ``x, y >= 0`` with at most one zero, ``z > 0``.

    Duplication with the series of Carlson (1995, eq. 2.29), fixed 16 steps.
    """
    x, y, z = jnp.broadcast_arrays(_inexact(x), _inexact(y), _inexact(z))

    def step(n, c):
        x, y, z, s = c
        sx, sy, sz = jnp.sqrt(x), jnp.sqrt(y), jnp.sqrt(z)
        lam = sx * sy + sx * sz + sy * sz
        s = s + 3 * 0.25**n / (sz * (z + lam))
        return (x + lam) / 4, (y + lam) / 4, (z + lam) / 4, s

    x, y, z, s = jax.lax.fori_loop(0, _DUPLICATIONS, step, (x, y, z, jnp.zeros_like(x)))
    a = (x + y + 3 * z) / 5
    dx, dy = 1 - x / a, 1 - y / a
    dz = -(dx + dy) / 3
    xy, zz = dx * dy, dz * dz
    e2 = xy - 6 * zz
    e3 = (3 * xy - 8 * zz) * dz
    e4 = 3 * (xy - zz) * zz
    e5 = xy * zz * dz
    series = (
        1 - 3 * e2 / 14 + e3 / 6 + 9 * e2 * e2 / 88 - 3 * e4 / 22 - 9 * e2 * e3 / 52 + 3 * e5 / 26
    )
    return s + 0.25**_DUPLICATIONS * series / (a * jnp.sqrt(a))


def _kd(m):
    """``K(m)`` and ``R_D(0, 1 - m, 1)``, from which ``E = K - m R_D / 3``."""
    m = _inexact(m)
    zero, one = jnp.zeros_like(m), jnp.ones_like(m)
    k = jnp.where(m == 1, jnp.inf, carlson_rf(zero, 1 - m, one))  # R_F(0, 0, 1) diverges
    return k, carlson_rd(zero, 1 - m, one)


def _e(m, k, rd):
    return jnp.where(m == 1, 1.0, k - m * rd / 3)


@jax.custom_jvp
def ellipk(m):
    """Complete elliptic integral of the first kind ``K(m) = R_F(0, 1 - m, 1)``.

    The argument is the parameter ``m = k**2 <= 1``, as in ``scipy.special.ellipk``. ``K`` diverges
    as ``log(4 / sqrt(1 - m))`` when ``m -> 1`` and is ``inf`` at 1. The derivative
    ``(E - (1 - m) K) / (2 m (1 - m))`` is evaluated as ``(K - R_D / 3) / (2 (1 - m))``, which has
    no 0/0 at ``m = 0``.
    """
    return _kd(m)[0]


@jax.custom_jvp
def ellipe(m):
    """Complete elliptic integral of the second kind ``E(m) = K(m) - m R_D(0, 1 - m, 1) / 3``.

    The argument is the parameter ``m = k**2 <= 1``, as in ``scipy.special.ellipe``. The derivative
    ``(E - K) / (2 m)`` is evaluated as ``-R_D(0, 1 - m, 1) / 6``.
    """
    m = _inexact(m)
    return _e(m, *_kd(m))


@ellipk.defjvp
def _ellipk_jvp(primals, tangents):
    (m,), (dm,) = primals, tangents
    k, rd = _kd(m)
    return k, (k - rd / 3) / (2 * (1 - m)) * dm


@ellipe.defjvp
def _ellipe_jvp(primals, tangents):
    (m,), (dm,) = primals, tangents
    k, rd = _kd(m)
    return _e(m, k, rd), -rd / 6 * dm


def _bessel01(x):
    """``J0(x)``, ``J1(x)``: series below 2, Miller recurrence below 25, Hankel expansion above."""
    ax = jnp.abs(x)
    # J_nu = sum (-x^2/4)^k (x/2)^nu / (k! (k+nu)!), 14 terms: last term < 1e-20 at x = 2.
    xs = jnp.minimum(ax, 2.0)
    q = -xs * xs / 4
    t0, t1 = jnp.ones_like(x), xs / 2
    s0, s1 = t0, t1
    for k in range(1, 15):
        t0, t1 = t0 * q / (k * k), t1 * q / (k * (k + 1))
        s0, s1 = s0 + t0, s1 + t1
    # Miller's backward recurrence from order 80, normalised by J0 + 2 (J2 + J4 + ...) = 1.
    xm = jnp.clip(ax, 2.0, 25.0)

    def back(i, c):
        bk1, bk, s = c
        k = 80 - i
        bm = 2 * k / xm * bk - bk1
        s = s + jnp.where((k % 2 == 1) & (k > 1), 2 * bm, 0.0)
        f = jnp.where(jnp.abs(bm) > 1e30, 1e-30, 1.0)  # growth per step <= 80: float32 is safe
        return bk * f, bm * f, s * f

    b1, b0, s = jax.lax.fori_loop(
        0, 80, back, (jnp.zeros_like(x), jnp.ones_like(x), jnp.zeros_like(x))
    )
    m0, m1 = b0 / (b0 + s), b1 / (b0 + s)
    # Hankel expansion (DLMF 10.17.3), 30 terms; at x >= 25 the omitted terms are below 1e-17.
    xa = jnp.maximum(ax, 25.0)
    p0 = p1 = jnp.ones_like(x)
    q0 = q1 = jnp.zeros_like(x)
    a0 = a1 = jnp.ones_like(x)
    for k in range(1, 31):
        a0 = a0 * (-((2 * k - 1) ** 2)) / (8 * k * xa)
        a1 = a1 * (4 - (2 * k - 1) ** 2) / (8 * k * xa)
        sign = (-1) ** (k // 2)
        if k % 2:
            q0, q1 = q0 + sign * a0, q1 + sign * a1
        else:
            p0, p1 = p0 + sign * a0, p1 + sign * a1
    # cos(x - pi/4) = (cos x + sin x) / sqrt(2) avoids shifting a large argument.
    c, sn = jnp.cos(xa), jnp.sin(xa)
    amp = jnp.sqrt(1 / (math.pi * xa))
    h0 = amp * (p0 * (c + sn) - q0 * (sn - c))
    h1 = amp * (p1 * (sn - c) + q1 * (sn + c))
    j0 = jnp.where(ax < 2, s0, jnp.where(ax < 25, m0, h0))
    j1 = jnp.where(ax < 2, s1, jnp.where(ax < 25, m1, h1))
    return j0, jnp.sign(x) * j1


@jax.custom_jvp
def bessel_j0(x):
    """Bessel function ``J0(x)`` of real argument, about 1e-15 absolute for ``|x| <= 1e3``.

    Drop-in for ``gkx.core.velocity.bessel_j0``. Derivative ``J0' = -J1``.
    """
    return _bessel01(_inexact(x))[0]


@jax.custom_jvp
def bessel_j1(x):
    """Bessel function ``J1(x)`` of real argument, about 1e-15 absolute for ``|x| <= 1e3``.

    Drop-in for ``gkx.core.velocity.bessel_j1``. Derivative ``J1' = J0 - J1 / x``.
    """
    return _bessel01(_inexact(x))[1]


@bessel_j0.defjvp
def _bessel_j0_jvp(primals, tangents):
    j0, j1 = _bessel01(_inexact(primals[0]))
    return j0, -j1 * tangents[0]


@bessel_j1.defjvp
def _bessel_j1_jvp(primals, tangents):
    x = _inexact(primals[0])
    j0, j1 = _bessel01(x)
    small = jnp.abs(x) < 1e-4
    ratio = jnp.where(small, 0.5 - x * x / 16, j1 / jnp.where(small, 1.0, x))
    return j1, (j0 - ratio) * tangents[0]


def _weideman_coefficients(n=40):
    """Coefficients of Weideman (1994, SIAM J. Numer. Anal. 31, eq. 3.13), highest power first."""
    big_m = 2 * n
    k = np.arange(-big_m + 1, big_m)
    scale = np.sqrt(n / np.sqrt(2.0))
    t = scale * np.tan(k * np.pi / (2 * big_m))
    f = np.r_[0.0, np.exp(-t * t) * (scale * scale + t * t)]
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * big_m)
    return float(scale), tuple(float(v) for v in a[n:0:-1])


_W_SCALE, _W_COEF = _weideman_coefficients()


def _faddeeva_upper(z):
    """``w(z)``, ``Im z >= 0``: Weideman N = 40 below ``|z| = 8``, continued fraction above."""
    zs = jnp.where(jnp.abs(z) < 8, z, 0.0)
    d = _W_SCALE - 1j * zs
    u = (_W_SCALE + 1j * zs) / d
    p = jnp.zeros_like(z)
    for a in _W_COEF:
        p = p * u + a
    near = 2 * p / (d * d) + 1 / (math.sqrt(math.pi) * d)
    zf = jnp.where(jnp.abs(z) < 8, 8.0, z)
    t = jnp.zeros_like(z)
    # Laplace: w = (i / sqrt(pi)) / (z - (1/2) / (z - 1 / (z - (3/2) / (z - ...))))
    for k in range(24, 0, -1):
        t = (k / 2) / (zf - t)
    far = 1j / (math.sqrt(math.pi) * (zf - t))
    return jnp.where(jnp.abs(z) < 8, near, far)


def _complex(z):
    z = jnp.asarray(z)
    if jnp.issubdtype(z.dtype, jnp.complexfloating):
        return z
    return z.astype(jnp.result_type(complex))


@jax.custom_jvp
def faddeeva(z):
    """Faddeeva function ``w(z) = exp(-z**2) erfc(-i z)``, entire, as ``scipy.special.wofz``.

    The lower half plane uses ``w(z) = 2 exp(-z**2) - w(-z)``. Against SciPy over
    ``|Re z|, |Im z| <= 30``: 8.3e-15 relative in float64; near the real axis SciPy itself errs by
    up to 2.5e-14. Derivative ``w' = -2 z w + 2 i / sqrt(pi)``, holomorphic.
    """
    z = _complex(z)
    lower = jnp.imag(z) < 0
    wu = _faddeeva_upper(jnp.where(lower, -z, z))
    return jnp.where(lower, 2 * jnp.exp(-z * z) - wu, wu)


@faddeeva.defjvp
def _faddeeva_jvp(primals, tangents):
    z = _complex(primals[0])
    w = faddeeva(z)
    return w, (-2 * z * w + 2j / math.sqrt(math.pi)) * tangents[0]


def plasma_dispersion(zeta):
    """Plasma dispersion function ``Z(zeta) = i sqrt(pi) w(zeta)`` (Fried and Conte 1961).

    This is the Landau-continued function, analytic in the whole plane: for ``Im zeta < 0`` it is
    not the integral ``pi**-0.5 int exp(-t**2) / (t - zeta) dt`` but its continuation, which differs
    from it by ``2 i sqrt(pi) exp(-zeta**2)``. ``Z(0) = i sqrt(pi)``, ``Z' = -2 (1 + zeta Z)``.
    """
    return 1j * math.sqrt(math.pi) * faddeeva(zeta)
