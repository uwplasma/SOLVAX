"""Special functions: Carlson elliptic integrals, Bessel J0 and J1, plasma dispersion function.

Every function is written in plain ``jax.numpy``, composes with ``jit``, ``vmap``
and ``grad``, and keeps the input precision. Accuracy is stated in float64; in
float32 expect a few 1e-6 relative. The coefficients are Python scalars applied
by Horner's rule, never a captured table indexed inside the program.

Convention for the complete elliptic integrals: the argument is the parameter
``m = k**2``, as in SciPy's ``ellipk`` and ``ellipe``, not the modulus ``k``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

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
    """``K(m)``, ``E(m)`` and ``R_D(0, 1 - m, 1)``."""
    m = _inexact(m)
    if jnp.iscomplexobj(m):
        raise TypeError("ellipk and ellipe take a real parameter m")
    zero, one = jnp.zeros_like(m), jnp.ones_like(m)
    k = jnp.where(m == 1, jnp.inf, carlson_rf(zero, 1 - m, one))  # R_F(0, 0, 1) diverges
    rd = carlson_rd(zero, 1 - m, one)
    return k, jnp.where(m == 1, 1.0, k - m * rd / 3), rd


@jax.custom_jvp
def ellipk(m):
    """Complete elliptic integral of the first kind ``K(m) = R_F(0, 1 - m, 1)``.

    Parameter ``m = k**2 <= 1``, as in ``scipy.special.ellipk``; ``K ~ log(4 / sqrt(1 - m))`` as
    ``m -> 1``. ``dK/dm = (E - (1 - m) K) / (2 m (1 - m))`` is evaluated as
    ``(K - R_D / 3) / (2 (1 - m))``, without the 0/0 at ``m = 0``. ``m > 1`` gives NaN; complex
    ``m`` raises ``TypeError``.
    """
    return _kd(m)[0]


@jax.custom_jvp
def ellipe(m):
    """Complete elliptic integral of the second kind ``E(m) = K(m) - m R_D(0, 1 - m, 1) / 3``.

    The argument is the parameter ``m = k**2 <= 1``, as in ``scipy.special.ellipe``. The derivative
    ``(E - K) / (2 m)`` is evaluated as ``-R_D(0, 1 - m, 1) / 6``; at ``m = 1``, where it is
    ``-inf``, the returned value is finite and not meaningful.
    """
    return _kd(m)[1]


@ellipk.defjvp
def _ellipk_jvp(primals, tangents):
    (m,), (dm,) = primals, tangents
    k, _, rd = _kd(m)
    return k, (k - rd / 3) / (2 * (1 - m)) * dm


@ellipe.defjvp
def _ellipe_jvp(primals, tangents):
    (m,), (dm,) = primals, tangents
    _, e, rd = _kd(m)
    return e, -rd / 6 * dm


# Rational approximations of Cephes j0.c and j1.c (S. L. Moshier, Cephes Math Library 2.8, 2000;
# BSD-3 as distributed in SciPy's xsf), highest power first; RQ and QQ have an implied leading 1.
def _table(text):
    return {
        k.strip(): tuple(float(v) for v in vals.split())
        for k, vals in (row.split(":") for row in text.strip().split(";"))
    }


_J0 = _table("""
RP: -4.79443220978201773821E9 1.95617491946556577543E12 -2.49248344360967716204E14
    9.70862251047306323952E15;
RQ: 4.99563147152651017219E2 1.73785401676374683123E5 4.84409658339962045305E7
    1.11855537045356834862E10 2.11277520115489217587E12 3.10518229857422583814E14
    3.18121955943204943306E16 1.71086294081043136091E18;
PP: 7.96936729297347051624E-4 8.28352392107440799803E-2 1.23953371646414299388E0
    5.44725003058768775090E0 8.74716500199817011941E0 5.30324038235394892183E0
    9.99999999999999997821E-1;
PQ: 9.24408810558863637013E-4 8.56288474354474431428E-2 1.25352743901058953537E0
    5.47097740330417105182E0 8.76190883237069594232E0 5.30605288235394617618E0
    1.00000000000000000218E0;
QP: -1.13663838898469149931E-2 -1.28252718670509318512E0 -1.95539544257735972385E1
    -9.32060152123768231369E1 -1.77681167980488050595E2 -1.47077505154951170175E2
    -5.14105326766599330220E1 -6.05014350600728481186E0;
QQ: 6.43178256118178023184E1 8.56430025976980587198E2 3.88240183605401609683E3
    7.24046774195652478189E3 5.93072701187316984827E3 2.06209331660327847417E3
    2.42005740240291393179E2;
Z: 5.78318596294678452118E0 3.04712623436620863991E1
""")
_J1 = _table("""
RP: -8.99971225705559398224E8 4.52228297998194034323E11 -7.27494245221818276015E13
    3.68295732863852883286E15;
RQ: 6.20836478118054335476E2 2.56987256757748830383E5 8.35146791431949253037E7
    2.21511595479792499675E10 4.74914122079991414898E12 7.84369607876235854894E14
    8.95222336184627338078E16 5.32278620332680085395E18;
PP: 7.62125616208173112003E-4 7.31397056940917570436E-2 1.12719608129684925192E0
    5.11207951146807644818E0 8.42404590141772420927E0 5.21451598682361504063E0
    1.00000000000000000254E0;
PQ: 5.71323128072548699714E-4 6.88455908754495404082E-2 1.10514232634061696926E0
    5.07386386128601488557E0 8.39985554327604159757E0 5.20982848682361821619E0
    9.99999999999999997461E-1;
QP: 5.10862594750176621635E-2 4.98213872951233449420E0 7.58238284132545283818E1
    3.66779609360150777800E2 7.10856304998926107277E2 5.97489612400613639965E2
    2.11688757100572135698E2 2.52070205858023719784E1;
QQ: 7.42373277035675149943E1 1.05644886038262816351E3 4.98641058337653607651E3
    9.56231892404756170795E3 7.99704160447350683650E3 2.82619278517639096600E3
    3.36093607810698293419E2;
Z: 1.46819706421238932572E1 4.92184563216946036703E1
""")


def _horner(coef, z, lead=None):
    p = jnp.ones_like(z) if lead else jnp.full_like(z, coef[0])
    for c in coef if lead else coef[1:]:
        p = p * z + c
    return p


def _bessel(x, t, order):
    """Cephes ``J0`` or ``J1``: rational in ``x**2`` up to 5, Hankel's form above."""
    ax = jnp.abs(x)
    xs = jnp.minimum(ax, 5.0)
    z = xs * xs
    small = (z - t["Z"][0]) * (z - t["Z"][1]) * _horner(t["RP"], z) / _horner(t["RQ"], z, 1)
    small = small * xs if order else small
    xl = jnp.maximum(ax, 5.0)
    w = 5.0 / xl
    p = _horner(t["PP"], w * w) / _horner(t["PQ"], w * w)
    q = w * _horner(t["QP"], w * w) / _horner(t["QQ"], w * w, 1)
    c, sn = jnp.cos(xl), jnp.sin(xl)  # cos(x - pi/4) = (c + sn) / sqrt(2): no shift of a large x
    big = (p * (c + sn) - q * (sn - c)) if order == 0 else (p * (sn - c) + q * (sn + c))
    out = jnp.where(ax <= 5, small, jnp.where(jnp.isinf(ax), 0.0, big / jnp.sqrt(math.pi * xl)))
    return jnp.sign(x) * out if order else out


@jax.custom_jvp
def bessel_j0(x):
    """Bessel function ``J0(x)`` of real argument, 1e-15 absolute for finite ``x``, 0 at ``inf``.

    Same call signature as ``gkx.core.velocity.bessel_j0``. Derivative ``J0' = -J1``.
    """
    return _bessel(_inexact(x), _J0, 0)


@jax.custom_jvp
def bessel_j1(x):
    """Bessel function ``J1(x)`` of real argument, 1e-15 absolute for finite ``x``, 0 at ``inf``.

    Same call signature as ``gkx.core.velocity.bessel_j1``. Derivative ``J1' = J0 - J1 / x``.
    """
    return _bessel(_inexact(x), _J1, 1)


@bessel_j0.defjvp
def _bessel_j0_jvp(primals, tangents):
    x = _inexact(primals[0])
    return bessel_j0(x), -bessel_j1(x) * tangents[0]


@bessel_j1.defjvp
def _bessel_j1_jvp(primals, tangents):
    x = _inexact(primals[0])
    j0, j1 = bessel_j0(x), bessel_j1(x)
    small = jnp.abs(x) < 1e-4
    ratio = jnp.where(small, 0.5 - x * x / 16, j1 / jnp.where(small, 1.0, x))
    return j1, (j0 - ratio) * tangents[0]


# Weideman (1994, SIAM J. Numer. Anal. 31, eq. 3.13), N = 40: scale and coefficients, highest first.
_W_SCALE = math.sqrt(40 / math.sqrt(2.0))
_t = _W_SCALE * np.tan(np.arange(-79, 80) * np.pi / 160)
_a = np.fft.fft(np.fft.fftshift(np.r_[0.0, np.exp(-_t * _t) * (_W_SCALE**2 + _t * _t)])).real / 160
_W_COEF = tuple(float(v) for v in _a[40:0:-1])


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


@jax.custom_jvp
def faddeeva(z):
    """Faddeeva function ``w(z) = exp(-z**2) erfc(-i z)``, entire, as ``scipy.special.wofz``.

    The lower half plane uses ``w(z) = 2 exp(-z**2) - w(-z)``. Over ``|Re z|, |Im z| <= 30`` the
    error is 1e-13 relative in modulus (float64). On the real axis ``Re w = exp(-x**2)``; just off
    it (``0 < |Im z| <= 1e-2``) the error of ``Re w`` is 6e-17 absolute, not relative, so for
    ``|Re z| > 4`` there ``Re w`` may have few correct digits. Derivative
    ``w' = -2 z w + 2 i / sqrt(pi)``, holomorphic; NaN where ``w`` overflows below the axis.
    """
    z = jnp.asarray(z) + 0j  # complex64 from float32, complex128 otherwise
    lower = jnp.imag(z) < 0
    wu = _faddeeva_upper(jnp.where(lower, -z, z))
    w = jnp.where(lower, 2 * jnp.exp(-z * z) - wu, wu)
    x = jnp.real(z)
    return jnp.where(
        jnp.imag(z) == 0, jnp.exp(-x * x) + 1j * jnp.imag(w), w
    )  # Re w(x) = exp(-x**2)


@faddeeva.defjvp
def _faddeeva_jvp(primals, tangents):
    z = jnp.asarray(primals[0]) + 0j
    w = faddeeva(z)
    return w, (-2 * z * w + 2j / math.sqrt(math.pi)) * tangents[0]


def plasma_dispersion(zeta):
    """Plasma dispersion function ``Z(zeta) = i sqrt(pi) w(zeta)`` (Fried and Conte 1961).

    This is the Landau-continued function, analytic in the whole plane: for ``Im zeta < 0`` it is
    not the integral ``pi**-0.5 int exp(-t**2) / (t - zeta) dt`` but its continuation, which differs
    from it by ``2 i sqrt(pi) exp(-zeta**2)``. ``Z(0) = i sqrt(pi)``, ``Z' = -2 (1 + zeta Z)``.
    """
    return 1j * math.sqrt(math.pi) * faddeeva(zeta)
