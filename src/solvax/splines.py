r"""Cubic interpolating splines in Hermite form: fit once, evaluate anywhere.

:func:`cubic_spline` returns the knots, the data and the knot slopes of a
piecewise cubic; :func:`cubic_spline_evaluate` gives its value, first or second
derivative at any points, or several of them from one interval search.
:func:`cubic_spline_coefficients` gives the power coefficients of each interval,
which :func:`piecewise_cubic_evaluate` evaluates without rebuilding them.
:func:`tensor_spline_coefficients` and :func:`piecewise_bicubic_evaluate` do the
same for the tensor-product spline on a rectilinear 2-D grid.
On the interval :math:`[x_i, x_{i+1}]` of width
:math:`h_i`, with secant :math:`\delta_i = (y_{i+1} - y_i)/h_i` and slopes
:math:`s_i`, the cubic in :math:`u = x - x_i` is

.. math::

    y_i + s_i u + \frac{3\delta_i - 2 s_i - s_{i+1}}{h_i} u^2
    + \frac{s_i + s_{i+1} - 2\delta_i}{h_i^2} u^3 .

For the C2 splines the slopes solve one tridiagonal system (one cyclic system
for ``bc="periodic"``), the continuity of the second derivative at the interior
knots, de Boor (2001), *A Practical Guide to Splines*, chapter IV, eq. (6)-(7),
closed by the end condition:

- ``"not-a-knot"``: the third derivative is continuous at the second and the
  next-to-last knot; a cubic is reproduced exactly.
- ``"natural"``: zero second derivative at both ends.
- ``"clamped"``: the first derivative at the ends is ``end_slopes``.
- ``"periodic"``: value, slope and curvature match across the seam; ``y[-1]``
  is ignored and taken as ``y[0]``, and the spline repeats with period
  ``x[-1] - x[0]``.

These are the conventions of :class:`scipy.interpolate.CubicSpline`, and the
systems are its rows. ``monotone=True`` takes instead the Fritsch-Carlson
slopes, Fritsch & Carlson, *SIAM J. Numer. Anal.* 17 (1980) 238, in the
weighted-harmonic-mean form of Fritsch & Butland (1984) with the three-point
end rule of Moler, *Numerical Computing with MATLAB* (2004), section 3.4: the
algorithm of :class:`scipy.interpolate.PchipInterpolator`. It is C1 only, and on
monotone data it cannot overshoot. ``slopes="akima"`` takes the slopes of Akima,
*J. ACM* 17 (1970) 589, with his end extension (two secants beyond each end,
continued linearly), and ``slopes="makima"`` the modified weights of Moler and
Ionita (2019): the rules of :class:`scipy.interpolate.Akima1DInterpolator`,
including its fallback to the mean secant where both weights are below 1e-9
of their largest value (taken per data column here, over all of them in SciPy).
An array ``slopes`` gives the knot slopes directly, a C1 Hermite spline as
:class:`scipy.interpolate.CubicHermiteSpline`.

Knots lie on the leading axis of ``y``; trailing axes are independent data
sets that share the knots and are fitted in one solve. Outside the knots a
non-periodic spline continues the cubic of the end interval. Everything is
written in ``jax.numpy`` and the tridiagonal solves of :mod:`solvax.tridiagonal`,
so fit and evaluation compose with ``jit``, ``vmap``, ``jvp`` and ``grad`` with
respect to the knots, the data and the evaluation points.
:func:`cubic_spline_integrate` gives the exact integral between two limits.

Knots must be real and strictly increasing; this is checked when ``x`` is
concrete and not inside a trace, where repeated or unsorted knots give NaN or
an undefined result. Data may be complex (except with ``monotone=True``);
the knots and points stay real. With knot spacings that differ by a factor
of 1e6 the values agree with SciPy to about 4e-10 of the data scale for every
end condition on 16 or more knots, and for all but not-a-knot on fewer; the
not-a-knot end rows lose more on few knots, up to 1e-4 on four.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from solvax.tridiagonal import cyclic_tridiagonal_solve, tridiagonal_solve

__all__ = [
    "CubicSpline",
    "cubic_spline",
    "cubic_spline_evaluate",
    "cubic_spline_integrate",
    "cubic_spline_antiderivative",
    "cubic_spline_coefficients",
    "piecewise_cubic_evaluate",
    "tensor_spline_coefficients",
    "piecewise_bicubic_evaluate",
]

_BCS = ("not-a-knot", "natural", "clamped", "periodic")


class CubicSpline(NamedTuple):
    """Piecewise cubic Hermite interpolant: knots, data and knot slopes.

    Attributes:
        x: knots, shape ``(n,)``, strictly increasing.
        y: data at the knots, shape ``(n, *batch)``.
        slopes: first derivative at the knots, shape of ``y``.
        period: ``x[-1] - x[0]`` for a periodic spline, zero otherwise.
        smooth: true when the slopes make the spline C2 (the end conditions of
            ``bc``), false for the C1 slopes of ``monotone=True``; it decides
            how :func:`cubic_spline_coefficients` forms the coefficients.
    """

    x: jax.Array
    y: jax.Array
    slopes: jax.Array
    period: jax.Array
    smooth: jax.Array


def _column(v: jax.Array, ndim: int) -> jax.Array:
    """Append unit axes so a knot-axis vector broadcasts against ``ndim``-d data."""
    return v.reshape(v.shape + (1,) * (ndim - v.ndim))


def _pchip_slopes(h: jax.Array, d: jax.Array) -> jax.Array:
    """Fritsch-Butland interior slopes and Moler's three-point end slopes."""
    hl, hr = _column(h[:-1], d.ndim), _column(h[1:], d.ndim)
    dl, dr = d[:-1], d[1:]
    w1, w2 = 2.0 * hr + hl, hr + 2.0 * hl
    same = dl * dr > 0
    safe_l, safe_r = jnp.where(same, dl, 1.0), jnp.where(same, dr, 1.0)
    interior = jnp.where(same, (w1 + w2) / (w1 / safe_l + w2 / safe_r), 0.0)

    def end(dk, dk1, hk, hk1):
        m = ((2.0 * hk + hk1) * dk - hk * dk1) / (hk + hk1)
        m = jnp.where(jnp.sign(m) != jnp.sign(dk), 0.0, m)
        flip = (jnp.sign(dk) != jnp.sign(dk1)) & (jnp.abs(m) > 3.0 * jnp.abs(dk))
        return jnp.where(flip, 3.0 * dk, m)

    first = end(d[0], d[1], h[0], h[1])
    last = end(d[-1], d[-2], h[-1], h[-2])
    return jnp.concatenate([first[None], interior, last[None]], axis=0)


def _akima_slopes(d: jax.Array, modified: bool) -> jax.Array:
    """Akima's weighted secants, two secants added beyond each end."""
    lo, hi = 2.0 * d[0] - d[1], 2.0 * d[-1] - d[-2]
    m = jnp.concatenate([(2.0 * lo - d[0])[None], lo[None], d, hi[None], (2.0 * hi - d[-1])[None]])
    w = jnp.abs(jnp.diff(m, axis=0))
    if modified:
        w = w + 0.5 * jnp.abs(m[1:] + m[:-1])
    f1, f2 = w[2:], w[:-2]
    f12 = f1 + f2
    ok = f12 > 1e-9 * jnp.max(f12, axis=0)
    weighted = m[1:-2] + f2 / jnp.where(ok, f12, 1.0) * (m[2:-1] - m[1:-2])
    return jnp.where(ok, weighted, 0.5 * (m[3:] + m[:-3]))


def cubic_spline(
    x: jax.Array,
    y: jax.Array,
    *,
    bc: str = "not-a-knot",
    end_slopes: tuple = (0.0, 0.0),
    monotone: bool = False,
    slopes: jax.Array | str | None = None,
) -> CubicSpline:
    """Fit a cubic spline through ``(x[i], y[i])``.

    Args:
        x: strictly increasing knots, shape ``(n,)``, ``n >= 2``.
        y: data, shape ``(n, *batch)``; trailing axes are fitted together.
        bc: ``"not-a-knot"``, ``"natural"``, ``"clamped"`` or ``"periodic"``.
            Not-a-knot needs ``n >= 4`` for a cubic: with three knots it is
            the parabola and with two the line through the data, as in SciPy.
            Periodic needs ``n >= 4`` and ignores ``y[-1]``.
        end_slopes: first derivative at ``x[0]`` and ``x[-1]`` for
            ``bc="clamped"``; scalars or arrays of the batch shape.
        monotone: Fritsch-Carlson (PCHIP) slopes instead of a C2 spline;
            ``bc`` must then be left at its default and is not used.
        slopes: ``"akima"`` or ``"makima"`` for Akima slopes (real data), or
            the knot slopes themselves, shape of ``y``; ``bc`` and ``monotone``
            must then be left at their defaults.

    Returns:
        A :class:`CubicSpline` for :func:`cubic_spline_evaluate`.
    """
    try:
        concrete = np.asarray(x)
    except jax.errors.TracerArrayConversionError:
        concrete = None
    if concrete is not None and not np.all(np.diff(concrete) > 0):
        raise ValueError("knots must be strictly increasing")
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, float))
    if jnp.iscomplexobj(x):
        raise ValueError("knots must be real")
    dtype = jnp.result_type(x, y)
    y = jnp.asarray(y, dtype)
    if (monotone or isinstance(slopes, str)) and jnp.iscomplexobj(y):
        raise ValueError("monotone=True and Akima slopes need real data")
    n = x.shape[0]
    if x.ndim != 1 or y.shape[:1] != (n,):
        raise ValueError("x must be one-dimensional and y must have len(x) rows")
    if bc not in _BCS:
        raise ValueError(f"unknown bc {bc!r}; expected one of {_BCS}")
    if (monotone or slopes is not None) and bc != "not-a-knot" or (monotone and slopes is not None):
        raise ValueError(
            "monotone=True and slopes take no boundary condition and exclude each other"
        )
    if isinstance(slopes, str) and slopes not in ("akima", "makima"):
        raise ValueError(f"unknown slopes {slopes!r}; expected 'akima', 'makima' or an array")
    if n < 2 or (bc == "periodic" and n < 4):
        raise ValueError(f"bc={bc!r} needs at least {4 if bc == 'periodic' else 2} knots")
    if bc == "periodic":
        y = y.at[-1].set(y[0])
    h = jnp.diff(x)
    d = jnp.diff(y, axis=0) / _column(h, y.ndim)
    smooth = not monotone and slopes is None
    zero, c2 = jnp.zeros((), x.dtype), jnp.asarray(smooth)
    if slopes is not None and not isinstance(slopes, str):
        return CubicSpline(x, y, jnp.broadcast_to(jnp.asarray(slopes, dtype), y.shape), zero, c2)
    if (n == 2 and bc != "clamped") or (n == 3 and bc == "not-a-knot" and smooth):
        # Line or parabola through the data: its slope at a knot is a weighted secant.
        if n == 2:
            return CubicSpline(x, y, jnp.concatenate([d, d]), zero, c2)
        w0, w1 = h[1] / (h[0] + h[1]), h[0] / (h[0] + h[1])
        mid = w0 * d[0] + w1 * d[1]
        s = jnp.stack([2 * d[0] - mid, mid, 2 * d[1] - mid])
        return CubicSpline(x, y, s, zero, c2)
    if monotone or slopes is not None:
        s = _pchip_slopes(h, d) if monotone else _akima_slopes(d, slopes == "makima")
        return CubicSpline(x, y, s, zero, c2)
    if bc == "periodic":
        hl, hr = jnp.roll(h, 1), h  # widths left and right of knots 0 .. n-2
        dl, dr = jnp.roll(d, 1, axis=0), d
        rhs = 3.0 * (_column(hr, y.ndim) * dl + _column(hl, y.ndim) * dr)
        s = cyclic_tridiagonal_solve(hr, 2.0 * (hl + hr), hl, rhs)
        return CubicSpline(x, y, jnp.concatenate([s, s[:1]], axis=0), x[-1] - x[0], c2)
    # Interior rows: continuity of the second derivative at knots 1 .. n-2.
    lower = jnp.concatenate([zero[None], h[1:], zero[None]])
    diag = jnp.concatenate([zero[None], 2.0 * (h[:-1] + h[1:]), zero[None]])
    upper = jnp.concatenate([zero[None], h[:-1], zero[None]])
    rhs = 3.0 * (_column(h[1:], y.ndim) * d[:-1] + _column(h[:-1], y.ndim) * d[1:])
    if bc == "not-a-knot":
        w0, w1 = x[2] - x[0], x[-1] - x[-3]
        diag = diag.at[0].set(h[1]).at[-1].set(h[-2])
        upper, lower = upper.at[0].set(w0), lower.at[-1].set(w1)
        r0 = ((h[0] + 2 * w0) * h[1] * d[0] + h[0] ** 2 * d[1]) / w0
        r1 = (h[-1] ** 2 * d[-2] + (2 * w1 + h[-1]) * h[-2] * d[-1]) / w1
    elif bc == "natural":
        diag = diag.at[0].set(2.0).at[-1].set(2.0)
        upper, lower = upper.at[0].set(1.0), lower.at[-1].set(1.0)
        r0, r1 = 3.0 * d[0], 3.0 * d[-1]
    else:
        diag = diag.at[0].set(1.0).at[-1].set(1.0)
        r0 = jnp.broadcast_to(jnp.asarray(end_slopes[0], dtype), y.shape[1:])
        r1 = jnp.broadcast_to(jnp.asarray(end_slopes[1], dtype), y.shape[1:])
    rhs = jnp.concatenate([r0[None], rhs, r1[None]], axis=0)
    return CubicSpline(x, y, tridiagonal_solve(lower, diag, upper, rhs), zero, c2)


def cubic_spline_evaluate(
    spline: CubicSpline, points: jax.Array, *, derivative: int | tuple[int, ...] = 0
) -> jax.Array | tuple[jax.Array, ...]:
    """Value (``derivative=0``), first or second derivative of a spline.

    A tuple such as ``derivative=(0, 1)`` returns a tuple of those orders from
    one interval search.

    Args:
        spline: the result of :func:`cubic_spline`.
        points: evaluation points of any shape.
        derivative: 0, 1 or 2, or a tuple of them.

    Returns:
        Array (or tuple of arrays) of shape ``points.shape + batch``. A periodic spline is
        evaluated at the points reduced into ``[x[0], x[-1])``; any other
        continues its end cubics beyond the knots. At an interior knot the
        cubic of the interval to its right is used, at ``x[-1]`` that of the
        last interval: this decides the second derivative of a monotone
        spline there and the derivative of any order with respect to the point.
    """
    _check_orders(derivative)
    i, u, (y0, s0, c2, c3) = _locate(spline, points)
    if isinstance(derivative, tuple):
        return tuple(_horner(k, (c3, c2, s0, y0), u) for k in derivative)
    if derivative == 0:
        return y0 + u * (s0 + u * (c2 + u * c3))
    if derivative == 1:
        return s0 + u * (2.0 * c2 + 3.0 * u * c3)
    return 2.0 * c2 + 6.0 * u * c3


def _check_orders(derivative) -> None:
    """Accept exactly the integers 0, 1, 2, alone or in a non-empty tuple."""
    orders = derivative if isinstance(derivative, tuple) else (derivative,)
    if not orders or any(type(k) is not int or k not in (0, 1, 2) for k in orders):
        raise ValueError("derivative must be 0, 1 or 2, or a tuple of them")


def _horner(k: int, c, u):
    """Order-``k`` derivative of ``c[0] u^3 + c[1] u^2 + c[2] u + c[3]``."""
    c3, c2, c1, c0 = c
    if k == 0:
        return c0 + u * (c1 + u * (c2 + u * c3))
    if k == 1:
        return c1 + u * (2.0 * c2 + 3.0 * u * c3)
    return 2.0 * c2 + 6.0 * u * c3


def _interval(x: jax.Array, p: jax.Array) -> jax.Array:
    """Index of the interval holding each point, the end intervals extended."""
    # Comparing with every knot is fastest for a few knots; binary search beyond.
    method = "compare_all" if x.shape[0] <= 32 else "scan_unrolled"
    i = jnp.searchsorted(x, p, side="right", method=method)
    return jnp.clip(i - 1, 0, x.shape[0] - 2)


def _locate(spline: CubicSpline, points):
    """Interval index, offset and power coefficients at the (wrapped) points."""
    x, y, s, period, _ = spline
    p = jnp.asarray(points, x.dtype)
    p = jnp.where(period > 0, x[0] + jnp.mod(p - x[0], jnp.where(period > 0, period, 1.0)), p)
    i = _interval(x, p)
    h = _column(x[i + 1] - x[i], y.ndim - 1 + p.ndim)
    u = _column(p - x[i], y.ndim - 1 + p.ndim)
    y0, y1, s0, s1 = y[i], y[i + 1], s[i], s[i + 1]
    delta = (y1 - y0) / h
    c2 = (3.0 * delta - 2.0 * s0 - s1) / h
    c3 = (s0 + s1 - 2.0 * delta) / (h * h)
    return i, u, (y0, s0, c2, c3)


def cubic_spline_antiderivative(spline: CubicSpline, points: jax.Array) -> jax.Array:
    """Integral of the spline from the first knot ``x[0]`` to each point.

    It vanishes at ``x[0]`` and is negative for points below it. Beyond the
    knots it integrates the end cubics; for a periodic spline it adds the
    integral over one period for each whole period between ``x[0]`` and the
    point. The result has shape ``points.shape + batch``; its derivative in
    the points is the spline value. One evaluation, half the work of
    ``cubic_spline_integrate(spline, x[0], points)``.
    """
    x, period = spline.x, spline.period
    p = jnp.asarray(points, x.dtype)
    _, _, coef = _locate(spline, x[:-1])
    h = _column(jnp.diff(x), spline.y.ndim)
    full = sum(c * h ** (k + 1) / (k + 1) for k, c in enumerate(coef))
    prefix = jnp.concatenate([jnp.zeros_like(full[:1]), jnp.cumsum(full, axis=0)])
    i, u, coef = _locate(spline, p)
    partial = sum(c * u ** (k + 1) / (k + 1) for k, c in enumerate(coef))
    turns = jnp.where(period > 0, jnp.floor((p - x[0]) / jnp.where(period > 0, period, 1.0)), 0)
    return prefix[i] + partial + _column(turns, partial.ndim) * prefix[-1]


def cubic_spline_integrate(spline: CubicSpline, a: jax.Array, b: jax.Array) -> jax.Array:
    """Exact integral of the spline from ``a`` to ``b`` (negative when ``a > b``).

    ``a`` and ``b`` broadcast together; the result has their shape plus the
    batch shape. Limits outside the knots integrate the extrapolated end
    cubics, or whole periods of a periodic spline. A definite integral is
    returned rather than an antiderivative because the latter needs a base
    point; ``a = x[0]`` gives the cumulative integral. The result is the
    difference of two antiderivatives, so an interval much shorter than the
    distance of its ends from ``x[0]`` loses digits to cancellation.
    """
    # Each limit keeps its own shape; the batch axes trail both, so they broadcast.
    return cubic_spline_antiderivative(spline, b) - cubic_spline_antiderivative(spline, a)


def cubic_spline_coefficients(spline: CubicSpline) -> jax.Array:
    """Power coefficients of each interval, in the layout of SciPy's ``PPoly.c``.

    Returns ``c`` of shape ``(4, n - 1, *batch)``: on ``[x[i], x[i+1]]`` the
    spline is ``sum_k c[k, i] (x - x[i]) ** (3 - k)``. For a C2 spline the
    knot second derivatives are solved from the moment equations, closed by
    the spline's own end slopes (the same spline for every end condition),
    and ``c[0]``, ``c[1]``, ``c[2]`` are formed from them: the slope formula
    for ``c[0]``, ``(s0 + s1 - 2 delta) / h^2``, cancels to O(h^2) of its terms
    and carries about five times more rounding. The moment equations take the
    second differences of the data before dividing by the spacings. A spline
    that is only C1 (``spline.smooth`` false) has its coefficients formed
    from the slopes.
    """
    return _coefficients(spline, natural=False)


def _coefficients(spline: CubicSpline, natural: bool) -> jax.Array:
    """:func:`cubic_spline_coefficients`; ``natural`` closes the moments with zero end curvature."""
    x, y, s = spline.x, spline.y, spline.slopes
    h = _column(jnp.diff(x), y.ndim)
    d = jnp.diff(y, axis=0) / h
    c1 = jnp.stack(
        [(s[:-1] + s[1:] - 2.0 * d) / h**2, (3.0 * d - 2.0 * s[:-1] - s[1:]) / h, s[:-1]]
    )
    zero = jnp.zeros((1,), x.dtype)
    hl, hr = jnp.concatenate([zero, jnp.diff(x)]), jnp.concatenate([jnp.diff(x), zero])
    # Second differences of the data before dividing: exact on equal spacings, where
    # differencing the secants d would lose the digits that d and d' share.
    dy = jnp.diff(y, axis=0)
    jump = (dy[1:] - dy[:-1] * (h[1:] / h[:-1])) / h[1:]
    rhs = 6.0 * jnp.concatenate([d[:1] - s[:1], jump, s[-1:] - d[-1:]], axis=0)
    if natural:  # exact zeros, not the rounding of 6 (d - s) / h at the ends
        hl, hr, rhs = hl.at[-1].set(0.0), hr.at[0].set(0.0), rhs.at[0].set(0.0).at[-1].set(0.0)
    m = tridiagonal_solve(hl, 2.0 * (hl + hr), hr, rhs)
    c2 = jnp.stack(
        [(m[1:] - m[:-1]) / (6.0 * h), 0.5 * m[:-1], d - h * (2.0 * m[:-1] + m[1:]) / 6.0]
    )
    return jnp.concatenate([jnp.where(spline.smooth, c2, c1), y[None, :-1]])


def piecewise_cubic_evaluate(
    x: jax.Array,
    coefficients: jax.Array,
    points: jax.Array,
    *,
    derivative: int | tuple[int, ...] = 0,
    uniform: bool = False,
    periodic: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
    """Evaluate stored interval coefficients, as returned by :func:`cubic_spline_coefficients`.

    Args:
        x: the knots, shape ``(n,)``.
        coefficients: shape ``(4, n - 1, *batch)``, highest power first.
        points: evaluation points of any shape.
        derivative: 0, 1 or 2, or a tuple of them.
        uniform: the caller promises equally spaced knots; the interval is
            then ``floor((p - x[0]) / h)`` instead of a search. Checked when
            ``x`` is concrete; under a trace a non-uniform grid gives wrong
            values. At a knot rounding may select either neighbouring
            interval, which matters only where the second derivative jumps.
        periodic: reduce the points into ``[x[0], x[-1])``, period
            ``x[-1] - x[0]``; otherwise the end cubics continue beyond the knots.

    Returns:
        Array (or tuple) of shape ``points.shape + batch``.
    """
    _check_orders(derivative)
    i, u = _index(x, points, uniform, periodic)
    c = coefficients[:, i]
    u = _column(u, c.ndim - 1)
    values = tuple(
        _horner(k, c, u) for k in (derivative if isinstance(derivative, tuple) else (derivative,))
    )
    return values if isinstance(derivative, tuple) else values[0]


def _index(x: jax.Array, points: jax.Array, uniform: bool, periodic: bool):
    """Interval index and offset of each point, as for :func:`piecewise_cubic_evaluate`."""
    x = jnp.asarray(x)
    p = jnp.asarray(points, x.dtype)
    n = x.shape[0]
    if periodic:
        p = x[0] + jnp.mod(p - x[0], x[-1] - x[0])
    if uniform:
        try:
            concrete = np.asarray(x)
        except jax.errors.TracerArrayConversionError:
            concrete = None
        if concrete is not None and not np.allclose(np.diff(concrete), concrete[1] - concrete[0]):
            raise ValueError("uniform=True needs equally spaced knots")
        step = (x[-1] - x[0]) / (n - 1)
        i = jnp.clip(jnp.floor((p - x[0]) / step), 0, n - 2).astype(jnp.int32)  # clip, then cast
    else:
        i = _interval(x, p)
    return i, p - x[i]


_FITS = {
    "monotone": {"monotone": True},
    "akima": {"slopes": "akima"},
    "makima": {"slopes": "makima"},
}


def tensor_spline_coefficients(
    x: jax.Array, y: jax.Array, f: jax.Array, *, bc: tuple[str, str] = ("not-a-knot", "not-a-knot")
) -> jax.Array:
    """Power coefficients of the tensor-product cubic spline through ``f[i, j]``.

    The spline is fitted along ``x`` (with ``bc[0]``), and its interval
    coefficients are fitted along ``y`` (with ``bc[1]``); by linearity this is
    the tensor-product interpolant, whatever the order. Each entry of ``bc`` is
    an end condition of :func:`cubic_spline` or ``"monotone"``, ``"akima"``,
    ``"makima"`` for those slopes. The C2 coefficients come from the knot
    second derivatives, as in :func:`cubic_spline_coefficients`: formed from
    the slopes, the highest coefficients carry one more power of ``1/h`` of
    rounding and the block of ``d_x^3 d_y^3`` loses a digit; a natural axis
    sets the end second derivatives to zero exactly, since through the end
    slopes their rounding is amplified by the cross block. A slope rule along
    ``y`` is not linear in the data, so it is applied instead to the knot
    values and the x-slopes, the corner data of a bicubic Hermite patch: the
    spline is then only C1 in ``x`` between the y-knots, and swapping the axes
    of the data and of ``bc`` gives a different interpolant. ``"clamped"`` sets
    the end slopes to zero. A periodic axis ignores the last row (``bc[0]``) or
    column (``bc[1]``) of ``f``.

    Args:
        x, y: strictly increasing knots, shapes ``(nx,)`` and ``(ny,)``.
        f: data, shape ``(nx, ny, *batch)``.
        bc: the fit along ``x`` and along ``y``.

    Returns:
        ``c`` of shape ``(4, nx - 1, 4, ny - 1, *batch)``: on the cell
        ``[x[i], x[i+1]] x [y[j], y[j+1]]`` the spline is
        ``sum_ab c[a, i, b, j] (x - x[i]) ** (3 - a) (y - y[j]) ** (3 - b)``.
    """

    def fit(t, v, kind):
        return cubic_spline(t, v, **_FITS.get(kind, {"bc": kind}))

    along_x = fit(x, f, bc[0])
    if bc[1] in _FITS:  # not linear: fit the knot values and x-slopes along y, then Hermite in x
        knots = jnp.moveaxis(jnp.stack([along_x.y, along_x.slopes], 2), 1, 0)  # (ny, nx, 2, ...)
        c = jnp.moveaxis(
            cubic_spline_coefficients(fit(y, knots, bc[1])), 2, 0
        )  # (nx, 4, ny - 1, 2, ...)
        hermite = CubicSpline(x, c[:, :, :, 0], c[:, :, :, 1], along_x.period, jnp.asarray(False))
        return cubic_spline_coefficients(hermite)
    c = _coefficients(along_x, bc[0] == "natural")  # (4, nx - 1, ny, *batch)
    c = _coefficients(fit(y, jnp.moveaxis(c, 2, 0), bc[1]), bc[1] == "natural")
    return jnp.moveaxis(c, (0, 1), (2, 3))


def piecewise_bicubic_evaluate(
    x: jax.Array,
    y: jax.Array,
    coefficients: jax.Array,
    px: jax.Array,
    py: jax.Array,
    *,
    derivative: tuple = (0, 0),
    uniform: tuple[bool, bool] = (False, False),
    periodic: tuple[bool, bool] = (False, False),
) -> jax.Array | tuple[jax.Array, ...]:
    """Evaluate :func:`tensor_spline_coefficients` or a mixed partial derivative.

    Args:
        x, y: the knots.
        coefficients: shape ``(4, nx - 1, 4, ny - 1, *batch)``.
        px, py: coordinates of the points; they broadcast together, so
            ``px[:, None]`` and ``py[None, :]`` evaluate on a grid.
        derivative: ``(i, j)`` for ``d^(i+j) / dx^i dy^j`` with ``i, j`` in
            0, 1, 2, or a tuple of such pairs, evaluated from one search.
        uniform, periodic: per axis, as in :func:`piecewise_cubic_evaluate`.

    Returns:
        Array (or tuple) of the broadcast shape of the points plus ``batch``.
    """
    pairs = derivative if derivative and isinstance(derivative[0], tuple) else (derivative,)
    if not isinstance(derivative, tuple) or not all(
        isinstance(d, tuple) and len(d) == 2 and all(type(k) is int and 0 <= k <= 2 for k in d)
        for d in pairs
    ):
        raise ValueError("derivative must be a pair of orders 0, 1, 2, or a tuple of pairs")
    i, u = _index(x, px, uniform[0], periodic[0])
    j, v = _index(y, py, uniform[1], periodic[1])
    i, u, j, v = jnp.broadcast_arrays(i, u, j, v)
    batch = coefficients.shape[4:]
    block = coefficients[:, i.ravel(), :, j.ravel()]  # (m, a, b, *batch)
    block = block.reshape(u.size, 4, 4, np.prod(batch, dtype=int))
    along_x = {a: _horner(a, jnp.unstack(block, axis=1), u.reshape(-1, 1, 1)) for a, _ in pairs}
    values = tuple(
        _horner(b, jnp.unstack(along_x[a], axis=1), v.reshape(-1, 1)).reshape(u.shape + batch)
        for a, b in pairs
    )
    return values if isinstance(derivative[0], tuple) else values[0]
