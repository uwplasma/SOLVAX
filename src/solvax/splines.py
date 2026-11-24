r"""Cubic interpolating splines in Hermite form: fit once, evaluate anywhere.

:func:`cubic_spline` returns the knots, the data and the knot slopes of a
piecewise cubic; :func:`cubic_spline_evaluate` gives its value, first or second
derivative at any points. On the interval :math:`[x_i, x_{i+1}]` of width
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
monotone data it cannot overshoot.

Knots lie on the leading axis of ``y``; trailing axes are independent data
sets that share the knots and are fitted in one solve. Outside the knots a
non-periodic spline continues the cubic of the end interval. Everything is
written in ``jax.numpy`` and the tridiagonal solves of :mod:`solvax.tridiagonal`,
so fit and evaluation compose with ``jit``, ``vmap``, ``jvp`` and ``grad`` with
respect to the knots, the data and the evaluation points.

Knots must be real and strictly increasing; this is checked when ``x`` is
concrete and not inside a trace, where repeated or unsorted knots give NaN or
an undefined result. Data may be complex (except with ``monotone=True``);
the knots and points stay real. With knot spacings that differ by a factor
of 1e6 the C2 slope systems lose about four digits (SciPy agrees to 1e-9).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from solvax.tridiagonal import cyclic_tridiagonal_solve, tridiagonal_solve

__all__ = ["CubicSpline", "cubic_spline", "cubic_spline_evaluate"]

_BCS = ("not-a-knot", "natural", "clamped", "periodic")


class CubicSpline(NamedTuple):
    """Piecewise cubic Hermite interpolant: knots, data and knot slopes.

    Attributes:
        x: knots, shape ``(n,)``, strictly increasing.
        y: data at the knots, shape ``(n, *batch)``.
        slopes: first derivative at the knots, shape of ``y``.
        period: ``x[-1] - x[0]`` for a periodic spline, zero otherwise.
    """

    x: jax.Array
    y: jax.Array
    slopes: jax.Array
    period: jax.Array


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


def cubic_spline(
    x: jax.Array,
    y: jax.Array,
    *,
    bc: str = "not-a-knot",
    end_slopes: tuple = (0.0, 0.0),
    monotone: bool = False,
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
    if monotone and jnp.iscomplexobj(y):
        raise ValueError("monotone=True needs real data")
    n = x.shape[0]
    if x.ndim != 1 or y.shape[:1] != (n,):
        raise ValueError("x must be one-dimensional and y must have len(x) rows")
    if bc not in _BCS:
        raise ValueError(f"unknown bc {bc!r}; expected one of {_BCS}")
    if monotone and bc != "not-a-knot":
        raise ValueError("monotone=True takes no boundary condition")
    if n < 2 or (bc == "periodic" and n < 4):
        raise ValueError(f"bc={bc!r} needs at least {4 if bc == 'periodic' else 2} knots")
    if bc == "periodic":
        y = y.at[-1].set(y[0])
    h = jnp.diff(x)
    d = jnp.diff(y, axis=0) / _column(h, y.ndim)
    zero = jnp.zeros((), x.dtype)
    if (n == 2 and bc != "clamped") or (n == 3 and bc == "not-a-knot" and not monotone):
        # Line or parabola through the data: its slope at a knot is a weighted secant.
        if n == 2:
            return CubicSpline(x, y, jnp.concatenate([d, d]), zero)
        w0, w1 = h[1] / (h[0] + h[1]), h[0] / (h[0] + h[1])
        mid = w0 * d[0] + w1 * d[1]
        s = jnp.stack([2 * d[0] - mid, mid, 2 * d[1] - mid])
        return CubicSpline(x, y, s, zero)
    if monotone:
        return CubicSpline(x, y, _pchip_slopes(h, d), zero)
    if bc == "periodic":
        hl, hr = jnp.roll(h, 1), h  # widths left and right of knots 0 .. n-2
        dl, dr = jnp.roll(d, 1, axis=0), d
        rhs = 3.0 * (_column(hr, y.ndim) * dl + _column(hl, y.ndim) * dr)
        s = cyclic_tridiagonal_solve(hr, 2.0 * (hl + hr), hl, rhs)
        return CubicSpline(x, y, jnp.concatenate([s, s[:1]], axis=0), x[-1] - x[0])
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
    return CubicSpline(x, y, tridiagonal_solve(lower, diag, upper, rhs), zero)


def cubic_spline_evaluate(
    spline: CubicSpline, points: jax.Array, *, derivative: int = 0
) -> jax.Array:
    """Value (``derivative=0``), first or second derivative of a spline.

    Args:
        spline: the result of :func:`cubic_spline`.
        points: evaluation points of any shape.
        derivative: 0, 1 or 2.

    Returns:
        Array of shape ``points.shape + batch``. A periodic spline is
        evaluated at the points reduced into ``[x[0], x[-1])``; any other
        continues its end cubics beyond the knots.
    """
    if derivative not in (0, 1, 2):
        raise ValueError("derivative must be 0, 1 or 2")
    i, u, (y0, s0, c2, c3) = _locate(spline, points)
    if derivative == 0:
        return y0 + u * (s0 + u * (c2 + u * c3))
    if derivative == 1:
        return s0 + u * (2.0 * c2 + 3.0 * u * c3)
    return 2.0 * c2 + 6.0 * u * c3


def _locate(spline: CubicSpline, points):
    """Interval index, offset and power coefficients at the (wrapped) points."""
    x, y, s, period = spline
    p = jnp.asarray(points, x.dtype)
    p = jnp.where(period > 0, x[0] + jnp.mod(p - x[0], jnp.where(period > 0, period, 1.0)), p)
    i = jnp.clip(jnp.searchsorted(x, p, side="right") - 1, 0, x.shape[0] - 2)
    h = _column(x[i + 1] - x[i], y.ndim - 1 + p.ndim)
    u = _column(p - x[i], y.ndim - 1 + p.ndim)
    y0, y1, s0, s1 = y[i], y[i + 1], s[i], s[i + 1]
    delta = (y1 - y0) / h
    c2 = (3.0 * delta - 2.0 * s0 - s1) / h
    c3 = (s0 + s1 - 2.0 * delta) / (h * h)
    return i, u, (y0, s0, c2, c3)
