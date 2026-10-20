"""Embedded explicit Runge-Kutta steps and a PI step-size controller.

All functions are pure and traceable, so they compose with ``jax.lax.scan`` and
``jax.lax.while_loop``. States are JAX arrays or pytrees of arrays.

An ``ERKPair`` stores the Butcher tableau of a propagated solution of order
``order`` and the weights ``e = b - b_hat`` of an embedded solution of order
``embedded_order``; ``h * sum_i e_i k_i`` estimates the local error of the
embedded solution. References: Hairer, Norsett, Wanner, *Solving ODEs I*, II.4;
Soderlind, *Digital filters in adaptive time-stepping*, ACM TOMS 29 (2003).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class ERKPair(NamedTuple):
    """Explicit Runge-Kutta tableau with an embedded error estimator."""

    a: tuple[tuple[float, ...], ...]  # strictly lower triangular rows 1..s-1
    b: tuple[float, ...]
    e: tuple[float, ...]  # b - b_hat
    c: tuple[float, ...]
    order: int
    embedded_order: int
    fsal: bool  # last stage equals f(y_new): it is the next step's first stage


# Classical RK4; the embedded solution is the explicit midpoint rule (stage 2),
# so the estimate costs no extra right-hand-side evaluation.
RK4_2 = ERKPair(
    a=((1 / 2,), (0.0, 1 / 2), (0.0, 0.0, 1.0)),
    b=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
    e=(1 / 6, 1 / 3 - 1.0, 1 / 3, 1 / 6),
    c=(0.0, 1 / 2, 1 / 2, 1.0),
    order=4,
    embedded_order=2,
    fsal=False,
)

# Bogacki-Shampine 3(2), FSAL: three evaluations per accepted step.
BS3_2 = ERKPair(
    a=((1 / 2,), (0.0, 3 / 4), (2 / 9, 1 / 3, 4 / 9)),
    b=(2 / 9, 1 / 3, 4 / 9, 0.0),
    e=(2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8),
    c=(0.0, 1 / 2, 3 / 4, 1.0),
    order=3,
    embedded_order=2,
    fsal=True,
)

# Dormand-Prince 5(4), FSAL: six evaluations per accepted step.
DOPRI5_4 = ERKPair(
    a=(
        (1 / 5,),
        (3 / 40, 9 / 40),
        (44 / 45, -56 / 15, 32 / 9),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
        (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
    ),
    b=(35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0),
    e=(
        35 / 384 - 5179 / 57600,
        0.0,
        500 / 1113 - 7571 / 16695,
        125 / 192 - 393 / 640,
        -2187 / 6784 + 92097 / 339200,
        11 / 84 - 187 / 2100,
        -1 / 40,
    ),
    c=(0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0),
    order=5,
    embedded_order=4,
    fsal=True,
)

PAIRS: dict[str, ERKPair] = {"rk4_2": RK4_2, "bs3_2": BS3_2, "dopri5_4": DOPRI5_4}


def _combine(y: Any, h: Any, weights: tuple[float, ...], ks: list[Any]) -> Any:
    """``y + h * sum_i weights_i k_i`` skipping zero weights."""

    def leaf(y_leaf: Any, *k_leaves: Any) -> Any:
        total = y_leaf
        for w, k in zip(weights, k_leaves, strict=False):
            if w != 0.0:
                total = total + (h * w) * k
        return total

    return jax.tree.map(leaf, y, *ks)


def erk_step(
    f: Callable[[Any, Any], Any],
    t: Any,
    y: Any,
    h: Any,
    pair: ERKPair,
    k1: Any | None = None,
    project: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, Any]:
    """One embedded explicit RK step.

    Returns ``(y_new, err, k_last)``. ``err`` is the embedded local-error
    estimate with the structure of ``y``. ``k_last`` is ``f(t + h, y_new)`` for
    an FSAL pair (pass it back as ``k1``) and the last stage otherwise.
    ``project`` is applied to every stage state, never to ``y_new``.
    """

    proj = project if project is not None else (lambda x: x)
    rows = pair.a[:-1] if pair.fsal else pair.a
    ks = [f(t, y) if k1 is None else k1]
    for row, ci in zip(rows, pair.c[1:], strict=False):
        ks.append(f(t + ci * h, proj(_combine(y, h, row, ks))))
    y_new = _combine(y, h, pair.b, ks)
    if pair.fsal:  # the last stage is the new state itself
        ks.append(f(t + h, y_new))
    err = jax.tree.map(
        lambda *k: h * sum(w * kk for w, kk in zip(pair.e, k, strict=False) if w != 0.0),
        *ks,
    )
    return y_new, err, ks[-1]


def error_norm(err: Any, y: Any, y_new: Any, rtol: float, atol: float) -> jax.Array:
    """Weighted RMS norm of ``err`` with weights ``atol + rtol max(|y|, |y_new|)``."""

    def leaf(e: Any, a: Any, b: Any) -> tuple[jax.Array, int]:
        scale = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
        return jnp.sum(jnp.abs(e / scale) ** 2), e.size

    parts = jax.tree.leaves(
        jax.tree.map(leaf, err, y, y_new), is_leaf=lambda x: isinstance(x, tuple)
    )
    total = sum(p[0] for p in parts)
    count = sum(p[1] for p in parts)
    return jnp.sqrt(total / count)


class PIController(NamedTuple):
    """Proportional-integral step-size controller.

    ``factor = safety * err**(-k_i/p) * err_prev**(k_p/p)`` with ``p`` the
    order of the embedded solution plus one, clipped to ``[fac_min, fac_max]``.
    The defaults ``k_i = 0.6``, ``k_p = 0.2`` are the Gustafsson-Lundh-Soderlind PI gains.
    """

    k_i: float = 0.6
    k_p: float = 0.2
    safety: float = 0.9
    fac_min: float = 0.2
    fac_max: float = 5.0


def step_factor(ctrl: PIController, err: Any, err_prev: Any, embedded_order: int) -> jax.Array:
    """Next-step multiplier from the current and previous error norms (normalized to 1)."""

    p = embedded_order + 1.0
    tiny = jnp.asarray(1e-10, jnp.result_type(err, jnp.float32))
    e = jnp.maximum(err, tiny)
    ep = jnp.maximum(err_prev, tiny)
    fac = ctrl.safety * e ** (-ctrl.k_i / p) * ep ** (ctrl.k_p / p)
    return jnp.clip(fac, ctrl.fac_min, ctrl.fac_max)


class ERKSolution(NamedTuple):
    y: Any
    t: jax.Array
    steps: jax.Array
    rejected: jax.Array
    evals: jax.Array


def erk_integrate(
    f: Callable[[Any, Any], Any],
    y0: Any,
    t0: float,
    t1: float,
    *,
    pair: ERKPair = DOPRI5_4,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float | None = None,
    max_steps: int = 100_000,
    controller: PIController | None = None,
) -> ERKSolution:
    """Integrate ``dy/dt = f(t, y)`` from ``t0`` to ``t1`` with PI step control."""

    controller = PIController() if controller is None else controller

    dtype = jnp.result_type(*jax.tree.leaves(y0), jnp.float32)
    real = jnp.finfo(dtype).dtype
    t0a, t1a = jnp.asarray(t0, real), jnp.asarray(t1, real)
    h_init = jnp.asarray(h0 if h0 is not None else 1e-2 * (t1 - t0), real)
    n_stage = len(pair.b)
    cost = n_stage - 1 if pair.fsal else n_stage

    def cond(s: tuple[Any, ...]) -> jax.Array:
        return (s[1] < t1a) & (s[4] + s[5] < max_steps)

    def body(s: tuple[Any, ...]) -> tuple[Any, ...]:
        y, t, h, err_prev, nacc, nrej, k1 = s
        last = h >= t1a - t
        h = jnp.where(last, t1a - t, h)
        y_new, err, k_last = erk_step(f, t, y, h, pair, k1=k1 if pair.fsal else None)
        en = error_norm(err, y, y_new, rtol, atol)
        ok = en <= 1.0
        fac = step_factor(controller, en, jnp.where(ok, err_prev, 1.0), pair.embedded_order)

        def pick(new: Any, old: Any) -> Any:
            return jax.tree.map(lambda a, b: jnp.where(ok, a, b), new, old)

        return (
            pick(y_new, y),
            jnp.where(ok, jnp.where(last, t1a, t + h), t),
            h * fac,
            jnp.where(ok, jnp.maximum(en, 1e-10), err_prev),
            nacc + ok,
            nrej + (~ok),
            pick(k_last, k1),
        )

    k_init = f(t0a, y0)
    state = (y0, t0a, h_init, jnp.asarray(1.0, real), jnp.int32(0), jnp.int32(0), k_init)
    y, t, _h, _e, nacc, nrej, _k = jax.lax.while_loop(cond, body, state)
    return ERKSolution(y, t, nacc, nrej, (nacc + nrej) * cost + 1)


__all__ = [
    "BS3_2",
    "DOPRI5_4",
    "ERKPair",
    "ERKSolution",
    "PAIRS",
    "PIController",
    "RK4_2",
    "erk_integrate",
    "erk_step",
    "error_norm",
    "step_factor",
]
