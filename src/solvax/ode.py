"""Time stepping: explicit, IMEX, Lawson and SSP Runge-Kutta steps, step control, a driver.

Steps are single-step maps (``erk_step``, ``imex_step``, ``lawson_step``,
``ssp_step``); ``cfl_dt``, ``step_factor`` and ``controlled`` choose the step
size; ``integrate`` runs a step in checkpointed chunks with progress lines and a
non-finite stop. All functions are pure and traceable, so they compose with ``jax.lax.scan`` and
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




def _scaled(h: Any, weights: tuple[float, ...], ks: list[Any]) -> Any:
    """``h * sum_i weights_i k_i`` skipping zero weights (at least one is nonzero)."""

    pairs = [(w, k) for w, k in zip(weights, ks, strict=False) if w != 0.0]
    return jax.tree.map(
        lambda *leaves: h * sum(w * leaf for (w, _), leaf in zip(pairs, leaves, strict=True)),
        *[k for _, k in pairs],
    )


def _add(*trees: Any) -> Any:
    return jax.tree.map(lambda *leaves: sum(leaves[1:], leaves[0]), *trees)


# --------------------------------------------------------------------------
# Implicit-explicit (additive) Runge-Kutta
# --------------------------------------------------------------------------
class IMEXTableau(NamedTuple):
    """Additive Runge-Kutta tableau for ``y' = f_exp(t, y) + f_imp(t, y)``.

    ``a_exp`` is strictly lower triangular and ``a_imp`` lower triangular, both
    as full ``s x s`` rows. ``b_hat`` holds embedded weights shared by both
    parts, or ``None``. ``c_exp`` and ``c_imp`` are the row sums.
    """

    a_exp: tuple[tuple[float, ...], ...]
    a_imp: tuple[tuple[float, ...], ...]
    b_exp: tuple[float, ...]
    b_imp: tuple[float, ...]
    c_exp: tuple[float, ...]
    c_imp: tuple[float, ...]
    order: int
    b_hat: tuple[float, ...] | None = None


def _imex(
    a_exp: Any, a_imp: Any, b_exp: Any, b_imp: Any, order: int, b_hat: Any = None
) -> IMEXTableau:
    a_exp = tuple(tuple(float(v) for v in r) for r in a_exp)
    a_imp = tuple(tuple(float(v) for v in r) for r in a_imp)
    return IMEXTableau(
        a_exp,
        a_imp,
        tuple(map(float, b_exp)),
        tuple(map(float, b_imp)),
        tuple(sum(r) for r in a_exp),
        tuple(sum(r) for r in a_imp),
        order,
        None if b_hat is None else tuple(map(float, b_hat)),
    )


# Forward-backward Euler, ARS(1,1,1).
IMEX_EULER = _imex(((0, 0), (1, 0)), ((0, 0), (0, 1)), (1, 0), (0, 1), 1)

# Pareschi-Russo IMEX-SSP2(2,2,2), J. Sci. Comput. 25 (2005), gamma = 1 - 1/sqrt(2).
_GS = 1.0 - 2.0**-0.5
IMEX_SSP2 = _imex(
    ((0, 0), (1, 0)), ((_GS, 0), (1 - 2 * _GS, _GS)), (0.5, 0.5), (0.5, 0.5), 2
)

# Ascher-Ruuth-Spiteri ARS(3,4,3), Appl. Numer. Math. 25 (1997); published digits.
_G3 = 0.4358665215
_B3 = (0.0, -1.5 * _G3**2 + 4 * _G3 - 0.25, 1.5 * _G3**2 - 5 * _G3 + 1.25, _G3)
ARS_343 = _imex(
    (
        (0, 0, 0, 0),
        (_G3, 0, 0, 0),
        (0.3212788860, 0.3966543747, 0, 0),
        (-0.105858296, 0.5529291479, 0.5529291479, 0),
    ),
    ((0, 0, 0, 0), (0, _G3, 0, 0), (0, (1 - _G3) / 2, _G3, 0), _B3),
    _B3,
    _B3,
    3,
)

# Kennedy-Carpenter ARK3(2)4L[2]SA, Appl. Numer. Math. 44 (2003), with its embedded pair.
_GK = 1767732205903 / 4055673282236
_BK = (
    1471266399579 / 7840856788654,
    -4482444167858 / 7529755066697,
    11266239266428 / 11593286722821,
    _GK,
)
ARK3_KC = _imex(
    (
        (0, 0, 0, 0),
        (2 * _GK, 0, 0, 0),
        (5535828885825 / 10492691773637, 788022342437 / 10882634858940, 0, 0),
        (
            6485989280629 / 16251701735622,
            -4246266847089 / 9704473918619,
            10755448449292 / 10357097424841,
            0,
        ),
    ),
    (
        (0, 0, 0, 0),
        (_GK, _GK, 0, 0),
        (2746238789719 / 10658868560708, -640167445237 / 6845629431997, _GK, 0),
        _BK,
    ),
    _BK,
    _BK,
    3,
    (
        2756255671327 / 12835298489170,
        -10771552573575 / 22201958757719,
        9247589265047 / 10645013368117,
        2193209047091 / 5459859503100,
    ),
)

# Spalart-Moser-Rogers low-storage RK3 with Crank-Nicolson substeps (J. Comput.
# Phys. 96, 1991), written as a stiffly accurate four-stage additive tableau.
_SG, _SZ, _SA = (8 / 15, 5 / 12, 3 / 4), (0.0, -17 / 60, -5 / 12), (4 / 15, 1 / 15, 1 / 6)
SMR_RK3_CN = _imex(
    (
        (0, 0, 0, 0),
        (_SG[0], 0, 0, 0),
        (_SG[0] + _SZ[1], _SG[1], 0, 0),
        (_SG[0] + _SZ[1], _SG[1] + _SZ[2], _SG[2], 0),
    ),
    (
        (0, 0, 0, 0),
        (_SA[0], _SA[0], 0, 0),
        (_SA[0], _SA[0] + _SA[1], _SA[1], 0),
        (_SA[0], _SA[0] + _SA[1], _SA[1] + _SA[2], _SA[2]),
    ),
    (_SG[0] + _SZ[1], _SG[1] + _SZ[2], _SG[2], 0),
    (_SA[0], _SA[0] + _SA[1], _SA[1] + _SA[2], _SA[2]),
    2,
)

IMEX_TABLEAUX: dict[str, IMEXTableau] = {
    "euler": IMEX_EULER,
    "ssp2": IMEX_SSP2,
    "ars343": ARS_343,
    "ark3": ARK3_KC,
    "smr_rk3_cn": SMR_RK3_CN,
}


def imex_step(
    f_exp: Callable[[Any, Any], Any],
    f_imp: Callable[[Any, Any], Any],
    solve: Callable[[Any, Any, Any], Any],
    t: Any,
    y: Any,
    h: Any,
    tableau: IMEXTableau,
    project: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any]:
    """One additive Runge-Kutta step for ``y' = f_exp(t, y) + f_imp(t, y)``.

    ``solve(t, r, gamma_h)`` returns ``z`` with ``z - gamma_h f_imp(t, z) = r``;
    for a linear stiff part ``L`` that is ``(I - gamma_h L)^{-1} r``, so any
    matrix-free Krylov, multigrid or fast-diagonal solve fits. It is called once
    per stage with a nonzero diagonal, with ``gamma_h = h a_imp[i][i]`` traced.
    ``project`` is applied to every stage state. Returns ``(y_new, err)``; ``err``
    is the embedded error estimate, or ``None`` when the tableau has no ``b_hat``.
    A stiffly accurate tableau returns its last stage without evaluating it.
    """

    proj = project if project is not None else (lambda x: x)
    s = len(tableau.b_exp)
    sa = tableau.b_hat is None and (
        tableau.b_exp == tableau.a_exp[-1] and tableau.b_imp == tableau.a_imp[-1]
    )
    ke: list[Any] = []
    ki: list[Any] = []
    z = y
    for i in range(s):
        r = _combine(_combine(y, h, tableau.a_exp[i][:i], ke), h, tableau.a_imp[i][:i], ki)
        g = tableau.a_imp[i][i]
        z = proj(solve(t + tableau.c_imp[i] * h, r, g * h) if g != 0.0 else r)
        if sa and i == s - 1:
            return z, None
        ke.append(f_exp(t + tableau.c_exp[i] * h, z))
        ki.append(f_imp(t + tableau.c_imp[i] * h, z))
    y_new = _combine(_combine(y, h, tableau.b_exp, ke), h, tableau.b_imp, ki)
    if tableau.b_hat is None:
        return y_new, None
    e_exp = tuple(b - bh for b, bh in zip(tableau.b_exp, tableau.b_hat, strict=True))
    e_imp = tuple(b - bh for b, bh in zip(tableau.b_imp, tableau.b_hat, strict=True))
    return y_new, _add(_scaled(h, e_exp, ke), _scaled(h, e_imp, ki))


# --------------------------------------------------------------------------
# Lawson (integrating-factor) Runge-Kutta
# --------------------------------------------------------------------------
def _lawson_combine(
    propagate: Callable[[Any, Any], Any],
    y: Any,
    h: Any,
    ci: float,
    weights: tuple[float, ...],
    ks: list[Any],
    c: tuple[float, ...],
) -> Any:
    """``exp(ci h L) y + h sum_j w_j exp((ci - c_j) h L) k_j``, one propagation per distinct c_j."""

    groups: dict[float, tuple[list[float], list[Any]]] = {}
    for w, k, cj in zip(weights, ks, c, strict=False):
        if w != 0.0:
            groups.setdefault(cj, ([], []))[0].append(w)
            groups[cj][1].append(k)
    terms = []
    if y is not None:
        ws, kk = groups.pop(0.0, ([], []))
        terms.append((ci, _combine(y, h, tuple(ws), kk)))
    terms += [(ci - cj, _scaled(h, tuple(ws), kk)) for cj, (ws, kk) in groups.items()]
    return _add(*[x if lag == 0.0 else propagate(x, lag * h) for lag, x in terms])


def lawson_step(
    f: Callable[[Any, Any], Any],
    propagate: Callable[[Any, Any], Any],
    t: Any,
    y: Any,
    h: Any,
    pair: ERKPair = RK4_2,
    k1: Any | None = None,
    project: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, Any]:
    """One Lawson step of an ``ERKPair`` for ``y' = L y + f(t, y)``.

    ``propagate(x, tau)`` returns ``exp(tau L) x`` for ``tau >= 0``: a diagonal
    decay or phase in Fourier space, or a rotation between two transforms. It
    is called with ``tau = (c_i - c_j) h`` once per distinct pair of nodes.
    Returns ``(y_new, err, k_last)`` as ``erk_step`` does; ``err`` is propagated
    to the end of the step, and the step is exact when ``f`` vanishes.
    """

    proj = project if project is not None else (lambda x: x)
    rows = pair.a[:-1] if pair.fsal else pair.a
    ks = [f(t, y) if k1 is None else k1]
    for row, ci in zip(rows, pair.c[1:], strict=False):
        ks.append(f(t + ci * h, proj(_lawson_combine(propagate, y, h, ci, row, ks, pair.c))))
    y_new = _lawson_combine(propagate, y, h, 1.0, pair.b, ks, pair.c)
    if pair.fsal:
        ks.append(f(t + h, y_new))
    err = _lawson_combine(propagate, None, h, 1.0, pair.e, ks, pair.c)
    return y_new, err, ks[-1]


# --------------------------------------------------------------------------
# Low-storage strong-stability-preserving Runge-Kutta
# --------------------------------------------------------------------------
class LowStorageSSP(NamedTuple):
    """Two-register SSP Runge-Kutta scheme.

    Each row ``(a, b, g, d, e)`` evaluates ``k = f(q1)``, sets
    ``q1 <- a q2 + b q1 + g h k`` and then ``q2 <- d q2 + e q1``; both
    registers start at ``y``. ``ssp`` is the SSP coefficient: the step is
    stable in any norm in which forward Euler is stable at ``h / ssp``.
    """

    rows: tuple[tuple[float, float, float, float, float], ...]
    order: int
    ssp: float


_EULER6 = (0.0, 1.0, 1 / 6, 1.0, 0.0)
SSPRK3 = LowStorageSSP(  # Shu-Osher (3,3)
    ((0.0, 1.0, 1.0, 1.0, 0.0), (0.75, 0.25, 0.25, 1.0, 0.0), (1 / 3, 2 / 3, 2 / 3, 1.0, 0.0)),
    3,
    1.0,
)
SSPRK43 = LowStorageSSP(  # Kraaijevanger (4,3)
    (
        (0.0, 1.0, 0.5, 1.0, 0.0),
        (0.0, 1.0, 0.5, 1.0, 0.0),
        (2 / 3, 1 / 3, 1 / 6, 1.0, 0.0),
        (0.0, 1.0, 0.5, 1.0, 0.0),
    ),
    3,
    2.0,
)
SSPRK104 = LowStorageSSP(  # Ketcheson (10,4), SIAM J. Sci. Comput. 30 (2008)
    (_EULER6,) * 4
    + ((0.6, 0.4, 1 / 15, -0.5, 0.9),)
    + (_EULER6,) * 4
    + ((1.0, 0.6, 0.1, 1.0, 0.0),),
    4,
    6.0,
)


def ssp_step(
    f: Callable[[Any, Any], Any],
    t: Any,
    y: Any,
    h: Any,
    scheme: LowStorageSSP = SSPRK3,
    project: Callable[[Any], Any] | None = None,
) -> Any:
    """One low-storage SSP Runge-Kutta step; ``project`` is applied to each new ``q1``."""

    proj = project if project is not None else (lambda x: x)
    q1 = q2 = y
    c1 = c2 = 0.0  # stage times, from the same recurrence applied to y' = 1
    for a, b, g, d, e in scheme.rows:
        k = f(t + c1 * h, q1)
        update = lambda p2, p1, kk, a=a, b=b, g=g: a * p2 + b * p1 + g * h * kk  # noqa: E731
        q1 = proj(jax.tree.map(update, q2, q1, k))
        c1 = a * c2 + b * c1 + g
        if (d, e) != (1.0, 0.0):
            q2 = jax.tree.map(lambda p2, p1, d=d, e=e: d * p2 + e * p1, q2, q1)
            c2 = d * c2 + e * c1
    return q1


# --------------------------------------------------------------------------
# Step control and a chunked driver
# --------------------------------------------------------------------------
def cfl_dt(rate: Any, cfl: float, dt_max: Any = jnp.inf, dt_min: Any = 0.0) -> jax.Array:
    """``cfl / rate`` clipped to ``[dt_min, dt_max]``; ``rate <= 0`` gives ``dt_max``.

    ``rate`` is the largest signal speed over grid spacing, or the largest
    eigenvalue modulus of the explicit part, evaluated on the current state.
    """

    rate = jnp.asarray(rate)
    safe = jnp.where(rate > 0, rate, 1.0)
    return jnp.clip(jnp.where(rate > 0, cfl / safe, dt_max), dt_min, dt_max)


def controlled(
    step: Callable[[Any, Any, Any], Any],
    *,
    dt_limit: Callable[[Any, Any], Any] | None = None,
    embedded_order: int | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    controller: PIController | None = None,
) -> Callable[[Any, Any, Any, Any], tuple[Any, Any, Any, Any]]:
    """Wrap a one-step map ``step(t, y, h)`` into the step contract of ``integrate``.

    Without ``embedded_order`` the step returns ``y_new``, is always accepted and
    keeps its step size. With it, the step returns ``(y_new, err)``, is accepted
    when the weighted error norm is at most 1, and the next step comes from the
    PI controller; the controller state (the previous error norm) is the carried
    ``aux``. ``dt_limit(t, y)``, for example a CFL bound from ``cfl_dt``, caps
    the next step and replaces it in the fixed-step case.
    """

    ctrl = PIController() if controller is None else controller

    def wrapped(t: Any, y: Any, h: Any, aux: Any) -> tuple[Any, Any, Any, Any]:
        out = step(t, y, h)
        if embedded_order is None:
            dt = h if dt_limit is None else dt_limit(t + h, out)
            return out, dt, jnp.asarray(True), aux
        y_new, err = out
        en = error_norm(err, y, y_new, rtol, atol)
        ok = en <= 1.0
        dt = h * step_factor(ctrl, en, jnp.where(ok, aux, 1.0), embedded_order)
        if dt_limit is not None:
            dt = jnp.minimum(dt, dt_limit(t + h, y_new))
        return y_new, dt, ok, jnp.where(ok, jnp.maximum(en, 1e-10), aux)

    return wrapped


class Trajectory(NamedTuple):
    y: Any  # last accepted, finite state
    t: jax.Array
    dt: jax.Array  # the next step size
    steps: jax.Array  # accepted steps
    rejected: jax.Array  # rejected attempts, by the controller or for a non-finite state
    ok: jax.Array  # False when more than max_retries consecutive attempts were rejected
    saved: Any  # observe(t, y) after each chunk, stacked; None without observe


def _report(chunk: Any, t: Any, dt: Any, steps: Any, rejected: Any, ok: Any) -> None:
    flag = "" if ok else "  stopped: non-finite or rejected state"
    print(
        f"chunk {int(chunk)}: step {int(steps)}  t = {float(t):.6g}  dt = {float(dt):.3g}  "
        f"rejected {int(rejected)}{flag}",
        flush=True,
    )


def integrate(
    step: Callable[[Any, Any, Any, Any], tuple[Any, Any, Any, Any]],
    y0: Any,
    t0: float,
    dt0: float,
    n_steps: int,
    *,
    t1: float | None = None,
    chunk: int = 100,
    aux0: Any = 1.0,
    max_retries: int = 10,
    checkpoint: bool = True,
    progress: bool | Callable[..., None] = False,
    observe: Callable[[Any, Any], Any] | None = None,
) -> Trajectory:
    """Advance ``y0`` by at most ``n_steps`` attempts of ``step``, in chunks of ``lax.scan``.

    ``step(t, y, h, aux) -> (y_new, dt_next, accept, aux_new)``; ``controlled``
    builds one from a stepper. An attempt whose state is non-finite is rejected
    and retried at half the step; one the step rejects is retried at
    ``dt_next``. After ``max_retries`` consecutive rejections the run stops with
    ``ok = False`` and returns the last good state. The run also stops at ``t1``,
    clipping the last step. ``n_steps`` and ``chunk`` are static.

    With ``checkpoint`` each chunk is a ``jax.checkpoint``, so reverse mode
    stores one carry per chunk and recomputes one chunk at a time; ``chunk``
    near ``sqrt(n_steps)`` balances the two. ``progress`` prints one flushed
    line per chunk from the running program (or calls the given function with
    ``chunk, t, dt, steps, rejected, ok``). The function is pure and traceable,
    so it differentiates, jits and vmaps; under ``vmap`` the step is evaluated
    for every lane until all lanes stop.
    """

    leaves = jax.tree.leaves(y0)
    real = jnp.finfo(jnp.result_type(*leaves, jnp.float32)).dtype
    t_end = jnp.asarray(jnp.inf if t1 is None else t1, real)
    n_chunks = -(-n_steps // chunk)
    report = _report if progress is True else progress

    def attempt(c: tuple[Any, ...]) -> tuple[Any, ...]:
        t, y, dt, aux, steps, rejected, retries, ok = c
        h = jnp.minimum(dt, t_end - t)
        y_new, dt_next, accept, aux_new = step(t, y, h, aux)
        finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(y_new)]))
        good = finite & accept

        def pick(cond: Any, a: Any, b: Any) -> Any:
            return jax.tree.map(lambda u, v: jnp.where(cond, u, v), a, b)

        retries = jnp.where(good, 0, retries + 1)
        return (
            jnp.where(good, t + h, t),
            pick(good, y_new, y),
            jnp.where(finite, jnp.asarray(dt_next, real), 0.5 * h).astype(real),
            pick(finite, aux_new, aux),
            steps + good,
            rejected + (~good),
            retries,
            ok & (retries <= max_retries),
        )

    def run_chunk(c: tuple[Any, ...], index: Any) -> tuple[Any, ...]:
        def body(cc: tuple[Any, ...], j: Any) -> tuple[Any, None]:
            live = cc[7] & (cc[0] < t_end) & (index * chunk + j < n_steps)
            return jax.lax.cond(live, attempt, lambda x: x, cc), None

        return jax.lax.scan(body, c, jnp.arange(chunk))[0]

    chunk_fn = jax.checkpoint(run_chunk) if checkpoint else run_chunk

    def outer(c: tuple[Any, ...], index: Any) -> tuple[Any, Any]:
        live = c[7] & (c[0] < t_end)
        c = chunk_fn(c, index)
        if report:
            jax.lax.cond(
                live,
                lambda: jax.debug.callback(report, index, c[0], c[2], c[4], c[5], c[7]),
                lambda: None,
            )
        return c, None if observe is None else observe(c[0], c[1])

    zero = jnp.int32(0)
    aux0 = jax.tree.map(jnp.asarray, aux0)
    t_0, dt_0 = jnp.asarray(t0, real), jnp.asarray(dt0, real)
    init = (t_0, y0, dt_0, aux0, zero, zero, zero, jnp.bool_(True))
    c, saved = jax.lax.scan(outer, init, jnp.arange(n_chunks))
    return Trajectory(c[1], c[0], c[2], c[4], c[5], c[7], saved)


__all__ = [
    "ARK3_KC",
    "ARS_343",
    "BS3_2",
    "DOPRI5_4",
    "ERKPair",
    "ERKSolution",
    "IMEX_EULER",
    "IMEX_SSP2",
    "IMEX_TABLEAUX",
    "IMEXTableau",
    "LowStorageSSP",
    "PAIRS",
    "PIController",
    "RK4_2",
    "SMR_RK3_CN",
    "SSPRK104",
    "SSPRK3",
    "SSPRK43",
    "Trajectory",
    "cfl_dt",
    "controlled",
    "erk_integrate",
    "erk_step",
    "error_norm",
    "imex_step",
    "integrate",
    "lawson_step",
    "ssp_step",
    "step_factor",
]
