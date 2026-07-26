"""Multigrid relaxation sweeps: point, block, line, plane and upwind-ordered.

A *smoother* is not a solver. Its only job is to annihilate the error
components a coarse grid cannot represent, so that what survives the sweep
is smooth enough for the coarse-grid correction to remove. Multigrid works
when the two are complementary, and it fails — at any cycle index, with
any transfer — when they are not (Brandt 1977; Trottenberg et al., ch. 2
and 4).

Every builder here returns a callable in the multigrid smoother protocol

    smooth(matvec, x, b) -> x,

which improves an existing iterate and is what :func:`solvax.precond.
multigrid` expects per level. This is deliberately *not* the
preconditioner protocol ``precond(b) -> x`` of :mod:`solvax.precond`: a
smoother is applied repeatedly to a running iterate, so it must see the
current ``x``. :func:`relaxation` converts any approximate inverse
``M^{-1}`` into a smoother by the damped correction

    x <- x + omega M^{-1} (b - A x),

and every other builder in this module is that adaptor wrapped around a
particular ``M``:

- :func:`jacobi_smoother` — ``M = diag(A)``. Cheap and perfectly
  parallel; the classical choice for isotropic operators, where
  ``omega = 2/3`` minimizes the smoothing factor of the model Laplacian.
- :func:`block_jacobi_smoother` — dense blocks along one axis (velocity,
  species, moments), LU-factored once and applied batched: the remedy
  when the *within-cell* physics is stiff.
- :func:`tridiagonal_smoother` — line relaxation: solve the strongly
  coupled grid direction exactly, batched over all others, using the
  vendor-aware :func:`solvax.tridiagonal.tridiagonal_solve` (cuSPARSE on
  GPU, Thomas on CPU) or its cyclic variant on a periodic axis. This is
  the standard cure for anisotropic coupling, where point smoothers stall
  (Trottenberg et al., ch. 5).
- :func:`plane_smoother` — solve whole two-dimensional planes exactly,
  batched over the remaining axes, via banded (or periodic banded) LU.
  Needed when the operator is strongly coupled in *two* directions at
  once, e.g. across both field-line angles.
- :func:`upwind_smoother` — line relaxation whose sweep *order follows the
  wind*. For a streaming/advection operator the error is transported along
  characteristics, so a relaxation ordered downstream removes it in
  essentially one sweep while the reverse ordering barely converges. The
  ordering is expressed pointwise: keeping only the coupling to the
  upstream neighbour makes the swept operator triangular in the flow
  direction, so one tridiagonal solve *is* the ordered Gauss-Seidel sweep
  — no sequential loop over grid points, no dynamic control flow.
- :func:`alternating_smoother` — compose the above multiplicatively
  (alternating-direction relaxation); each component refreshes the
  residual itself.

:func:`smoothing_factor` measures what a smoother actually does: the
asymptotic error-reduction rate restricted to the high-frequency modes a
given coarsening cannot represent, estimated by power iteration on the
Fourier-projected error-propagation operator. This is the empirical
counterpart of Brandt's local Fourier analysis smoothing factor ``mu``,
and it is the number to tune ``omega``, sweep counts and orderings
against — a smoother with ``mu`` near 1 will not be rescued by the rest
of the cycle.

References
----------
- A. Brandt, "Multi-level adaptive solutions to boundary-value problems",
  Math. Comp. 31, 333 (1977) — local Fourier analysis, the smoothing
  factor, and smoother/coarse-grid complementarity.
- U. Trottenberg, C. W. Oosterlee & A. Schüller, *Multigrid*, Academic
  Press (2001) — chapters 2, 4 and 5: damped Jacobi and its optimal
  ``omega``, line and plane relaxation, alternating directions,
  anisotropy and semicoarsening.
- A. Brandt & I. Yavneh, "Accelerated multigrid convergence and
  high-Reynolds recirculating flows", SIAM J. Sci. Comput. 14, 607 (1993)
  — downstream (upwind) relaxation ordering for convection-dominated
  operators.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
  (2003), ch. 4 — Jacobi, Gauss-Seidel and block relaxation as splittings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import lu_factor, lu_solve

from solvax.banded import (
    lu_factor_banded,
    lu_factor_banded_periodic,
    lu_solve_banded,
    lu_solve_banded_periodic,
)
from solvax.tridiagonal import (
    _reusable_tridiagonal_solver,
    cyclic_tridiagonal_solve,
)

MatVec = Callable[[jax.Array], jax.Array]
Smoother = Callable[[MatVec, jax.Array, jax.Array], jax.Array]

#: Sweep orderings understood by :func:`upwind_smoother`.
ORDERINGS = ("upwind", "downwind", "forward", "backward")


def _axis_permutation(axis: int, ndim: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Permutation bringing ``axis`` to the front, and its inverse."""
    axis = int(axis) % ndim
    permutation = (axis,) + tuple(i for i in range(ndim) if i != axis)
    inverse = tuple(permutation.index(i) for i in range(ndim))
    return permutation, inverse


def _shift(values: jax.Array, offset: int) -> jax.Array:
    """Zero-filled shift along the last axis: ``out[..., j] = values[..., j + offset]``."""
    if offset == 0:
        return values
    pad = jnp.zeros_like(values[..., : abs(offset)])
    if offset > 0:
        return jnp.concatenate([values[..., offset:], pad], axis=-1)
    return jnp.concatenate([pad, values[..., :offset]], axis=-1)


def relaxation(solve: MatVec, *, omega: float = 1.0, sweeps: int = 1) -> Smoother:
    """Turn an approximate inverse into a damped relaxation sweep.

    Applies ``x <- x + omega * solve(b - A x)`` ``sweeps`` times, refreshing
    the residual between sweeps. With ``solve = M^{-1}`` this is the
    stationary iteration of the splitting ``A = M - (M - A)``, whose error
    propagation operator is ``I - omega M^{-1} A``; ``omega`` trades
    asymptotic (low-frequency) convergence for high-frequency damping,
    which is the only thing a smoother is asked to provide.

    Args:
        solve: approximate inverse action ``r -> M^{-1} r``, e.g. a line or
            plane solve; must preserve the shape of ``r``.
        omega: relaxation weight.
        sweeps: number of sweeps per smoother application (static int).

    Returns:
        A callable ``smooth(matvec, x, b) -> x``.
    """
    if sweeps < 1:
        raise ValueError("sweeps must be >= 1")
    omega = float(omega)

    def smooth(matvec: MatVec, x: jax.Array, b: jax.Array) -> jax.Array:
        for _ in range(sweeps):
            x = x + omega * solve(b - matvec(x))
        return x

    return smooth


def jacobi_smoother(
    diagonal: jax.Array, *, omega: float = 2.0 / 3.0, sweeps: int = 1
) -> Smoother:
    """Damped point-Jacobi relaxation ``x <- x + omega diag(A)^{-1} r``.

    The default ``omega = 2/3`` is the classical optimum for the model
    Laplacian: it minimizes the worst high-frequency amplification factor,
    giving a smoothing factor of ``1/3`` per sweep in any dimension
    (Trottenberg et al., section 2.1). Undamped Jacobi (``omega = 1``)
    leaves the highest frequencies untouched and is *not* a smoother.

    Args:
        diagonal: ``diag(A)`` with the shape of the iterate.
        omega: relaxation weight.
        sweeps: number of sweeps per application.

    Returns:
        A callable ``smooth(matvec, x, b) -> x``.
    """
    inverse = 1.0 / jnp.asarray(diagonal)
    return relaxation(lambda residual: inverse * residual, omega=omega, sweeps=sweeps)


def block_jacobi_smoother(
    blocks: jax.Array,
    *,
    axis: int = -1,
    omega: float = 1.0,
    sweeps: int = 1,
) -> Smoother:
    """Damped block-Jacobi relaxation with caller-supplied dense blocks.

    Each block couples all entries along ``axis`` — a velocity, pitch,
    species or moment index — at one grid point, and is inverted exactly.
    The blocks are LU-factored once at build time (batched, with partial
    pivoting) and applied as one batched ``lu_solve`` per sweep, so the
    within-cell physics is treated implicitly while the grid coupling is
    left to the outer cycle.

    Args:
        blocks: dense diagonal blocks, shape ``(*batch, m, m)`` where
            ``batch`` is the iterate's shape with ``axis`` removed and
            ``m`` is its length along ``axis``. A single ``(m, m)`` block
            is shared by every grid point.
        axis: iterate axis spanned by each block.
        omega: relaxation weight.
        sweeps: number of sweeps per application.

    Returns:
        A callable ``smooth(matvec, x, b) -> x``.
    """
    blocks = jnp.asarray(blocks)
    if blocks.ndim < 2 or blocks.shape[-1] != blocks.shape[-2]:
        raise ValueError("blocks must have shape (*batch, m, m)")
    size = blocks.shape[-1]
    shared = blocks.ndim == 2
    if shared:
        factors = lu_factor(blocks)
    else:
        factors = jax.vmap(lu_factor)(blocks.reshape(-1, size, size))

    def solve(residual: jax.Array) -> jax.Array:
        moved = jnp.moveaxis(residual, axis, -1)
        flat = moved.reshape(-1, size)
        with jax.named_scope("solvax.block_jacobi_smoother.lu_solve"):
            if shared:
                solved = lu_solve(factors, flat.T).T
            else:
                solved = jax.vmap(lambda lu, piv, rhs: lu_solve((lu, piv), rhs))(
                    factors[0], factors[1], flat
                )
        return jnp.moveaxis(solved.reshape(moved.shape), -1, axis)

    return relaxation(solve, omega=omega, sweeps=sweeps)


def tridiagonal_smoother(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    *,
    axis: int = 0,
    periodic: bool = False,
    omega: float = 1.0,
    sweeps: int = 1,
) -> Smoother:
    """Line relaxation along one grid axis, batched over the others.

    Solves the tridiagonal operator restricted to every line parallel to
    ``axis`` exactly, treating the coupling to neighbouring lines
    explicitly. That is the standard response to anisotropy: whichever
    direction dominates the operator is inverted, so the error left behind
    is smooth in that direction rather than merely averaged
    (Trottenberg et al., section 5.1). All other axes ride along as batch
    columns of one :func:`solvax.tridiagonal.tridiagonal_solve` call, which
    is exactly the layout the fused vendor kernels want; on a CPU the
    Thomas factors are computed once at build time and reused by every
    sweep.

    Args:
        lower: sub-diagonal along ``axis``, iterate-shaped; entry ``j``
            couples line point ``j`` to ``j - 1``.
        diag: main diagonal, iterate-shaped.
        upper: super-diagonal along ``axis``, iterate-shaped.
        axis: grid axis holding the lines.
        periodic: solve cyclic lines, i.e. ``lower`` at the first point and
            ``upper`` at the last one wrap around (needs at least 3 points).
        omega: relaxation weight.
        sweeps: number of sweeps per application.

    Returns:
        A callable ``smooth(matvec, x, b) -> x``.
    """
    lower, diag, upper = (jnp.asarray(value) for value in (lower, diag, upper))
    if not (lower.shape == diag.shape == upper.shape):
        raise ValueError("lower, diag and upper must share the iterate shape")
    permutation, inverse = _axis_permutation(axis, diag.ndim)
    bands = tuple(jnp.transpose(value, permutation) for value in (lower, diag, upper))

    if periodic:

        def solve(residual: jax.Array) -> jax.Array:
            with jax.named_scope("solvax.tridiagonal_smoother.cyclic_solve"):
                solved = cyclic_tridiagonal_solve(
                    *bands, jnp.transpose(residual, permutation)
                )
            return jnp.transpose(solved, inverse)

    else:
        lines = _reusable_tridiagonal_solver(*bands)

        def solve(residual: jax.Array) -> jax.Array:
            with jax.named_scope("solvax.tridiagonal_smoother.tridiagonal_solve"):
                solved = lines(jnp.transpose(residual, permutation))
            return jnp.transpose(solved, inverse)

    return relaxation(solve, omega=omega, sweeps=sweeps)


def upwind_smoother(
    wind: jax.Array,
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    *,
    axis: int = 0,
    order: str = "upwind",
    periodic: bool = False,
    omega: float = 1.0,
    sweeps: int = 1,
) -> Smoother:
    r"""Line relaxation whose sweep order follows an advection field.

    For a streaming operator such as ``v_par b.grad`` the error is
    transported along characteristics, so relaxation must visit grid points
    *downstream*: a Gauss-Seidel sweep ordered with the flow reduces the
    error by a factor that grows with the mesh Peclet number, while the
    same sweep ordered against the flow stagnates (Brandt & Yavneh 1993;
    Trottenberg et al., section 7.4).

    The ordering is expressed pointwise rather than as a loop. Sweeping in
    increasing index means the *upstream* neighbour is already updated
    while the downstream one is not, i.e. the sweep solves

        (D + L) x = r      where wind > 0,
        (D + U) x = r      where wind < 0,

    keeping only the coupling to the upstream neighbour. Selecting those
    bands with :func:`jax.numpy.where` makes the swept operator triangular
    in the flow direction wherever the wind has one sign, so a *single*
    :func:`solvax.tridiagonal.tridiagonal_solve` performs the whole ordered
    sweep — batched over every other axis, with no sequential grid loop and
    no data-dependent control flow. Where the sign changes, the remaining
    2x2 coupling is simply solved exactly too, which is stronger than
    Gauss-Seidel, never weaker.

    Args:
        wind: advection velocity along ``axis``, iterate-shaped. Only its
            sign is used.
        lower: sub-diagonal along ``axis``, iterate-shaped.
        diag: main diagonal, iterate-shaped.
        upper: super-diagonal along ``axis``, iterate-shaped.
        axis: grid axis holding the lines.
        order: ``"upwind"`` follows the wind; ``"downwind"`` is its exact
            reverse; ``"forward"`` and ``"backward"`` are the wind-agnostic
            fixed orderings (increasing and decreasing index). The last
            three exist to measure what the ordering is worth — at high
            mesh Peclet number ``"upwind"`` beats all of them.
        periodic: solve cyclic lines.
        omega: relaxation weight.
        sweeps: number of sweeps per application.

    Returns:
        A callable ``smooth(matvec, x, b) -> x``.
    """
    if order not in ORDERINGS:
        raise ValueError(f"unknown order {order!r}; expected one of {ORDERINGS}")
    wind, lower, diag, upper = (
        jnp.asarray(value) for value in (wind, lower, diag, upper)
    )
    if wind.shape != diag.shape:
        raise ValueError("wind must share the iterate shape")
    if order == "forward":
        keep_lower = jnp.ones(diag.shape, bool)
    elif order == "backward":
        keep_lower = jnp.zeros(diag.shape, bool)
    elif order == "upwind":
        keep_lower = wind >= 0
    else:
        keep_lower = wind < 0
    zero = jnp.zeros((), diag.dtype)
    return tridiagonal_smoother(
        jnp.where(keep_lower, lower, zero),
        diag,
        jnp.where(keep_lower, zero, upper),
        axis=axis,
        periodic=periodic,
        omega=omega,
        sweeps=sweeps,
    )


def _plane_bands(
    diagonal: jax.Array,
    lower_first: jax.Array,
    upper_first: jax.Array,
    lower_second: jax.Array,
    upper_second: jax.Array,
    n_first: int,
    n_second: int,
    periodic_second: bool,
) -> jax.Array:
    """Banded storage of a five-point plane operator in row-major order.

    Unknowns are ordered first-axis-major, so the second-axis couplings sit
    on the first off-diagonals and the first-axis couplings ``n_second``
    away: the bandwidth is ``n_second`` either side. Entries that would
    join two different lines are masked out, and the periodic second-axis
    wrap lands inside the band at offset ``n_second - 1``.
    """
    size = n_first * n_second
    index = jnp.arange(size)
    bands = jnp.zeros((*diagonal.shape[:-1], 2 * n_second + 1, size), diagonal.dtype)

    def place(offset: int, values: jax.Array, mask: jax.Array | None = None) -> None:
        nonlocal bands
        shifted = _shift(values, offset)
        if mask is not None:
            shifted = jnp.where(mask, shifted, 0.0)
        bands = bands.at[..., n_second + offset, :].add(shifted)

    place(0, diagonal)
    place(1, lower_second, ((index + 1) % n_second) != 0)
    place(-1, upper_second, (index % n_second) != 0)
    place(n_second, lower_first)
    place(-n_second, upper_first)
    if periodic_second:
        place(-(n_second - 1), lower_second, (index % n_second) == (n_second - 1))
        place(n_second - 1, upper_second, (index % n_second) == 0)
    return bands


def plane_smoother(
    diagonal: jax.Array,
    first: tuple[jax.Array, jax.Array],
    second: tuple[jax.Array, jax.Array],
    *,
    axes: tuple[int, int] = (-2, -1),
    periodic: tuple[bool, bool] = (False, False),
    omega: float = 1.0,
    sweeps: int = 1,
) -> Smoother:
    """Exact two-dimensional plane relaxation, batched over the other axes.

    When an operator is strongly coupled along *two* axes at once — both
    field-line angles of a transport operator, say — no line smoother
    covers it and the alternating-direction composition degrades as the
    coupling grows. Plane relaxation inverts the whole plane instead
    (Trottenberg et al., section 5.2). The five-point plane operator is
    assembled in banded storage with bandwidth equal to the second plane
    axis and factored once per plane with the non-pivoted banded LU of
    :mod:`solvax.banded`; a periodic first axis is handled by the
    capacitance (Woodbury) variant, and a periodic second axis costs
    nothing at all because its wrap-around already lies inside the band.

    The exact plane solve costs ``O(n_first * n_second^2)`` per plane
    instead of the ``O((n_first * n_second)^3)`` of a dense block inverse,
    but its factors are ``O(n_first * n_second^2)`` per plane — order the
    ``axes`` so the *shorter* one is second.

    Args:
        diagonal: main diagonal, iterate-shaped.
        first: ``(lower, upper)`` couplings along ``axes[0]``,
            iterate-shaped.
        second: ``(lower, upper)`` couplings along ``axes[1]``,
            iterate-shaped.
        axes: the two grid axes spanned by each plane.
        periodic: whether each plane axis wraps around. A periodic second
            axis needs at least three points.
        omega: relaxation weight.
        sweeps: number of sweeps per application.

    Returns:
        A callable ``smooth(matvec, x, b) -> x``.
    """
    diagonal = jnp.asarray(diagonal)
    lower_first, upper_first = (jnp.asarray(value) for value in first)
    lower_second, upper_second = (jnp.asarray(value) for value in second)
    shape = diagonal.shape
    ndim = diagonal.ndim
    axis_first, axis_second = (int(value) % ndim for value in axes)
    if axis_first == axis_second:
        raise ValueError("axes must be two distinct grid axes")
    for value in (lower_first, upper_first, lower_second, upper_second):
        if value.shape != shape:
            raise ValueError("every band must share the iterate shape")
    n_first, n_second = shape[axis_first], shape[axis_second]
    if periodic[1] and n_second < 3:
        raise ValueError("a periodic second plane axis needs at least three points")

    rest = tuple(i for i in range(ndim) if i not in (axis_first, axis_second))
    permutation = (*rest, axis_first, axis_second)
    inverse = tuple(permutation.index(i) for i in range(ndim))
    batch = tuple(shape[i] for i in rest)
    size = n_first * n_second

    def flatten(value: jax.Array) -> jax.Array:
        return jnp.transpose(value, permutation).reshape(-1, size)

    bands = _plane_bands(
        flatten(diagonal),
        flatten(lower_first),
        flatten(upper_first),
        flatten(lower_second),
        flatten(upper_second),
        n_first,
        n_second,
        periodic[1],
    )

    if periodic[0]:
        eye = jnp.eye(n_second, dtype=diagonal.dtype)
        corner_ul = flatten(lower_first)[:, :n_second, None] * eye
        corner_lr = flatten(upper_first)[:, size - n_second :, None] * eye
        factors = jax.vmap(
            lambda band, upper_right, lower_left: lu_factor_banded_periodic(
                band, n_second, n_second, upper_right, lower_left
            )
        )(bands, corner_ul, corner_lr)
        apply_factors = jax.vmap(lu_solve_banded_periodic)
    else:
        factors = jax.vmap(lambda band: lu_factor_banded(band, n_second, n_second))(bands)
        apply_factors = jax.vmap(lu_solve_banded)

    def solve(residual: jax.Array) -> jax.Array:
        with jax.named_scope("solvax.plane_smoother.banded_solve"):
            solved = apply_factors(factors, flatten(residual))
        return jnp.transpose(solved.reshape(*batch, n_first, n_second), inverse)

    return relaxation(solve, omega=omega, sweeps=sweeps)


def alternating_smoother(smoothers: Sequence[Smoother]) -> Smoother:
    """Compose smoothers multiplicatively, in order.

    Each component sees the iterate the previous one produced and refreshes
    the residual itself, so this is a multiplicative (Gauss-Seidel-like)
    composition, not an average. Alternating the directions of line
    relaxation covers anisotropy of unknown or mixed orientation, and
    alternating a line sweep with a block sweep covers grid and within-cell
    stiffness at once (Trottenberg et al., section 5.1). For the additive,
    PCG-safe composition of *fixed symmetric* inverse actions use
    :func:`solvax.precond.additive_preconditioner` instead.

    Args:
        smoothers: nonempty sequence of smoothers in the
            ``smooth(matvec, x, b) -> x`` protocol.

    Returns:
        A callable ``smooth(matvec, x, b) -> x`` applying them in order.
    """
    smoothers = tuple(smoothers)
    if not smoothers:
        raise ValueError("smoothers must contain at least one sweep")

    def smooth(matvec: MatVec, x: jax.Array, b: jax.Array) -> jax.Array:
        for component in smoothers:
            x = component(matvec, x, b)
        return x

    return smooth


def high_frequency_mask(
    shape: Sequence[int], coarsen: Sequence[bool] | None = None
) -> jax.Array:
    """Boolean mask of the Fourier modes a coarsening cannot represent.

    On a periodic grid, mode ``k`` along an axis has frequency
    ``theta = 2 pi k / n``; halving that axis aliases everything with
    ``|theta| >= pi / 2`` onto the coarse grid. A mode is therefore *high
    frequency* — the smoother's responsibility, not the coarse grid's — as
    soon as it is high along at least one coarsened axis.

    Args:
        shape: grid shape.
        coarsen: one boolean per axis; defaults to every axis (standard
            full coarsening).

    Returns:
        A boolean array of shape ``shape``, True on high-frequency modes in
        :func:`jax.numpy.fft.fftn` index order.
    """
    shape = tuple(int(n) for n in shape)
    coarsen = (True,) * len(shape) if coarsen is None else tuple(bool(c) for c in coarsen)
    if len(coarsen) != len(shape):
        raise ValueError("coarsen must have one entry per axis of shape")
    if not any(coarsen):
        raise ValueError("at least one axis must be coarsened")
    mask = jnp.zeros(shape, bool)
    for axis, (n, flag) in enumerate(zip(shape, coarsen, strict=True)):
        if not flag:
            continue
        wave = jnp.arange(n)
        folded = jnp.minimum(wave, n - wave)
        high = (4 * folded >= n).reshape((n,) + (1,) * (len(shape) - axis - 1))
        mask = mask | high
    return mask


def smoothing_factor(
    smooth: Smoother,
    matvec: MatVec,
    shape: Sequence[int],
    *,
    key: jax.Array,
    coarsen: Sequence[bool] | None = None,
    steps: int = 24,
    dtype: Any = None,
) -> jax.Array:
    """Measured high-frequency error-reduction rate of a smoother.

    Estimates Brandt's smoothing factor

        mu = rho( Q_high (I - M^{-1} A) Q_high ),

    the asymptotic rate at which the sweep reduces exactly those error
    components the coarse grid cannot represent, with ``Q_high`` the
    Fourier projector of :func:`high_frequency_mask`. The estimate is a
    power iteration: start from a random high-frequency error, apply the
    smoother to it with a zero right-hand side (so the iterate *is* the
    error), project back onto the high-frequency space, renormalize, and
    report the last growth factor.

    Unlike analytic local Fourier analysis this needs no stencil algebra
    and works for any smoother in this module, including plane, block and
    upwind-ordered ones; unlike a plain residual-reduction measurement it
    cannot be flattered by low-frequency components the coarse grid was
    going to remove anyway. ``mu < 1`` is necessary for a working cycle,
    ``mu`` around ``0.5`` or below is what makes grid-independent
    convergence possible.

    The projector is periodic (an FFT), so on a non-periodic grid the
    result is the usual interior-mode idealization.

    Args:
        smooth: the smoother to measure, ``smooth(matvec, x, b) -> x``.
        matvec: the operator being smoothed.
        shape: grid shape of the iterate.
        key: PRNG key for the starting error.
        coarsen: axes that the hierarchy coarsens; defaults to all of them.
        steps: power-iteration steps.
        dtype: iterate dtype; defaults to the platform default float.

    Returns:
        The scalar smoothing factor estimate.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    shape = tuple(int(n) for n in shape)
    dtype = jnp.result_type(float) if dtype is None else dtype
    mask = high_frequency_mask(shape, coarsen)
    zero = jnp.zeros(shape, dtype)

    def project(error: jax.Array) -> jax.Array:
        spectrum = jnp.fft.fftn(error)
        return jnp.real(jnp.fft.ifftn(jnp.where(mask, spectrum, 0.0))).astype(dtype)

    def normalize(error: jax.Array) -> tuple[jax.Array, jax.Array]:
        norm = jnp.linalg.norm(error)
        return error / jnp.where(norm > 0, norm, 1.0), norm

    start, _ = normalize(project(jax.random.normal(key, shape, dtype)))

    def body(_, state):
        error, _ = state
        return normalize(project(smooth(matvec, error, zero)))

    _, rate = lax.fori_loop(0, int(steps), body, (start, jnp.zeros((), dtype)))
    return rate
