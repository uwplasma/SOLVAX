"""Grid transfer operators for geometric and semicoarsened multigrid.

A geometric multigrid hierarchy needs two transfers per level: a
*restriction* ``R`` carrying a fine residual to the coarse grid and a
*prolongation* ``P`` carrying a coarse correction back. On a structured
tensor-product grid both are separable — the ``d``-dimensional operator is
the Kronecker product of ``d`` one-dimensional stencils — so this module
never forms an N-D transfer matrix. It builds one small ``(n_c, n_f)``
matrix per coarsened axis and applies them as a sequence of per-axis
contractions

    x <- tensordot(M_axis, x, axes=[[1], [axis]])   (then moved back),

which is jit-, vmap- and grad-transparent, costs ``O(prod(shape) * n_f)``
per axis instead of ``O(prod(shape)^2)``, and leaves untouched axes
literally untouched (no matmul is emitted for them at all).

Leaving axes untouched is the point. *Semicoarsening* — coarsening only
the axes along which the operator is strongly coupled, keeping the others
at full resolution — is the standard cure for anisotropic and
convection-dominated operators, where a full-coarsening hierarchy loses
the coupling that made the smoother work (Trottenberg et al., ch. 5;
Brandt 1977, section 4). Every builder here therefore takes a boolean
axis mask.

Three grid closures are supported per axis, each with its own coarse-grid
alignment:

- ``"periodic"``: ``n_f`` even, ``n_c = n_f / 2``, coarse point ``i`` at
  fine point ``2i``, stencils wrap around.
- ``"dirichlet"``: ``n_f`` odd, ``n_c = (n_f - 1) / 2``; the grid holds
  interior unknowns only and values outside are zero, so coarse point
  ``i`` sits at fine point ``2i + 1`` — the classical vertex-centered
  hierarchy for which no stencil is ever truncated.
- ``"reflective"``: ``n_f`` even, ``n_c = n_f / 2``, cell-centered with a
  mirror closure (``x_{-1} = x_0``); coarse cell ``i`` covers fine cells
  ``2i`` and ``2i + 1``.

and two stencil families per transfer:

- restriction ``"full_weighting"`` — ``[1/4, 1/2, 1/4]`` on the
  vertex-centered closures, ``[1/8, 3/8, 3/8, 1/8]`` cell-centered — or
  ``"injection"``, which simply samples the coincident fine point;
- prolongation ``"linear"`` — multilinear interpolation, i.e.
  ``[1/2, 1, 1/2]`` vertex-centered and ``[1/4, 3/4, 3/4, 1/4]``
  cell-centered — or ``"injection"``, the adjoint of the injection
  restriction.

The stencil tables are written independently of one another, but they are
chosen to satisfy the *variational* (adjoint) relation exactly:

    R = c P^T,   c = 2^{-d}   for ``full_weighting`` / ``linear``,
                 c = 1        for ``injection`` / ``injection``,

with ``d`` the number of coarsened axes. That relation is what makes the
coarse-grid correction a projection in the ``A``-inner product when the
coarse operator is Galerkin, and it is pinned by the test suite.

Coarse operators are *not* built here. This module deliberately supports
**rediscretization** — rebuilding the operator directly on each coarse
grid — which is cheaper than the Galerkin triple product ``R A P``, keeps
the coarse operator in the same structured (banded, tridiagonal) form the
smoothers need, and is the standard choice for non-symmetric
convection-dominated operators (Trottenberg et al., section 2.8.3).

References
----------
- A. Brandt, "Multi-level adaptive solutions to boundary-value problems",
  Math. Comp. 31, 333 (1977) — the multigrid algorithm, local Fourier
  analysis, and the smoothing/coarse-grid complementarity principle.
- U. Trottenberg, C. W. Oosterlee & A. Schüller, *Multigrid*, Academic
  Press (2001) — chapters 2 and 5: transfer operators, the variational
  relation ``R = c P^T``, semicoarsening, and rediscretized coarse
  operators.
- P. Wesseling, *An Introduction to Multigrid Methods*, Wiley (1992) —
  cell-centered transfers and their mirror closures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

MatVec = Callable[[jax.Array], jax.Array]

#: Grid closures understood by every builder in this module.
BOUNDARIES = ("periodic", "dirichlet", "reflective")

#: Restriction stencils, as ``(offset from the coincident fine point, weight)``.
_RESTRICTION_STENCILS: dict[tuple[str, str], tuple[tuple[int, float], ...]] = {
    ("full_weighting", "periodic"): ((-1, 0.25), (0, 0.5), (1, 0.25)),
    ("full_weighting", "dirichlet"): ((-1, 0.25), (0, 0.5), (1, 0.25)),
    ("full_weighting", "reflective"): ((-1, 0.125), (0, 0.375), (1, 0.375), (2, 0.125)),
    ("injection", "periodic"): ((0, 1.0),),
    ("injection", "dirichlet"): ((0, 1.0),),
    ("injection", "reflective"): ((0, 1.0),),
}

#: Prolongation stencils, same convention; ``P[center + offset, i] = weight``.
_PROLONGATION_STENCILS: dict[tuple[str, str], tuple[tuple[int, float], ...]] = {
    ("linear", "periodic"): ((-1, 0.5), (0, 1.0), (1, 0.5)),
    ("linear", "dirichlet"): ((-1, 0.5), (0, 1.0), (1, 0.5)),
    ("linear", "reflective"): ((-1, 0.25), (0, 0.75), (1, 0.75), (2, 0.25)),
    ("injection", "periodic"): ((0, 1.0),),
    ("injection", "dirichlet"): ((0, 1.0),),
    ("injection", "reflective"): ((0, 1.0),),
}


class CoarseningPlan(NamedTuple):
    """Shapes and per-level axis masks of a (semi)coarsening hierarchy.

    Attributes:
        shapes: grid shapes, finest first; ``len(shapes) == len(masks) + 1``.
            ``shapes[-1]`` is the coarsest grid, the one the caller must
            supply an exact solve for.
        masks: per-level boolean tuples; ``masks[l][axis]`` is True when
            ``axis`` is coarsened going from level ``l`` to level ``l + 1``.
            An axis whose length is odd (periodic/reflective), even
            (dirichlet), or already at ``min_size`` stops coarsening while
            the others continue.
    """

    shapes: tuple[tuple[int, ...], ...]
    masks: tuple[tuple[bool, ...], ...]


class _AxisTransfer(eqx.Module):
    """Separable transfer: one ``(out, in)`` matmul per coarsened axis.

    Applying the matrices one axis at a time keeps the cost linear in the
    grid size and emits nothing at all for untouched axes. Arrays may carry
    extra *trailing* axes beyond ``input_shape`` (stacked fields, species,
    right-hand sides); they ride along untouched, as elsewhere in solvax.
    """

    matrices: tuple[jax.Array, ...]
    axes: tuple[int, ...] = eqx.field(static=True)
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        rank = len(self.input_shape)
        if x.ndim < rank or x.shape[:rank] != self.input_shape:
            raise ValueError(
                f"transfer expects an array whose leading axes are "
                f"{self.input_shape}, got shape {x.shape}"
            )
        for axis, matrix in zip(self.axes, self.matrices, strict=True):
            contracted = jnp.tensordot(matrix, x, axes=[[1], [axis]])
            x = jnp.moveaxis(contracted, 0, axis)
        return x


def _check_boundary(boundary: str) -> str:
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {BOUNDARIES}")
    return boundary


def _per_axis(value: Any, ndim: int, name: str) -> tuple[Any, ...]:
    """Broadcast a scalar option to one entry per axis."""
    if isinstance(value, str):
        return (value,) * ndim
    values = tuple(value)
    if len(values) != ndim:
        raise ValueError(f"{name} must be a string or have one entry per axis ({ndim})")
    return values


def _center(index: jax.Array, boundary: str) -> jax.Array:
    """Fine index of the coincident (or first covered) point of coarse ``index``."""
    return 2 * index + 1 if boundary == "dirichlet" else 2 * index


def _close(columns: jax.Array, n_fine: int, boundary: str) -> jax.Array:
    """Map out-of-range fine indices onto the grid for the given closure."""
    if boundary == "periodic":
        return jnp.mod(columns, n_fine)
    if boundary == "reflective":  # mirror about the outer cell faces
        columns = jnp.where(columns < 0, -1 - columns, columns)
        return jnp.where(columns >= n_fine, 2 * n_fine - 1 - columns, columns)
    return columns  # dirichlet stencils never leave the grid


def coarse_axis_size(n_fine: int, *, boundary: str = "periodic") -> int:
    """Length of one coarsened axis.

    Args:
        n_fine: fine-grid length of the axis.
        boundary: ``"periodic"``, ``"dirichlet"`` or ``"reflective"``.

    Returns:
        ``n_fine // 2`` for the periodic and reflective closures,
        ``(n_fine - 1) // 2`` for the dirichlet closure.

    Raises:
        ValueError: if ``boundary`` is unknown or ``n_fine`` has the wrong
            parity for it (even is required by ``"periodic"`` and
            ``"reflective"``, odd by ``"dirichlet"``), which would leave the
            coarse grid misaligned with the fine one.
    """
    _check_boundary(boundary)
    n_fine = int(n_fine)
    if boundary == "dirichlet":
        if n_fine % 2 != 1 or n_fine < 3:
            raise ValueError(
                f"the dirichlet closure needs an odd axis length >= 3, got {n_fine}"
            )
        return (n_fine - 1) // 2
    if n_fine % 2 != 0 or n_fine < 2:
        raise ValueError(
            f"the {boundary} closure needs an even axis length >= 2, got {n_fine}"
        )
    return n_fine // 2


def coarsenable(n_fine: int, *, boundary: str = "periodic", min_size: int = 4) -> bool:
    """Whether an axis may be coarsened once more.

    Args:
        n_fine: fine-grid length of the axis.
        boundary: grid closure, see :func:`coarse_axis_size`.
        min_size: smallest coarse length still considered useful.

    Returns:
        True when the parity fits the closure and the coarse length would
        still be at least ``min_size``.
    """
    _check_boundary(boundary)
    n_fine = int(n_fine)
    if boundary == "dirichlet":
        return n_fine % 2 == 1 and (n_fine - 1) // 2 >= min_size
    return n_fine % 2 == 0 and n_fine // 2 >= min_size


def coarse_grid_shape(
    shape: Sequence[int],
    coarsen: Sequence[bool],
    *,
    boundary: str | Sequence[str] = "periodic",
) -> tuple[int, ...]:
    """Grid shape after coarsening the masked axes once.

    Args:
        shape: fine grid shape.
        coarsen: one boolean per axis; False leaves the axis at full
            resolution (semicoarsening).
        boundary: one closure for every axis, or one per axis.

    Returns:
        The coarse shape, with unmasked axes unchanged.
    """
    shape = tuple(int(n) for n in shape)
    coarsen = tuple(bool(flag) for flag in coarsen)
    if len(coarsen) != len(shape):
        raise ValueError("coarsen must have one entry per axis of shape")
    boundaries = _per_axis(boundary, len(shape), "boundary")
    return tuple(
        coarse_axis_size(n, boundary=side) if flag else n
        for n, flag, side in zip(shape, coarsen, boundaries, strict=True)
    )


def restriction_matrix(
    n_fine: int,
    *,
    kind: str = "full_weighting",
    boundary: str = "periodic",
    dtype: Any = None,
) -> jax.Array:
    """One-dimensional restriction matrix, shape ``(n_coarse, n_fine)``.

    ``"full_weighting"`` is the standard smoothing restriction: it averages
    the residual over the fine points a coarse point represents, so
    high-frequency residual content is damped rather than aliased onto the
    coarse grid (Trottenberg et al., section 2.3). ``"injection"`` samples
    the coincident fine point only; it is cheaper, aliases, and is mainly
    useful with a full-weighting-like smoother already in place or for
    exact-adjoint diagnostics.

    Rows sum to one for every closure and kind, so a constant restricts to
    the same constant.

    Args:
        n_fine: fine-grid length of the axis.
        kind: ``"full_weighting"`` or ``"injection"``.
        boundary: ``"periodic"``, ``"dirichlet"`` or ``"reflective"``.
        dtype: result dtype; defaults to the platform default float.

    Returns:
        The dense one-dimensional restriction matrix.
    """
    _check_boundary(boundary)
    if kind not in ("full_weighting", "injection"):
        raise ValueError(
            f"unknown restriction {kind!r}; expected 'full_weighting' or 'injection'"
        )
    n_coarse = coarse_axis_size(n_fine, boundary=boundary)
    dtype = jnp.result_type(float) if dtype is None else dtype
    rows = jnp.arange(n_coarse)
    center = _center(rows, boundary)
    matrix = jnp.zeros((n_coarse, int(n_fine)), dtype)
    for offset, weight in _RESTRICTION_STENCILS[(kind, boundary)]:
        columns = _close(center + offset, int(n_fine), boundary)
        matrix = matrix.at[rows, columns].add(jnp.asarray(weight, dtype))
    return matrix


def prolongation_matrix(
    n_fine: int,
    *,
    kind: str = "linear",
    boundary: str = "periodic",
    dtype: Any = None,
) -> jax.Array:
    """One-dimensional prolongation matrix, shape ``(n_fine, n_coarse)``.

    ``"linear"`` is linear interpolation: coincident fine points copy their
    coarse value and intermediate ones average their two coarse neighbours
    (three-quarters/one-quarter for the cell-centered reflective closure).
    Together with ``"full_weighting"`` restriction it satisfies the
    variational relation ``R = 2^{-d} P^T`` over ``d`` coarsened axes.
    ``"injection"`` is the exact adjoint of injection restriction and
    leaves non-coincident fine points untouched.

    Args:
        n_fine: fine-grid length of the axis.
        kind: ``"linear"`` or ``"injection"``.
        boundary: ``"periodic"``, ``"dirichlet"`` or ``"reflective"``.
        dtype: result dtype; defaults to the platform default float.

    Returns:
        The dense one-dimensional prolongation matrix.
    """
    _check_boundary(boundary)
    if kind not in ("linear", "injection"):
        raise ValueError(f"unknown prolongation {kind!r}; expected 'linear' or 'injection'")
    n_coarse = coarse_axis_size(n_fine, boundary=boundary)
    dtype = jnp.result_type(float) if dtype is None else dtype
    columns = jnp.arange(n_coarse)
    center = _center(columns, boundary)
    matrix = jnp.zeros((int(n_fine), n_coarse), dtype)
    for offset, weight in _PROLONGATION_STENCILS[(kind, boundary)]:
        rows = _close(center + offset, int(n_fine), boundary)
        matrix = matrix.at[rows, columns].add(jnp.asarray(weight, dtype))
    return matrix


def grid_transfer(
    shape: Sequence[int],
    coarsen: Sequence[bool],
    *,
    boundary: str | Sequence[str] = "periodic",
    restriction: str | Sequence[str] = "full_weighting",
    prolongation: str | Sequence[str] = "linear",
    dtype: Any = None,
) -> tuple[MatVec, MatVec]:
    """Build the ``(restrict, prolong)`` pair for one coarsening step.

    The returned callables act on N-D arrays whose *leading* axes match
    ``shape`` (restriction) or the coarse shape (prolongation); extra
    trailing axes — stacked fields, species, right-hand sides — are carried
    through untouched. Axes with ``coarsen[axis] = False`` are never
    contracted, which is what makes this a semicoarsening transfer.

    Both callables are equinox modules, hence valid pytrees: they may be
    closed over, passed as arguments into ``jax.jit``, or stored in a
    hierarchy without leaving traceable code.

    Args:
        shape: fine grid shape.
        coarsen: one boolean per axis selecting the axes to coarsen.
        boundary: grid closure for every axis, or one per axis; see
            :func:`coarse_axis_size`.
        restriction: ``"full_weighting"`` or ``"injection"``, shared or per
            axis.
        prolongation: ``"linear"`` or ``"injection"``, shared or per axis.
        dtype: transfer-matrix dtype; defaults to the platform default float.

    Returns:
        ``(restrict, prolong)``: ``restrict`` maps fine arrays to coarse,
        ``prolong`` maps coarse arrays to fine. With the default stencils
        ``restrict == 2**-d * prolong^T`` over the ``d`` coarsened axes.
    """
    shape = tuple(int(n) for n in shape)
    coarsen = tuple(bool(flag) for flag in coarsen)
    if len(coarsen) != len(shape):
        raise ValueError("coarsen must have one entry per axis of shape")
    ndim = len(shape)
    boundaries = _per_axis(boundary, ndim, "boundary")
    restrictions = _per_axis(restriction, ndim, "restriction")
    prolongations = _per_axis(prolongation, ndim, "prolongation")
    coarse = coarse_grid_shape(shape, coarsen, boundary=boundaries)

    axes: list[int] = []
    restrict_matrices: list[jax.Array] = []
    prolong_matrices: list[jax.Array] = []
    for axis, flag in enumerate(coarsen):
        if not flag:
            continue
        axes.append(axis)
        restrict_matrices.append(
            restriction_matrix(
                shape[axis],
                kind=restrictions[axis],
                boundary=boundaries[axis],
                dtype=dtype,
            )
        )
        prolong_matrices.append(
            prolongation_matrix(
                shape[axis],
                kind=prolongations[axis],
                boundary=boundaries[axis],
                dtype=dtype,
            )
        )
    restrict = _AxisTransfer(tuple(restrict_matrices), tuple(axes), shape, coarse)
    prolong = _AxisTransfer(tuple(prolong_matrices), tuple(axes), coarse, shape)
    return restrict, prolong


def coarsening_plan(
    shape: Sequence[int],
    coarsen: Sequence[bool],
    *,
    levels: int,
    boundary: str | Sequence[str] = "periodic",
    min_size: int = 4,
) -> CoarseningPlan:
    """Plan a semicoarsening hierarchy, stopping axes as they run out.

    Starting from ``shape``, each level coarsens every axis that is both
    selected by ``coarsen`` and still :func:`coarsenable` — an axis of odd
    length under a periodic closure, or one already down at ``min_size``,
    simply stops while the others continue. The plan is truncated early
    (fewer than ``levels`` transitions) when no axis can be coarsened at
    all, so the caller always gets a consistent hierarchy rather than an
    error.

    Args:
        shape: finest grid shape.
        coarsen: one boolean per axis; axes marked False stay fine on every
            level.
        levels: maximum number of coarsening steps (the hierarchy then has
            ``levels + 1`` grids at most).
        boundary: grid closure for every axis, or one per axis.
        min_size: smallest coarse length an axis may reach.

    Returns:
        A :class:`CoarseningPlan` with ``levels + 1`` shapes at most and one
        axis mask per realized transition.
    """
    shape = tuple(int(n) for n in shape)
    coarsen = tuple(bool(flag) for flag in coarsen)
    if len(coarsen) != len(shape):
        raise ValueError("coarsen must have one entry per axis of shape")
    if levels < 0:
        raise ValueError("levels must be >= 0")
    boundaries = _per_axis(boundary, len(shape), "boundary")

    shapes = [shape]
    masks: list[tuple[bool, ...]] = []
    for _ in range(int(levels)):
        current = shapes[-1]
        mask = tuple(
            flag and coarsenable(n, boundary=side, min_size=min_size)
            for n, flag, side in zip(current, coarsen, boundaries, strict=True)
        )
        if not any(mask):
            break
        masks.append(mask)
        shapes.append(coarse_grid_shape(current, mask, boundary=boundaries))
    return CoarseningPlan(tuple(shapes), tuple(masks))
