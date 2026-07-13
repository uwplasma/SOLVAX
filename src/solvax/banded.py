"""Banded LU solvers: non-pivoted factorization and periodic (circulant-banded) systems.

Storage convention (identical to :func:`scipy.linalg.solve_banded`): a matrix
``A`` with ``lower_bw`` sub-diagonals and ``upper_bw`` super-diagonals is held
in a ``bands`` array of shape ``(n_diags, n)`` with
``n_diags = lower_bw + upper_bw + 1``, where row ``r`` holds the diagonal with
offset ``upper_bw - r``:

    bands[upper_bw + i - j, j] = A[i, j]   for max(0, j - upper_bw) <= i <= min(n - 1, j + lower_bw)

Entries of ``bands`` outside that range are ignored. The LU factorization is
Doolittle elimination *without pivoting*, carried out column by column in
banded storage with a ``jax.lax.scan`` (static shapes, so everything is
jit/vmap/grad-transparent — XLA handles row pivoting poorly, and avoiding it
is the point of this module). Two safeguards substitute for pivoting:

- *row equilibration*: each row is pre-scaled by ``1 / max|row|``;
- *static pivoting*: any pivot with ``|pivot| < floor`` is clamped to
  ``sign(pivot) * floor``, and the number of clamps is recorded in the factors
  so callers can detect near-singularity and fall back to iterative
  refinement (see ``solvax.refine``).

LU without pivoting is guaranteed backward-stable only for diagonally
dominant (or block-dominant) systems (Demmel, Higham & Schreiber, Numer.
Linear Algebra Appl. 2, 173 (1995)); with equilibration and static pivoting it
is a robust practical choice for the advection-dominated periodic 1-D
operators these routines target, but callers should monitor the clamp counter
in weakly-dominant regimes.

Periodic (circulant-banded) systems ``A = B + U V^T`` — a banded core ``B``
plus wrap-around corner blocks expressed as a low-rank update — are solved
with the Sherman-Morrison-Woodbury capacitance-matrix identity

    (B + U V^T)^{-1} = B^{-1} - B^{-1} U (I + V^T B^{-1} U)^{-1} V^T B^{-1},

where the small dense capacitance matrix ``I + V^T B^{-1} U`` is LU-factored
once (with partial pivoting — it is tiny) and ``B^{-1} U`` is precomputed with
the banded factorization, so each periodic solve costs one banded solve plus
O(bw) dense work.

References
----------
- G. H. Golub & C. F. Van Loan, *Matrix Computations*, 4th ed., section 4.3
  (banded Gaussian elimination) and section 2.1.4 (Sherman-Morrison-Woodbury).
- J. W. Demmel, N. J. Higham & R. S. Schreiber, "Stability of block LU
  factorization", Numer. Linear Algebra Appl. 2, 173 (1995) — stability
  caveats for elimination without pivoting.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve


def _band_indices(n_diags: int, n: int, upper_bw: int):
    """Row-index grid and validity mask for the banded layout.

    Entry ``bands[r, j]`` represents ``A[i, j]`` with ``i = j + r - upper_bw``;
    it is valid iff ``0 <= i < n``.
    """
    r = jnp.arange(n_diags)[:, None]
    j = jnp.arange(n)[None, :]
    i = j + r - upper_bw
    return i, (i >= 0) & (i < n)


def _shift_rows(v: jax.Array, d: int) -> jax.Array:
    """Zero-padded shift along axis 0: ``out[i] = v[i + d]`` where defined."""
    if d == 0:
        return v
    if d > 0:
        return jnp.concatenate([v[d:], jnp.zeros_like(v[:d])], axis=0)
    return jnp.concatenate([jnp.zeros_like(v[d:]), v[:d]], axis=0)


def banded_matvec(
    bands: jax.Array, lower_bw: int, upper_bw: int, x: jax.Array
) -> jax.Array:
    """Matrix-vector product ``A @ x`` with ``A`` in banded storage.

    Vectorized over the matrix dimension (the only Python loop is over the
    ``n_diags`` diagonals, which is static).

    Args:
        bands: banded storage of ``A``, shape ``(lower_bw + upper_bw + 1, n)``
            (see module docstring); out-of-range entries are ignored.
        lower_bw: number of sub-diagonals.
        upper_bw: number of super-diagonals.
        x: vector ``(n,)`` or block of vectors ``(n, k)``.

    Returns:
        ``A @ x`` with the same shape as ``x``.
    """
    bands = jnp.asarray(bands)
    x = jnp.asarray(x)
    n_diags, n = bands.shape
    if n_diags != lower_bw + upper_bw + 1:
        raise ValueError("bands must have lower_bw + upper_bw + 1 rows")
    _, mask = _band_indices(n_diags, n, upper_bw)
    ab = jnp.where(mask, bands, 0.0)

    y = jnp.zeros(x.shape, dtype=jnp.result_type(ab.dtype, x.dtype))
    for r in range(n_diags):
        w = ab[r]
        c = w[:, None] * x if x.ndim > 1 else w * x
        # bands[r, j] multiplies x[j] and contributes to y[i], i = j - (upper_bw - r).
        y = y + _shift_rows(c, upper_bw - r)
    return y


class BandedLUFactors(NamedTuple):
    """Packed non-pivoted banded LU from :func:`lu_factor_banded`.

    The bandwidths are recovered from the array shapes, so the factors stay
    jit/vmap-transparent (no static metadata crosses transform boundaries).

    Attributes:
        l_bands: unit-lower-triangular multipliers, shape ``(lower_bw, n)``;
            ``l_bands[t - 1, k]`` holds ``L[k + t, k]``.
        u_bands: upper factor in banded storage, shape ``(upper_bw + 1, n)``;
            ``u_bands[upper_bw + i - j, j]`` holds ``U[i, j]``.
        row_scale: equilibration scales applied to the rows of ``A`` (all ones
            when ``equilibrate=False``), shape ``(n,)``.
        n_clamped: int32 count of pivots clamped by static pivoting; a
            nonzero value signals near-singularity of the (equilibrated) core.
    """

    l_bands: jax.Array
    u_bands: jax.Array
    row_scale: jax.Array
    n_clamped: jax.Array


def lu_factor_banded(
    bands: jax.Array,
    lower_bw: int,
    upper_bw: int,
    *,
    equilibrate: bool = True,
    static_pivot_floor: float | None = None,
) -> BandedLUFactors:
    """Non-pivoted (Doolittle) LU of a banded matrix, in banded storage.

    Columns are eliminated left to right with a ``jax.lax.scan``; the carry
    holds the multipliers of the previous ``upper_bw`` steps, so shapes are
    static and the factorization is jit/vmap/grad-transparent. Row
    equilibration and static pivoting (see module docstring) substitute for
    the row pivoting that XLA handles poorly.

    Args:
        bands: banded storage of ``A``, shape ``(lower_bw + upper_bw + 1, n)``;
            out-of-range entries are ignored.
        lower_bw: number of sub-diagonals.
        upper_bw: number of super-diagonals.
        equilibrate: scale each row by ``1 / max|row|`` before factoring
            (the scales are stored and applied to the right-hand side by
            :func:`lu_solve_banded`).
        static_pivot_floor: clamp threshold for pivots; ``None`` (default)
            uses ``sqrt(machine eps) * max|bands|`` of the (equilibrated)
            matrix.

    Returns:
        Factors for :func:`lu_solve_banded`, including the clamp counter.
    """
    bands = jnp.asarray(bands)
    kl, ku = int(lower_bw), int(upper_bw)
    n_diags, n = bands.shape
    if n_diags != kl + ku + 1:
        raise ValueError("bands must have lower_bw + upper_bw + 1 rows")
    i_idx, mask = _band_indices(n_diags, n, ku)
    ab = jnp.where(mask, bands, 0.0)
    dtype = ab.dtype

    if equilibrate:
        # Row i of A is scattered across the diagonals: |A[i, j]| = |ab[r, i + ku - r]|.
        rows = jnp.stack([_shift_rows(jnp.abs(ab[r]), ku - r) for r in range(n_diags)])
        row_max = rows.max(axis=0)
        safe = jnp.where(row_max > 0, row_max, 1.0)
        row_scale = jnp.where(row_max > 0, 1.0 / safe, 1.0)
        ab = jnp.where(mask, ab * row_scale[jnp.clip(i_idx, 0, n - 1)], 0.0)
    else:
        row_scale = jnp.ones(n, dtype)

    if static_pivot_floor is None:
        floor = jnp.sqrt(jnp.finfo(dtype).eps) * jnp.max(jnp.abs(ab))
    else:
        floor = jnp.asarray(static_pivot_floor, dtype)

    def step(carry, c):
        # c = ab[:, j]: c[r] holds A[j + r - ku, j]. m_hist[:, ku - t] holds the
        # multipliers of elimination step k = j - t (zeros for k < 0).
        m_hist, n_clamped = carry
        for t in range(ku, 0, -1):  # oldest step first
            u_kj = c[ku - t]  # U[j - t, j], final after older updates
            c = c.at[ku - t + 1 : ku - t + 1 + kl].add(-m_hist[:, ku - t] * u_kj)
        p = c[ku]
        small = jnp.abs(p) < floor
        p = jnp.where(small, jnp.where(p >= 0, floor, -floor), p)
        mults = c[ku + 1 :] / p
        c = c.at[ku].set(p).at[ku + 1 :].set(mults)
        if ku > 0:
            m_hist = jnp.concatenate([m_hist[:, 1:], mults[:, None]], axis=1)
        return (m_hist, n_clamped + small.astype(jnp.int32)), c

    carry0 = (jnp.zeros((kl, ku), dtype), jnp.int32(0))
    (_, n_clamped), cols = jax.lax.scan(step, carry0, ab.T)
    packed = cols.T  # LAPACK-style packed LU in the input layout
    return BandedLUFactors(packed[ku + 1 :], packed[: ku + 1], row_scale, n_clamped)


def lu_solve_banded(factors: BandedLUFactors, b: jax.Array) -> jax.Array:
    """Solve ``A x = b`` using precomputed banded LU factors.

    Forward substitution with the unit-lower multipliers, then backward
    substitution with the banded upper factor, each as a ``jax.lax.scan``
    carrying a bandwidth-sized window of the solution.

    Args:
        factors: output of :func:`lu_factor_banded`.
        b: right-hand side, shape ``(n,)`` or ``(n, k)``.

    Returns:
        Solution with the same shape as ``b``.
    """
    l_bands, u_bands, row_scale, _ = factors
    kl = l_bands.shape[0]
    ku = u_bands.shape[0] - 1
    b = jnp.asarray(b)
    vector = b.ndim == 1
    bb = (b[:, None] if vector else b).astype(jnp.result_type(b, l_bands, u_bands))
    bb = bb * row_scale[:, None]

    # Forward: y[i] = b[i] - sum_t L[i, i - t] y[i - t], L[i, i - t] = l_bands[t - 1, i - t].
    if kl > 0:
        lcoef = jnp.stack(
            [_shift_rows(l_bands[t - 1], -t) for t in range(1, kl + 1)], axis=1
        )

        def fwd_step(y_win, inputs):  # y_win[s] = y[i - kl + s]
            coeffs, b_i = inputs
            y_i = b_i - coeffs[::-1] @ y_win
            return jnp.concatenate([y_win[1:], y_i[None]], axis=0), y_i

        y0 = jnp.zeros((kl, bb.shape[1]), bb.dtype)
        _, y = jax.lax.scan(fwd_step, y0, (lcoef, bb))
    else:
        y = bb

    # Backward: x[i] = (y[i] - sum_t U[i, i + t] x[i + t]) / U[i, i].
    diag = u_bands[ku]
    if ku > 0:
        ucoef = jnp.stack(
            [_shift_rows(u_bands[ku - t], t) for t in range(1, ku + 1)], axis=1
        )

        def bwd_step(x_win, inputs):  # x_win[s] = x[i + 1 + s]
            coeffs, y_i, d_i = inputs
            x_i = (y_i - coeffs @ x_win) / d_i
            return jnp.concatenate([x_i[None], x_win[:-1]], axis=0), x_i

        x0 = jnp.zeros((ku, y.shape[1]), y.dtype)
        _, x = jax.lax.scan(bwd_step, x0, (ucoef, y, diag), reverse=True)
    else:
        x = y / diag[:, None]

    return x[:, 0] if vector else x


class PeriodicBandedLUFactors(NamedTuple):
    """Woodbury factors for a periodic banded system, from
    :func:`lu_factor_banded_periodic`.

    Attributes:
        core: banded LU of the (non-periodic) banded core ``B``.
        z_ul: ``B^{-1} U`` columns generated by the top-right corner block,
            shape ``(n, bw_ul)``.
        z_lr: ``B^{-1} U`` columns generated by the bottom-left corner block,
            shape ``(n, bw_lr)``.
        cap_lu: dense LU of the capacitance matrix ``I + V^T B^{-1} U``,
            shape ``(bw_ul + bw_lr, bw_ul + bw_lr)``.
        cap_piv: matching pivot indices.
    """

    core: BandedLUFactors
    z_ul: jax.Array
    z_lr: jax.Array
    cap_lu: jax.Array
    cap_piv: jax.Array


def lu_factor_banded_periodic(
    bands: jax.Array,
    lower_bw: int,
    upper_bw: int,
    corner_ul: jax.Array,
    corner_lr: jax.Array,
    *,
    equilibrate: bool = True,
    static_pivot_floor: float | None = None,
) -> PeriodicBandedLUFactors:
    """Factor a periodic (circulant-banded) matrix via the capacitance method.

    The matrix is ``A = B + U V^T``: a banded core ``B`` (given in ``bands``)
    plus the periodic wrap-around corners as a rank-``(bw_ul + bw_lr)``
    update, where ``U`` carries the corner blocks and ``V`` selects the
    coupled columns. ``B`` is factored with the non-pivoted banded LU,
    ``Z = B^{-1} U`` is precomputed, and the small dense capacitance matrix
    ``I + V^T Z`` is LU-factored so :func:`lu_solve_banded_periodic` can apply
    the Woodbury identity (module docstring) at the cost of one banded solve.

    Args:
        bands: banded storage of the core ``B``, shape
            ``(lower_bw + upper_bw + 1, n)``.
        lower_bw: number of sub-diagonals of ``B``.
        upper_bw: number of super-diagonals of ``B``.
        corner_ul: top-right corner block ``A[:bw, n - bw:]`` coupling the
            first rows to the last columns, shape ``(bw, bw)``.
        corner_lr: bottom-left corner block ``A[n - bw:, :bw]`` coupling the
            last rows to the first columns, shape ``(bw, bw)``.
        equilibrate: passed to :func:`lu_factor_banded` for the core.
        static_pivot_floor: passed to :func:`lu_factor_banded` for the core.

    Returns:
        Factors for :func:`lu_solve_banded_periodic`.
    """
    bands = jnp.asarray(bands)
    corner_ul = jnp.asarray(corner_ul)
    corner_lr = jnp.asarray(corner_lr)
    if corner_ul.ndim != 2 or corner_ul.shape[0] != corner_ul.shape[1]:
        raise ValueError("corner_ul must be a square matrix")
    if corner_lr.ndim != 2 or corner_lr.shape[0] != corner_lr.shape[1]:
        raise ValueError("corner_lr must be a square matrix")

    core = lu_factor_banded(
        bands,
        lower_bw,
        upper_bw,
        equilibrate=equilibrate,
        static_pivot_floor=static_pivot_floor,
    )
    n = bands.shape[1]
    r_ul, r_lr = corner_ul.shape[0], corner_lr.shape[0]
    dtype = jnp.result_type(bands.dtype, corner_ul.dtype, corner_lr.dtype)

    # U stacks the corner blocks; V^T picks the last r_ul then first r_lr rows.
    u_cols = jnp.zeros((n, r_ul + r_lr), dtype)
    u_cols = u_cols.at[:r_ul, :r_ul].set(corner_ul)
    u_cols = u_cols.at[n - r_lr :, r_ul:].set(corner_lr)
    z = lu_solve_banded(core, u_cols)
    vt_z = jnp.concatenate([z[n - r_ul :], z[:r_lr]], axis=0)
    cap_lu, cap_piv = lu_factor(jnp.eye(r_ul + r_lr, dtype=dtype) + vt_z)
    return PeriodicBandedLUFactors(core, z[:, :r_ul], z[:, r_ul:], cap_lu, cap_piv)


def lu_solve_banded_periodic(
    factors: PeriodicBandedLUFactors, b: jax.Array
) -> jax.Array:
    """Solve a periodic banded system using precomputed Woodbury factors.

    Applies ``x = B^{-1} b - Z (I + V^T Z)^{-1} V^T B^{-1} b`` with
    ``Z = B^{-1} U`` from :func:`lu_factor_banded_periodic`.

    Args:
        factors: output of :func:`lu_factor_banded_periodic`.
        b: right-hand side, shape ``(n,)`` or ``(n, k)``.

    Returns:
        Solution with the same shape as ``b``.
    """
    core, z_ul, z_lr, cap_lu, cap_piv = factors
    r_ul = z_ul.shape[1]
    n = z_ul.shape[0]
    b = jnp.asarray(b)
    vector = b.ndim == 1
    bb = b[:, None] if vector else b

    y = lu_solve_banded(core, bb)
    vt_y = jnp.concatenate([y[n - r_ul :], y[: z_lr.shape[1]]], axis=0)
    t = lu_solve((cap_lu, cap_piv), vt_y)
    x = y - z_ul @ t[:r_ul] - z_lr @ t[r_ul:]
    return x[:, 0] if vector else x
