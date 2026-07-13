"""Structured direct solvers: block-tridiagonal Schur elimination (block Thomas).

For a system with blocks ``L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k``
(k = 0..N-1), a Schur-complement sweep from the last block down to the first,

    Delta_{N-1} = D_{N-1}
    Delta_k     = D_k - U_k Delta_{k+1}^{-1} L_{k+1}
    sigma_k     = b_k - U_k Delta_{k+1}^{-1} sigma_{k+1}

followed by substitution upward from block 0:

    x_0 = Delta_0^{-1} sigma_0
    x_k = Delta_k^{-1} (sigma_k - L_k x_{k-1})

Each step costs one dense LU solve plus one matrix product — never an explicit
inverse. Cost is O(N m^3) time; the factor/solve split lets several right-hand
sides share one elimination. ``block_thomas_truncated`` additionally exploits a
common kinetic-equation structure: when the right-hand side vanishes for
k >= K and only the lowest K blocks of the solution are needed (e.g. velocity
moments touching only the first few spectral modes), the upward substitution
can stop at block K and the downward sweep needs no storage above it, so peak
memory is O(K m^2), *independent of N*.

Stability note: block LU without pivoting is guaranteed stable only for
block-diagonally-dominant systems (Demmel, Higham & Schreiber, Numer. Linear
Algebra Appl. 2, 173 (1995)); each block here is factored with partial
pivoting, and callers should monitor conditioning in weakly-dominant regimes
(see ``solvax.refine`` for iterative-refinement fallbacks).

References
----------
- G. H. Golub & C. F. Van Loan, *Matrix Computations*, 4th ed., section 4.5.
- F. J. Escoto, PhD thesis (2025), https://arxiv.org/abs/2510.27513 —
  block-tridiagonal elimination over Legendre modes for kinetic equations,
  including the truncated-storage observation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

from solvax.refine import iterative_refinement


class BlockTridiagFactors(NamedTuple):
    """Reusable elimination state from :func:`block_thomas_factor`.

    Attributes:
        delta_lu: LU factors of every Schur complement ``Delta_k``,
            shape ``(n_blocks, m, m)``.
        delta_piv: matching pivot indices, shape ``(n_blocks, m)``.
        lower: sub-diagonal blocks ``L_k`` (``lower[0]`` unused).
        upper: super-diagonal blocks ``U_k`` (``upper[-1]`` unused).
    """

    delta_lu: jax.Array
    delta_piv: jax.Array
    lower: jax.Array
    upper: jax.Array


def block_thomas_factor(
    lower: jax.Array, diag: jax.Array, upper: jax.Array, factor_dtype=None
) -> BlockTridiagFactors:
    """Run the downward Schur sweep once, for reuse across right-hand sides.

    Args:
        lower: sub-diagonal blocks ``L_k``, shape ``(n_blocks, m, m)``;
            ``lower[0]`` is ignored.
        diag: diagonal blocks ``D_k``, shape ``(n_blocks, m, m)``.
        upper: super-diagonal blocks ``U_k``, shape ``(n_blocks, m, m)``;
            ``upper[-1]`` is ignored.
        factor_dtype: if given, the Schur-complement LU factorizations and
            their triangular solves run in this lower precision, while the
            block products ``U_k Delta^{-1} L_k`` and the stored off-diagonal
            bands stay in the working precision of ``diag``. The returned
            ``delta_lu`` is then low precision; pair with
            :func:`block_thomas_solve` under :func:`iterative_refinement` (see
            :func:`mixed_precision_block_thomas`) to recover working-precision
            accuracy on fast low-precision hardware. Practically this is
            ``jnp.float32``: ``lu_factor`` dispatches to LAPACK/cuSOLVER
            ``getrf``, which has float32 and float64 kernels but no half
            precision, so bfloat16/float16 raise ``NotImplementedError``.

    Returns:
        Factors for :func:`block_thomas_solve`.
    """
    work = jnp.result_type(diag)
    fdt = work if factor_dtype is None else factor_dtype

    def down_step(carry, inputs):
        delta_next = carry
        d_k, u_k, l_next = inputs
        x = lu_solve(delta_next, l_next.astype(fdt)).astype(work)
        delta_k = lu_factor((d_k - u_k @ x).astype(fdt))
        return delta_k, delta_k

    last = lu_factor(diag[-1].astype(fdt))
    inputs = (diag[:-1], upper[:-1], lower[1:])  # steps k = n-2 .. 0
    _, (lus, pivs) = jax.lax.scan(down_step, last, inputs, reverse=True)

    delta_lu = jnp.concatenate([lus, last[0][None]], axis=0)
    delta_piv = jnp.concatenate([pivs, last[1][None]], axis=0)
    return BlockTridiagFactors(delta_lu, delta_piv, lower, upper)


def block_thomas_factor_fn(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    factor_dtype=None,
) -> BlockTridiagFactors:
    """Factor generated block rows once for reusable primal/transpose solves.

    Unlike :func:`block_thomas_factor`, this entry point never materializes the
    diagonal band. ``block_fn`` is evaluated exactly once per block index; the
    returned state stores Schur LU factors and the two off-diagonal bands needed
    by :func:`block_thomas_solve`.

    Args:
        block_fn: maps a traced int32 index to ``(lower, diagonal, upper)``
            blocks of identical square shape.
        n_blocks: static positive number of block rows.
        factor_dtype: optional lower precision for Schur LU factorizations, with
            the same contract as :func:`block_thomas_factor`.

    Returns:
        Reusable factors accepted by :func:`block_thomas_solve`, including its
        exact ``transpose=True`` path.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")

    l_last, d_last, u_last = block_fn(jnp.int32(n_blocks - 1))
    work = jnp.result_type(d_last)
    fdt = work if factor_dtype is None else factor_dtype
    last = lu_factor(d_last.astype(fdt))

    def down_step(carry, index):
        delta_next, l_next = carry
        lower, diagonal, upper = block_fn(index)
        solved_lower = lu_solve(delta_next, l_next.astype(fdt)).astype(work)
        delta = lu_factor((diagonal - upper @ solved_lower).astype(fdt))
        return (delta, lower), (delta[0], delta[1], lower, upper)

    _, (lus, pivs, lowers, uppers) = jax.lax.scan(
        down_step,
        (last, l_last),
        jnp.arange(n_blocks - 1, dtype=jnp.int32),
        reverse=True,
    )
    return BlockTridiagFactors(
        delta_lu=jnp.concatenate([lus, last[0][None]], axis=0),
        delta_piv=jnp.concatenate([pivs, last[1][None]], axis=0),
        lower=jnp.concatenate([lowers, l_last[None]], axis=0),
        upper=jnp.concatenate([uppers, u_last[None]], axis=0),
    )


def block_thomas_solve(
    factors: BlockTridiagFactors, rhs: jax.Array, transpose: bool = False
) -> jax.Array:
    """Solve using precomputed factors.

    With ``transpose=True`` this solves the *transposed* system
    ``A^T x = rhs`` reusing the same factors: for the same elimination order
    the Schur complements of ``A^T`` are exactly ``Delta_k^T`` (inductively,
    ``Delta'_{N-1} = D_{N-1}^T`` and ``Delta'_k = D_k^T - L_{k+1}^T
    Delta_{k+1}^{-T} U_k^T = Delta_k^T``), so the stored LU factors serve
    both directions via ``trans=1`` triangular solves. The off-diagonal
    blocks swap roles and transpose: the downward sweep uses ``L_{k+1}^T``
    where the forward solve used ``U_k``, and the upward substitution uses
    ``U_{k-1}^T`` where it used ``L_k``. One elimination thus covers the
    forward and the adjoint solve — exactly what implicit differentiation
    needs.

    Args:
        factors: output of :func:`block_thomas_factor`.
        rhs: ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``.
        transpose: if True, solve ``A^T x = rhs`` instead of ``A x = rhs``.

    Returns:
        Solution with the same shape as ``rhs``.
    """
    delta_lu, delta_piv, lower, upper = factors
    if transpose:
        down_blocks = jnp.swapaxes(lower[1:], -1, -2)
        up_blocks = jnp.swapaxes(upper[:-1], -1, -2)
        trans = 1
    else:
        down_blocks = upper[:-1]
        up_blocks = lower[1:]
        trans = 0

    # When the factors were built with a low ``factor_dtype`` (delta_lu below
    # the working precision), run each triangular solve in that low precision
    # and cast the result back up, so the band products keep working precision.
    # For all-working-precision factors both casts are no-ops.
    fdt = delta_lu.dtype
    work = jnp.result_type(rhs, lower)

    def tsolve(lu, piv, v):
        return lu_solve((lu, piv), v.astype(fdt), trans=trans).astype(work)

    # Static Python loops deliberately expose the linear recurrence to JAX.
    # In JAX 0.9, reverse-transposing a scan that emits every intermediate can
    # leak an internal ValAccum into the scan inputs. Unrolling restores the
    # advertised linear_transpose/VJP contract and is also faster after
    # compilation for representative 8--64-block systems; compilation grows
    # with the static block count, as expected for an unrolled recurrence.
    sigma = [None] * rhs.shape[0]
    sigma[-1] = rhs[-1]
    for k in range(rhs.shape[0] - 2, -1, -1):
        sigma[k] = rhs[k] - down_blocks[k] @ tsolve(
            delta_lu[k + 1], delta_piv[k + 1], sigma[k + 1]
        )

    solution = [tsolve(delta_lu[0], delta_piv[0], sigma[0])]
    for k in range(1, rhs.shape[0]):
        solution.append(
            tsolve(
                delta_lu[k],
                delta_piv[k],
                sigma[k] - up_blocks[k - 1] @ solution[-1],
            )
        )
    return jnp.stack(solution)


def block_thomas(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Solve a block-tridiagonal system by Schur-complement elimination.

    Convenience wrapper: :func:`block_thomas_factor` then
    :func:`block_thomas_solve`. For repeated solves with the same matrix,
    call the two stages directly and reuse the factors.

    Args:
        lower: ``L_k`` blocks, ``(n_blocks, m, m)``; ``lower[0]`` ignored.
        diag: ``D_k`` blocks, ``(n_blocks, m, m)``.
        upper: ``U_k`` blocks, ``(n_blocks, m, m)``; ``upper[-1]`` ignored.
        rhs: ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``.

    Returns:
        ``x`` with the same shape as ``rhs``.
    """
    return block_thomas_solve(block_thomas_factor(lower, diag, upper), rhs)


def block_tridiag_matvec(
    lower: jax.Array, diag: jax.Array, upper: jax.Array, x: jax.Array
) -> jax.Array:
    """Apply a block-tridiagonal operator without forming a dense matrix.

    ``(A x)_k = L_k x_{k-1} + D_k x_k + U_k x_{k+1}``, evaluated for every
    block at once. ``x`` and the result share the layout of the right-hand
    side, ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``. This independent
    operator action is also used by residual diagnostics.
    """
    sub = "kij,kj...->ki..."
    y = jnp.einsum(sub, diag, x)
    y = y.at[1:].add(jnp.einsum(sub, lower[1:], x[:-1]))
    y = y.at[:-1].add(jnp.einsum(sub, upper[:-1], x[1:]))
    return y


def block_tridiag_relative_residual(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    solution: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Return ``||b - A x||_2 / max(||b||_2, tiny)`` per right-hand side.

    A vector right-hand side returns a scalar. Multiple right-hand sides return
    one value per final column. Every block row is included, so this diagnostic
    cannot silently omit a truncated high-mode tail.
    """
    residual = rhs - block_tridiag_matvec(lower, diag, upper, solution)
    axes = tuple(range(residual.ndim - 1)) if residual.ndim > 2 else None
    residual_norm = jnp.linalg.norm(residual, axis=axes)
    rhs_norm = jnp.linalg.norm(rhs, axis=axes)
    tiny = jnp.finfo(residual.real.dtype).tiny
    return residual_norm / jnp.maximum(rhs_norm, tiny)


def _solve_matrix_and_rhs(delta, matrix_rhs, rhs):
    """Apply one LU solve to a matrix block and one or more RHS columns."""
    vector_rhs = rhs.ndim == 1
    rhs_columns = rhs[:, None] if vector_rhs else rhs
    width = matrix_rhs.shape[1]
    solved = lu_solve(delta, jnp.concatenate([matrix_rhs, rhs_columns], axis=1))
    solved_matrix = solved[:, :width]
    solved_rhs = solved[:, width:]
    return solved_matrix, solved_rhs[:, 0] if vector_rhs else solved_rhs


def mixed_precision_block_thomas(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
    *,
    factor_dtype=jnp.float32,
    refine_steps: int = 2,
) -> jax.Array:
    """Block-tridiagonal solve with a low-precision factorization + refinement.

    Factors once with :func:`block_thomas_factor` in ``factor_dtype`` — the
    dense Schur-complement LU factorizations, the dominant cost of the sweep,
    then run at (e.g.) float32 throughput, up to 32x that of float64 on
    consumer GPUs — and recovers working-precision accuracy with
    ``refine_steps`` sweeps of :func:`solvax.refine.iterative_refinement`. Each
    sweep forms the residual with the working-precision operator and corrects
    it with one low-precision :func:`block_thomas_solve`, so the result matches
    the full-precision solve to working-precision accuracy whenever
    ``kappa(A) * u_low < 1`` (Carson & Higham 2018; see :mod:`solvax.refine`).

    This composes the existing factor/solve with iterative refinement — no
    parallel scan — and stays jit/vmap/grad-transparent like the rest of the
    module.

    Args:
        lower: ``L_k`` blocks, ``(n_blocks, m, m)``; ``lower[0]`` ignored.
        diag: ``D_k`` blocks, ``(n_blocks, m, m)``.
        upper: ``U_k`` blocks, ``(n_blocks, m, m)``; ``upper[-1]`` ignored.
        rhs: ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``.
        factor_dtype: precision for the LU factorizations (default float32,
            the fast low precision supported by the LAPACK/cuSOLVER ``getrf``
            backend; half precision is not — see :func:`block_thomas_factor`).
        refine_steps: number of refinement sweeps (static int); ``0`` returns
            the bare low-precision solve.

    Returns:
        ``x`` with the same shape and precision as ``rhs``.
    """
    residual_dtype = jnp.result_type(rhs)
    factors = block_thomas_factor(lower, diag, upper, factor_dtype=factor_dtype)
    matvec = lambda x: block_tridiag_matvec(lower, diag, upper, x)  # noqa: E731
    approx_solve = lambda r: block_thomas_solve(factors, r)  # noqa: E731
    x, _ = iterative_refinement(
        matvec,
        rhs,
        approx_solve,
        iterations=refine_steps,
        residual_dtype=residual_dtype,
    )
    return x


def block_thomas_truncated(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs_low: jax.Array,
    keep_lowest: int,
) -> jax.Array:
    """Block-tridiagonal solve returning only the lowest ``keep_lowest`` blocks.

    Requires the right-hand side to vanish for ``k >= keep_lowest``
    (``rhs_low`` holds the nonzero head). The downward Schur sweep runs over
    all blocks but stores nothing above ``keep_lowest``; the upward
    substitution stops there. Peak memory O(keep_lowest * m^2), independent
    of ``n_blocks``.

    Args:
        lower, diag, upper: as in :func:`block_thomas`.
        rhs_low: nonzero head of the right-hand side, shape
            ``(keep_lowest, m)`` or ``(keep_lowest, m, n_rhs)``.
        keep_lowest: static number of solution blocks to compute
            (1 <= keep_lowest <= n_blocks; equality recovers the full solve).

    Returns:
        The lowest ``keep_lowest`` solution blocks, same layout as ``rhs_low``.
    """
    k = keep_lowest
    n = diag.shape[0]
    if not 1 <= k <= n:
        raise ValueError("need 1 <= keep_lowest <= n_blocks")
    if rhs_low.shape[0] != k:
        raise ValueError("rhs_low must have keep_lowest leading blocks")

    # Tail sweep (blocks n-1 .. k): carry only the running Schur complement.
    def tail_step(carry, inputs):
        delta_next = carry
        d_j, u_j, l_next = inputs
        delta_j = lu_factor(d_j - u_j @ lu_solve(delta_next, l_next))
        return delta_j, None

    last = lu_factor(diag[-1])
    tail_inputs = (diag[k:-1], upper[k:-1], lower[k + 1 :])
    delta_tail, _ = jax.lax.scan(tail_step, last, tail_inputs, reverse=True)

    # Head sweep (blocks k-1 .. 0): rhs is nonzero here; the sigma feeding in
    # from above is zero because the rhs vanishes for j >= k.
    def head_step(carry, inputs):
        delta_next_lu, delta_next_piv, sigma_next = carry
        d_j, u_j, l_next, b_j = inputs
        x, solved_sigma = _solve_matrix_and_rhs(
            (delta_next_lu, delta_next_piv), l_next, sigma_next
        )
        sigma_j = b_j - u_j @ solved_sigma
        lu_j, piv_j = lu_factor(d_j - u_j @ x)
        return (lu_j, piv_j, sigma_j), (lu_j, piv_j, sigma_j)

    upper_head = upper[:k]
    lower_next = lower[1 : k + 1]
    if k == n:
        # The head sweep now covers every block; its top step has no block
        # above, encoded as U_{n-1} = 0 (the padded lower partner is then
        # annihilated, and the initial carry acts as a dummy).
        upper_head = upper_head.at[-1].set(0.0)
        lower_next = jnp.concatenate([lower_next, lower[:1]], axis=0)

    carry0 = (delta_tail[0], delta_tail[1], jnp.zeros_like(rhs_low[0]))
    head_inputs = (diag[:k], upper_head, lower_next, rhs_low)
    _, (lus, pivs, sigmas) = jax.lax.scan(
        head_step, carry0, head_inputs, reverse=True
    )

    def up_step(x_prev, inputs):
        lu_j, piv_j, l_j, sigma_j = inputs
        x_j = lu_solve((lu_j, piv_j), sigma_j - l_j @ x_prev)
        return x_j, x_j

    x0 = lu_solve((lus[0], pivs[0]), sigmas[0])
    inputs_up = (lus[1:], pivs[1:], lower[1:k], sigmas[1:])
    _, xs = jax.lax.scan(up_step, x0, inputs_up)
    return jnp.concatenate([x0[None], xs], axis=0)


def _block_thomas_truncated_fn_state(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs_low: jax.Array,
    keep_lowest: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return a generated truncated solution and retained elimination state.

    Identical semantics to :func:`block_thomas_truncated`, but the blocks are
    produced per index by ``block_fn(k) -> (lower_k, diag_k, upper_k)`` inside
    the sweeps, so the full ``(n_blocks, m, m)`` band arrays are never
    materialized: peak memory is O(keep_lowest * m^2) plus a single block
    triple. This is the right entry point when blocks are assembled from
    compact physics coefficients (collision/streaming factors) and
    ``n_blocks`` is large.

    Args:
        block_fn: maps a (traced) int32 index ``k`` to the three ``(m, m)``
            blocks ``(L_k, D_k, U_k)``. ``L_0`` and ``U_{n_blocks-1}`` are
            ignored.
        n_blocks: static total number of blocks (>= 1).
        rhs_low: nonzero head of the right-hand side, shape
            ``(keep_lowest, m)`` or ``(keep_lowest, m, n_rhs)``; the right-hand
            side must vanish for ``k >= keep_lowest``.
        keep_lowest: static number of solution blocks to compute
            (1 <= keep_lowest <= n_blocks; equality recovers the full solve).

    Returns:
        The lowest ``keep_lowest`` solution blocks, same layout as ``rhs_low``.
    """
    k = keep_lowest
    n = n_blocks
    if not 1 <= k <= n:
        raise ValueError("need 1 <= keep_lowest <= n_blocks")
    if rhs_low.shape[0] != k:
        raise ValueError("rhs_low must have keep_lowest leading blocks")

    m = rhs_low.shape[1]
    dtype = rhs_low.dtype

    if k < n:
        # Tail sweep (blocks n-1 .. k): carry the running Schur complement
        # and the L block of the row just processed (needed one step below).
        l_last, d_last, _ = block_fn(jnp.int32(n - 1))
        carry0 = (lu_factor(d_last), l_last)

        def tail_step(carry, j):
            delta_next, l_next = carry
            l_j, d_j, u_j = block_fn(j)
            x = lu_solve(delta_next, l_next)
            delta_j = lu_factor(d_j - u_j @ x)
            return (delta_j, l_j), None

        (delta_head, l_head), _ = jax.lax.scan(
            tail_step, carry0, jnp.arange(k, n - 1, dtype=jnp.int32), reverse=True
        )
    else:
        # No tail: the head's top step has no block above; a dummy identity
        # carry works because that step's U is annihilated below.
        eye = jnp.eye(m, dtype=dtype)
        delta_head = lu_factor(eye)
        l_head = jnp.zeros((m, m), dtype=dtype)

    def head_step(carry, inputs):
        delta_next, l_next, sigma_next = carry
        j, b_j = inputs
        l_j, d_j, u_j = block_fn(j)
        if k == n:
            # Top block couples to nothing above.
            u_j = jnp.where(j == n - 1, jnp.zeros_like(u_j), u_j)
        x, solved_sigma = _solve_matrix_and_rhs(delta_next, l_next, sigma_next)
        sigma_j = b_j - u_j @ solved_sigma
        delta_j = lu_factor(d_j - u_j @ x)
        return (delta_j, l_j, sigma_j), (delta_j[0], delta_j[1], sigma_j, l_j)

    carry0 = (delta_head, l_head, jnp.zeros_like(rhs_low[0]))
    head_inputs = (jnp.arange(k, dtype=jnp.int32), rhs_low)
    _, (lus, pivs, sigmas, ls) = jax.lax.scan(
        head_step, carry0, head_inputs, reverse=True
    )

    def up_step(x_prev, inputs):
        lu_j, piv_j, l_j, sigma_j = inputs
        x_j = lu_solve((lu_j, piv_j), sigma_j - l_j @ x_prev)
        return x_j, x_j

    x0 = lu_solve((lus[0], pivs[0]), sigmas[0])
    inputs_up = (lus[1:], pivs[1:], ls[1:], sigmas[1:])
    _, xs = jax.lax.scan(up_step, x0, inputs_up)
    solution = jnp.concatenate([x0[None], xs], axis=0)
    return solution, lus, pivs, sigmas, ls


def block_thomas_truncated_fn(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs_low: jax.Array,
    keep_lowest: int,
) -> jax.Array:
    """Truncated block-tridiagonal solve with on-the-fly block assembly.

    The full band arrays are never materialized, and each block index is
    assembled once. See :func:`block_thomas_truncated_fn_with_residual` when an
    algebraic residual of the retained Schur system is also required.
    """
    solution, _, _, _, _ = _block_thomas_truncated_fn_state(
        block_fn, n_blocks, rhs_low, keep_lowest
    )
    return solution


def block_thomas_truncated_fn_with_residual(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs_low: jax.Array,
    keep_lowest: int,
    *,
    residual_rhs_index: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the generated truncated solution and its Schur-system RMS residual.

    The residual is evaluated from the retained pivoted LU factors as
    ``L @ (U @ x) - P @ rhs``. It therefore includes the eliminated high-mode
    tail without reconstructing another solution block or materializing the
    original diagonal band.
    """
    solution, lus, pivs, sigmas, lowers = _block_thomas_truncated_fn_state(
        block_fn, n_blocks, rhs_low, keep_lowest
    )
    effective_rhs = sigmas
    effective_rhs = effective_rhs.at[1:].add(
        -jnp.einsum("kij,kj...->ki...", lowers[1:], solution[:-1])
    )

    if residual_rhs_index is not None:
        if rhs_low.ndim != 3:
            raise ValueError("residual_rhs_index requires multiple right-hand sides")
        if not 0 <= residual_rhs_index < rhs_low.shape[-1]:
            raise ValueError("residual_rhs_index is out of range")
        residual_solution = solution[..., residual_rhs_index]
        residual_rhs = effective_rhs[..., residual_rhs_index]
    else:
        residual_solution = solution
        residual_rhs = effective_rhs

    def factor_residual(lu, piv, value, rhs):
        size = lu.shape[0]
        lower = jnp.tril(lu, -1) + jnp.eye(size, dtype=lu.dtype)
        upper = jnp.triu(lu)
        permutation = jax.lax.linalg.lu_pivots_to_permutation(piv, size)
        return lower @ (upper @ value) - rhs[permutation]

    residual = jax.vmap(factor_residual)(
        lus, pivs, residual_solution, residual_rhs
    )
    return solution, jnp.linalg.norm(residual) / jnp.sqrt(residual.size)
