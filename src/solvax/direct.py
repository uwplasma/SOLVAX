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
memory is O(K m^2), *independent of N*. This is a **selected-head** solve, not
a principal-submatrix approximation: every one of the N rows is still visited
and eliminated into the running Schur complement, so the returned blocks are
exact blocks of the full solution and the arithmetic remains O(N m^3). Only
storage and the final substitution are restricted to the head.

Differentiating that solve with respect to generated block parameters uses the
exact-window adjoint: with W = min(K+w, N) retained rows, the source cotangent
and every retained row cotangent are exact at any window, and the only
approximation is the omission of rows j >= W (see
``_block_thomas_selected_fn_state`` and ``_retained_row_cotangents``).

When the whole solution is needed and the same matrix is solved many times, the
factors themselves are what has to fit in memory. Reusable factors nominally
hold three (N, m, m) arrays: the Schur LU factors and the two off-diagonal
bands the substitution reads. Only the first cannot be recovered.
``block_thomas_factor_fn(..., store_offdiagonals=False)`` therefore keeps the
Schur LU alone and has ``block_thomas_solve`` regenerate L_k and U_k from the
same block generator, a third of the state (a sixth against float64 bands when
the LU is float32) for two extra generator evaluations per row per solve. The
elimination still runs exactly once, so this stays a factor/solve split and not
a one-shot solver like ``block_thomas_checkpointed_fn``.

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

import dataclasses
import operator
import warnings
from collections.abc import Callable
from functools import partial
from math import isqrt
from typing import NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
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


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class GeneratedBlockTridiagFactors:
    """Reusable elimination state that keeps *only* the Schur LU factors.

    Returned by :func:`block_thomas_factor_fn` with ``store_offdiagonals=False``
    and accepted by :func:`block_thomas_solve` exactly like
    :class:`BlockTridiagFactors`. The two off-diagonal bands are not stored;
    the substitution sweeps call ``block_fn`` again to rebuild ``L_k`` and
    ``U_k`` one block at a time.

    That is a factor-three cut in retained state --- three ``(n_blocks, m, m)``
    arrays become one --- and a factor six when the LU factors are float32 and
    the bands would have been float64. For a family of chains preconditioned
    together it is often the difference between fitting on a device and not;
    see :func:`block_thomas_factor_fn` for the cost that buys it.

    Attributes:
        delta_lu: LU factors of every Schur complement ``Delta_k``,
            shape ``(n_blocks, m, m)``.
        delta_piv: matching pivot indices, shape ``(n_blocks, m)``.
        block_fn: the row generator the factors were built from. It is carried
            here rather than passed at solve time so that factors and generator
            cannot be paired up wrongly, which would be silently wrong rather
            than an error.
        n_blocks: number of block rows.
        work_dtype: working precision of the generated blocks, recorded at
            factorization so that a low ``factor_dtype`` does not change the
            precision the substitution accumulates in.

    As a pytree, ``delta_lu`` and ``delta_piv`` are the children and the other
    three fields are static, so the state can cross a ``jit`` boundary. A
    ``block_fn`` that closes over arrays makes them compile-time constants of
    the consuming trace, and a freshly created closure is a fresh static value:
    build the generator once if the factors are passed as an argument.
    """

    delta_lu: jax.Array
    delta_piv: jax.Array
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]
    n_blocks: int
    work_dtype: object

    def tree_flatten(self):
        return (
            (self.delta_lu, self.delta_piv),
            (self.block_fn, self.n_blocks, self.work_dtype),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


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
            precision, so bfloat16/float16 fail at tracing or execution.

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
    *,
    store_offdiagonals: bool = True,
) -> BlockTridiagFactors | GeneratedBlockTridiagFactors:
    """Factor generated block rows once for reusable primal/transpose solves.

    Unlike :func:`block_thomas_factor`, this entry point never materializes the
    diagonal band. ``block_fn`` is evaluated exactly once per block index.

    With ``store_offdiagonals=False`` the returned state keeps *only* the Schur
    LU factors and pivots; :func:`block_thomas_solve` then regenerates ``L_k``
    and ``U_k`` from the same ``block_fn`` during its two substitution sweeps.
    Retained state falls from three ``(n_blocks, m, m)`` arrays to one --- a
    third, or a sixth against float64 bands when ``factor_dtype`` is
    ``jnp.float32`` --- which is what makes a chain family fit in device memory
    when the full band state does not.

    The cost is generator evaluations, not arithmetic: ``block_fn`` is called
    once per index to factor, and then **twice per index per solve**, once in
    the downward sweep and once in the upward one. Triangular-solve and
    matrix-product counts are unchanged. The factors stay reusable --- this is
    not :func:`block_thomas_checkpointed_fn`, which re-eliminates on every
    application; here the elimination happens exactly once, so a solve costs
    two block regenerations per row and no factorization at all.

    Args:
        block_fn: maps a traced int32 index to ``(lower, diagonal, upper)``
            blocks of identical square shape. It must be a pure function of its
            index: the substitution assumes a regenerated block equals the one
            the factorization saw.
        n_blocks: static positive number of block rows.
        factor_dtype: optional lower precision for Schur LU factorizations, with
            the same contract as :func:`block_thomas_factor`. It composes with
            ``store_offdiagonals=False``.
        store_offdiagonals: keep the two off-diagonal bands (default) and return
            :class:`BlockTridiagFactors`, or drop them and return
            :class:`GeneratedBlockTridiagFactors`.

    Returns:
        Reusable factors accepted by :func:`block_thomas_solve`, including its
        exact ``transpose=True`` path, under either storage policy.
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
        # With the bands dropped nothing of size ``(n_blocks, m, m)`` beyond the
        # Schur factors is ever stacked; only the running ``L`` sits in the
        # carry, exactly as it must for the elimination itself.
        emitted = (delta[0], delta[1])
        if store_offdiagonals:
            emitted += (lower, upper)
        return (delta, lower), emitted

    _, emitted = jax.lax.scan(
        down_step,
        (last, l_last),
        jnp.arange(n_blocks - 1, dtype=jnp.int32),
        reverse=True,
    )
    delta_lu = jnp.concatenate([emitted[0], last[0][None]], axis=0)
    delta_piv = jnp.concatenate([emitted[1], last[1][None]], axis=0)
    if not store_offdiagonals:
        return GeneratedBlockTridiagFactors(
            delta_lu=delta_lu,
            delta_piv=delta_piv,
            block_fn=block_fn,
            n_blocks=n_blocks,
            work_dtype=work,
        )
    return BlockTridiagFactors(
        delta_lu=delta_lu,
        delta_piv=delta_piv,
        lower=jnp.concatenate([emitted[2], l_last[None]], axis=0),
        upper=jnp.concatenate([emitted[3], u_last[None]], axis=0),
    )


def _block_thomas_checkpointed_impl(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs: jax.Array,
    checkpoint_size: int,
    *,
    transpose: bool,
) -> jax.Array:
    """Generated one-shot solve retaining one segment plus radial checkpoints."""
    lower_last, diag_last, upper_last = block_fn(jnp.int32(n_blocks - 1))
    last = lu_factor(diag_last)
    sigma_last = rhs[-1]
    n_segments = (n_blocks + checkpoint_size - 1) // checkpoint_size

    def eliminate(carry, index):
        delta_next, lower_next, sigma_next = carry
        lower, diagonal, upper = block_fn(index)
        solved_lower = lu_solve(delta_next, lower_next)
        delta = lu_factor(diagonal - upper @ solved_lower)
        down = jnp.swapaxes(lower_next, -1, -2) if transpose else upper
        solved_sigma = lu_solve(delta_next, sigma_next, trans=1 if transpose else 0)
        sigma = rhs[index] - down @ solved_sigma
        return (delta, lower, sigma), upper

    checkpoint_lu = jnp.zeros((n_segments,) + last[0].shape, last[0].dtype)
    checkpoint_piv = jnp.zeros((n_segments,) + last[1].shape, last[1].dtype)
    checkpoint_sigma = jnp.zeros((n_segments,) + sigma_last.shape, sigma_last.dtype)
    checkpoint_lu = checkpoint_lu.at[-1].set(last[0])
    checkpoint_piv = checkpoint_piv.at[-1].set(last[1])
    checkpoint_sigma = checkpoint_sigma.at[-1].set(sigma_last)

    def downward(step, carry):
        delta, lower_next, sigma, lus, pivs, sigmas = carry
        index = jnp.int32(n_blocks - 2 - step)
        (delta, lower_next, sigma), _ = eliminate((delta, lower_next, sigma), index)

        def save(checkpoints):
            lu_values, piv_values, sigma_values = checkpoints
            segment = index // checkpoint_size
            return (
                lu_values.at[segment].set(delta[0]),
                piv_values.at[segment].set(delta[1]),
                sigma_values.at[segment].set(sigma),
            )

        lus, pivs, sigmas = jax.lax.cond(
            (index + 1) % checkpoint_size == 0,
            save,
            lambda values: values,
            (lus, pivs, sigmas),
        )
        return delta, lower_next, sigma, lus, pivs, sigmas

    *_, checkpoint_lu, checkpoint_piv, checkpoint_sigma = jax.lax.fori_loop(
        0,
        n_blocks - 1,
        downward,
        (
            last,
            lower_last,
            sigma_last,
            checkpoint_lu,
            checkpoint_piv,
            checkpoint_sigma,
        ),
    )

    def solve_segment(segment, carry):
        x_previous, upper_previous, solution = carry
        start = segment * checkpoint_size
        end = jnp.minimum(start + checkpoint_size - 1, n_blocks - 1)
        length = end - start + 1
        lower_end, _, upper_end = block_fn(end)
        delta_end = (checkpoint_lu[segment], checkpoint_piv[segment])
        sigma_end = checkpoint_sigma[segment]
        lus = jnp.zeros((checkpoint_size,) + last[0].shape, last[0].dtype)
        pivs = jnp.zeros((checkpoint_size,) + last[1].shape, last[1].dtype)
        sigmas = jnp.zeros((checkpoint_size,) + sigma_last.shape, sigma_last.dtype)
        lowers = jnp.zeros((checkpoint_size,) + lower_last.shape, lower_last.dtype)
        uppers = jnp.zeros((checkpoint_size,) + upper_last.shape, upper_last.dtype)
        final = length - 1
        lus = lus.at[final].set(delta_end[0])
        pivs = pivs.at[final].set(delta_end[1])
        sigmas = sigmas.at[final].set(sigma_end)
        lowers = lowers.at[final].set(lower_end)
        uppers = uppers.at[final].set(upper_end)

        def recompute(step, state):
            delta, lower_next, sigma, lu_values, piv_values, sigma_values, lo, up = state

            def active(values):
                delta_next, lower_next, sigma_next, lu_values, piv_values, sigma_values, lo, up = (
                    values
                )
                index = end - 1 - step
                (delta, lower, sigma), upper = eliminate(
                    (delta_next, lower_next, sigma_next), index
                )
                local = index - start
                return (
                    delta,
                    lower,
                    sigma,
                    lu_values.at[local].set(delta[0]),
                    piv_values.at[local].set(delta[1]),
                    sigma_values.at[local].set(sigma),
                    lo.at[local].set(lower),
                    up.at[local].set(upper),
                )

            return jax.lax.cond(step < length - 1, active, lambda values: values, state)

        _, _, _, lus, pivs, sigmas, lowers, uppers = jax.lax.fori_loop(
            0,
            checkpoint_size - 1,
            recompute,
            (delta_end, lower_end, sigma_end, lus, pivs, sigmas, lowers, uppers),
        )

        def substitute(local, state):
            x_prev, upper_prev, values = state

            def active(carry_up):
                x_prev, upper_prev, values = carry_up
                index = start + local
                sigma = sigmas[local]
                lower = lowers[local]
                upper = uppers[local]
                delta = (lus[local], pivs[local])
                up = jnp.swapaxes(upper_prev, -1, -2) if transpose else lower
                correction = jax.lax.cond(
                    index == 0,
                    lambda _: jnp.zeros_like(sigma),
                    lambda _: up @ x_prev,
                    operand=None,
                )
                x = lu_solve(delta, sigma - correction, trans=1 if transpose else 0)
                return x, upper, values.at[index].set(x)

            return jax.lax.cond(local < length, active, lambda value: value, state)

        return jax.lax.fori_loop(
            0,
            checkpoint_size,
            substitute,
            (x_previous, upper_previous, solution),
        )

    return jax.lax.fori_loop(
        0,
        n_segments,
        solve_segment,
        (jnp.zeros_like(rhs[0]), jnp.zeros_like(upper_last), jnp.zeros_like(rhs)),
    )[2]


def block_thomas_checkpointed_fn(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs: jax.Array,
    checkpoint_size: int | None = None,
    *,
    transpose: bool = False,
) -> jax.Array:
    """Solve generated full block rows with exact radial checkpointing.

    The downward elimination retains one Schur/RHS checkpoint per
    ``checkpoint_size`` rows. Upward substitution recomputes and discards one
    segment at a time. Peak dense-factor storage is therefore
    ``O((n_blocks / checkpoint_size + checkpoint_size) * m**2)`` instead of
    ``O(n_blocks * m**2)``; ``checkpoint_size ~= sqrt(n_blocks)`` minimizes
    it. Every block row is generated twice and no band is materialized.

    :func:`jax.lax.custom_linear_solve` supplies implicit JVP/VJP rules: the
    reverse pass uses the same checkpointed algorithm on the transposed block
    system instead of differentiating through and taping the elimination.

    Args:
        block_fn: traced row generator returning ``(lower, diagonal, upper)``.
        n_blocks: static positive number of block rows.
        rhs: ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``.
        checkpoint_size: static positive segment width. The default
            ``ceil(sqrt(n_blocks))`` minimizes the factor-storage bound.
        transpose: solve ``A.T x = rhs`` when true.

    Returns:
        Solution with the same shape as ``rhs``.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    checkpoint_size = isqrt(n_blocks - 1) + 1 if checkpoint_size is None else checkpoint_size
    if checkpoint_size < 1:
        raise ValueError("checkpoint_size must be positive")
    checkpoint_size = min(checkpoint_size, n_blocks)
    if int(rhs.shape[0]) != n_blocks:
        raise ValueError("rhs leading dimension must equal n_blocks")

    def matvec(x, *, transposed):
        values = [jnp.zeros_like(x[0]) for _ in range(n_blocks)]
        for index in range(n_blocks):
            lower, diagonal, upper = block_fn(jnp.int32(index))
            if transposed:
                values[index] += jnp.swapaxes(diagonal, -1, -2) @ x[index]
                if index > 0:
                    values[index - 1] += jnp.swapaxes(lower, -1, -2) @ x[index]
                if index < n_blocks - 1:
                    values[index + 1] += jnp.swapaxes(upper, -1, -2) @ x[index]
            else:
                values[index] = diagonal @ x[index]
                if index > 0:
                    values[index] += lower @ x[index - 1]
                if index < n_blocks - 1:
                    values[index] += upper @ x[index + 1]
        return jnp.stack(values)

    def operator(x):
        return matvec(x, transposed=transpose)

    return jax.lax.custom_linear_solve(
        operator,
        rhs,
        solve=lambda _, b: _block_thomas_checkpointed_impl(
            block_fn, n_blocks, b, checkpoint_size, transpose=transpose
        ),
        transpose_solve=lambda _, b: _block_thomas_checkpointed_impl(
            block_fn, n_blocks, b, checkpoint_size, transpose=not transpose
        ),
        symmetric=False,
    )


def _block_thomas_solve_regenerated(
    factors: GeneratedBlockTridiagFactors, rhs: jax.Array, transpose: bool
) -> jax.Array:
    """Substitution that rebuilds the off-diagonal blocks instead of reading them.

    The recurrences are the ones in :func:`block_thomas_solve`, re-associated so
    that step ``k`` of each sweep touches only ``Delta_k`` and one block of row
    ``k`` --- never row ``k+1``'s Schur factor. That matters twice over.

    First, it lets both sweeps take the *whole* ``delta_lu`` band as scan inputs
    rather than a shifted slice of it. A slice would materialize a second
    ``(n_blocks, m, m)`` array during the solve and hand back most of the memory
    this storage policy exists to save.

    Second, it keeps every scan carry linear in the right-hand side: the
    regenerated blocks are consumed where they are produced and never enter a
    carry. Reverse mode and :func:`jax.linear_transpose` therefore transpose
    these scans in the ordinary way.

    Writing ``z_k = Delta_k^{-1} sigma_k``, the downward sweep carries ``z``
    instead of ``sigma`` and emits ``sigma``,

        sigma_k = b_k - C_k z_{k+1},   z_k = Delta_k^{-1} sigma_k,

    with ``z_N = 0`` and ``C_k`` the regenerated coupling (``U_k`` for the primal
    system, ``L_{k+1}^T`` for the transposed one). The upward sweep is unchanged
    apart from where its blocks come from. These are the stored-band operations
    in the stored-band order, so the two paths agree at round-off rather than at
    a solver tolerance --- what little they differ by is how XLA fuses a scan
    against an unrolled loop. The one extra triangular solve is ``z_0``, which
    the upward sweep recomputes as ``x_0`` so that its inputs stay unsliced too.
    """
    delta_lu, delta_piv = factors.delta_lu, factors.delta_piv
    block_fn, n_blocks = factors.block_fn, factors.n_blocks
    if int(rhs.shape[0]) != n_blocks:
        raise ValueError("rhs leading dimension must equal n_blocks")

    fdt = delta_lu.dtype
    work = jnp.result_type(rhs, factors.work_dtype)
    trans = 1 if transpose else 0
    swap = lambda block: jnp.swapaxes(block, -1, -2)  # noqa: E731

    def tsolve(lu, piv, v):
        return lu_solve((lu, piv), v.astype(fdt), trans=trans).astype(work)

    def down_step(z_next, inputs):
        index, lu, piv, b_k = inputs
        if transpose:
            # Row ``k+1``'s sub-diagonal block; at the top row it multiplies the
            # zero carry, and the clamp keeps the generator index in range.
            lower_next, _, _ = block_fn(jnp.minimum(index + 1, n_blocks - 1))
            coupling = swap(lower_next)
        else:
            _, _, upper_k = block_fn(index)
            coupling = upper_k
        # ``select`` rather than relying on a zero carry: ``upper[-1]`` is
        # documented as unused, so a generator may return anything there.
        sigma_k = jnp.where(index == n_blocks - 1, b_k, b_k - coupling @ z_next)
        return tsolve(lu, piv, sigma_k), sigma_k

    indices = jnp.arange(n_blocks, dtype=jnp.int32)
    zero = jnp.zeros(rhs.shape[1:], work)
    _, sigma = jax.lax.scan(
        down_step, zero, (indices, delta_lu, delta_piv, rhs), reverse=True
    )

    def up_step(x_previous, inputs):
        index, lu, piv, sigma_k = inputs
        if transpose:
            _, _, upper_previous = block_fn(jnp.maximum(index - 1, 0))
            coupling = swap(upper_previous)
        else:
            lower_k, _, _ = block_fn(index)
            coupling = lower_k
        value = jnp.where(index == 0, sigma_k, sigma_k - coupling @ x_previous)
        x_k = tsolve(lu, piv, value)
        return x_k, x_k

    _, solution = jax.lax.scan(
        up_step, zero, (indices, delta_lu, delta_piv, sigma)
    )
    return solution


def block_thomas_solve(
    factors: BlockTridiagFactors | GeneratedBlockTridiagFactors,
    rhs: jax.Array,
    transpose: bool = False,
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

    :class:`GeneratedBlockTridiagFactors` is accepted here on the same terms.
    Those factors carry no off-diagonal bands, so both sweeps call the factored
    ``block_fn`` again, one block at a time, in place of reading them; the
    transposed solve is exact under that policy too, because it needs the same
    two blocks per row.

    Args:
        factors: output of :func:`block_thomas_factor` or
            :func:`block_thomas_factor_fn`, under either storage policy.
        rhs: ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``.
        transpose: if True, solve ``A^T x = rhs`` instead of ``A x = rhs``.

    Returns:
        Solution with the same shape as ``rhs``.
    """
    if isinstance(factors, GeneratedBlockTridiagFactors):
        return _block_thomas_solve_regenerated(factors, rhs, transpose)
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
    sigma: list[jax.Array] = [None] * rhs.shape[0]  # type: ignore[list-item]
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
    implicit_adjoint: bool = False,
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
        implicit_adjoint: if True, reverse mode uses a custom VJP that solves
            the adjoint system ``A^T lam = cotangent`` by the *same* refinement
            reusing the transposed low-precision factors — zero additional
            factorizations, no differentiation through the factorization, and
            the gradient inherits the refined working-precision forward error
            rather than the factorization precision. If False (default),
            reverse mode differentiates through the refinement loop directly.

    Returns:
        ``x`` with the same shape and precision as ``rhs``.
    """
    if implicit_adjoint:
        return _mixed_precision_block_thomas_amortized(
            lower, diag, upper, rhs, factor_dtype, refine_steps
        )
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


def _band_gradients(lam: jax.Array, x: jax.Array):
    """Band gradients of a solve: ``bar A = -lambda x^T`` restricted to the band.

    ``bar D_j = -lam_j x_j^T``, ``bar L_j = -lam_j x_{j-1}^T`` (``j >= 1``),
    ``bar U_j = -lam_j x_{j+1}^T`` (``j <= N-2``); trailing right-hand-side
    axes are summed out.
    """
    lam2 = lam.reshape(lam.shape[0], lam.shape[1], -1)
    x2 = x.reshape(x.shape[0], x.shape[1], -1)
    outer = lambda a, b: -jnp.einsum("jmr,jnr->jmn", a, b)  # noqa: E731
    x_below = jnp.concatenate([jnp.zeros_like(x2[:1]), x2[:-1]], axis=0)
    x_above = jnp.concatenate([x2[1:], jnp.zeros_like(x2[:1])], axis=0)
    diag_bar = outer(lam2, x2)
    lower_bar = outer(lam2, x_below).at[0].set(0.0)
    upper_bar = outer(lam2, x_above).at[-1].set(0.0)
    return lower_bar, diag_bar, upper_bar


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _mixed_precision_block_thomas_amortized(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
    factor_dtype,
    refine_steps: int,
) -> jax.Array:
    x, _ = _mixed_precision_forward(lower, diag, upper, rhs, factor_dtype, refine_steps)
    return x


def _mixed_precision_forward(lower, diag, upper, rhs, factor_dtype, refine_steps):
    residual_dtype = jnp.result_type(rhs)
    factors = block_thomas_factor(lower, diag, upper, factor_dtype=factor_dtype)
    x, _ = iterative_refinement(
        lambda v: block_tridiag_matvec(lower, diag, upper, v),
        rhs,
        lambda r: block_thomas_solve(factors, r),
        iterations=refine_steps,
        residual_dtype=residual_dtype,
    )
    return x, factors


def _mixed_precision_fwd(lower, diag, upper, rhs, factor_dtype, refine_steps):
    x, factors = _mixed_precision_forward(
        lower, diag, upper, rhs, factor_dtype, refine_steps
    )
    return x, (lower, diag, upper, x, factors)


def _mixed_precision_bwd(factor_dtype, refine_steps, residuals, cotangent):
    lower, diag, upper, x, factors = residuals
    # Adjoint solve A^T lam = cotangent by the same refinement, reusing the
    # low-precision factors transposed: zero extra factorizations, and the
    # gradient inherits the refined (working-precision) forward error rather
    # than the factorization precision.
    lower_t, diag_t, upper_t = _transpose_bands(lower, diag, upper)
    lam, _ = iterative_refinement(
        lambda v: block_tridiag_matvec(lower_t, diag_t, upper_t, v),
        cotangent,
        lambda r: block_thomas_solve(factors, r, transpose=True),
        iterations=refine_steps,
        residual_dtype=jnp.result_type(cotangent),
    )
    lower_bar, diag_bar, upper_bar = _band_gradients(lam, x)
    return lower_bar, diag_bar, upper_bar, lam


_mixed_precision_block_thomas_amortized.defvjp(
    _mixed_precision_fwd, _mixed_precision_bwd
)


def _block_thomas_truncated_impl(
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


def _transpose_bands(
    lower: jax.Array, diag: jax.Array, upper: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bands of ``A^T`` for a block-tridiagonal ``A``.

    ``(A^T)_{j,j-1} = (A_{j-1,j})^T = U_{j-1}^T`` and
    ``(A^T)_{j,j+1} = (A_{j+1,j})^T = L_{j+1}^T``: the off-diagonals swap and
    each block transposes. The out-of-domain slots ``lower[0]``/``upper[-1]``
    stay zero.
    """
    t = lambda a: jnp.swapaxes(a, -1, -2)  # noqa: E731
    lower_t = jnp.concatenate([jnp.zeros_like(upper[:1]), t(upper[:-1])], axis=0)
    upper_t = jnp.concatenate([t(lower[1:]), jnp.zeros_like(lower[:1])], axis=0)
    return lower_t, t(diag), upper_t


def _windowed_leading_solve(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs_low: jax.Array,
    keep_lowest: int,
    window: int,
) -> tuple[jax.Array, int]:
    """Solve the leading ``W = keep_lowest + window`` principal subsystem.

    Uses a homogeneous closure at block ``W`` (the tail is dropped): with the
    right-hand side zero above ``keep_lowest``, block ``j`` of the full solution
    is reproduced with error ``O(rho^{W-j})`` under block diagonal dominance
    (Demko--Moss--Smith decay). Returns the ``W`` leading solution blocks and
    ``W``; peak memory ``O(W m^2)``, independent of the block count.
    """
    n = diag.shape[0]
    w = min(keep_lowest + window, n)
    pad = jnp.zeros((w - keep_lowest, *rhs_low.shape[1:]), rhs_low.dtype)
    rhs = jnp.concatenate([rhs_low, pad], axis=0)
    return block_thomas(lower[:w], diag[:w], upper[:w], rhs), w


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _block_thomas_truncated_bounded(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs_low: jax.Array,
    keep_lowest: int,
    adjoint_window: int,
) -> jax.Array:
    return _block_thomas_truncated_impl(lower, diag, upper, rhs_low, keep_lowest)


def _bounded_fwd(lower, diag, upper, rhs_low, keep_lowest, adjoint_window):
    x_low = _block_thomas_truncated_impl(lower, diag, upper, rhs_low, keep_lowest)
    return x_low, (lower, diag, upper, rhs_low)


def _bounded_bwd(keep_lowest, adjoint_window, residual, cotangent):
    lower, diag, upper, rhs_low = residual
    lower_t, diag_t, upper_t = _transpose_bands(lower, diag, upper)

    # rhs-adjoint (exact): b_bar = P_K A^{-T} E_K x_bar is itself a truncated
    # solve of the transpose -- O(keep_lowest * m^2), no truncation error.
    rhs_bar = _block_thomas_truncated_impl(
        lower_t, diag_t, upper_t, cotangent, keep_lowest
    )

    # band-adjoint (windowed): the full primal/adjoint spread over all blocks but
    # decay away from the lowest block window; a leading-window re-solve gives the
    # band gradients at O((keep_lowest + adjoint_window) m^2) with O(rho^{2 w})
    # error (Demko decay). Grad blocks G_j = -lambda_j x_{.}^T.
    n, m = diag.shape[0], diag.shape[-1]
    x_win, w = _windowed_leading_solve(
        lower, diag, upper, rhs_low, keep_lowest, adjoint_window
    )
    lam_win, _ = _windowed_leading_solve(
        lower_t, diag_t, upper_t, cotangent, keep_lowest, adjoint_window
    )
    outer = jax.vmap(lambda a, b: jnp.einsum("i...,j...->ij", a, b))
    x_below = jnp.concatenate([jnp.zeros_like(x_win[:1]), x_win[:-1]], axis=0)
    x_above = jnp.concatenate([x_win[1:], jnp.zeros_like(x_win[:1])], axis=0)
    diag_bar = -outer(lam_win, x_win)
    lower_bar = (-outer(lam_win, x_below)).at[0].set(0.0)
    upper_bar = (-outer(lam_win, x_above)).at[-1].set(0.0)

    tail = jnp.zeros((n - w, m, m), diag.dtype)
    pad = lambda g: jnp.concatenate([g, tail], axis=0)  # noqa: E731
    return pad(lower_bar), pad(diag_bar), pad(upper_bar), rhs_bar


_block_thomas_truncated_bounded.defvjp(_bounded_fwd, _bounded_bwd)



def _as_window(adjoint_window):
    """Coerce a window argument to an int, accepting :class:`LocalizationWindow`.

    The advisor returns a record rather than a bare integer, and its whole
    point is to be handed straight back to the solver. Comparing it against a
    bound before coercing it is what made that documented call fail, so the
    coercion happens first, through ``operator.index`` so that anything with
    ``__index__`` --- the advisor's record, a NumPy integer --- is accepted and
    a float is not.
    """
    try:
        return operator.index(adjoint_window)
    except TypeError as exc:
        raise TypeError(
            "adjoint_window must be an integer or expose __index__ "
            f"(got {type(adjoint_window).__name__})"
        ) from exc


def block_thomas_truncated(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs_low: jax.Array,
    keep_lowest: int,
    *,
    adjoint_window: int | None = None,
) -> jax.Array:
    """Block-tridiagonal solve returning only the lowest ``keep_lowest`` blocks.

    Requires the right-hand side to vanish for ``k >= keep_lowest``
    (``rhs_low`` holds the nonzero head). The downward Schur sweep runs over all
    blocks but stores nothing above ``keep_lowest``; the upward substitution
    stops there. Peak memory ``O(keep_lowest * m^2)``, independent of
    ``n_blocks``.

    Args:
        lower, diag, upper: as in :func:`block_thomas`.
        rhs_low: nonzero head of the right-hand side, shape
            ``(keep_lowest, m)`` or ``(keep_lowest, m, n_rhs)``.
        keep_lowest: static number of solution blocks to compute
            (1 <= keep_lowest <= n_blocks; equality recovers the full solve).
        adjoint_window: if ``None`` (default), reverse mode differentiates the
            elimination directly, taping the sweep at ``O(n_blocks * m^2)``. If
            an integer ``w`` --- or any object exposing ``__index__``, such as
            the :class:`LocalizationWindow` returned by
            :func:`localization_crossover_window` --- the exact-window custom
            VJP is used. The bands are routed through the same generated
            construction as :func:`block_thomas_truncated_fn`, so both entry
            points give bitwise-identical finite-window gradients: the
            right-hand-side cotangent is exact at every window, every retained
            row ``j < keep_lowest + w`` contributes exactly, and the only error
            is the omission of the rows above. The differentiated solve runs at
            ``O((keep_lowest + w) m^2)`` memory, independent of ``n_blocks``.
            That omitted tail decays geometrically in ``w`` for block
            diagonally dominant systems; ``w >= n_blocks`` reproduces the exact
            gradient.

    Returns:
        The lowest ``keep_lowest`` solution blocks, same layout as ``rhs_low``.
    """
    if not 1 <= keep_lowest <= diag.shape[0]:
        raise ValueError("need 1 <= keep_lowest <= n_blocks")
    if rhs_low.shape[0] != keep_lowest:
        raise ValueError("rhs_low must have keep_lowest leading blocks")
    if adjoint_window is None:
        return _block_thomas_truncated_impl(lower, diag, upper, rhs_low, keep_lowest)
    adjoint_window = _as_window(adjoint_window)
    if adjoint_window < 0:
        raise ValueError("adjoint_window must be non-negative")
    # Route the windowed reverse mode through the generated exact-window rule so
    # that both public entry points give the same finite-window semantics. The
    # bands are handed over as the differentiable parameter pytree and indexed
    # by the generator, which costs nothing here (they are already
    # materialized) and buys the exactness decomposition: exact right-hand-side
    # cotangent at every window, exact cotangents for every retained row, and an
    # error equal to the omitted rows alone. The superseded leading-principal
    # closure remains reachable through ``_block_thomas_truncated_bounded`` for
    # the ablation reported in the tests. The accuracy costs nothing: at
    # N=256, m=16, w=6 both paths compile to the same 116.3 KiB of reverse-mode
    # temporaries and run in the same 0.61 ms, because indexing a materialized
    # band is what the old closure did anyway.
    return block_thomas_truncated_fn(
        _band_block_fn,
        diag.shape[0],
        rhs_low,
        keep_lowest,
        params=(lower, diag, upper),
        adjoint_window=adjoint_window,
    )


def _band_block_fn(params, k):
    """Index stored bands as a generated row: ``(L_k, D_k, U_k)``.

    Used to give :func:`block_thomas_truncated` the same exact-window reverse
    rule as the generated entry point; the parameter pytree is the band triple
    itself, so the row cotangents pull back to the arrays directly.
    """
    lower, diag, upper = params
    return lower[k], diag[k], upper[k]


def _block_thomas_selected_fn_state(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs_low: jax.Array,
    source_blocks: int,
    retain_blocks: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exact leading ``retain_blocks`` blocks of the *full* generated solve.

    This is the selected-head primitive: the elimination visits and eliminates
    **every** one of the ``n_blocks`` rows -- the tail sweep folds the whole
    remainder into the running Schur complement -- so the returned blocks are
    exact blocks of the full ``n_blocks``-row solution, not a leading-principal
    (homogeneous-closure) approximation of them. Only storage and the final
    upward substitution are restricted to the head.

    The two lengths are deliberately independent:

    * ``source_blocks`` is the support of the right-hand side (it must vanish
      above it), and
    * ``retain_blocks`` is how many exact solution blocks are returned.

    Decoupling them is what lets a reverse rule request an extra *halo* block
    beyond the physically returned head, which the exact-window parameter
    adjoint needs to form the upper-block cotangent of its last retained row.

    Blocks are produced per index by ``block_fn(k) -> (L_k, D_k, U_k)`` inside
    the sweeps, so the ``(n_blocks, m, m)`` band arrays are never materialized:
    dense workspace is ``O(retain_blocks * m^2)`` plus a single block triple,
    independent of ``n_blocks``. No explicit inverse is formed.

    Args:
        block_fn: maps a (traced) int32 index ``k`` to the three ``(m, m)``
            blocks ``(L_k, D_k, U_k)``. ``L_0`` and ``U_{n_blocks-1}`` are
            ignored.
        n_blocks: static total number of blocks (>= 1).
        rhs_low: nonzero head of the right-hand side, shape
            ``(source_blocks, m)`` or ``(source_blocks, m, n_rhs)``; the
            right-hand side must vanish for ``k >= source_blocks``.
        source_blocks: static support of the right-hand side.
        retain_blocks: static number of exact solution blocks to return
            (``source_blocks <= retain_blocks <= n_blocks``; equality with
            ``n_blocks`` recovers the full solve).

    Returns:
        The lowest ``retain_blocks`` exact solution blocks, plus the retained
        head factors used by residual diagnostics.
    """
    n = n_blocks
    if not 1 <= source_blocks <= retain_blocks <= n:
        raise ValueError("need 1 <= source_blocks <= retain_blocks <= n_blocks")
    if rhs_low.shape[0] != source_blocks:
        raise ValueError("rhs_low must have source_blocks leading blocks")

    # Zero-pad the source up to the retained length: the extra rows carry no
    # forcing, so padding changes nothing mathematically and lets the head
    # sweep below run over the retained window uniformly.
    if retain_blocks > source_blocks:
        pad = jnp.zeros(
            (retain_blocks - source_blocks,) + rhs_low.shape[1:], rhs_low.dtype
        )
        rhs_low = jnp.concatenate([rhs_low, pad], axis=0)

    k = retain_blocks
    m = rhs_low.shape[1]
    dtype = rhs_low.dtype

    if k < n:
        # Tail sweep (blocks n-1 .. k): carry the running Schur complement
        # and the L block of the row just processed (needed one step below).
        l_last, d_last, _ = block_fn(jnp.int32(n - 1))
        tail_carry0 = (lu_factor(d_last), l_last)

        def tail_step(carry, j):
            delta_next, l_next = carry
            l_j, d_j, u_j = block_fn(j)
            x = lu_solve(delta_next, l_next)
            delta_j = lu_factor(d_j - u_j @ x)
            return (delta_j, l_j), None

        (delta_head, l_head), _ = jax.lax.scan(
            tail_step, tail_carry0, jnp.arange(k, n - 1, dtype=jnp.int32), reverse=True
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

    head_carry0 = (delta_head, l_head, jnp.zeros_like(rhs_low[0]))
    head_inputs = (jnp.arange(k, dtype=jnp.int32), rhs_low)
    _, (lus, pivs, sigmas, ls) = jax.lax.scan(
        head_step, head_carry0, head_inputs, reverse=True
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


def _block_thomas_truncated_fn_state(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs_low: jax.Array,
    keep_lowest: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Selected-head solve whose source support equals its retained length.

    Thin compatibility wrapper over :func:`_block_thomas_selected_fn_state`
    with ``source_blocks = retain_blocks = keep_lowest``; the behaviour of
    :func:`block_thomas_truncated_fn` is unchanged.
    """
    return _block_thomas_selected_fn_state(
        block_fn, n_blocks, rhs_low, keep_lowest, keep_lowest
    )


def _window_lengths(n_blocks: int, keep_lowest: int, adjoint_window: int):
    """Retained row count ``W`` and primal window ``M`` of the exact-window rule.

    ``W = min(K + w, N)`` rows contribute exactly to the parameter gradient.
    Row ``W-1`` needs ``x_W`` to form its upper-block cotangent, so the primal
    window carries one halo block: ``M = min(W + 1, N)``.
    """
    w_rows = min(keep_lowest + adjoint_window, n_blocks)
    return w_rows, min(w_rows + 1, n_blocks)


def _adjoint_t(a: jax.Array) -> jax.Array:
    """Transpose used to build the adjoint chain, in JAX's cotangent convention.

    The mathematical statement of the method transposes with a conjugate
    (``A^*``) under a sesquilinear pairing. JAX, however, represents the
    cotangent of a complex value so that reverse mode propagates through the
    *plain* transpose: for a real objective the incoming output cotangent
    already carries the conjugation, and re-conjugating here would apply it
    twice. This is verified directly against ordinary reverse mode by the
    complex tests rather than inferred from the real case -- with a conjugate
    transpose the complex gradient is wrong at **every** window, including the
    full window where it must be exact.
    """
    return jnp.swapaxes(a, -1, -2)


def _generated_transpose_block_fn(block_fn, params, n_blocks: int):
    """Row ``j`` of ``A^H`` for a generated chain: ``(U_{j-1}^H, D_j^H, L_{j+1}^H)``.

    The out-of-domain slots (``L_0`` of the transpose at ``j = 0`` and its
    ``U_{N-1}`` at ``j = N-1``) are never read by the elimination, so clamping
    the neighbour index is safe and costs no branch.
    """

    def transposed_block_fn(j):
        _, _, u_prev = block_fn(params, jnp.maximum(j - 1, 0))
        _, d_j, _ = block_fn(params, j)
        l_next, _, _ = block_fn(params, jnp.minimum(j + 1, n_blocks - 1))
        return _adjoint_t(u_prev), _adjoint_t(d_j), _adjoint_t(l_next)

    return transposed_block_fn


def _retained_row_cotangents(lam_win: jax.Array, x_win: jax.Array):
    """Exact block cotangents of the retained rows ``j < W``.

    With exact adjoint blocks ``lam_win`` (length ``W``) and exact primal
    blocks ``x_win`` (length ``M = min(W+1, N)``), the implicit rule
    ``bar A = -lambda x^T`` (JAX cotangent convention; see :func:`_adjoint_t`)
    restricted to row ``j`` gives

        bar L_j = -lam_j x_{j-1}^T,
        bar D_j = -lam_j x_j^T,
        bar U_j = -lam_j x_{j+1}^T,

    with ``x_{-1} = 0`` and, when ``W = N``, ``x_N = 0``. Trailing
    right-hand-side axes are contracted. Because both windows hold exact blocks
    of the full solutions, every returned row cotangent is exact -- no decay
    assumption is used here.
    """
    w_rows = lam_win.shape[0]
    lam2 = lam_win.reshape(w_rows, lam_win.shape[1], -1)
    x2 = x_win.reshape(x_win.shape[0], x_win.shape[1], -1)
    zero = jnp.zeros((1,) + lam2.shape[1:], x2.dtype)

    x_at = x2[:w_rows]
    x_below = jnp.concatenate([zero, x2[: w_rows - 1]], axis=0) if w_rows > 1 else zero
    above = x2[1 : w_rows + 1]
    if above.shape[0] < w_rows:  # W == N: no halo, x_N = 0
        above = jnp.concatenate([above, zero], axis=0)

    def outer(a, b):
        # Plain (not conjugated) outer product: see _adjoint_t for why JAX's
        # cotangent convention must not re-conjugate here.
        return -jnp.einsum("jmr,jnr->jmn", a, b)

    return outer(lam2, x_below), outer(lam2, x_at), outer(lam2, above)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 3, 4))
def _truncated_fn_bounded(
    block_fn,
    n_blocks: int,
    params,
    keep_lowest: int,
    adjoint_window: int,
    rhs_low: jax.Array,
) -> jax.Array:
    solution, _, _, _, _ = _block_thomas_truncated_fn_state(
        lambda j: block_fn(params, j), n_blocks, rhs_low, keep_lowest
    )
    return solution


def _truncated_fn_fwd(block_fn, n_blocks, params, keep_lowest, adjoint_window, rhs_low):
    # Exact-window forward rule: run the full-tail selected-head elimination
    # once, retaining the exact primal window x_0..x_{M-1} (physical head plus
    # the one-block halo the reverse rule needs). Only x_0..x_{K-1} is returned.
    _, primal_window = _window_lengths(n_blocks, keep_lowest, adjoint_window)
    x_window, _, _, _, _ = _block_thomas_selected_fn_state(
        lambda j: block_fn(params, j),
        n_blocks,
        rhs_low,
        keep_lowest,
        primal_window,
    )
    return x_window[:keep_lowest], (params, rhs_low, x_window)


def _truncated_fn_bwd(block_fn, n_blocks, keep_lowest, adjoint_window, residuals, ct):
    """Exact-window reverse rule.

    Both the right-hand-side cotangent and every retained row cotangent are
    *exact* at any window: the adjoint is obtained from a full-tail
    selected-head solve of the conjugate-transposed generated chain, so its
    retained blocks are exact blocks of the full adjoint, and they are paired
    with the exact primal window saved by the forward rule. The only
    approximation is the omission of the operator rows ``j >= W``; nothing
    inside the window is approximated (contrast the leading-principal closure
    kept for ablation in :func:`_leading_principal_params_bar`).
    """
    params, rhs_low, x_window = residuals
    n, k = n_blocks, keep_lowest
    retained_rows, _ = _window_lengths(n, k, adjoint_window)

    # Exact adjoint window: full-tail selected-head solve of A^H with the
    # output cotangent as its (K-supported) source, retaining W exact blocks.
    transposed_block_fn = _generated_transpose_block_fn(block_fn, params, n)
    lam_window, _, _, _, _ = _block_thomas_selected_fn_state(
        transposed_block_fn, n, ct, k, retained_rows
    )

    # Exact source cotangent: bar b = P_K lambda, independent of the window.
    rhs_bar = lam_window[:k]

    # Exact retained-row cotangents, pulled back through the generator's own
    # VJP at each index. Live state is O((W+1) m^2) plus the params cotangent,
    # independent of n_blocks.
    lower_bar, diag_bar, upper_bar = _retained_row_cotangents(lam_window, x_window)

    def accumulate(carry, inputs):
        j, l_bar, d_bar, u_bar = inputs
        _, pullback = jax.vjp(lambda p: block_fn(p, j), params)
        (contribution,) = pullback((l_bar, d_bar, u_bar))
        return jax.tree.map(jnp.add, carry, contribution), None

    zero = jax.tree.map(jnp.zeros_like, params)
    indices = jnp.arange(retained_rows, dtype=jnp.int32)
    params_bar, _ = jax.lax.scan(
        accumulate, zero, (indices, lower_bar, diag_bar, upper_bar)
    )
    return params_bar, rhs_bar


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 3, 4))
def _truncated_fn_masked(
    block_fn,
    n_blocks: int,
    params,
    keep_lowest: int,
    adjoint_window: int,
    rhs_low: jax.Array,
    row_mask: jax.Array,
) -> jax.Array:
    """Windowed solve whose retained rows are selected by a *traced* mask.

    Identical to :func:`_truncated_fn_bounded` in the forward direction --- the
    mask does not enter the primal at all --- and identical in reverse when the
    mask is all ones. Its reason to exist is ``vmap``: ``adjoint_window`` is
    static and therefore shared by every chain in a batch, whereas a mask is
    ordinary data and can differ per chain.
    """
    solution, _, _, _, _ = _block_thomas_selected_fn_state(
        lambda k: block_fn(params, k), n_blocks, rhs_low, keep_lowest, keep_lowest
    )
    return solution


def _truncated_fn_masked_fwd(
    block_fn, n_blocks, params, keep_lowest, adjoint_window, rhs_low, row_mask
):
    _, primal_window = _window_lengths(n_blocks, keep_lowest, adjoint_window)
    x_window, _, _, _, _ = _block_thomas_selected_fn_state(
        lambda k: block_fn(params, k), n_blocks, rhs_low, keep_lowest, primal_window
    )
    return x_window[:keep_lowest], (params, rhs_low, x_window, row_mask)


def _truncated_fn_masked_bwd(
    block_fn, n_blocks, keep_lowest, adjoint_window, residuals, ct
):
    """Exact-window reverse rule with a per-chain row selection.

    The mask multiplies the *retained row* cotangents only. It must not touch
    the source cotangent ``bar b = P_K lambda``, which is exact at every window
    and would be silently corrupted by narrowing it -- that gradient does not
    depend on the window and masking it would introduce an error the uniform
    path does not have.
    """
    params, rhs_low, x_window, row_mask = residuals
    n, k = n_blocks, keep_lowest
    retained_rows, _ = _window_lengths(n, k, adjoint_window)

    transposed_block_fn = _generated_transpose_block_fn(block_fn, params, n)
    lam_window, _, _, _, _ = _block_thomas_selected_fn_state(
        transposed_block_fn, n, ct, k, retained_rows
    )
    rhs_bar = lam_window[:k]

    lower_bar, diag_bar, upper_bar = _retained_row_cotangents(lam_window, x_window)
    weights = row_mask.astype(lower_bar.dtype).reshape((-1,) + (1,) * (lower_bar.ndim - 1))
    lower_bar = lower_bar * weights
    diag_bar = diag_bar * weights
    upper_bar = upper_bar * weights

    def accumulate(carry, inputs):
        j, l_bar, d_bar, u_bar = inputs
        _, pullback = jax.vjp(lambda p: block_fn(p, j), params)
        (contribution,) = pullback((l_bar, d_bar, u_bar))
        return jax.tree.map(jnp.add, carry, contribution), None

    zero = jax.tree.map(jnp.zeros_like, params)
    indices = jnp.arange(retained_rows, dtype=jnp.int32)
    params_bar, _ = jax.lax.scan(
        accumulate, zero, (indices, lower_bar, diag_bar, upper_bar)
    )
    # The mask selects an approximation; it is not a quantity the primal
    # depends on, so its cotangent is exactly zero.
    return params_bar, rhs_bar, jnp.zeros_like(row_mask)


_truncated_fn_masked.defvjp(_truncated_fn_masked_fwd, _truncated_fn_masked_bwd)


def _leading_principal_params_bar(
    block_fn, n_blocks: int, params, keep_lowest: int, adjoint_window: int, rhs_low, ct
):
    """Superseded leading-principal closure, retained only for ablation.

    Reconstructs the retained primal and adjoint states from the leading
    ``W x W`` principal submatrix instead of the full chain. Its retained
    blocks are therefore *not* blocks of the full solution, so its gradient
    error contains an interface-induced term inside the window in addition to
    the omitted tail. This is not the production path; it is retained only so
    that the cost of the superseded closure can be measured against the
    exact-window construction.
    """
    n, k = n_blocks, keep_lowest
    w_rows = min(k + adjoint_window, n)
    indices = jnp.arange(w_rows, dtype=jnp.int32)
    lower_w, diag_w, upper_w = jax.lax.map(lambda j: block_fn(params, j), indices)

    x_win, _ = _windowed_leading_solve(lower_w, diag_w, upper_w, rhs_low, k, w_rows - k)
    lower_t, diag_t, upper_t = _transpose_bands(lower_w, diag_w, upper_w)
    lam_win, _ = _windowed_leading_solve(lower_t, diag_t, upper_t, ct, k, w_rows - k)
    lower_bar, diag_bar, upper_bar = _band_gradients(lam_win, x_win)

    def accumulate(carry, inputs):
        j, l_bar, d_bar, u_bar = inputs
        _, pullback = jax.vjp(lambda p: block_fn(p, j), params)
        (contribution,) = pullback((l_bar, d_bar, u_bar))
        return jax.tree.map(jnp.add, carry, contribution), None

    zero = jax.tree.map(jnp.zeros_like, params)
    params_bar, _ = jax.lax.scan(
        accumulate, zero, (indices, lower_bar, diag_bar, upper_bar)
    )
    return params_bar


_truncated_fn_bounded.defvjp(_truncated_fn_fwd, _truncated_fn_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 3, 4, 6))
def _truncated_fn_bounded_with_residual(
    block_fn,
    n_blocks: int,
    params,
    keep_lowest: int,
    adjoint_window: int,
    rhs_low: jax.Array,
    residual_rhs_index,
):
    """Exact-window solve that also reports the residual of its own sweep.

    The residual is a by-product of factors the forward elimination already
    forms, so it costs no extra sweep. It is *not* differentiated: the reverse
    rule below returns cotangents from the solution alone, which is sound only
    because the public wrapper hands the residual back through
    ``stop_gradient``, making its incoming cotangent structurally zero.
    """
    solution, lus, pivs, sigmas, lowers = _block_thomas_truncated_fn_state(
        lambda j: block_fn(params, j), n_blocks, rhs_low, keep_lowest
    )
    residual = _residual_from_state(
        solution, lus, pivs, sigmas, lowers, rhs_low, residual_rhs_index
    )
    return solution, residual


def _truncated_fn_residual_fwd(
    block_fn, n_blocks, params, keep_lowest, adjoint_window, rhs_low,
    residual_rhs_index,
):
    _, primal_window = _window_lengths(n_blocks, keep_lowest, adjoint_window)
    x_window, lus, pivs, sigmas, lowers = _block_thomas_selected_fn_state(
        lambda j: block_fn(params, j),
        n_blocks,
        rhs_low,
        keep_lowest,
        primal_window,
    )
    residual = _residual_from_state(
        x_window[:keep_lowest],
        lus[:keep_lowest],
        pivs[:keep_lowest],
        sigmas[:keep_lowest],
        lowers[:keep_lowest],
        rhs_low,
        residual_rhs_index,
    )
    return (x_window[:keep_lowest], residual), (params, rhs_low, x_window)


def _truncated_fn_residual_bwd(
    block_fn, n_blocks, keep_lowest, adjoint_window, residual_rhs_index,
    residuals, cotangents,
):
    solution_cotangent, _ = cotangents  # the residual's cotangent is zero
    return _truncated_fn_bwd(
        block_fn, n_blocks, keep_lowest, adjoint_window, residuals,
        solution_cotangent,
    )


_truncated_fn_bounded_with_residual.defvjp(
    _truncated_fn_residual_fwd, _truncated_fn_residual_bwd
)



_FORWARD_MODE_HINT = (
    "the exact-window rule is registered as a custom vector--Jacobian product, "
    "which defines reverse mode only, so `jax.jacfwd` and `jax.jvp` cannot be "
    "pushed through it -- they raise here rather than falling back to a taped "
    "path.\n\n"
    "Two ways forward, depending on what you need:\n"
    "  * for a *derivative audit* against reverse mode, differentiate the "
    "untruncated entry point `block_thomas_checkpointed_fn`, which supports "
    "both modes and is exact;\n"
    "  * for forward sensitivities as the production derivative, call this "
    "function without `params` and differentiate the band arrays, or use "
    "`block_thomas_truncated`.\n\n"
    "A code that takes reverse-mode gradients but forward-mode audits needs "
    "both paths available; that is expected, not a misuse."
)


def _reraise_forward_mode(error: Exception) -> None:
    """Turn JAX's generic custom_vjp message into an actionable one.

    JAX says only \"can't apply forward-mode autodiff (jvp) to a custom_vjp
    function\", which is true and tells the caller nothing about what to do
    instead. Naming the entry point that does support forward mode is the whole
    content of the answer, so it belongs in the error.
    """
    text = str(error)
    if "custom_vjp" in text and ("jvp" in text or "forward-mode" in text):
        raise TypeError(f"{text}\n\n{_FORWARD_MODE_HINT}") from error
    raise


def block_thomas_truncated_fn(
    block_fn: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    rhs_low: jax.Array,
    keep_lowest: int,
    *,
    params=None,
    adjoint_window: int | None = None,
    chain_window: jax.Array | None = None,
) -> jax.Array:
    """Truncated block-tridiagonal solve with on-the-fly block assembly.

    The full band arrays are never materialized, and each block index is
    assembled once. See :func:`block_thomas_truncated_fn_with_residual` when an
    algebraic residual of the retained Schur system is also required.

    By default ``block_fn`` maps an index to blocks, ``k -> (L_k, D_k, U_k)``,
    and reverse mode differentiates through the generated sweeps (taping them
    at ``O(n_blocks m^2)``). Passing ``params`` (any pytree) switches
    ``block_fn`` to the explicit form ``(params, k) -> (L_k, D_k, U_k)`` and
    selects a structure-preserving custom VJP governed by ``adjoint_window``:
    the right-hand-side gradient is an exactly generated truncated solve of
    ``A^T`` (three block assemblies per index), and the ``params`` gradient
    pulls the windowed band cotangents back through ``block_fn``'s own VJP at
    each of the leading ``keep_lowest + adjoint_window`` indices. Forward and
    reverse then both run at memory independent of ``n_blocks`` — the band
    arrays are never materialized in either direction.

    Args:
        block_fn: ``k -> (L_k, D_k, U_k)``, or ``(params, k) -> ...`` when
            ``params`` is given. ``L_0`` and ``U_{n_blocks-1}`` are ignored.
        n_blocks: static total number of blocks (>= 1).
        rhs_low: nonzero head of the right-hand side, shape
            ``(keep_lowest, m)`` or ``(keep_lowest, m, n_rhs)``.
        keep_lowest: static number of solution blocks to compute.
        params: optional differentiable parameters consumed by ``block_fn``.
        adjoint_window: window ``w`` for the ``params`` gradient (required with
            ``params``); band-gradient error decays as ``O(rho^{2w})`` for
            block diagonally dominant systems, exactly as for
            :func:`block_thomas_truncated`.
        chain_window: optional *traced* per-chain window, at most
            ``adjoint_window``. Because ``adjoint_window`` is static it is
            shared by every chain under ``vmap``, which forces one worst-case
            width on the whole batch; ``chain_window`` is ordinary data and so
            can differ per chain. ``adjoint_window`` then bounds the retained
            rows (and hence the memory) while each chain contributes only its
            own. Passing ``chain_window=adjoint_window`` reproduces the uniform
            gradient exactly.

    Returns:
        The lowest ``keep_lowest`` solution blocks, same layout as ``rhs_low``.
    """
    if params is not None:
        if adjoint_window is None:
            raise ValueError("params requires a non-negative adjoint_window")
        adjoint_window = _as_window(adjoint_window)
        if adjoint_window < 0:
            raise ValueError("params requires a non-negative adjoint_window")
        try:
            if chain_window is not None:
                retained_rows, _ = _window_lengths(
                    n_blocks, keep_lowest, adjoint_window
                )
                rows = jnp.arange(retained_rows, dtype=jnp.int32)
                mask = (rows < keep_lowest + jnp.asarray(chain_window)).astype(
                    jnp.result_type(rhs_low).type(0).real.dtype
                )
                return _truncated_fn_masked(
                    block_fn, n_blocks, params, keep_lowest, adjoint_window,
                    rhs_low, mask,
                )
            return _truncated_fn_bounded(
                block_fn, n_blocks, params, keep_lowest, adjoint_window, rhs_low
            )
        except TypeError as error:  # forward-mode through a custom_vjp
            _reraise_forward_mode(error)
            raise
    if chain_window is not None:
        raise ValueError("chain_window requires params")
    solution, _, _, _, _ = _block_thomas_truncated_fn_state(
        block_fn, n_blocks, rhs_low, keep_lowest
    )
    return solution


def _residual_from_state(solution, lus, pivs, sigmas, lowers, rhs_low,
                         residual_rhs_index):
    """RMS residual of the Schur system, from factors the sweep already formed.

    Evaluates ``L @ (U @ x) - P @ rhs`` on the retained rows. The eliminated
    tail is already folded into ``sigmas``, so this measures the whole system
    without reconstructing a block above the window or materializing a band.
    Shared by both entry points so the diagnostic means the same thing on the
    taped and the exact-window paths.
    """
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

    residual = jax.vmap(factor_residual)(lus, pivs, residual_solution, residual_rhs)
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)


def block_thomas_truncated_fn_with_residual(
    block_fn,
    n_blocks: int,
    rhs_low: jax.Array,
    keep_lowest: int,
    *,
    params=None,
    adjoint_window=None,
    residual_rhs_index: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the generated truncated solution and its Schur-system RMS residual.

    The residual is evaluated from the retained pivoted LU factors as
    ``L @ (U @ x) - P @ rhs``. It therefore includes the eliminated high-mode
    tail without reconstructing another solution block or materializing the
    original diagonal band.

    Args:
        block_fn: generator of row ``j``. Takes ``(j)`` when ``params`` is
            ``None`` and ``(params, j)`` otherwise.
        n_blocks: chain length.
        rhs_low: right-hand side on the leading ``keep_lowest`` blocks.
        keep_lowest: number of leading blocks returned.
        params: compact parameters the rows are generated from. Supplying them
            selects the exact-window reverse rule of
            :func:`block_thomas_truncated_fn` for the solution, instead of
            taping the elimination, and requires ``adjoint_window``.
        adjoint_window: retained adjoint rows beyond ``keep_lowest``. Accepts an
            integer or a :class:`LocalizationWindow`. ``adjoint_window >=
            n_blocks`` retains every row and is exact.
        residual_rhs_index: which right-hand-side column the residual measures.

    Returns:
        ``(solution, residual)``. The residual is a diagnostic and carries no
        derivative: it is returned through ``stop_gradient`` so that
        differentiating a function of it yields zero rather than a wrong
        number. Differentiate the solution.
    """
    if params is None:
        solution, lus, pivs, sigmas, lowers = _block_thomas_truncated_fn_state(
            block_fn, n_blocks, rhs_low, keep_lowest
        )
        residual = _residual_from_state(
            solution, lus, pivs, sigmas, lowers, rhs_low, residual_rhs_index
        )
        return solution, residual

    if adjoint_window is None:
        raise ValueError("params requires a non-negative adjoint_window")
    window = _as_window(adjoint_window)
    if window < 0:
        raise ValueError("params requires a non-negative adjoint_window")

    solution, residual = _truncated_fn_bounded_with_residual(
        block_fn, n_blocks, params, keep_lowest, window, rhs_low,
        residual_rhs_index,
    )
    return solution, jax.lax.stop_gradient(residual)


def localization_profile_fn(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
) -> jax.Array:
    """Per-row localization factors ``rho_k = ||Delta_k^{-1} L_k||_2`` of a chain.

    These are the *exact* per-step factors of the block inverse, not an
    asymptotic envelope. Writing the downward Schur recursion that the
    selected-head solve already runs,

        Delta_{N-1} = D_{N-1},
        Delta_k     = D_k - U_k Delta_{k+1}^{-1} L_{k+1},

    the block inverse factorizes as an ordered product of the local ratios
    ``-Delta_k^{-1} L_k`` (Meurant, SIAM J. Matrix Anal. Appl. 13, 707 (1992),
    block form), so ``rho_k`` measures how much the influence of a source
    shrinks when it is propagated across row ``k``.

    This matters because a single uniform decay rate is the wrong model for
    operators whose coupling and diagonal scale differently with the row index
    -- for a Legendre-mode kinetic chain the collisional diagonal grows like
    ``nu * k^2`` while the streaming coupling stays ``O(1)``, so the chain is
    *not* localized at low ``k`` and becomes strongly localized above a
    crossover. Fitting one ``rho`` to such a chain reports the non-localized
    head and badly underestimates how well a window will work.

    The profile runs the *same* Schur recursion the forward elimination
    performs, but as a separate sweep: the solve does not currently hand its
    factors back, so this costs one extra downward pass plus a norm estimate
    per row. Sharing the factors would remove that pass; it has not been done.
    No    extra factorization is needed beyond the norm estimates.

    Args:
        block_fn: maps a traced int32 index ``k`` to ``(L_k, D_k, U_k)``.
        n_blocks: static number of block rows (>= 2).

    Returns:
        ``rho`` of shape ``(n_blocks,)``. Entry ``k`` is
        ``||Delta_k^{-1} L_k||_2`` for ``k >= 1``; entry ``0`` is set to
        ``inf`` because row 0 has no sub-diagonal block to propagate.

    See also:
        :func:`localization_crossover_window`, which turns this profile into a
        heuristic starting window, and :func:`check_localized_gradient`, which
        establishes accuracy afterwards.
    """
    if n_blocks < 2:
        raise ValueError("need n_blocks >= 2")

    _, d_last, _ = block_fn(jnp.int32(n_blocks - 1))

    def scan_step(delta, k):
        lower, _, _ = block_fn(k + 1)
        solved = jnp.linalg.solve(delta, lower)
        rho_next = jnp.linalg.norm(solved, ord=2)
        _, diag_k, upper_k = block_fn(k)
        return diag_k - upper_k @ solved, rho_next

    _, rhos = jax.lax.scan(
        scan_step, d_last, jnp.arange(n_blocks - 2, -1, -1, dtype=jnp.int32)
    )
    # scan ran k = N-2 .. 0 and produced rho_{k+1}; restore ascending order and
    # pad the missing row-0 entry.
    rho_ascending = rhos[::-1]
    return jnp.concatenate([jnp.array([jnp.inf], dtype=rho_ascending.dtype), rho_ascending])


@dataclasses.dataclass(frozen=True)
class LocalizationWindow:
    """What the localization diagnostic found, and what it does not promise.

    Attributes:
        window: the suggested ``adjoint_window``. A starting point, not a
            certified width for any tolerance.
        crossover_row: first row at which the primal envelope ``rho_k`` falls
            below ``threshold``, or ``n_blocks`` if it never does.
        localized: whether such a row exists within the chain.
        primal_profile: the full ``rho_k`` array the decision was read from.
        certified: whether the window carries a proof that the realized
            gradient error is within ``tolerance``. ``False`` from
            :func:`localization_crossover_window`, which is a diagnostic;
            ``True`` from :func:`certified_adjoint_window`.
        status: ``"heuristic"``, ``"certified"`` or ``"full-window"``.
        tolerance: the relative tolerance certified against, when certified.
        tail_bound: absolute bound on ``||grad - g_W||`` at this window.
        gradient_lower_bound: the certified lower bound on ``||grad||`` that
            made the relative statement possible.
        certified_relative_error: ``tail_bound / gradient_lower_bound`` -- the
            proven relative gradient error, at or below ``tolerance``.
    """

    window: int
    crossover_row: int
    localized: bool
    primal_profile: np.ndarray
    certified: bool = False
    status: str = "heuristic"
    tolerance: float | None = None
    tail_bound: float | None = None
    gradient_lower_bound: float | None = None
    certified_relative_error: float | None = None

    def __int__(self) -> int:
        return int(self.window)

    def __index__(self) -> int:
        return int(self.window)


def localization_crossover_window(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    keep_lowest: int,
    *,
    threshold: float = 1.0,
    margin: int = 2,
) -> LocalizationWindow:
    """Where the chain starts to contract, as a heuristic starting window.

    This is a **diagnostic, not an error certificate.** It accepts no gradient
    tolerance and cannot return one, because it uses only part of what the tail
    bound needs. Specifically it reads the *primal* envelope
    ``rho_k = ||Delta_k^{-1} L_k||`` and reports the first row where that falls
    below ``threshold``, plus a margin. The bound on the omitted tail,

        ||grad - g_W|| <= sum_{j>=W} gamma_j Lambda_j (X_{j-1} + X_j + X_{j+1}),

    additionally requires the *transposed* envelopes ``Lambda_j``, the
    generator sensitivity ``gamma_j``, the scales of the source and output
    cotangent, and a cumulative sum over the omitted rows rather than the first
    index at which one factor crosses one. None of those enter here.

    What it is good for is the thing needed first: an initial window that is on
    the right order, obtained before any differentiated solve, on operators
    where fitting a single decay rate to the leading rows is wrong by orders of
    magnitude. Establish accuracy afterwards with
    :func:`check_localized_gradient`, which compares nested windows.

    Args:
        block_fn: row generator, as in :func:`localization_profile_fn`.
        n_blocks: static number of block rows.
        keep_lowest: source/output support ``K``.
        threshold: criterion on ``rho_k`` (default 1).
        margin: extra rows kept beyond the crossing.

    Returns:
        A :class:`LocalizationWindow`. It converts to ``int``, so it can be
        passed straight to ``adjoint_window=``. If the chain never localizes
        within ``n_blocks`` the full window is returned, which makes the
        gradient exact rather than silently wrong.
    """
    rho = np.asarray(localization_profile_fn(block_fn, n_blocks))
    localized = np.flatnonzero(np.isfinite(rho) & (rho < threshold))
    full = n_blocks - keep_lowest
    if localized.size == 0:
        return LocalizationWindow(
            window=full, crossover_row=n_blocks, localized=False,
            primal_profile=rho, status="full-window",
        )
    crossing = int(localized[0])
    return LocalizationWindow(
        window=int(min(max(crossing + margin - keep_lowest, 0), full)),
        crossover_row=crossing, localized=True, primal_profile=rho,
    )


def suggest_adjoint_window(
    block_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    keep_lowest: int,
    *,
    threshold: float = 1.0,
    margin: int = 2,
) -> int:
    """Deprecated alias for :func:`localization_crossover_window`.

    The old name suggested a recommendation backed by an accuracy guarantee,
    which this rule does not provide. Returns a plain ``int`` for backward
    compatibility. Scheduled for removal one release after 0.9.x.
    """
    warnings.warn(
        "suggest_adjoint_window is deprecated; use "
        "localization_crossover_window, which returns a LocalizationWindow "
        "diagnostic and is explicit that the result is heuristic rather than "
        "a certified window for a tolerance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return int(
        localization_crossover_window(
            block_fn, n_blocks, keep_lowest, threshold=threshold, margin=margin
        )
    )


def _generator_sensitivity(
    block_fn: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    params,
    n_blocks: int,
) -> jax.Array:
    """``gamma_j = ||D B_j(p)||_F`` for every row of a generated chain.

    ``B_j`` is the map from the parameters to row ``j``'s block triple, so
    ``gamma_j`` bounds how much of a row cotangent the generator can transmit
    back into parameter space: ``||D B_j(p)^*[M]||_2 <= gamma_j ||M||_F``. The
    Frobenius norm is used rather than the induced 2-norm because it is an
    upper bound on it and is computable exactly, where the induced norm would
    need an iteration that converges from *below* -- the wrong side for a
    certificate.

    Cost is one full Jacobian per row, mapped so that only one is live at a
    time. That is affordable as a setup diagnostic and prohibitive inside a
    solve, which is why :func:`certified_adjoint_window` accepts a precomputed
    ``sensitivity`` for callers whose parameter vector is large or whose
    generator has a known analytic bound.
    """

    flat, unravel = jax.flatten_util.ravel_pytree(params)
    basis = jnp.eye(flat.size, dtype=flat.dtype)
    # A complex parameter is differentiated as R^2, not as C: the generator is
    # not assumed holomorphic, and `jacfwd` refuses complex input for exactly
    # that reason. Summing over both real tangent directions gives the
    # Frobenius norm of the real Jacobian, which is the object the bound needs.
    # For a generator that *is* holomorphic this over-counts by sqrt(2), which
    # is the safe direction for a certificate.
    if jnp.issubdtype(flat.dtype, jnp.complexfloating):
        basis = jnp.concatenate([basis, 1j * basis], axis=0)

    def row_norm(j):
        def column(tangent_flat):
            _, tangent = jax.jvp(
                lambda p: block_fn(p, j), (params,), (unravel(tangent_flat),)
            )
            leaves = jax.tree_util.tree_leaves(tangent)
            squares = [jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves]
            return sum(squares[1:], squares[0])

        return jnp.sqrt(jnp.sum(jax.vmap(column)(basis)))

    return jax.lax.map(row_norm, jnp.arange(n_blocks, dtype=jnp.int32))


def _tail_bound_profile(
    block_fn,
    params,
    n_blocks: int,
    keep_lowest: int,
    sensitivity: jax.Array | None,
) -> np.ndarray:
    """Log of the unscaled tail bound ``T(W)`` for every retained-row count.

    Returns ``log T`` indexed by ``W`` in ``[keep_lowest, n_blocks]``, where

        ||grad - g_W|| <= S * T(W),
        T(W) = sum_{j>=W} gamma_j Lambda_j (X_{j-1} + X_j + X_{j+1}),

    ``S = ||lambda_{K-1}||_F ||x_{K-1}||_F`` is the scale supplied separately by
    the caller's own data, and ``X``, ``Lambda`` are the primal and transposed
    envelopes normalized to one at row ``K-1``.

    The envelopes are ordered products of per-row transfer norms. Above the
    source support the primal recursion is homogeneous, ``x_j = -Delta_j^{-1}
    L_j x_{j-1}``, so ``||x_j|| <= rho_j ||x_{j-1}||`` with ``rho`` exactly the
    profile :func:`localization_profile_fn` already computes; the adjoint obeys
    the same recursion on the conjugate-transposed chain. Everything is
    accumulated in logs, because these products run over the whole chain and a
    chain that is not localized in its head has ``rho > 1`` there -- a direct
    product overflows long before the interesting rows are reached.
    """
    n, k = n_blocks, keep_lowest
    rho = np.asarray(localization_profile_fn(lambda j: block_fn(params, j), n))
    transposed = _generated_transpose_block_fn(block_fn, params, n)
    rho_t = np.asarray(localization_profile_fn(transposed, n))
    if sensitivity is None:
        gamma = np.asarray(_generator_sensitivity(block_fn, params, n))
    else:
        gamma = np.asarray(sensitivity)
        if gamma.shape != (n,):
            raise ValueError(f"sensitivity must have shape ({n},); got {gamma.shape}")

    with np.errstate(divide="ignore"):
        log_rho = np.log(np.abs(rho).astype(np.float64))
        log_rho_t = np.log(np.abs(rho_t).astype(np.float64))
        log_gamma = np.log(np.abs(gamma).astype(np.float64))

    # X_j and Lambda_j for j in [K-1, N-1], with index N reserved for the
    # out-of-chain block x_N = 0 that the last retained row pairs against.
    log_x = np.full(n + 1, -np.inf)
    log_lam = np.full(n + 1, -np.inf)
    log_x[k - 1] = 0.0
    log_lam[k - 1] = 0.0
    for j in range(k, n):
        log_x[j] = log_x[j - 1] + log_rho[j]
        log_lam[j] = log_lam[j - 1] + log_rho_t[j]

    # Row j contributes gamma_j * ||lambda_j|| * (||x_{j-1}|| + ||x_j|| + ||x_{j+1}||).
    neighbourhood = np.logaddexp(
        np.logaddexp(log_x[k - 1 : n - 1], log_x[k:n]), log_x[k + 1 : n + 1]
    )
    terms = log_gamma[k:n] + log_lam[k:n] + neighbourhood

    # Suffix log-sum-exp: entry w is the bound for retaining rows [0, K+w).
    suffix = np.full(n - k + 1, -np.inf)
    for i in range(n - k - 1, -1, -1):
        suffix[i] = np.logaddexp(terms[i], suffix[i + 1])
    return suffix


@dataclasses.dataclass(frozen=True)
class ChainWindowPlan:
    """How to run a batch of chains whose certified windows differ.

    Attributes:
        buckets: ``(window, chain_indices)`` pairs. Every chain in a bucket is
            solved with that bucket's static ``adjoint_window``, so each bucket
            is one traced shape.
        retained_rows: total retained block rows under this plan.
        uniform_retained_rows: total under the single worst-case window, which
            is what one static window over the whole batch costs.
        reduction: fraction of retained rows the plan removes.
        ideal_retained_rows: total if every chain had its own window --- the
            bound no bucketing can beat.
    """

    buckets: tuple[tuple[int, np.ndarray], ...]
    retained_rows: int
    uniform_retained_rows: int
    reduction: float
    ideal_retained_rows: int


def plan_chain_windows(
    windows,
    keep_lowest: int,
    *,
    max_buckets: int = 4,
) -> ChainWindowPlan:
    """Group chains with different windows into a few static ones.

    ``adjoint_window`` is static, so one ``vmap`` over a batch runs every chain
    at the widest window any chain needs. When the chains are a collisionality
    scan that is badly wasteful: the criterion gives up on the least collisional
    chain and drags the whole batch to the full window, while most chains would
    localize in a fraction of it.

    Padding to the maximum does not fix this --- it *is* the problem. What fixes
    it is a segmented layout: sort the chains by window, cut them into at most
    ``max_buckets`` groups, and trace one shape per group. This function chooses
    those cuts optimally, by dynamic programming over the distinct window
    values, minimizing total retained rows ``sum_b |b| (K + W_b)``.

    Args:
        windows: per-chain windows, e.g. from :func:`certified_adjoint_window`
            applied to each chain.
        keep_lowest: source and output support ``K``.
        max_buckets: how many distinct traced shapes to allow. More buckets
            approach the per-chain ideal at the cost of more compilations.

    Returns:
        A :class:`ChainWindowPlan`. Run each bucket separately::

            for window, chains in plan.buckets:
                grads = jax.vmap(lambda c: gradient(c, adjoint_window=window))(chains)

        Use ``chain_window=`` inside a bucket if the chains within it should
        still use their individual windows; that changes which rows contribute
        but not the retained state, which the bucket's static window fixes.
    """
    windows = np.asarray(windows, dtype=np.int64)
    if windows.ndim != 1 or windows.size == 0:
        raise ValueError("windows must be a non-empty one-dimensional array")
    if max_buckets < 1:
        raise ValueError("need max_buckets >= 1")
    k = int(keep_lowest)

    values, counts = np.unique(windows, return_counts=True)  # ascending
    n_values = values.size
    buckets = min(int(max_buckets), n_values)

    # cost[i][j] = rows for chains with window in values[i..j], all run at
    # values[j] -- the widest in the group.
    prefix = np.concatenate([[0], np.cumsum(counts)])

    def segment_cost(i, j):
        return int(prefix[j + 1] - prefix[i]) * (k + int(values[j]))

    # best[b][j]: minimum rows covering values[0..j] with b+1 buckets.
    best = np.full((buckets, n_values), np.inf)
    cut = np.zeros((buckets, n_values), dtype=np.int64)
    for j in range(n_values):
        best[0, j] = segment_cost(0, j)
    for b in range(1, buckets):
        for j in range(n_values):
            for i in range(1, j + 1):
                candidate = best[b - 1, i - 1] + segment_cost(i, j)
                if candidate < best[b, j]:
                    best[b, j] = candidate
                    cut[b, j] = i

    edges, j, b = [], n_values - 1, buckets - 1
    while b > 0:
        i = int(cut[b, j])
        edges.append((i, j))
        j, b = i - 1, b - 1
    edges.append((0, j))
    edges.reverse()

    assignment = []
    for i, j in edges:
        window = int(values[j])
        members = np.flatnonzero((windows >= values[i]) & (windows <= values[j]))
        assignment.append((window, members))

    retained = int(sum(len(m) * (k + w) for w, m in assignment))
    uniform = int(windows.size * (k + int(values[-1])))
    ideal = int(np.sum(windows + k))
    return ChainWindowPlan(
        buckets=tuple(assignment),
        retained_rows=retained,
        uniform_retained_rows=uniform,
        reduction=1.0 - retained / uniform,
        ideal_retained_rows=ideal,
    )


def certified_adjoint_window(
    block_fn: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    n_blocks: int,
    keep_lowest: int,
    params,
    rhs_low: jax.Array,
    cotangent: jax.Array,
    *,
    rtol: float = 1.0e-6,
    sensitivity: jax.Array | None = None,
    probe_window: int | None = None,
) -> LocalizationWindow:
    r"""Smallest window whose gradient error is *provably* within ``rtol``.

    This is the certificate that :func:`localization_crossover_window` declines
    to give. It is possible because the exact-window rule leaves exactly one
    approximation to bound: every retained row cotangent and the source
    cotangent are exact blocks of the full solve, so

    .. math::
        \nabla_p J - g_W = \sum_{j \ge W} DB_j(p)^*[\bar L_j, \bar D_j, \bar U_j]

    holds with equality, and the row cotangents are the rank-one products
    :math:`\bar L_j = -\lambda_j x_{j-1}^*` and its neighbours. Bounding each
    factor by its envelope gives a fully computable

    .. math::
        \|\nabla_p J - g_W\| \le S \sum_{j\ge W} \gamma_j \Lambda_j
            (X_{j-1} + X_j + X_{j+1}) =: B(W),

    with :math:`S = \|\lambda_{K-1}\|\,\|x_{K-1}\|` read off one selected-head
    solve of the caller's own data and :math:`\gamma_j = \|DB_j(p)\|_F`.

    Making that *relative* needs a lower bound on the gradient itself, which
    norms of the summands cannot supply -- they cannot see cancellation. So one
    windowed gradient is evaluated at a probe window, and the reverse triangle
    inequality turns it into one: :math:`\|\nabla_p J\| \ge \|g_{W_0}\| -
    B(W_0)`. That bound is valid whichever window produced it, so the search
    over windows costs no further differentiated solves.

    The returned window therefore satisfies
    :math:`\|\nabla_p J - g_W\| \le \mathtt{rtol}\,\|\nabla_p J\|` --- a
    statement about the realized gradient error, not about the bound.

    Cost is two localization sweeps, ``n_blocks`` generator Jacobians, and one
    or two differentiated solves. It is a setup computation: call it when the
    operator or parameters change materially, not inside an optimizer step.

    Args:
        block_fn: row generator ``(params, k) -> (L_k, D_k, U_k)``.
        n_blocks: static number of block rows.
        keep_lowest: source and output support ``K`` (>= 1).
        params: the parameters the gradient is taken with respect to.
        rhs_low: the ``(K, m, ...)`` source, as passed to the solver.
        cotangent: the output cotangent, same shape as the returned head. This
            is problem data: the certified window depends on what is being
            differentiated, not on the operator alone.
        rtol: target relative gradient error.
        sensitivity: precomputed ``gamma`` of shape ``(n_blocks,)``, if the
            per-row Jacobians are too expensive or an analytic bound is known.
        probe_window: window at which to evaluate the probe gradient. Defaults
            to the heuristic crossover window, widening automatically if the
            resulting lower bound is not positive.

    Returns:
        A :class:`LocalizationWindow` with ``certified=True``, carrying the
        absolute bound, the gradient lower bound, and the certified relative
        error. If no proper window is certifiable the full window is returned,
        whose bound is zero because its tail is empty -- an over-estimate,
        never an under-estimate.

    See also:
        :func:`localization_crossover_window` for the cheap uncertified
        starting point, and :func:`check_localized_gradient` for the empirical
        refinement check that needs no generator Jacobians.
    """
    if n_blocks < 2:
        raise ValueError("need n_blocks >= 2")
    if not 1 <= keep_lowest <= n_blocks:
        raise ValueError("need 1 <= keep_lowest <= n_blocks")
    if not 0.0 < rtol < 1.0:
        raise ValueError("rtol must lie strictly between zero and one")

    n, k = n_blocks, keep_lowest
    full = n - k
    log_tail = _tail_bound_profile(block_fn, params, n, k, sensitivity)

    # S: the scale the envelopes are relative to. Both heads are exact blocks
    # of the full solutions, so a K-row selected-head solve suffices for each.
    primal_fn = lambda j: block_fn(params, j)  # noqa: E731
    x_head, *_ = _block_thomas_selected_fn_state(primal_fn, n, rhs_low, k, k)
    transposed = _generated_transpose_block_fn(block_fn, params, n)
    lam_head, *_ = _block_thomas_selected_fn_state(transposed, n, cotangent, k, k)
    scale = float(jnp.linalg.norm(x_head[k - 1])) * float(jnp.linalg.norm(lam_head[k - 1]))

    def bound(window: int) -> float:
        return scale * float(np.exp(log_tail[min(int(window), full)]))

    def gradient_norm(window: int) -> float:
        def head(p):
            return block_thomas_truncated_fn(
                block_fn, n, rhs_low, keep_lowest=k, params=p, adjoint_window=int(window)
            )

        _, pullback = jax.vjp(head, params)
        (grad,) = pullback(cotangent)
        squares = [
            float(jnp.sum(jnp.abs(leaf) ** 2))
            for leaf in jax.tree_util.tree_leaves(grad)
        ]
        return float(np.sqrt(sum(squares)))

    heuristic = localization_crossover_window(primal_fn, n, k)
    if probe_window is None:
        probe = min(int(heuristic.window), full)
    else:
        probe = min(max(int(probe_window), 0), full)

    # The reverse triangle inequality needs ||g_probe|| > B(probe) to say
    # anything. Widen until it does; the full window always does, because its
    # tail is empty and its bound is therefore exactly zero.
    lower = gradient_norm(probe) - bound(probe)
    while lower <= 0.0 and probe < full:
        probe = min(max(probe, 1) * 2, full)
        lower = gradient_norm(probe) - bound(probe)

    if lower <= 0.0:
        # The gradient is numerically zero: no proper window is certifiable
        # against a relative tolerance, and the exact one is the honest answer.
        return LocalizationWindow(
            window=full,
            crossover_row=heuristic.crossover_row,
            localized=heuristic.localized,
            primal_profile=heuristic.primal_profile,
            certified=True,
            status="full-window",
            tolerance=float(rtol),
            tail_bound=0.0,
            gradient_lower_bound=float(max(lower, 0.0)),
            certified_relative_error=0.0,
        )

    target = rtol * lower
    admissible = np.flatnonzero(scale * np.exp(log_tail) <= target)
    window = int(admissible[0]) if admissible.size else full
    achieved = bound(window)
    return LocalizationWindow(
        window=window,
        crossover_row=heuristic.crossover_row,
        localized=heuristic.localized,
        primal_profile=heuristic.primal_profile,
        certified=True,
        status="certified" if window < full else "full-window",
        tolerance=float(rtol),
        tail_bound=achieved,
        gradient_lower_bound=lower,
        certified_relative_error=achieved / lower,
    )


def check_localized_gradient(
    gradient_fn: Callable[[int], object],
    window: int,
    *,
    increment: int = 2,
    rtol: float = 1e-6,
    atol: float = 0.0,
) -> dict[str, object]:
    """Compare a windowed gradient against a wider one: an honest accuracy gate.

    The localization diagnostic cannot certify a tolerance, so accuracy has to
    be established the ordinary way --- by refining the window until the
    gradient stops moving. This runs that check once and reports it, at the
    cost of one extra differentiated solve.

    It is a convergence check, not a proof: agreement between two nested
    windows is evidence that the omitted tail is small, not a bound on it. What
    *is* guaranteed, by construction, is that the full window is exact, so
    refinement converges to the true gradient.

    Args:
        gradient_fn: maps an ``adjoint_window`` to a gradient pytree.
        window: the window under test.
        increment: how many further rows to retain for the comparison.
        rtol, atol: tolerances for the relative difference.

    Returns:
        A dict with the two gradients, their relative difference, whether the
        comparison passed, and a recommendation.
    """
    g_w = gradient_fn(int(window))
    g_wide = gradient_fn(int(window) + int(increment))
    leaves_w = jax.tree_util.tree_leaves(g_w)
    leaves_wide = jax.tree_util.tree_leaves(g_wide)
    num = sum(
        float(jnp.sum(jnp.abs(a - b) ** 2))
        for a, b in zip(leaves_w, leaves_wide, strict=True)
    )
    den = sum(float(jnp.sum(jnp.abs(b) ** 2)) for b in leaves_wide)
    rel = float(np.sqrt(num) / max(np.sqrt(den), np.finfo(float).tiny))
    passed = bool(np.sqrt(num) <= atol + rtol * np.sqrt(den))
    return {
        "window": int(window),
        "comparison_window": int(window) + int(increment),
        "gradient": g_w,
        "comparison_gradient": g_wide,
        "relative_difference": rel,
        "passed": passed,
        "recommendation": (
            "window appears sufficient at this tolerance"
            if passed
            else f"increase adjoint_window beyond {int(window) + int(increment)}"
        ),
    }
