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

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve


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
    lower: jax.Array, diag: jax.Array, upper: jax.Array
) -> BlockTridiagFactors:
    """Run the downward Schur sweep once, for reuse across right-hand sides.

    Args:
        lower: sub-diagonal blocks ``L_k``, shape ``(n_blocks, m, m)``;
            ``lower[0]`` is ignored.
        diag: diagonal blocks ``D_k``, shape ``(n_blocks, m, m)``.
        upper: super-diagonal blocks ``U_k``, shape ``(n_blocks, m, m)``;
            ``upper[-1]`` is ignored.

    Returns:
        Factors for :func:`block_thomas_solve`.
    """

    def down_step(carry, inputs):
        delta_next = carry
        d_k, u_k, l_next = inputs
        x = lu_solve(delta_next, l_next)
        delta_k = lu_factor(d_k - u_k @ x)
        return delta_k, delta_k

    last = lu_factor(diag[-1])
    inputs = (diag[:-1], upper[:-1], lower[1:])  # steps k = n-2 .. 0
    _, (lus, pivs) = jax.lax.scan(down_step, last, inputs, reverse=True)

    delta_lu = jnp.concatenate([lus, last[0][None]], axis=0)
    delta_piv = jnp.concatenate([pivs, last[1][None]], axis=0)
    return BlockTridiagFactors(delta_lu, delta_piv, lower, upper)


def block_thomas_solve(factors: BlockTridiagFactors, rhs: jax.Array) -> jax.Array:
    """Solve using precomputed factors.

    Args:
        factors: output of :func:`block_thomas_factor`.
        rhs: ``(n_blocks, m)`` or ``(n_blocks, m, n_rhs)``.

    Returns:
        Solution with the same shape as ``rhs``.
    """
    delta_lu, delta_piv, lower, upper = factors

    def down_step(sigma_next, inputs):
        u_k, lu_next, piv_next, b_k = inputs
        sigma_k = b_k - u_k @ lu_solve((lu_next, piv_next), sigma_next)
        return sigma_k, sigma_k

    inputs = (upper[:-1], delta_lu[1:], delta_piv[1:], rhs[:-1])
    _, sigmas = jax.lax.scan(down_step, rhs[-1], inputs, reverse=True)
    sigma = jnp.concatenate([sigmas, rhs[-1][None]], axis=0)

    def up_step(x_prev, inputs):
        lu_k, piv_k, l_k, sigma_k = inputs
        x_k = lu_solve((lu_k, piv_k), sigma_k - l_k @ x_prev)
        return x_k, x_k

    x0 = lu_solve((delta_lu[0], delta_piv[0]), sigma[0])
    inputs_up = (delta_lu[1:], delta_piv[1:], lower[1:], sigma[1:])
    _, xs = jax.lax.scan(up_step, x0, inputs_up)
    return jnp.concatenate([x0[None], xs], axis=0)


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
            (1 <= keep_lowest < n_blocks).

    Returns:
        The lowest ``keep_lowest`` solution blocks, same layout as ``rhs_low``.
    """
    k = keep_lowest
    n = diag.shape[0]
    if not 1 <= k < n:
        raise ValueError("need 1 <= keep_lowest < n_blocks")
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
        x = lu_solve((delta_next_lu, delta_next_piv), l_next)
        sigma_j = b_j - u_j @ lu_solve((delta_next_lu, delta_next_piv), sigma_next)
        lu_j, piv_j = lu_factor(d_j - u_j @ x)
        return (lu_j, piv_j, sigma_j), (lu_j, piv_j, sigma_j)

    carry0 = (delta_tail[0], delta_tail[1], jnp.zeros_like(rhs_low[0]))
    head_inputs = (diag[:k], upper[:k], lower[1 : k + 1], rhs_low)
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
