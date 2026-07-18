"""Randomized Nyström preconditioning for SPD systems.

For a symmetric positive semidefinite operator ``A`` and a regularized system
``(A + mu I) x = b``, a rank-``ell`` randomized Nystrom approximation
``A_nys = U diag(lam) U^T`` built from ``ell`` operator applications yields the
preconditioner

    P^{-1} v = U diag((lam_ell + mu) / (lam + mu)) U^T v + (v - U U^T v),

which is symmetric positive definite, costs one ``(n, ell)`` matmul pair per
application, and — when ``ell`` exceeds roughly twice the ``mu``-effective
dimension of ``A`` — bounds the preconditioned condition number by a small
constant *in expectation, independently of the spectrum's decay rate*
(Frangella, Tropp & Udell, SIAM J. Matrix Anal. Appl. 44, 718 (2023)). It is
the scalable coarse-correction alternative when no grid hierarchy or
structured coarse operator exists.

The sketch uses an explicit PRNG key, so construction is deterministic,
jit-able, and differentiable through both the sketch and the eigenfactors; the
stabilized-shift construction follows Frangella et al., Algorithm 2.1.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

MatVec = Callable[[jax.Array], jax.Array]


def nystrom_preconditioner(
    matvec: MatVec,
    n: int,
    rank: int,
    key: jax.Array,
    *,
    mu: float = 0.0,
    dtype=None,
) -> MatVec:
    """Build a randomized Nystrom preconditioner for ``matvec + mu I``.

    Args:
        matvec: symmetric positive semidefinite operator ``v -> A v`` on flat
            ``(n,)`` arrays; must be pure JAX. Symmetry is assumed, not
            checked.
        n: static operand dimension.
        rank: static sketch size ``ell`` (1 <= rank <= n). Effective when it
            exceeds about twice the ``mu``-effective dimension of ``A``.
        key: PRNG key for the Gaussian test matrix; fixing it makes the
            preconditioner deterministic and differentiable.
        mu: regularization shift of the system being solved (``A + mu I``).
        dtype: sketch dtype (defaults to float32/float64 per x64 mode).

    Returns:
        A symmetric-positive-definite inverse action suitable as ``precond=``
        for :func:`solvax.pcg.pcg` on the system ``(A + mu I) x = b``.
    """
    if not 1 <= rank <= n:
        raise ValueError("need 1 <= rank <= n")
    dtype = jnp.zeros(0).dtype if dtype is None else dtype

    omega = jax.random.normal(key, (n, rank), dtype)
    omega, _ = jnp.linalg.qr(omega)
    sketch = jax.vmap(matvec, in_axes=1, out_axes=1)(omega)

    # Stabilized Nystrom factorization (Frangella-Tropp-Udell, Alg. 2.1):
    # shift by nu ~ eps ||Y|| so the core Cholesky exists for psd A.
    nu = jnp.finfo(dtype).eps * jnp.linalg.norm(sketch)
    shifted = sketch + nu * omega
    core = jnp.linalg.cholesky(omega.T @ shifted)
    half = jax.scipy.linalg.solve_triangular(core, shifted.T, lower=True).T
    basis, singular, _ = jnp.linalg.svd(half, full_matrices=False)
    eigenvalues = jnp.maximum(singular**2 - nu, 0.0)

    smallest = eigenvalues[-1]

    def precond(v: jax.Array) -> jax.Array:
        projected = basis.T @ v
        scaled = (smallest + mu) / (eigenvalues + mu) * projected
        return basis @ (scaled - projected) + v

    return precond
