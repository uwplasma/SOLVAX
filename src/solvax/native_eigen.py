"""Bounded-memory host bridge for sparse nonsymmetric eigenpairs.

The JAX operator is sampled in small column batches, sparsified immediately,
and passed to SciPy's implicitly restarted Arnoldi implementation.  This is an
eager CPU bridge for operators whose exact sparsity makes a coupled
shift-invert factorization preferable to a long matrix-free cold transient.

References
----------
R. B. Lehoucq, D. C. Sorensen, and C. Yang, *ARPACK Users' Guide*, SIAM
(1998), DOI 10.1137/1.9780898719628.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from solvax.native import SpluFactorization, _check_not_traced, _import_scipy_sparse

__all__ = ["SparseEigenSolution", "sparse_eigenpairs", "sparse_operator_matrix"]


class SparseEigenSolution(NamedTuple):
    """Host-solved eigenpairs and sparse-matrix residual certificates."""

    eigenvalues: jax.Array
    eigenvectors: jax.Array
    residuals: jax.Array
    converged: jax.Array


def sparse_operator_matrix(
    apply: Callable[[jax.Array], jax.Array],
    prototype: jax.Array,
    *,
    batch_size: int = 64,
    drop_tolerance: float = 0.0,
):
    """Sample a shape-preserving JAX operator into CSR without a dense matrix."""

    _check_not_traced(prototype, "sparse_operator_matrix")
    if batch_size < 1 or drop_tolerance < 0.0:
        raise ValueError("batch_size must be positive and drop_tolerance non-negative")
    sparse, _ = _import_scipy_sparse()
    shape, size = prototype.shape, int(prototype.size)

    def column(vector):
        return jnp.reshape(apply(jnp.reshape(vector, shape)), (size,))

    columns = jax.jit(jax.vmap(column))
    chunks = []
    for start in range(0, size, batch_size):
        indices = jnp.arange(start, min(start + batch_size, size))
        block = np.asarray(
            columns(jax.nn.one_hot(indices, size, dtype=prototype.dtype))
        ).T.copy()
        block[np.abs(block) <= drop_tolerance] = 0.0
        chunks.append(sparse.csr_matrix(block))
    return sparse.hstack(chunks, format="csr")


def sparse_eigenpairs(
    matrix,
    *,
    candidates: int = 6,
    shift: complex | None = None,
    initial: jax.Array | None = None,
    tolerance: float = 1.0e-10,
    maxiter: int | None = None,
    residual_tolerance: float = 1.0e-8,
    factorization: SpluFactorization | None = None,
    adjoint: bool = False,
) -> SparseEigenSolution:
    """Solve a sparse eigenproblem, optionally reusing one shifted LU for its adjoint.

    When ``factorization`` is supplied it must factor ``matrix - shift * I``.
    The adjoint path applies its conjugate-transpose factors, avoiding a second
    factorization during implicit reverse-mode differentiation.
    """

    sparse, sparse_linalg = _import_scipy_sparse()
    if not sparse.issparse(matrix):
        raise TypeError("sparse_eigenpairs expects a scipy sparse matrix")
    if initial is not None:
        _check_not_traced(initial, "sparse_eigenpairs")
    n = int(matrix.shape[0])
    if matrix.shape != (n, n) or not 1 <= candidates < n - 1:
        raise ValueError("matrix must be square and 1 <= candidates < size - 1")
    if tolerance <= 0.0 or residual_tolerance <= 0.0:
        raise ValueError("tolerances must be positive")
    if factorization is not None and shift is None:
        raise ValueError("a reused shift factorization requires shift")
    if factorization is not None and factorization.shape != matrix.shape:
        raise ValueError("factorization and matrix shapes must match")
    if maxiter is not None and maxiter < 1:
        raise ValueError("maxiter must be positive")

    operator = matrix.getH() if adjoint else matrix
    target = None if shift is None else (np.conj(shift) if adjoint else shift)
    opinv = None
    if factorization is not None:
        trans = "H" if adjoint else "N"
        opinv = sparse_linalg.LinearOperator(
            matrix.shape,
            matvec=lambda vector: factorization._solve_numpy(vector, trans=trans),
            dtype=matrix.dtype,
        )
    v0 = None if initial is None else np.asarray(initial).reshape(-1)
    values, vectors = sparse_linalg.eigs(
        operator,
        k=candidates,
        sigma=target,
        which="LR" if target is None else "LM",
        v0=v0,
        tol=tolerance,
        maxiter=maxiter,
        OPinv=opinv,
    )
    images = operator @ vectors
    numerators = np.linalg.norm(images - vectors * values[None, :], axis=0)
    denominators = np.maximum(
        np.linalg.norm(images, axis=0),
        np.abs(values) * np.linalg.norm(vectors, axis=0),
    )
    residuals = numerators / np.maximum(denominators, np.finfo(float).tiny)
    finite = np.isfinite(values.real) & np.isfinite(values.imag) & np.isfinite(residuals)
    return SparseEigenSolution(
        jnp.asarray(values),
        jnp.asarray(vectors.T),
        jnp.asarray(residuals),
        jnp.asarray(finite & (residuals < residual_tolerance)),
    )
