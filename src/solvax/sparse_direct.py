"""Traced sparse-direct solves and eigenvalues on host factors, with adjoints that reuse them.

:mod:`solvax.native` factors a SciPy sparse matrix with SuperLU or MUMPS but
runs eagerly: it refuses tracers and defines no derivative. This module puts
those factors behind ``jax.pure_callback`` so they can be staged under ``jit``
and ``vmap`` and differentiated:

* :func:`sparse_solve` solves ``A x = b`` for ``A`` given as a static
  :class:`CsrPattern` plus traced values. It is a ``jax.lax.custom_linear_solve``
  whose matrix-vector product is the traced CSR product, so forward- and
  reverse-mode derivatives with respect to both the values and ``b`` follow
  from implicit differentiation. The tangent solve and the transposed solve of
  reverse mode go to the *same* host factorization as the forward solve: the
  factors are cached by a digest of the values, so a ``value_and_grad`` factors
  once. Several right-hand sides, and ``vmap`` over the right-hand side, are one
  multi-right-hand-side solve against one factorization.
* :func:`sparse_eigenvalue` returns the eigenvalue of ``A(params)`` selected
  among those nearest a shift, with the right and left eigenvectors from one
  shifted factorization (the left vector from conjugate-transposed solves on
  it). Its derivative is the standard first-order perturbation
  ``d lambda = y^H (dA) x / (y^H x)``, evaluated as *one* forward-mode product
  of the caller's matrix-free operator, so differentiating does not
  differentiate the factorization or the sparse assembly.
* :func:`csr_data_from_products` recovers CSR values from a traced batch of
  compressed products (see :mod:`solvax.compression`), so the values the host
  factors can be assembled from a matrix-free operator inside the same trace.

Everything numerical runs on the host CPU. On an accelerator the values and
right-hand sides are copied to the host and the solution copied back, per call.

The factor cache is process-global and holds at most
:func:`set_factor_cache_size` factorizations (default 2), least recently used
first out; :func:`clear_factor_cache` releases them. Cached factors hold native
memory, so a large cache is a memory decision.

References
----------
- J. H. Wilkinson, *The Algebraic Eigenvalue Problem*, Oxford (1965), ch. 2:
  first-order perturbation of a simple eigenvalue.
- P. R. Amestoy, I. S. Duff, J.-Y. L'Excellent, and J. Koster, SIAM J. Matrix
  Anal. Appl. 23(1), 15 (2001), DOI 10.1137/S0895479899358194 (MUMPS).
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from solvax.native import SpluFactorization, _import_scipy_sparse
from solvax.native_eigen import sparse_eigenpairs

__all__ = [
    "CsrPattern",
    "HostFactorOptions",
    "SparseEigenvalue",
    "clear_factor_cache",
    "csr_data_from_products",
    "csr_matvec",
    "factor_cache_info",
    "set_factor_cache_size",
    "sparse_eigenvalue",
    "sparse_solve",
]

_TOKENS = itertools.count()


class CsrPattern:
    """A fixed CSR sparsity pattern whose values are supplied separately.

    Patterns compare and hash by identity, so one pattern object can be a static
    argument of a jitted function; build it once and reuse it. ``rows[e]`` is the
    row of stored entry ``e``.
    """

    __slots__ = ("indptr", "indices", "rows", "shape", "_token")

    def __init__(self, indptr, indices, shape: tuple[int, int]):
        indptr = np.asarray(indptr, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        n_rows, n_cols = (int(s) for s in shape)
        if indptr.ndim != 1 or indptr.size != n_rows + 1 or indptr[0] != 0:
            raise ValueError("indptr must have n_rows + 1 entries starting at 0")
        if np.any(np.diff(indptr) < 0) or indptr[-1] != indices.size:
            raise ValueError("indptr must be non-decreasing and end at len(indices)")
        if indices.ndim != 1 or (indices.size and (indices.min() < 0 or indices.max() >= n_cols)):
            raise ValueError("indices must be one-dimensional column indices in range")
        self.indptr = indptr
        self.indices = indices
        self.rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
        self.shape = (n_rows, n_cols)
        self._token = next(_TOKENS)

    @property
    def nnz(self) -> int:
        """Number of stored entries."""
        return int(self.indices.size)

    @classmethod
    def from_scipy(cls, matrix, *, include_diagonal: bool = False):
        """Pattern and values of a SciPy sparse matrix.

        Duplicates are summed and column indices sorted. ``include_diagonal``
        stores every diagonal entry (explicit zeros where absent), which a
        diagonal shift such as ``A - sigma I`` requires.

        Returns:
            ``(pattern, values)`` with ``values`` a NumPy array of length ``nnz``.
        """
        sparse, _ = _import_scipy_sparse()
        if not sparse.issparse(matrix):
            raise TypeError("CsrPattern.from_scipy expects a scipy sparse matrix")
        csr = sparse.csr_matrix(matrix, copy=True)
        csr.sum_duplicates()
        if include_diagonal:
            if csr.shape[0] != csr.shape[1]:
                raise ValueError("include_diagonal requires a square matrix")
            marker = sparse.csr_matrix(
                (np.ones(csr.nnz), csr.indices, csr.indptr), shape=csr.shape
            ) + sparse.identity(csr.shape[0], format="csr")
            marker.sort_indices()
            values = np.asarray(csr[marker.nonzero()]).reshape(-1)
            pattern = cls(marker.indptr, marker.indices, marker.shape)
            return pattern, values.astype(csr.dtype)
        csr.sort_indices()
        return cls(csr.indptr, csr.indices, csr.shape), np.asarray(csr.data)

    def to_scipy(self, values):
        """A SciPy CSR matrix with this pattern and concrete ``values``."""
        sparse, _ = _import_scipy_sparse()
        values = np.asarray(values)
        if values.shape != (self.nnz,):
            raise ValueError(f"values must have shape ({self.nnz},)")
        return sparse.csr_matrix((values, self.indices, self.indptr), shape=self.shape)

    def diagonal_positions(self) -> np.ndarray:
        """Stored-entry index of every diagonal element; raises if one is missing."""
        n = self.shape[0]
        if self.shape[1] != n:
            raise ValueError("diagonal positions need a square pattern")
        hits = np.flatnonzero(self.rows == self.indices)
        if hits.size != n:
            raise ValueError(
                "the pattern does not store every diagonal entry; build it with "
                "CsrPattern.from_scipy(..., include_diagonal=True)"
            )
        return hits


@dataclass(frozen=True)
class HostFactorOptions:
    """How the host factors a matrix: see :class:`solvax.SpluFactorization`."""

    backend: str = "superlu"
    memory_limit_bytes: int | None = None
    memory_safety_factor: float | None = None

    def __post_init__(self):
        if self.backend not in {"superlu", "mumps"}:
            raise ValueError("backend must be 'superlu' or 'mumps'")


# --------------------------------------------------------------------------
# Factor cache
# --------------------------------------------------------------------------

_CACHE: OrderedDict[tuple, SpluFactorization] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_STATE = {"size": 2, "hits": 0, "misses": 0}


def set_factor_cache_size(size: int) -> None:
    """Keep at most ``size`` host factorizations (0 disables reuse)."""
    if int(size) < 0:
        raise ValueError("size must be non-negative")
    with _CACHE_LOCK:
        _CACHE_STATE["size"] = int(size)
        _evict_locked()


def clear_factor_cache() -> None:
    """Release every cached factorization and reset the hit/miss counters."""
    with _CACHE_LOCK:
        while _CACHE:
            _CACHE.popitem(last=False)[1].close()
        _CACHE_STATE["hits"] = 0
        _CACHE_STATE["misses"] = 0


def factor_cache_info() -> dict[str, int]:
    """Current cache size limit, entries, hits and misses (factorizations)."""
    with _CACHE_LOCK:
        return {
            "size": _CACHE_STATE["size"],
            "entries": len(_CACHE),
            "hits": _CACHE_STATE["hits"],
            "misses": _CACHE_STATE["misses"],
        }


def _evict_locked() -> None:
    while len(_CACHE) > _CACHE_STATE["size"]:
        _CACHE.popitem(last=False)[1].close()


def _factor(pattern: CsrPattern, options: HostFactorOptions, values: np.ndarray):
    """The factorization of ``pattern`` with ``values``, from the cache if present."""
    values = np.ascontiguousarray(values)
    digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
    key = (pattern._token, options, values.dtype.str, digest)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            _CACHE_STATE["hits"] += 1
            return cached
        _CACHE_STATE["misses"] += 1
    factor = SpluFactorization(
        pattern.to_scipy(values),
        backend=options.backend,
        memory_limit_bytes=options.memory_limit_bytes,
        memory_safety_factor=options.memory_safety_factor,
    )
    with _CACHE_LOCK:
        if _CACHE_STATE["size"] > 0:
            _CACHE[key] = factor
            _evict_locked()
    return factor


# --------------------------------------------------------------------------
# Traced pieces
# --------------------------------------------------------------------------


def csr_matvec(pattern: CsrPattern, values: jax.Array, x: jax.Array) -> jax.Array:
    """``A @ x`` for ``A`` with ``pattern`` and traced ``values``; ``x`` is (n,) or (n, k)."""
    values = jnp.asarray(values)
    x = jnp.asarray(x)
    if values.shape != (pattern.nnz,):
        raise ValueError(f"values must have shape ({pattern.nnz},), got {values.shape}")
    if x.shape[:1] != (pattern.shape[1],) or x.ndim not in (1, 2):
        raise ValueError(f"x must have shape ({pattern.shape[1]},) or ({pattern.shape[1]}, k)")
    gathered = x[pattern.indices]
    weights = values if x.ndim == 1 else values[:, None]
    return jax.ops.segment_sum(
        weights * gathered,
        jnp.asarray(pattern.rows),
        num_segments=pattern.shape[0],
        indices_are_sorted=True,
    )


def csr_data_from_products(
    pattern: CsrPattern, groups: Sequence[np.ndarray], products: jax.Array
) -> jax.Array:
    """CSR values from compressed products, traceably.

    ``products[g]`` must be ``A @ s_g`` where ``s_g`` is the 0/1 seed of column
    group ``groups[g]`` (for example from :func:`solvax.column_groups`), and no
    two columns of a group may share a row of ``pattern``. Entry ``(i, j)`` is
    then ``products[group(j), i]``. The gather is traced, so derivatives flow
    back into whatever produced ``products``.
    """
    products = jnp.asarray(products)
    n_rows, n_cols = pattern.shape
    if products.ndim != 2 or products.shape != (len(groups), n_rows):
        raise ValueError(f"products must have shape ({len(groups)}, {n_rows})")
    owner = np.full(n_cols, -1, dtype=np.int64)
    for g, columns in enumerate(groups):
        columns = np.asarray(columns, dtype=np.int64)
        if np.any(owner[columns] >= 0):
            raise ValueError("a column appears in more than one group")
        owner[columns] = g
    entry_group = owner[pattern.indices]
    if np.any(entry_group < 0):
        raise ValueError("every stored column must belong to a group")
    return products[entry_group, pattern.rows]


def _host_solve(pattern, options, core_ndim, trans, values, rhs):
    """Host callback: one factorization per distinct batch of values, multi-RHS solves."""
    values = np.asarray(values)
    rhs = np.asarray(rhs)
    dtype = np.result_type(values.dtype, rhs.dtype)
    n = pattern.shape[0]
    core = rhs.shape[rhs.ndim - core_ndim :]
    value_batch = values.shape[:-1]
    rhs_batch = rhs.shape[: rhs.ndim - core_ndim]
    batch = np.broadcast_shapes(value_batch, rhs_batch)
    count = int(np.prod(batch, dtype=np.int64))
    stacked = np.broadcast_to(rhs, batch + core).reshape((count,) + core).astype(dtype)
    if int(np.prod(value_batch, dtype=np.int64)) == 1:
        factor = _factor(pattern, options, values.reshape(-1).astype(dtype))
        columns = np.moveaxis(stacked, 0, -1).reshape(n, -1)
        solved = np.asarray(factor._solve_numpy(columns, trans=trans))
        solved = np.moveaxis(solved.reshape(core + (count,)), -1, 0)
    else:
        per_value = np.broadcast_to(values, batch + values.shape[-1:]).reshape(count, -1)
        solved = np.stack(
            [
                np.asarray(
                    _factor(pattern, options, per_value[i].astype(dtype))._solve_numpy(
                        stacked[i], trans=trans
                    )
                )
                for i in range(count)
            ]
        )
    return solved.reshape(batch + core).astype(dtype)


def _traced_solve(pattern, options, core_ndim, trans, values, rhs):
    return jax.pure_callback(
        partial(_host_solve, pattern, options, core_ndim, trans),
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        values,
        rhs,
        vmap_method="expand_dims",
    )


def sparse_solve(
    pattern: CsrPattern,
    values: jax.Array,
    b: jax.Array,
    *,
    options: HostFactorOptions | None = None,
) -> jax.Array:
    """Solve ``A x = b`` on a host factorization; traceable and differentiable.

    Args:
        pattern: the square sparsity pattern of ``A``.
        values: the ``nnz`` stored values, traced or concrete.
        b: right-hand side(s), shape ``(n,)`` or ``(n, k)``.
        options: backend and memory controls, :class:`HostFactorOptions`.

    Returns:
        ``x`` with the shape of ``b`` and the promoted dtype of ``values`` and ``b``.

    Derivatives with respect to ``values`` and ``b`` (``jvp``, ``vjp``, ``grad``)
    are implicit: ``dx = A^{-1} (db - dA x)`` and, in reverse mode, a solve with
    ``A^T`` on the same cached factors. ``vmap`` over ``b`` alone is one
    multi-right-hand-side solve; ``vmap`` over ``values`` factors each distinct
    matrix once.
    """
    options = HostFactorOptions() if options is None else options
    if pattern.shape[0] != pattern.shape[1]:
        raise ValueError("sparse_solve needs a square pattern")
    values = jnp.asarray(values)
    b = jnp.asarray(b)
    if values.shape != (pattern.nnz,):
        raise ValueError(f"values must have shape ({pattern.nnz},), got {values.shape}")
    if b.ndim not in (1, 2) or b.shape[0] != pattern.shape[0]:
        raise ValueError(f"b must have shape ({pattern.shape[0]},) or ({pattern.shape[0]}, k)")
    dtype = jnp.result_type(values.dtype, b.dtype)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(dtype, jnp.float32)
    values = values.astype(dtype)
    b = b.astype(dtype)
    core_ndim = b.ndim

    def matvec(x):
        return csr_matvec(pattern, values, x)

    def solve(_matvec, rhs):
        return _traced_solve(pattern, options, core_ndim, "N", values, rhs)

    def transpose_solve(_vecmat, rhs):
        return _traced_solve(pattern, options, core_ndim, "T", values, rhs)

    return jax.lax.custom_linear_solve(matvec, b, solve, transpose_solve=transpose_solve)


# --------------------------------------------------------------------------
# Eigenvalue with a one-product derivative
# --------------------------------------------------------------------------


class SparseEigenvalue(NamedTuple):
    """Selected eigenvalue with its right/left eigenvectors and residual certificates.

    ``right`` and ``left`` are returned under ``stop_gradient``: only ``value``
    carries a derivative. ``value`` is NaN when either residual exceeds the
    requested tolerance (fail closed).
    """

    value: jax.Array
    right: jax.Array
    left: jax.Array
    residual: jax.Array
    left_residual: jax.Array


@dataclass(frozen=True)
class _EigenConfig:
    candidates: int
    select: str
    tolerance: float
    residual_tolerance: float
    maxiter: int | None
    options: HostFactorOptions


def _failed_eigen(n, dtype):
    nan = np.asarray(np.nan + 1j * np.nan, dtype=dtype)
    vec = np.full(n, nan, dtype=dtype)
    inf = np.asarray(np.inf, dtype=np.real(nan).dtype)
    return nan, vec, vec.copy(), inf, inf


def _host_eigen(pattern, config, values, shift, initial):
    """Host callback: shifted factor, right pairs, selection, left pair on the same factor."""
    _, sparse_linalg = _import_scipy_sparse()
    values = np.asarray(values)
    shift = complex(np.asarray(shift))
    dtype = np.result_type(values.dtype, np.complex64)
    n = pattern.shape[0]
    matrix = pattern.to_scipy(values.astype(dtype))
    shifted = values.astype(dtype).copy()
    shifted[pattern.diagonal_positions()] -= shift
    factor = _factor(pattern, config.options, shifted)
    arpack = dict(
        candidates=config.candidates,
        shift=shift,
        tolerance=config.tolerance,
        maxiter=config.maxiter,
        residual_tolerance=config.residual_tolerance,
        factorization=factor,
    )
    try:
        right = sparse_eigenpairs(matrix, initial=np.asarray(initial, dtype=dtype), **arpack)
    except sparse_linalg.ArpackNoConvergence:
        return _failed_eigen(n, dtype)
    lams = np.asarray(right.eigenvalues)
    usable = np.asarray(right.converged)
    if not usable.any():
        return _failed_eigen(n, dtype)
    if config.select == "max_real":
        index = int(np.argmax(np.where(usable, lams.real, -np.inf)))
    else:
        index = int(np.argmin(np.where(usable, np.abs(lams - shift), np.inf)))
    lam = lams[index]
    x = np.asarray(right.eigenvectors)[index]
    try:
        left = sparse_eigenpairs(matrix, initial=np.conj(x), adjoint=True, **arpack)
    except sparse_linalg.ArpackNoConvergence:
        return _failed_eigen(n, dtype)
    left_values = np.conj(np.asarray(left.eigenvalues))
    pair = int(np.argmin(np.abs(left_values - lam)))
    y = np.asarray(left.eigenvectors)[pair]
    residual = np.asarray(right.residuals)[index]
    left_residual = np.asarray(left.residuals)[pair]
    if not (residual <= config.residual_tolerance and left_residual <= config.residual_tolerance):
        lam = np.nan + 1j * np.nan
    real = np.real(np.zeros((), dtype=dtype)).dtype
    return (
        np.asarray(lam, dtype=dtype),
        x.astype(dtype),
        y.astype(dtype),
        np.asarray(residual, dtype=real),
        np.asarray(left_residual, dtype=real),
    )


def _eigen_callback(pattern, config, values, shift, initial):
    n = pattern.shape[0]
    dtype = jnp.result_type(values.dtype, jnp.complex64)
    real = jnp.finfo(dtype).dtype
    shapes = (
        jax.ShapeDtypeStruct((), dtype),
        jax.ShapeDtypeStruct((n,), dtype),
        jax.ShapeDtypeStruct((n,), dtype),
        jax.ShapeDtypeStruct((), real),
        jax.ShapeDtypeStruct((), real),
    )
    return jax.pure_callback(
        partial(_host_eigen, pattern, config),
        shapes,
        values,
        shift,
        initial,
        vmap_method="sequential",
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _eigenvalue(operator, pattern, config, params, values, shift, initial):
    return _eigen_callback(pattern, config, values, shift, initial)


@_eigenvalue.defjvp
def _eigenvalue_jvp(operator, pattern, config, primals, tangents):
    params, values, shift, initial = primals
    out = _eigen_callback(pattern, config, values, shift, initial)
    value, right, left = out[0], out[1], out[2]
    _, image = jax.jvp(lambda p: operator(p, right), (params,), (tangents[0],))
    tangent = jnp.vdot(left, image) / jnp.vdot(left, right)
    zero = jnp.zeros((), out[3].dtype)
    return out, (
        tangent.astype(value.dtype),
        jnp.zeros_like(right),
        jnp.zeros_like(left),
        zero,
        zero,
    )


def sparse_eigenvalue(
    operator: Callable[[Any, jax.Array], jax.Array],
    params: Any,
    pattern: CsrPattern,
    values: jax.Array,
    shift: complex | jax.Array,
    *,
    initial: jax.Array | None = None,
    candidates: int = 6,
    select: str = "max_real",
    tolerance: float = 1.0e-12,
    residual_tolerance: float = 1.0e-8,
    maxiter: int | None = None,
    options: HostFactorOptions | None = None,
) -> SparseEigenvalue:
    """An eigenvalue of ``A(params)`` from one shifted host factorization.

    Args:
        operator: ``operator(params, x) -> A(params) @ x`` for ``x`` of shape
            ``(n,)``, traceable in ``params``.
        params: pytree the eigenvalue is differentiated with respect to.
        pattern: square pattern of ``A``, storing every diagonal entry.
        values: CSR values of ``A(params)`` (used only for the factorization;
            they are not differentiated -- pass the assembly of ``operator`` at
            ``params``, for instance via :func:`csr_data_from_products`).
        shift: factorization shift ``sigma``; ``A - sigma I`` is factored once.
        initial: Arnoldi starting vector (default: all ones).
        candidates: eigenvalues nearest ``shift`` computed by shift-invert Arnoldi.
        select: ``"max_real"`` (largest real part among converged candidates,
            e.g. a growth rate) or ``"nearest"`` (closest to ``shift``).
        tolerance: ARPACK tolerance.
        residual_tolerance: certificate for the right and left pairs
            ``||A v - lambda v|| / max(||A v||, |lambda| ||v||)``; the value is
            NaN if either misses it.
        maxiter: ARPACK iteration cap.
        options: host factorization backend, :class:`HostFactorOptions`.

    Returns:
        :class:`SparseEigenvalue`. The left vector solves ``A^H y = conj(lambda) y``
        with the conjugate-transposed solves of the same factorization. The
        derivative of ``value`` is ``y^H (d_params A) x / (y^H x)`` from one
        ``jax.jvp`` of ``operator`` at the right eigenvector, so reverse mode
        costs one operator VJP. Valid for a simple eigenvalue.
    """
    if select not in {"max_real", "nearest"}:
        raise ValueError("select must be 'max_real' or 'nearest'")
    n = pattern.shape[0]
    if pattern.shape[1] != n or not 1 <= int(candidates) < n - 1:
        raise ValueError("need a square pattern and 1 <= candidates < n - 1")
    pattern.diagonal_positions()
    values = jnp.asarray(values)
    if values.shape != (pattern.nnz,):
        raise ValueError(f"values must have shape ({pattern.nnz},), got {values.shape}")
    dtype = jnp.result_type(values.dtype, jnp.complex64)
    values = jax.lax.stop_gradient(values.astype(dtype))
    shift = jax.lax.stop_gradient(jnp.asarray(shift, dtype=dtype))
    initial = jnp.ones((n,), dtype) if initial is None else jnp.asarray(initial, dtype)
    initial = jax.lax.stop_gradient(initial.reshape(n))
    config = _EigenConfig(
        candidates=int(candidates),
        select=select,
        tolerance=float(tolerance),
        residual_tolerance=float(residual_tolerance),
        maxiter=maxiter,
        options=HostFactorOptions() if options is None else options,
    )
    value, right, left, residual, left_residual = _eigenvalue(
        operator, pattern, config, params, values, shift, initial
    )
    return SparseEigenvalue(
        value,
        jax.lax.stop_gradient(right),
        jax.lax.stop_gradient(left),
        jax.lax.stop_gradient(residual),
        jax.lax.stop_gradient(left_residual),
    )
