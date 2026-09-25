"""Recovering a sparse matrix from products with it, given its pattern.

A matrix-free operator can always be read out one column at a time: apply it to
each unit vector and keep the result. That costs ``n`` products, which is what
:func:`solvax.native_eigen.sparse_operator_matrix` does and why it is confined
to small problems.

When the sparsity pattern is known the cost collapses. Partition the columns
into groups such that no two columns of a group share a row. One product with
the sum of a group's unit vectors then carries every entry of every column in
it, because in each row at most one of those columns contributes: the entry
simply appears in the result at that row. This is Curtis, Powell and Reid's
seeding, and the number of products is the number of groups rather than the
number of columns.

Finding the fewest groups is a distance-2 colouring of the column intersection
graph, which is NP-hard; the greedy largest-first pass in :func:`column_groups`
is the standard practical choice. It never needs fewer groups than the densest
row has entries, and on structured operators it usually lands near that bound.

The pattern must be a **superset** of the true nonzeros. Entries outside it are
not recovered, and they corrupt the entries that are: a stray coupling lands in
a row where another column of the same group already contributes, and the two
are added. :func:`verify_products` exists to catch exactly that, by comparing
the recovered matrix against the operator on random vectors.

References
----------
- A. R. Curtis, M. J. D. Powell & J. K. Reid, *On the Estimation of Sparse
  Jacobian Matrices*, J. Inst. Maths Applics 13, 117 (1974),
  DOI 10.1093/imamat/13.1.117.
- T. F. Coleman & J. J. Moré, *Estimation of Sparse Jacobian Matrices and Graph
  Coloring Problems*, SIAM J. Numer. Anal. 20(1), 187 (1983),
  DOI 10.1137/0720013.
- A. H. Gebremedhin, F. Manne & A. Pothen, *What Color Is Your Jacobian? Graph
  Coloring for Computing Derivatives*, SIAM Review 47(4), 629 (2005),
  DOI 10.1137/S0036144504444711.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def _import_scipy_sparse():
    """Import scipy.sparse lazily with an actionable error message."""
    try:
        import scipy.sparse as sparse
    except ImportError as err:  # pragma: no cover - exercised by packaging tests
        raise ImportError(
            "solvax.compression requires SciPy; install it with "
            "`pip install solvax[native]` (or `pip install scipy`)."
        ) from err
    return sparse


def column_groups(pattern) -> list[np.ndarray]:
    """Group columns so that no two in a group share a row.

    Greedy largest-first distance-2 colouring of the column intersection graph:
    columns are visited densest first, and each takes the lowest group not used
    by a column it shares a row with. The intersection graph ``|P|^T |P|`` is
    formed once by a compiled sparse product, so the Python loop visits each
    column's neighbours once; its memory is the graph's edge count (each column
    times the columns it shares a row with). For a structured operator whose
    groups follow from its stencil, pass them to :func:`matrix_from_products`
    directly instead of paying for this.

    Args:
        pattern: scipy sparse matrix whose nonzeros are the entries to recover.

    Returns:
        One index array per group, together covering every column exactly once.
    """
    sparse = _import_scipy_sparse()
    if not sparse.issparse(pattern):
        raise TypeError(f"pattern must be a scipy sparse matrix, got {type(pattern).__name__}")
    csc = pattern.tocsc()
    n_columns = csc.shape[1]
    structure = sparse.csc_matrix(
        (np.ones(csc.nnz, dtype=np.int32), csc.indices, csc.indptr), shape=csc.shape
    )
    # Column j's neighbours: every column sharing a row with it (itself included,
    # harmlessly, since it is still uncoloured when visited).
    graph = (structure.T @ structure).tocsr()
    group_of_column = np.full(n_columns, -1, dtype=np.int64)
    # Timestamps, so the forbidden set is cleared in O(1) per column; a column
    # with d neighbours always finds a free group among the first d + 1.
    forbidden = np.full(n_columns + 1, -1, dtype=np.int64)
    counts = np.diff(csc.indptr)
    for column in np.argsort(-counts, kind="stable"):
        neighbours = graph.indices[graph.indptr[column] : graph.indptr[column + 1]]
        used = group_of_column[neighbours]
        forbidden[used[used >= 0]] = column
        window = forbidden[: neighbours.size + 1]
        group_of_column[column] = int(np.argmax(window != column))
    order = np.argsort(group_of_column, kind="stable")
    boundaries = np.flatnonzero(np.diff(group_of_column[order])) + 1
    return [np.asarray(part) for part in np.split(order, boundaries)]


def matrix_from_products(
    apply: Callable[[jax.Array], jax.Array],
    pattern,
    *,
    groups: Sequence[np.ndarray] | None = None,
    dtype=None,
):
    """Recover a sparse matrix from one product per group of columns.

    Args:
        apply: the operator, mapping a vector of length ``n`` to length ``m``.
            It is called once per group, on the sum of that group's unit
            vectors.
        pattern: scipy sparse matrix holding the entries to recover; it must be
            a superset of the operator's nonzeros.
        groups: column groups from :func:`column_groups`, or any partition with
            the same property. Computed from ``pattern`` when omitted.
        dtype: dtype of the probe vectors; the pattern's dtype by default.

    Returns:
        A scipy CSR matrix carrying the recovered values on ``pattern``.

    Raises:
        ValueError: if a group contains two columns that share a row, which
            would add their entries together instead of recovering them.
    """
    sparse = _import_scipy_sparse()
    if not sparse.issparse(pattern):
        raise TypeError(f"pattern must be a scipy sparse matrix, got {type(pattern).__name__}")
    csc = pattern.tocsc()
    rows, columns = csc.shape
    if groups is None:
        groups = column_groups(csc)
    probe_dtype = np.float64 if dtype is None else dtype
    values = np.zeros(csc.nnz, dtype=probe_dtype)
    for group in groups:
        group = np.asarray(group, dtype=np.int64)
        seed = np.zeros(columns, dtype=probe_dtype)
        seed[group] = 1.0
        result = np.asarray(apply(jnp.asarray(seed)), dtype=probe_dtype).reshape(-1)
        if result.size != rows:
            raise ValueError(
                f"apply returned {result.size} values for a {rows}-row pattern"
            )
        # Every row of the group's columns must be distinct, or two entries
        # arrive summed in one row and neither is recoverable.
        touched = np.concatenate(
            [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in group]
        ) if group.size else np.empty(0, dtype=np.int64)
        if np.unique(touched).size != touched.size:
            raise ValueError(
                "a column group shares a row between two of its columns; the "
                "groups must come from column_groups(pattern) or satisfy the "
                "same property"
            )
        for j in group:
            span = slice(csc.indptr[j], csc.indptr[j + 1])
            values[span] = result[csc.indices[span]]
    recovered = sparse.csc_matrix((values, csc.indices, csc.indptr), shape=csc.shape)
    return recovered.tocsr()


def verify_products(
    matrix,
    apply: Callable[[jax.Array], jax.Array],
    *,
    samples: int = 3,
    seed: int = 0,
    dtype=None,
) -> float:
    """Largest relative difference between ``matrix @ v`` and ``apply(v)``.

    The recovery is only as good as the pattern it was given, and a missing
    coupling shows up nowhere else: the factorization of a matrix that is not
    the operator succeeds and returns the wrong answer. Random vectors catch
    it, because a dropped entry moves almost every product.

    Args:
        matrix: the recovered matrix.
        apply: the operator it should reproduce.
        samples: how many random vectors to compare on.
        seed: seed of the random vectors.
        dtype: dtype of the random vectors; float64 by default.

    Returns:
        The largest relative difference over the samples, in the 2-norm.
    """
    rng = np.random.default_rng(seed)
    probe_dtype = np.float64 if dtype is None else dtype
    worst = 0.0
    for _ in range(int(samples)):
        v = rng.standard_normal(matrix.shape[1]).astype(probe_dtype)
        reference = np.asarray(apply(jnp.asarray(v)), dtype=np.float64).reshape(-1)
        recovered = np.asarray(matrix @ v, dtype=np.float64).reshape(-1)
        scale = float(np.linalg.norm(reference))
        difference = float(np.linalg.norm(recovered - reference))
        worst = max(worst, difference / scale if scale > 0.0 else difference)
    return worst
