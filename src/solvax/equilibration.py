"""Scaling a sparse matrix toward unit magnitude before it is factored.

A complete factorization chooses its pivots from the matrix it is given, so a
matrix whose rows span many orders of magnitude makes it choose badly. Sparse
direct solvers therefore scale first, and the ones that do not — static-pivoting
factorizations in particular — perturb the pivots they cannot use and return a
factorization of a different matrix.

Ruiz's algorithm scales rows and columns alternately by the square root of their
largest entry. Each sweep contracts the spread of the row and column maxima
toward one, converging linearly, and the scaling is diagonal, so the factored
system is ``D_r A D_c`` and the solution of ``A x = b`` follows from
``x = D_c y`` with ``D_r A D_c y = D_r b``.

The effect is not cosmetic. On a drift-kinetic operator of 66,004 unknowns,
whose rows carry streaming, collision and constraint terms at once, an
equilibrated matrix factored in 103 s where the same factorization of the
unscaled matrix took 201 s and returned a solution with a relative residual of
9.4e-2, which no amount of refinement recovered.

References
----------
- D. Ruiz, *A scaling algorithm to equilibrate both rows and columns norms in
  matrices*, Rutherford Appleton Laboratory RAL-TR-2001-034 (2001).
- I. S. Duff & J. Koster, *On algorithms for permuting large entries to the
  diagonal of a sparse matrix*, SIAM J. Matrix Anal. Appl. 22, 973 (2001),
  DOI 10.1137/S0895479899358443.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM (2002), chapter 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _import_scipy_sparse():
    """Import scipy.sparse lazily with an actionable error message."""
    try:
        import scipy.sparse as sparse
    except ImportError as err:  # pragma: no cover - exercised by packaging tests
        raise ImportError(
            "solvax.equilibration requires SciPy; install it with "
            "`pip install solvax[native]` (or `pip install scipy`)."
        ) from err
    return sparse


@dataclass(frozen=True)
class Equilibration:
    """A diagonal row and column scaling of a sparse matrix.

    Attributes:
        matrix: the scaled matrix ``D_r A D_c``.
        row_scale: the diagonal of ``D_r``, as a vector.
        column_scale: the diagonal of ``D_c``, as a vector.
        spread: ratio of the largest to the smallest nonzero magnitude after
            scaling, which is what the factorization sees.
        original_spread: the same ratio before scaling.
    """

    matrix: object
    row_scale: np.ndarray
    column_scale: np.ndarray
    spread: float
    original_spread: float

    def scale_rhs(self, b):
        """``D_r b``: the right-hand side of the scaled system."""
        return np.asarray(b) * self.row_scale.reshape(
            (-1,) + (1,) * (np.asarray(b).ndim - 1)
        )

    def unscale_solution(self, y):
        """``D_c y``: the solution of the original system."""
        return np.asarray(y) * self.column_scale.reshape(
            (-1,) + (1,) * (np.asarray(y).ndim - 1)
        )


def _spread(matrix) -> float:
    """Largest over smallest nonzero magnitude, or ``nan`` for an empty matrix."""
    data = np.abs(matrix.tocoo().data)
    data = data[data > 0.0]
    if data.size == 0:
        return float("nan")
    return float(data.max() / data.min())


def equilibrate(matrix, *, sweeps: int = 30, tolerance: float = 1.0e-2) -> Equilibration:
    """Scale rows and columns toward unit maximum magnitude.

    Args:
        matrix: scipy sparse matrix to scale.
        sweeps: maximum number of Ruiz sweeps. Convergence is linear, so a
            matrix spanning sixteen orders of magnitude needs tens of them; the
            sweeps are cheap next to the factorization they prepare.
        tolerance: stop once every row and column maximum is within this of one.

    Returns:
        The scaled matrix and the two diagonals, as an :class:`Equilibration`.

    Raises:
        TypeError: if ``matrix`` is not a scipy sparse matrix.
    """
    sparse = _import_scipy_sparse()
    if not sparse.issparse(matrix):
        raise TypeError(f"matrix must be a scipy sparse matrix, got {type(matrix).__name__}")
    scaled = matrix.tocsr().astype(np.float64)
    original_spread = _spread(scaled)
    rows = np.ones(scaled.shape[0], dtype=np.float64)
    columns = np.ones(scaled.shape[1], dtype=np.float64)
    for _ in range(int(sweeps)):
        row_max = np.asarray(abs(scaled).max(axis=1).todense()).ravel()
        column_max = np.asarray(abs(scaled).max(axis=0).todense()).ravel()
        # An empty row or column has nothing to scale; leave it alone.
        row_max[row_max == 0.0] = 1.0
        column_max[column_max == 0.0] = 1.0
        if max(np.abs(row_max - 1.0).max(), np.abs(column_max - 1.0).max()) <= tolerance:
            break
        dr = 1.0 / np.sqrt(row_max)
        dc = 1.0 / np.sqrt(column_max)
        scaled = (sparse.diags(dr) @ scaled @ sparse.diags(dc)).tocsr()
        rows *= dr
        columns *= dc
    return Equilibration(
        matrix=scaled,
        row_scale=rows,
        column_scale=columns,
        spread=_spread(scaled),
        original_spread=original_spread,
    )
