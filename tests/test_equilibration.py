"""Scaling a matrix before it is factored.

A factorization chooses its pivots from the matrix it is handed, so one whose
rows span many orders of magnitude makes it choose badly. These tests pin what
the scaling does and, more importantly, that it changes nothing about the
system's solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from solvax.equilibration import equilibrate

scipy_sparse = pytest.importorskip("scipy.sparse")
scipy_linalg = pytest.importorskip("scipy.sparse.linalg")


def _badly_scaled(n: int = 60, seed: int = 0):
    """A matrix whose rows carry wildly different magnitudes, as an operator
    mixing streaming, collision and constraint terms does."""
    rng = np.random.default_rng(seed)
    base = scipy_sparse.random(n, n, density=0.1, format="csr", random_state=rng)
    base.data = rng.standard_normal(base.nnz) + 1.0
    base = base + scipy_sparse.identity(n, format="csr")
    scales = 10.0 ** rng.integers(-8, 8, size=n)
    return (scipy_sparse.diags(scales) @ base).tocsr()


def test_it_pulls_the_magnitudes_together() -> None:
    matrix = _badly_scaled()
    result = equilibrate(matrix)
    assert result.spread < result.original_spread
    row_max = np.asarray(abs(result.matrix).max(axis=1).todense()).ravel()
    column_max = np.asarray(abs(result.matrix).max(axis=0).todense()).ravel()
    np.testing.assert_allclose(row_max, 1.0, atol=1e-2)
    np.testing.assert_allclose(column_max, 1.0, atol=1e-2)


def test_the_scaled_system_has_the_same_solution() -> None:
    """The whole point: scaling changes the arithmetic, not the answer."""
    matrix = _badly_scaled(seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(matrix.shape[0])
    b = matrix @ x
    result = equilibrate(matrix)
    y = scipy_linalg.spsolve(result.matrix.tocsc(), result.scale_rhs(b))
    recovered = result.unscale_solution(y)
    assert np.linalg.norm(recovered - x) / np.linalg.norm(x) < 1e-8


def test_the_scaling_is_diagonal_and_exact() -> None:
    matrix = _badly_scaled(seed=3)
    result = equilibrate(matrix)
    expected = (
        scipy_sparse.diags(result.row_scale) @ matrix @ scipy_sparse.diags(result.column_scale)
    ).tocsr()
    difference = (result.matrix - expected)
    difference.eliminate_zeros()
    assert np.max(np.abs(difference.data), initial=0.0) < 1e-12


def test_an_already_scaled_matrix_is_left_alone() -> None:
    matrix = scipy_sparse.identity(20, format="csr") * 1.0
    result = equilibrate(matrix)
    np.testing.assert_allclose(result.row_scale, 1.0)
    np.testing.assert_allclose(result.column_scale, 1.0)


def test_an_empty_row_does_not_divide_by_zero() -> None:
    """A structurally singular matrix must not produce infinities here; the
    factorization is where that is diagnosed."""
    matrix = scipy_sparse.csr_matrix(np.diag([1.0, 0.0, 3.0]))
    result = equilibrate(matrix)
    assert np.all(np.isfinite(result.row_scale))
    assert np.all(np.isfinite(result.column_scale))


def test_it_refuses_a_dense_array() -> None:
    with pytest.raises(TypeError, match="scipy sparse"):
        equilibrate(np.eye(3))
