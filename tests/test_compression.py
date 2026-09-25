"""Recovering a sparse matrix from products, given its pattern.

The operator is matrix-free; the factorization needs a matrix. Sampling every
column pays one product per column, which is why it is confined to small
problems. These tests pin the alternative: one product per group of columns
that share no row.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax.compression import column_groups, matrix_from_products, verify_products

jax.config.update("jax_enable_x64", True)
scipy_sparse = pytest.importorskip("scipy.sparse")


def _random_sparse(rows: int, columns: int, density: float, seed: int):
    rng = np.random.default_rng(seed)
    matrix = scipy_sparse.random(
        rows, columns, density=density, format="csr", random_state=rng
    )
    matrix.data = rng.standard_normal(matrix.nnz) + 1.0  # no accidental zeros
    return matrix


def _apply_of(matrix):
    def apply(vector):
        return jnp.asarray(matrix @ np.asarray(vector))

    return apply


def test_a_group_never_shares_a_row_between_its_columns() -> None:
    """The property the whole method rests on: one product, one entry per row."""
    matrix = _random_sparse(120, 120, 0.04, seed=0)
    csc = matrix.tocsc()
    groups = column_groups(matrix)
    covered = np.sort(np.concatenate(groups))
    np.testing.assert_array_equal(covered, np.arange(matrix.shape[1]))
    for group in groups:
        rows = np.concatenate(
            [csc.indices[csc.indptr[j] : csc.indptr[j + 1]] for j in group]
        )
        assert np.unique(rows).size == rows.size


def test_recovery_reproduces_the_matrix_entry_for_entry() -> None:
    matrix = _random_sparse(150, 150, 0.03, seed=1)
    recovered = matrix_from_products(_apply_of(matrix), matrix)
    difference = (recovered - matrix).tocoo()
    assert np.max(np.abs(difference.data), initial=0.0) == 0.0
    assert recovered.nnz == matrix.nnz


def test_it_costs_one_product_per_group_not_one_per_column() -> None:
    """The point of the method, and the bound it cannot beat: the densest row."""
    matrix = _random_sparse(200, 200, 0.02, seed=2)
    calls = []
    apply = _apply_of(matrix)

    def counted(vector):
        calls.append(1)
        return apply(vector)

    groups = column_groups(matrix)
    matrix_from_products(counted, matrix, groups=groups)
    assert len(calls) == len(groups)
    densest_row = int(np.diff(matrix.tocsr().indptr).max())
    assert densest_row <= len(groups) < matrix.shape[1]


def test_structured_groups_are_accepted_without_colouring() -> None:
    """A caller whose operator has a stencil knows its groups already."""
    stride = 5
    rows = columns = 40
    diagonals = [np.ones(columns - abs(k)) * (k + 2.0) for k in (-1, 0, 1)]
    matrix = scipy_sparse.diags(diagonals, (-1, 0, 1), format="csr")
    groups = [np.arange(start, columns, stride) for start in range(stride)]
    recovered = matrix_from_products(_apply_of(matrix), matrix, groups=groups)
    assert (recovered - matrix).nnz == 0
    assert matrix.shape == (rows, columns)


def test_a_group_that_shares_a_row_is_refused() -> None:
    """Two columns of one group meeting in a row add there; neither survives."""
    matrix = scipy_sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]]))
    with pytest.raises(ValueError, match="shares a row"):
        matrix_from_products(_apply_of(matrix), matrix, groups=[np.array([0, 1])])


def test_a_pattern_missing_an_entry_is_caught_by_the_products() -> None:
    """The failure that has no other symptom: the factorization of a matrix
    that is not the operator succeeds, and answers the wrong question."""
    matrix = _random_sparse(80, 80, 0.05, seed=3)
    pattern = matrix.copy().tolil()
    row, column = matrix.tocoo().row[0], matrix.tocoo().col[0]
    pattern[row, column] = 0.0
    pattern = pattern.tocsr()
    pattern.eliminate_zeros()
    recovered = matrix_from_products(_apply_of(matrix), pattern)
    assert verify_products(recovered, _apply_of(matrix)) > 1e-8
    exact = matrix_from_products(_apply_of(matrix), matrix)
    assert verify_products(exact, _apply_of(matrix)) < 1e-12


def _complex_banded(n: int = 40):
    rng = np.random.default_rng(3)
    real = scipy_sparse.random(n, n, density=0.08, random_state=1, format="csr")
    real = real + scipy_sparse.eye(n, format="csr")
    matrix = real.astype(np.complex128)
    matrix.data = matrix.data + 1j * rng.standard_normal(matrix.nnz)
    return matrix


def test_a_complex_operator_is_recovered_with_its_imaginary_part() -> None:
    matrix = _complex_banded()
    dense = jnp.asarray(matrix.toarray())
    recovered = matrix_from_products(lambda v: dense @ v, abs(matrix))
    assert np.iscomplexobj(recovered.data)
    np.testing.assert_allclose(recovered.toarray(), matrix.toarray(), rtol=0.0, atol=1e-14)
    assert verify_products(recovered, lambda v: dense @ v) < 1e-14
    assert verify_products(recovered, lambda v: dense @ v, dtype=np.complex128) < 1e-14


def test_verification_catches_a_wrong_imaginary_part() -> None:
    matrix = _complex_banded()
    dense = jnp.asarray(matrix.toarray())
    corrupted = matrix.copy()
    corrupted.data = corrupted.data.real + 0j
    assert verify_products(corrupted, lambda v: dense @ v) > 1e-2
    assert verify_products(corrupted.real, lambda v: dense @ v) > 1e-2


def test_recovery_widens_when_a_later_product_is_complex() -> None:
    matrix = _complex_banded()
    dense = matrix.toarray()
    groups = column_groups(abs(matrix))
    first = set(groups[0].tolist())

    def apply(v):
        # The first group's columns are real; complex entries appear later.
        product = np.asarray(dense @ np.asarray(v))
        return jnp.asarray(product.real if set(np.flatnonzero(np.asarray(v))) == first else product)

    recovered = matrix_from_products(apply, abs(matrix), groups=groups)
    assert np.iscomplexobj(recovered.data)
    expected = matrix.toarray()
    expected[:, groups[0]] = expected[:, groups[0]].real
    np.testing.assert_allclose(recovered.toarray(), expected, rtol=0.0, atol=1e-14)


def test_an_empty_pattern_recovers_an_empty_matrix() -> None:
    empty = scipy_sparse.csr_matrix((3, 0))
    recovered = matrix_from_products(lambda v: jnp.zeros(3), empty, groups=[])
    assert recovered.shape == (3, 0) and recovered.nnz == 0
