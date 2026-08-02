"""Tests for bounded-memory sparse eigenpairs and implicit AD composition."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    SpluFactorization,
    eigenpair_reverse,
    sparse_eigenpairs,
    sparse_operator_matrix,
)

jax.config.update("jax_enable_x64", True)

sparse = pytest.importorskip("scipy.sparse")


def _matrix() -> np.ndarray:
    diagonal = np.asarray(
        [-2.0 - 0.4j, -1.0 + 0.2j, 0.25 + 0.35j, 0.7 - 0.1j]
    )
    return np.diag(diagonal) + np.diag([0.4, -0.2j, 0.3], 1)


def test_sparse_operator_matrix_preserves_shape_and_drops_small_entries() -> None:
    dense = jnp.asarray(_matrix()).at[3, 0].set(1.0e-15)
    prototype = jnp.ones((2, 2), dtype=jnp.complex128)

    matrix = sparse_operator_matrix(
        lambda state: (dense @ state.reshape(-1)).reshape(state.shape),
        prototype,
        batch_size=3,
        drop_tolerance=1.0e-14,
    )

    expected = np.asarray(dense).copy()
    expected[3, 0] = 0.0
    np.testing.assert_allclose(matrix.toarray(), expected)


def test_sparse_shift_invert_reuses_one_factorization_for_adjoint() -> None:
    matrix = sparse.csr_matrix(_matrix())
    shift = 0.4 + 0.2j
    factor = SpluFactorization(matrix - shift * sparse.eye(4))
    right = sparse_eigenpairs(
        matrix,
        candidates=2,
        shift=shift,
        initial=jnp.ones((4,)),
        factorization=factor,
        residual_tolerance=1e-11,
    )
    left = sparse_eigenpairs(
        matrix,
        candidates=2,
        shift=shift,
        factorization=factor,
        adjoint=True,
        residual_tolerance=1e-11,
    )

    assert np.all(np.asarray(right.converged))
    assert np.all(np.asarray(left.converged))
    target = complex(np.asarray(right.eigenvalues)[0])
    assert np.min(np.abs(np.asarray(left.eigenvalues) - np.conj(target))) < 1.0e-12
    unshifted = sparse_eigenpairs(matrix, candidates=1)
    assert float(jnp.real(unshifted.eigenvalues[0])) == pytest.approx(0.7)


def test_sparse_primal_composes_with_implicit_reverse_mode() -> None:
    cache: dict[str, object] = {}
    initial = jnp.ones((3,), dtype=jnp.complex128)

    def build(theta):
        diagonal = jnp.asarray((theta, -1.0, -2.0))
        return lambda vector: diagonal * vector

    def primal(_theta, apply, seed):
        matrix = sparse_operator_matrix(apply, seed, batch_size=2)
        factor = SpluFactorization(matrix - 0.2 * sparse.eye(3))
        modes = sparse_eigenpairs(
            matrix, candidates=1, shift=0.2, factorization=factor
        )
        cache.update(matrix=matrix, factor=factor)
        return modes.eigenvalues[0], modes.eigenvectors[0]

    def left(_theta, _apply, _seed, _value):
        modes = sparse_eigenpairs(
            cache["matrix"],
            candidates=1,
            shift=0.2,
            factorization=cache["factor"],
            adjoint=True,
        )
        return modes.eigenvectors[0]

    def objective(theta):
        value, _ = eigenpair_reverse(
            theta, build, initial, primal_solver=primal, left_solver=left
        )
        return jnp.real(value)

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(0.25))
    assert float(value) == pytest.approx(0.25)
    assert float(gradient) == pytest.approx(1.0)


def test_sparse_native_validation_and_trace_guard() -> None:
    prototype = jnp.ones((2,), dtype=jnp.complex128)

    def apply(vector):
        return vector

    with pytest.raises(RuntimeError, match="must not be called under jit"):
        jax.jit(lambda value: sparse_operator_matrix(apply, value))(prototype)
    with pytest.raises(ValueError, match="batch_size"):
        sparse_operator_matrix(apply, prototype, batch_size=0)
    with pytest.raises(TypeError, match="scipy sparse"):
        sparse_eigenpairs(np.eye(4))
    with pytest.raises(ValueError, match="candidates"):
        sparse_eigenpairs(sparse.eye(4), candidates=3)
    with pytest.raises(ValueError, match="requires shift"):
        sparse_eigenpairs(
            sparse.eye(4), candidates=1, factorization=SpluFactorization(sparse.eye(4))
        )
    with pytest.raises(ValueError, match="shapes"):
        sparse_eigenpairs(
            sparse.eye(4),
            candidates=1,
            shift=0.0,
            factorization=SpluFactorization(sparse.eye(3)),
        )
    with pytest.raises(ValueError, match="maxiter"):
        sparse_eigenpairs(sparse.eye(4), candidates=1, maxiter=0)
