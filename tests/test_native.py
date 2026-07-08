"""Tests for solvax.native: host-side SuperLU bridge."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import SpluFactorization, splu_solve

jax.config.update("jax_enable_x64", True)

scipy_sparse = pytest.importorskip("scipy.sparse")


def make_sparse_system(n=80, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = scipy_sparse.random(
        n, n, density=density, random_state=rng, format="csr"
    ) + 10.0 * scipy_sparse.eye(n, format="csr")
    b = rng.standard_normal(n)
    return a.tocsr(), jnp.asarray(b)


def test_splu_solve_matches_dense():
    a, b = make_sparse_system()
    x = splu_solve(a, b)
    assert isinstance(x, jax.Array)
    x_dense = np.linalg.solve(a.toarray(), np.asarray(b))
    assert np.allclose(np.asarray(x), x_dense, atol=1e-10)


def test_splu_solve_accepts_csc_and_multiple_rhs():
    a, _ = make_sparse_system(seed=1)
    rng = np.random.default_rng(2)
    rhs = jnp.asarray(rng.standard_normal((a.shape[0], 3)))
    x = splu_solve(a.tocsc(), rhs)
    assert np.allclose(a.toarray() @ np.asarray(x), np.asarray(rhs), atol=1e-10)


def test_factorization_reuse_identical():
    a, b = make_sparse_system(seed=3)
    lu = SpluFactorization(a)
    x_once = splu_solve(a, b)
    x1 = lu.solve(b)
    x2 = lu.solve(b)
    assert np.array_equal(np.asarray(x1), np.asarray(x2))
    assert np.array_equal(np.asarray(x1), np.asarray(x_once))
    # A second right-hand side reuses the same factors.
    x3 = lu.solve(2.0 * b)
    assert np.allclose(np.asarray(x3), 2.0 * np.asarray(x1), atol=1e-10)


def test_splu_solve_raises_under_jit():
    a, b = make_sparse_system(seed=4)

    with pytest.raises(RuntimeError, match="must not be called under jit"):
        jax.jit(lambda v: splu_solve(a, v))(b)

    lu = SpluFactorization(a)
    with pytest.raises(RuntimeError, match="must not be called under jit"):
        jax.jit(lambda v: lu.solve(v))(b)


def test_rejects_dense_input():
    _, b = make_sparse_system(seed=5)
    with pytest.raises(TypeError, match="scipy sparse"):
        splu_solve(np.eye(b.shape[0]), b)
