"""Tests for solvax.direct: block-tridiagonal elimination vs dense reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    block_thomas,
    block_thomas_factor,
    block_thomas_solve,
    block_thomas_truncated,
)

jax.config.update("jax_enable_x64", True)


def make_system(n_blocks, m, n_rhs=None, seed=0, dominance=4.0):
    """Random well-conditioned block-tridiagonal system + its dense form."""
    rng = np.random.default_rng(seed)
    lower = rng.standard_normal((n_blocks, m, m))
    diag = rng.standard_normal((n_blocks, m, m)) + dominance * m * np.eye(m)
    upper = rng.standard_normal((n_blocks, m, m))
    shape = (n_blocks, m) if n_rhs is None else (n_blocks, m, n_rhs)
    rhs = rng.standard_normal(shape)

    dense = np.zeros((n_blocks * m, n_blocks * m))
    for k in range(n_blocks):
        s = slice(k * m, (k + 1) * m)
        dense[s, s] = diag[k]
        if k > 0:
            dense[s, slice((k - 1) * m, k * m)] = lower[k]
        if k < n_blocks - 1:
            dense[s, slice((k + 1) * m, (k + 2) * m)] = upper[k]
    return map(jnp.asarray, (lower, diag, upper, rhs)), dense


@pytest.mark.parametrize("n_rhs", [None, 3])
@pytest.mark.parametrize("n_blocks,m", [(4, 3), (12, 5), (40, 2)])
def test_block_thomas_matches_dense(n_blocks, m, n_rhs):
    (lower, diag, upper, rhs), dense = make_system(n_blocks, m, n_rhs)
    x = block_thomas(lower, diag, upper, rhs)
    x_dense = np.linalg.solve(dense, np.asarray(rhs).reshape(n_blocks * m, -1))
    assert np.allclose(np.asarray(x).reshape(n_blocks * m, -1), x_dense, atol=1e-12)


def test_factor_solve_reuse():
    (lower, diag, upper, rhs), dense = make_system(8, 4, seed=1)
    factors = block_thomas_factor(lower, diag, upper)
    x1 = block_thomas_solve(factors, rhs)
    x2 = block_thomas_solve(factors, 2.0 * rhs)
    assert np.allclose(np.asarray(x2), 2.0 * np.asarray(x1), atol=1e-12)
    x_dense = np.linalg.solve(dense, np.asarray(rhs).reshape(-1))
    assert np.allclose(np.asarray(x1).reshape(-1), x_dense, atol=1e-12)


@pytest.mark.parametrize("keep", [1, 3])
def test_truncated_matches_full(keep):
    n_blocks, m = 16, 4
    (lower, diag, upper, rhs), _ = make_system(n_blocks, m, seed=2)
    # Zero the rhs above the kept blocks, as the truncated solve assumes.
    rhs = rhs.at[keep:].set(0.0)
    x_full = block_thomas(lower, diag, upper, rhs)
    x_trunc = block_thomas_truncated(lower, diag, upper, rhs[:keep], keep)
    assert np.allclose(np.asarray(x_trunc), np.asarray(x_full[:keep]), atol=1e-12)


def test_vmap_over_batch():
    def solve_one(seed):
        (lower, diag, upper, rhs), _ = make_system(6, 3, seed=seed)
        return lower, diag, upper, rhs

    systems = [solve_one(s) for s in range(4)]
    stacked = [jnp.stack(arrs) for arrs in zip(*systems)]
    x_batch = jax.vmap(block_thomas)(*stacked)
    for i, (lower, diag, upper, rhs) in enumerate(systems):
        x_i = block_thomas(lower, diag, upper, rhs)
        assert np.allclose(np.asarray(x_batch[i]), np.asarray(x_i), atol=1e-12)


def test_gradient_through_solve():
    (lower, diag, upper, rhs), _ = make_system(5, 3, seed=3)

    def loss(d):
        return jnp.sum(block_thomas(lower, d, upper, rhs) ** 2)

    g = jax.grad(loss)(diag)
    # Central finite difference on one entry.
    eps = 1e-6
    e = jnp.zeros_like(diag).at[2, 1, 1].set(eps)
    fd = (loss(diag + e) - loss(diag - e)) / (2 * eps)
    assert np.isclose(float(g[2, 1, 1]), float(fd), rtol=1e-5)
