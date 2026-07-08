"""Tests for solvax.banded: banded LU and periodic Woodbury vs scipy/dense."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from solvax import (
    banded_matvec,
    lu_factor_banded,
    lu_factor_banded_periodic,
    lu_solve_banded,
    lu_solve_banded_periodic,
)

jax.config.update("jax_enable_x64", True)


def dense_from_bands(bands, lower_bw, upper_bw):
    """Expand scipy-layout banded storage to a dense numpy matrix."""
    bands = np.asarray(bands)
    n_diags, n = bands.shape
    dense = np.zeros((n, n))
    for r in range(n_diags):
        for j in range(n):
            i = j + r - upper_bw
            if 0 <= i < n:
                dense[i, j] = bands[r, j]
    return dense


def make_banded(n, lower_bw, upper_bw, n_rhs=None, seed=0, dominance=4.0):
    """Random diagonally-dominant banded system + its dense form."""
    rng = np.random.default_rng(seed)
    n_diags = lower_bw + upper_bw + 1
    bands = rng.standard_normal((n_diags, n))
    bands[upper_bw] = dominance * n_diags + rng.random(n)
    for r in range(n_diags):  # zero out-of-range entries so all references agree
        for j in range(n):
            if not 0 <= j + r - upper_bw < n:
                bands[r, j] = 0.0
    dense = dense_from_bands(bands, lower_bw, upper_bw)
    shape = (n,) if n_rhs is None else (n, n_rhs)
    rhs = rng.standard_normal(shape)
    return jnp.asarray(bands), jnp.asarray(rhs), dense


def make_periodic(n, bw, n_rhs=None, seed=0, dominance=4.0):
    """Random dominant periodic banded system: core bands, corners, dense form."""
    rng = np.random.default_rng(seed)
    dense = np.zeros((n, n))
    for d in range(-bw, bw + 1):
        v = rng.standard_normal(n) if d else dominance * (2 * bw + 1) + rng.random(n)
        for i in range(n):
            dense[i, (i + d) % n] += v[i]
    bands = np.zeros((2 * bw + 1, n))
    for r in range(2 * bw + 1):
        for j in range(n):
            i = j + r - bw
            if 0 <= i < n:
                bands[r, j] = dense[i, j]
    corner_ul = dense[:bw, n - bw :].copy()
    corner_lr = dense[n - bw :, :bw].copy()
    shape = (n,) if n_rhs is None else (n, n_rhs)
    rhs = rng.standard_normal(shape)
    return map(jnp.asarray, (bands, corner_ul, corner_lr, rhs)), dense


@pytest.mark.parametrize("n_rhs", [None, 3])
@pytest.mark.parametrize("n,kl,ku", [(5, 1, 1), (20, 2, 1), (20, 1, 3), (50, 3, 2)])
def test_factor_solve_matches_scipy(n, kl, ku, n_rhs):
    bands, rhs, _ = make_banded(n, kl, ku, n_rhs)
    x = lu_solve_banded(lu_factor_banded(bands, kl, ku), rhs)
    x_ref = scipy.linalg.solve_banded((kl, ku), np.asarray(bands), np.asarray(rhs))
    assert np.allclose(np.asarray(x), x_ref, atol=1e-12)


def test_no_equilibration_matches_dense():
    bands, rhs, dense = make_banded(20, 2, 2, seed=1)
    factors = lu_factor_banded(bands, 2, 2, equilibrate=False)
    x = lu_solve_banded(factors, rhs)
    assert int(factors.n_clamped) == 0
    assert np.allclose(np.asarray(x), np.linalg.solve(dense, np.asarray(rhs)), atol=1e-12)


def test_factor_solve_reuse():
    bands, rhs, dense = make_banded(16, 2, 1, seed=2)
    factors = lu_factor_banded(bands, 2, 1)
    x1 = lu_solve_banded(factors, rhs)
    x2 = lu_solve_banded(factors, 2.0 * rhs)
    assert np.allclose(np.asarray(x2), 2.0 * np.asarray(x1), atol=1e-12)
    assert np.allclose(np.asarray(x1), np.linalg.solve(dense, np.asarray(rhs)), atol=1e-12)


def test_tiny_pivot_clamped():
    n = 8
    bands, rhs, dense = make_banded(n, 1, 1, seed=3)
    # A[0, 0] tiny: fatal for unprotected no-pivot LU, harmless for the matrix.
    bands = bands.at[1, 0].set(1e-16).at[0, 1].set(1.0).at[2, 0].set(1.0)
    dense[0, 0], dense[0, 1], dense[1, 0] = 1e-16, 1.0, 1.0
    factors = lu_factor_banded(bands, 1, 1)
    assert int(factors.n_clamped) > 0
    x = lu_solve_banded(factors, rhs)
    x_ref = np.linalg.solve(dense, np.asarray(rhs))
    assert np.allclose(np.asarray(x), x_ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n,kl,ku", [(6, 1, 1), (15, 2, 3), (12, 3, 0)])
def test_banded_matvec_matches_dense(n, kl, ku):
    bands, _, dense = make_banded(n, kl, ku, seed=4)
    rng = np.random.default_rng(5)
    for shape in [(n,), (n, 2)]:
        v = rng.standard_normal(shape)
        y = banded_matvec(bands, kl, ku, jnp.asarray(v))
        assert np.allclose(np.asarray(y), dense @ v, atol=1e-12)


@pytest.mark.parametrize("n_rhs", [None, 4])
@pytest.mark.parametrize("n,bw", [(20, 1), (30, 2)])
def test_periodic_matches_dense(n, bw, n_rhs):
    (bands, corner_ul, corner_lr, rhs), dense = make_periodic(n, bw, n_rhs, seed=6)
    factors = lu_factor_banded_periodic(bands, bw, bw, corner_ul, corner_lr)
    x = lu_solve_banded_periodic(factors, rhs)
    x_ref = np.linalg.solve(dense, np.asarray(rhs))
    assert np.allclose(np.asarray(x), x_ref, atol=1e-10)


@pytest.mark.parametrize("n_rhs", [None, 3])
def test_periodic_advection_diffusion(n_rhs):
    # (-D d^2/dx^2 + a d/dx + c) u on a periodic grid, central differences.
    n, diff, a, c = 64, 0.05, 1.0, 1.0
    h = 2.0 * np.pi / n
    lo = -diff / h**2 - a / (2.0 * h)
    di = 2.0 * diff / h**2 + c
    up = -diff / h**2 + a / (2.0 * h)
    dense = np.zeros((n, n))
    for i in range(n):
        dense[i, (i - 1) % n] += lo
        dense[i, i] += di
        dense[i, (i + 1) % n] += up
    bands = np.zeros((3, n))
    bands[0, 1:], bands[1, :], bands[2, :-1] = up, di, lo
    corner_ul = np.array([[lo]])  # A[0, n-1]: sub-diagonal wrapping around
    corner_lr = np.array([[up]])  # A[n-1, 0]: super-diagonal wrapping around
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal((n,) if n_rhs is None else (n, n_rhs))

    factors = lu_factor_banded_periodic(
        jnp.asarray(bands), 1, 1, jnp.asarray(corner_ul), jnp.asarray(corner_lr)
    )
    x = lu_solve_banded_periodic(factors, jnp.asarray(rhs))
    assert np.allclose(np.asarray(x), np.linalg.solve(dense, rhs), atol=1e-10)


def test_jit_compose():
    bands, rhs, dense = make_banded(16, 1, 2, seed=8)
    solve = jax.jit(lambda ab, b: lu_solve_banded(lu_factor_banded(ab, 1, 2), b))
    assert np.allclose(
        np.asarray(solve(bands, rhs)), np.linalg.solve(dense, np.asarray(rhs)), atol=1e-12
    )


def test_vmap_over_batch():
    kl, ku = 2, 1
    systems = [make_banded(12, kl, ku, seed=s) for s in range(4)]
    bands_b = jnp.stack([s[0] for s in systems])
    rhs_b = jnp.stack([s[1] for s in systems])

    def solve(ab, b):
        return lu_solve_banded(lu_factor_banded(ab, kl, ku), b)

    x_batch = jax.vmap(solve)(bands_b, rhs_b)
    y_batch = jax.vmap(lambda ab, v: banded_matvec(ab, kl, ku, v))(bands_b, x_batch)
    for i, (bands, rhs, _) in enumerate(systems):
        assert np.allclose(np.asarray(x_batch[i]), np.asarray(solve(bands, rhs)), atol=1e-12)
        assert np.allclose(np.asarray(y_batch[i]), np.asarray(rhs), atol=1e-10)


def test_vmap_periodic_over_batch():
    bw = 1
    systems = [tuple(make_periodic(10, bw, seed=s)[0]) for s in range(3)]
    stacked = [jnp.stack(arrs) for arrs in zip(*systems)]

    def solve(ab, cul, clr, b):
        return lu_solve_banded_periodic(lu_factor_banded_periodic(ab, bw, bw, cul, clr), b)

    x_batch = jax.vmap(solve)(*stacked)
    for i, (bands, cul, clr, rhs) in enumerate(systems):
        x_i = solve(bands, cul, clr, rhs)
        assert np.allclose(np.asarray(x_batch[i]), np.asarray(x_i), atol=1e-12)


def test_gradient_through_solve():
    bands, rhs, _ = make_banded(10, 1, 1, seed=9)

    def loss(ab):
        return jnp.sum(lu_solve_banded(lu_factor_banded(ab, 1, 1), rhs) ** 2)

    g = jax.grad(loss)(bands)
    # Central finite differences on in-band entries (diagonal, super, sub).
    eps = 1e-6
    for idx in [(1, 4), (0, 3), (2, 5)]:
        e = jnp.zeros_like(bands).at[idx].set(eps)
        fd = (loss(bands + e) - loss(bands - e)) / (2 * eps)
        assert np.isclose(float(g[idx]), float(fd), rtol=1e-5)
