"""Fast core-solver checks for the `gpu` CI job (they also pass on CPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import block_thomas, lu_factor_banded, lu_solve_banded, pcg, tridiagonal_solve

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.gpu


def _on_default_device(x):
    return {d.platform for d in x.devices()} == {jax.default_backend()}


def test_tridiagonal_solve_matches_dense():
    n = 64
    rng = np.random.default_rng(0)
    lower, upper = rng.standard_normal(n), rng.standard_normal(n)
    diag = 5.0 + rng.random(n)
    rhs = rng.standard_normal((n, 3))
    x = tridiagonal_solve(*map(jnp.asarray, (lower, diag, upper, rhs)))
    dense = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    assert _on_default_device(x)
    assert np.allclose(np.asarray(x), np.linalg.solve(dense, rhs), atol=1e-10)


def test_block_thomas_matches_dense():
    n_blocks, m = 16, 4
    rng = np.random.default_rng(1)
    lower, upper = rng.standard_normal((2, n_blocks, m, m))
    diag = rng.standard_normal((n_blocks, m, m)) + 4.0 * m * np.eye(m)
    rhs = rng.standard_normal((n_blocks, m))
    dense = np.zeros((n_blocks * m, n_blocks * m))
    for k in range(n_blocks):
        s = slice(k * m, (k + 1) * m)
        dense[s, s] = diag[k]
        if k > 0:
            dense[s, (k - 1) * m : k * m] = lower[k]
        if k < n_blocks - 1:
            dense[s, (k + 1) * m : (k + 2) * m] = upper[k]
    x = block_thomas(*map(jnp.asarray, (lower, diag, upper, rhs)))
    assert _on_default_device(x)
    assert np.allclose(np.asarray(x).ravel(), np.linalg.solve(dense, rhs.ravel()), atol=1e-10)


def test_banded_lu_solves_pentadiagonal_system():
    n, lo, up = 48, 2, 2
    rng = np.random.default_rng(2)
    bands = rng.standard_normal((lo + up + 1, n))
    bands[up] = 20.0 + rng.random(n)
    dense = np.zeros((n, n))
    for r in range(lo + up + 1):
        for j in range(n):
            i = j + r - up
            if 0 <= i < n:
                dense[i, j] = bands[r, j]
            else:
                bands[r, j] = 0.0
    rhs = rng.standard_normal(n)
    x = lu_solve_banded(lu_factor_banded(jnp.asarray(bands), lo, up), jnp.asarray(rhs))
    assert _on_default_device(x)
    assert np.allclose(np.asarray(x), np.linalg.solve(dense, rhs), atol=1e-10)


def test_pcg_jit_solves_spd_system():
    n = 100
    rng = np.random.default_rng(3)
    q = rng.standard_normal((n, n))
    matrix = jnp.asarray(q @ q.T / n + np.eye(n))
    rhs = jnp.asarray(rng.standard_normal(n))
    solve = jax.jit(lambda b: pcg(lambda v: matrix @ v, b, rtol=1e-12, max_steps=300))
    solution = solve(rhs)
    assert bool(solution.converged)
    assert _on_default_device(solution.x)
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    assert np.allclose(np.asarray(solution.x), expected, atol=1e-8)
