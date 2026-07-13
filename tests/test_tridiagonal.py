"""Tests for solvax.tridiagonal: batched Thomas / fused solve vs dense reference.

The Thomas kernel is ported verbatim from the parity-proven vmec_jax radial
preconditioner (``vmec_jax/core/preconditioner.py``); these tests pin it
against a dense reference, a hand-written numpy Thomas sweep, and the fused
``jax.lax.linalg`` backend.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import tridiagonal_solve

jax.config.update("jax_enable_x64", True)


def make_tridiag(n, columns=(), n_fields=None, seed=0, dominance=5.0):
    """Random diagonally-dominant tridiagonal system + a dense per-column form."""
    rng = np.random.default_rng(seed)
    sys_shape = (n, *columns)
    lower = rng.standard_normal(sys_shape)
    upper = rng.standard_normal(sys_shape)
    diag = dominance + rng.random(sys_shape)  # dominant -> well conditioned
    rhs_shape = sys_shape if n_fields is None else sys_shape + (n_fields,)
    rhs = rng.standard_normal(rhs_shape)
    return (
        jnp.asarray(lower),
        jnp.asarray(diag),
        jnp.asarray(upper),
        jnp.asarray(rhs),
    )


def dense_solve(lower, diag, upper, rhs):
    """Reference: build the dense matrix per column and solve with numpy."""
    lower = np.asarray(lower)
    diag = np.asarray(diag)
    upper = np.asarray(upper)
    rhs = np.asarray(rhs)
    n = diag.shape[0]
    columns = diag.shape[1:]
    out = np.zeros_like(rhs)
    for idx in np.ndindex(*columns):
        a = np.diag(diag[(slice(None), *idx)])
        a += np.diag(upper[(slice(None), *idx)][:-1], 1)
        a += np.diag(lower[(slice(None), *idx)][1:], -1)
        b = rhs[(slice(None), *idx)]
        out[(slice(None), *idx)] = np.linalg.solve(a, b.reshape(n, -1)).reshape(b.shape)
    return out


def numpy_thomas(lower, diag, upper, rhs):
    """A textbook scalar Thomas sweep (single system, single rhs)."""
    lower, diag, upper, rhs = map(np.asarray, (lower, diag, upper, rhs))
    n = diag.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for j in range(1, n):
        denom = diag[j] - lower[j] * cp[j - 1]
        cp[j] = upper[j] / denom
        dp[j] = (rhs[j] - lower[j] * dp[j - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for j in range(n - 2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j + 1]
    return x


@pytest.mark.parametrize("method", ["thomas", "lax", "auto"])
@pytest.mark.parametrize(
    "n,columns,n_fields",
    [(5, (), None), (10, (3,), None), (8, (2, 2), 2), (20, (4,), 3), (3, (), None)],
)
def test_matches_dense(method, n, columns, n_fields):
    lower, diag, upper, rhs = make_tridiag(n, columns, n_fields)
    x = tridiagonal_solve(lower, diag, upper, rhs, method=method)
    x_ref = dense_solve(lower, diag, upper, rhs)
    assert x.shape == rhs.shape
    assert np.allclose(np.asarray(x), x_ref, atol=1e-10)


def test_thomas_matches_textbook_reference():
    lower, diag, upper, rhs = make_tridiag(12, (), None, seed=3)
    x = tridiagonal_solve(lower, diag, upper, rhs, method="thomas")
    x_ref = numpy_thomas(lower, diag, upper, rhs)
    assert np.allclose(np.asarray(x), x_ref, atol=1e-13)


def test_auto_is_thomas_on_cpu_bit_identical():
    # On the CPU lowering platform, "auto" must select the bit-reproducible
    # Thomas path (this whole suite runs on CPU in CI).
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-only backend-selection identity contract")
    lower, diag, upper, rhs = make_tridiag(15, (3,), 2, seed=4)
    x_auto = tridiagonal_solve(lower, diag, upper, rhs, method="auto")
    x_thomas = tridiagonal_solve(lower, diag, upper, rhs, method="thomas")
    assert np.array_equal(np.asarray(x_auto), np.asarray(x_thomas))


def test_lax_matches_thomas():
    lower, diag, upper, rhs = make_tridiag(30, (5,), 4, seed=5)
    x_lax = tridiagonal_solve(lower, diag, upper, rhs, method="lax")
    x_thomas = tridiagonal_solve(lower, diag, upper, rhs, method="thomas")
    assert np.allclose(np.asarray(x_lax), np.asarray(x_thomas), atol=1e-10)


@pytest.mark.parametrize("method", ["thomas", "lax", "auto"])
def test_complex_rhs_promotes_real_bands_and_is_differentiable(method):
    lower, diag, upper, rhs = make_tridiag(12, (3,), 2, seed=12)
    rhs = rhs + 1j * jnp.roll(rhs, 1, axis=0)
    solve = jax.jit(
        lambda value: tridiagonal_solve(lower, diag, upper, value, method=method)
    )
    solved = solve(rhs)
    assert solved == pytest.approx(dense_solve(lower, diag, upper, rhs), abs=1.0e-10)

    value, gradient = jax.value_and_grad(
        lambda scale: jnp.real(jnp.vdot(solve(scale * rhs), solve(scale * rhs)))
    )(jnp.asarray(1.0))
    assert gradient == pytest.approx(2.0 * value, rel=1.0e-10)


@pytest.mark.parametrize("n", [1, 2])
def test_small_systems_use_thomas(n):
    # cuSPARSE needs n >= 3; auto/lax on tiny systems fall back to Thomas.
    lower, diag, upper, rhs = make_tridiag(n, (2,), None, seed=6)
    for method in ("auto", "lax", "thomas"):
        x = tridiagonal_solve(lower, diag, upper, rhs, method=method)
        assert np.allclose(np.asarray(x), dense_solve(lower, diag, upper, rhs), atol=1e-12)


def test_empty_system_returns_rhs():
    empty = jnp.zeros((0, 3))
    out = tridiagonal_solve(empty, empty, empty, empty)
    assert out.shape == (0, 3)


def test_unknown_method_raises():
    lower, diag, upper, rhs = make_tridiag(5)
    with pytest.raises(ValueError, match="unknown method"):
        tridiagonal_solve(lower, diag, upper, rhs, method="bogus")


def test_vmap_over_batch():
    def solve_one(seed):
        return make_tridiag(9, (), None, seed=seed)

    systems = [solve_one(s) for s in range(4)]
    stacked = [jnp.stack(arrs) for arrs in zip(*systems, strict=True)]
    x_batch = jax.vmap(lambda lo, di, up, b: tridiagonal_solve(lo, di, up, b))(*stacked)
    for i, (lo, di, up, b) in enumerate(systems):
        x_i = tridiagonal_solve(lo, di, up, b)
        assert np.allclose(np.asarray(x_batch[i]), np.asarray(x_i), atol=1e-12)


def test_jit_static_method():
    lower, diag, upper, rhs = make_tridiag(12, (2,), 2, seed=7)
    solve = jax.jit(lambda lo, di, up, b: tridiagonal_solve(lo, di, up, b, method="thomas"))
    x = solve(lower, diag, upper, rhs)
    assert np.allclose(np.asarray(x), dense_solve(lower, diag, upper, rhs), atol=1e-10)


def test_gradient_through_solve():
    lower, diag, upper, rhs = make_tridiag(8, (), None, seed=8)

    def loss(d):
        return jnp.sum(tridiagonal_solve(lower, d, upper, rhs, method="thomas") ** 2)

    g = jax.grad(loss)(diag)
    eps = 1e-6
    e = jnp.zeros_like(diag).at[3].set(eps)
    fd = (loss(diag + e) - loss(diag - e)) / (2 * eps)
    assert np.isclose(float(g[3]), float(fd), rtol=1e-5)


def test_multiple_field_axes_solved_together():
    # rhs carries two trailing field axes beyond the system shape.
    lower, diag, upper, _ = make_tridiag(10, (3,), None, seed=9)
    rng = np.random.default_rng(10)
    rhs = jnp.asarray(rng.standard_normal((10, 3, 2, 4)))
    x = tridiagonal_solve(lower, diag, upper, rhs, method="thomas")
    assert x.shape == (10, 3, 2, 4)
    # Cross-check one field slice against a fresh single-field solve.
    x_slice = tridiagonal_solve(lower, diag, upper, rhs[..., 1, 2], method="thomas")
    assert np.allclose(np.asarray(x[..., 1, 2]), np.asarray(x_slice), atol=1e-12)
