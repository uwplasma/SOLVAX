"""Tests for solvax.autodiff: chunked Jacobians match jax.jacfwd/jacrev."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    auto_chunk_size,
    chunk_map,
    chunked_jacfwd,
    chunked_jacobian,
    chunked_jacrev,
)

jax.config.update("jax_enable_x64", True)


def vector_fun(x):
    """R^n -> R^m with a genuinely dense Jacobian."""
    return jnp.stack(
        [
            jnp.sum(x**2),
            jnp.sum(jnp.sin(x)),
            x[0] * x[-1],
            jnp.sum(jnp.cumsum(x)),
        ]
    )


def matrix_in_out_fun(x):
    """(p, q) input -> (r,) output, to exercise shape bookkeeping."""
    return jnp.array([jnp.sum(x**2), jnp.sum(x @ x.T), x[0, 0] * x[-1, -1]])


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 3, 5, 100, "auto"])
def test_jacfwd_matches_jax(chunk_size):
    x = jnp.asarray(np.random.default_rng(0).standard_normal(5))
    ref = jax.jacfwd(vector_fun)(x)
    got = chunked_jacfwd(vector_fun, chunk_size=chunk_size)(x)
    assert got.shape == ref.shape
    assert np.allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 3, 4, 100, "auto"])
def test_jacrev_matches_jax(chunk_size):
    x = jnp.asarray(np.random.default_rng(1).standard_normal(6))
    ref = jax.jacrev(vector_fun)(x)
    got = chunked_jacrev(vector_fun, chunk_size=chunk_size)(x)
    assert got.shape == ref.shape
    assert np.allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def test_chunk_none_is_bit_identical_to_jax():
    x = jnp.asarray(np.random.default_rng(2).standard_normal(7))
    assert np.array_equal(
        np.asarray(chunked_jacfwd(vector_fun, chunk_size=None)(x)),
        np.asarray(jax.jacfwd(vector_fun)(x)),
    )
    assert np.array_equal(
        np.asarray(chunked_jacrev(vector_fun, chunk_size=None)(x)),
        np.asarray(jax.jacrev(vector_fun)(x)),
    )


def test_matrix_input_shape_convention():
    x = jnp.asarray(np.random.default_rng(3).standard_normal((3, 2)))
    ref = jax.jacfwd(matrix_in_out_fun)(x)  # shape (3,) + (3, 2)
    got = chunked_jacfwd(matrix_in_out_fun, chunk_size=2)(x)
    assert got.shape == ref.shape == (3, 3, 2)
    assert np.allclose(np.asarray(got), np.asarray(ref), atol=1e-12)

    ref_r = jax.jacrev(matrix_in_out_fun)(x)
    got_r = chunked_jacrev(matrix_in_out_fun, chunk_size=1)(x)
    assert got_r.shape == ref_r.shape
    assert np.allclose(np.asarray(got_r), np.asarray(ref_r), atol=1e-12)


@pytest.mark.parametrize("mode", ["fwd", "rev", "auto"])
def test_chunked_jacobian_modes(mode):
    x = jnp.asarray(np.random.default_rng(4).standard_normal(5))
    ref = jax.jacobian(vector_fun)(x)
    got = chunked_jacobian(vector_fun, mode=mode, chunk_size=2)(x)
    assert np.allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def test_chunked_jacobian_auto_picks_by_shape():
    # Tall map (n < m): auto should still match jax.jacobian.
    def tall(x):  # R^2 -> R^5
        return jnp.array([x[0], x[1], x[0] * x[1], x[0] ** 2, x[1] ** 2])

    x = jnp.asarray([1.3, -0.7])
    got = chunked_jacobian(tall, mode="auto", chunk_size=1)(x)
    assert np.allclose(np.asarray(got), np.asarray(jax.jacobian(tall)(x)), atol=1e-12)


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        chunked_jacobian(vector_fun, mode="sideways")(jnp.ones(3))


@pytest.mark.parametrize("bad", [0, -3])
def test_bad_chunk_size_raises(bad):
    with pytest.raises(ValueError, match="chunk_size must be"):
        chunked_jacfwd(vector_fun, chunk_size=bad)(jnp.ones(4))


def test_bad_chunk_string_raises():
    with pytest.raises(ValueError, match="chunk_size string"):
        chunked_jacrev(vector_fun, chunk_size="biggest")(jnp.ones(4))


def test_argnums_selects_argument():
    def f(a, x):
        return jnp.stack([jnp.sum(a * x), jnp.sum(x**2)])

    a = jnp.asarray([2.0, 3.0, 4.0])
    x = jnp.asarray([1.0, -1.0, 0.5])
    got = chunked_jacfwd(f, argnums=1, chunk_size=2)(a, x)
    ref = jax.jacfwd(f, argnums=1)(a, x)
    assert np.allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def test_jit_through_jacobian():
    x = jnp.asarray(np.random.default_rng(5).standard_normal(5))
    jac = jax.jit(chunked_jacrev(vector_fun, chunk_size=2))
    assert np.allclose(np.asarray(jac(x)), np.asarray(jax.jacrev(vector_fun)(x)), atol=1e-12)


def test_second_order_grad_through_chunked_jacobian():
    x = jnp.asarray(np.random.default_rng(6).standard_normal(4))

    def scalar(x):
        return jnp.sum(chunked_jacrev(vector_fun, chunk_size=2)(x) ** 2)

    g = jax.grad(scalar)(x)

    def scalar_ref(x):
        return jnp.sum(jax.jacrev(vector_fun)(x) ** 2)

    assert np.allclose(np.asarray(g), np.asarray(jax.grad(scalar_ref)(x)), atol=1e-10)


# --- chunk_map ---------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 3, 4, 10])
def test_chunk_map_matches_vmap(chunk_size):
    xs = jnp.asarray(np.random.default_rng(7).standard_normal((7, 3)))
    def fun(row):
        return jnp.sum(row**2)
    got = chunk_map(fun, xs, chunk_size=chunk_size)
    assert np.allclose(np.asarray(got), np.asarray(jax.vmap(fun)(xs)), atol=1e-12)


# --- auto_chunk_size ---------------------------------------------------------


def test_auto_chunk_size_heuristic_is_sqrt_balanced():
    assert auto_chunk_size(100) == 10
    assert auto_chunk_size(101) == 11  # ceil(sqrt)
    assert auto_chunk_size(1) == 1
    assert auto_chunk_size(0) == 1


def test_auto_chunk_size_budget_mode_and_clamp():
    # budget = 0.5 * 8000 / (out=10 * 8 bytes) = 50, clamped to dim.
    assert auto_chunk_size(1000, 10, max_memory_bytes=8000) == 50
    assert auto_chunk_size(20, 10, max_memory_bytes=8000) == 20  # clamp to dim
    # A tiny budget still yields at least 1.
    assert auto_chunk_size(1000, 1000, max_memory_bytes=8) == 1


def test_auto_string_resolves_in_builder():
    x = jnp.asarray(np.random.default_rng(8).standard_normal(9))
    got = chunked_jacfwd(vector_fun, chunk_size="auto")(x)
    assert np.allclose(np.asarray(got), np.asarray(jax.jacfwd(vector_fun)(x)), atol=1e-12)


def test_device_memory_limit_reads_bytes_limit(monkeypatch):
    from solvax import autodiff

    class _FakeDevice:
        def memory_stats(self):
            return {"bytes_limit": 2_000_000_000}

    monkeypatch.setattr(autodiff.jax, "local_devices", lambda: [_FakeDevice()])
    assert autodiff._device_memory_limit() == 2_000_000_000
    # ...and the device budget then drives auto_chunk_size (no explicit budget).
    # dim large, per-vector = out(1) * 8 bytes -> chunk clamped to dim.
    assert auto_chunk_size(64) == 64


def test_device_memory_limit_absent_stats(monkeypatch):
    from solvax import autodiff

    class _NoStatsDevice:
        def memory_stats(self):
            return {}

    monkeypatch.setattr(autodiff.jax, "local_devices", lambda: [_NoStatsDevice()])
    assert autodiff._device_memory_limit() is None
