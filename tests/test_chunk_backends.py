"""Parity between the native and the optional adv-jax-math chunking backends.

The contract these tests defend is that ``backend=`` selects an implementation
and nothing else: for every input the two paths must agree to floating-point
tolerance, and the presence or absence of the optional dependency must not
change any default. Everything specific to the optional backend is skipped
rather than assumed, so the file passes on a JAX-only install.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax.autodiff import (
    available_backends,
    chunk_map,
    chunked_jacfwd,
    chunked_jacobian,
    chunked_jacrev,
    default_backend,
)

HAS_ADV = "adv" in available_backends()
adv_only = pytest.mark.skipif(not HAS_ADV, reason="adv-jax-math not installed")
BACKENDS = ["native"] + (["adv"] if HAS_ADV else [])


def _f(x):
    return jnp.sin(x) * jnp.sum(x**2) + jnp.cos(x[::-1])


# --------------------------------------------------------------- defaults ---
def test_default_backend_is_native_even_when_adv_present():
    """Installing the extra must not silently change what anyone gets."""
    assert default_backend() == "native"
    assert "native" in available_backends()


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("SOLVAX_CHUNK_BACKEND", "auto")
    assert default_backend() == "auto"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown chunking backend"):
        chunk_map(lambda a: a, jnp.arange(4.0), backend="nonsense")


# ------------------------------------------------------------ correctness ---
X64 = jax.config.jax_enable_x64


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("chunk", [1, 3, 4, 8, 32, None])
def test_jacfwd_matches_jax(backend, chunk):
    """Chunk width must not change the answer, at any width.

    Widths cover exact divisibility, a short final chunk, a chunk equal to the
    dimension and one larger than it. The dtype follows the session's x64
    setting, so CI covers both stacks through its existing matrix.
    """
    x = jnp.arange(1.0, 9.0)
    ref = np.asarray(jax.jacfwd(_f)(x))
    got = np.asarray(chunked_jacfwd(_f, chunk_size=chunk, backend=backend)(x))
    tol = 1e-12 if X64 else 1e-6
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("chunk", [1, 3, 8, None])
def test_jacrev_matches_jax(backend, chunk):
    x = jnp.arange(1.0, 9.0)
    ref = np.asarray(jax.jacrev(_f)(x))
    got = np.asarray(chunked_jacrev(_f, chunk_size=chunk, backend=backend)(x))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


@adv_only
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_backends_agree_bitwise_where_they_can(chunk):
    """The two paths evaluate the same JVPs; they should agree very tightly."""
    x = jnp.arange(1.0, 9.0)
    a = np.asarray(chunked_jacfwd(_f, chunk_size=chunk, backend="native")(x))
    b = np.asarray(chunked_jacfwd(_f, chunk_size=chunk, backend="adv")(x))
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_complex_input(backend):
    def g(z):
        return z * jnp.sum(z)

    z = jnp.arange(1.0, 5.0) + 1j * jnp.arange(4.0)
    ref = np.asarray(jax.jacfwd(g, holomorphic=True)(z))
    got = np.asarray(
        chunked_jacfwd(g, chunk_size=2, backend=backend, **(
            {"holomorphic": True} if backend == "adv" else {}))(z)
    ) if backend == "adv" else np.asarray(
        chunked_jacfwd(g, chunk_size=2, backend=backend)(z))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------- transformations ---
@pytest.mark.parametrize("backend", BACKENDS)
def test_under_jit(backend):
    x = jnp.arange(1.0, 9.0)
    jac = jax.jit(chunked_jacfwd(_f, chunk_size=3, backend=backend))
    np.testing.assert_allclose(
        np.asarray(jac(x)), np.asarray(jax.jacfwd(_f)(x)), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_under_vmap(backend):
    xs = jnp.arange(1.0, 17.0).reshape(2, 8)
    got = jax.vmap(chunked_jacfwd(_f, chunk_size=3, backend=backend))(xs)
    ref = jax.vmap(jax.jacfwd(_f))(xs)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_differentiable_through(backend):
    """The chunked Jacobian must itself be differentiable."""
    def loss(x):
        return jnp.sum(chunked_jacfwd(_f, chunk_size=3, backend=backend)(x) ** 2)

    x = jnp.arange(1.0, 9.0)
    g = np.asarray(jax.grad(loss)(x))
    ref = np.asarray(jax.grad(lambda v: jnp.sum(jax.jacfwd(_f)(v) ** 2))(x))
    np.testing.assert_allclose(g, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_extra_positional_args_held_fixed(backend):
    def h(x, c):
        return jnp.sin(x) * c

    x = jnp.arange(1.0, 5.0)
    got = chunked_jacfwd(h, argnums=0, chunk_size=2, backend=backend)(x, 3.0)
    ref = jax.jacfwd(h, argnums=0)(x, 3.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_chunked_jacobian_mode_selection(backend):
    x = jnp.arange(1.0, 9.0)
    for mode in ("fwd", "rev", "auto"):
        got = chunked_jacobian(_f, mode=mode, chunk_size=3, backend=backend)(x)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(jax.jacfwd(_f)(x)), rtol=1e-5, atol=1e-6
        )


# -------------------------------------------------------- optional features ---
def test_native_rejects_adv_only_keywords():
    """An option the native path cannot honour must raise, never be ignored."""
    with pytest.raises(TypeError, match="backend='native'"):
        chunk_map(lambda a: a, jnp.arange(8.0).reshape(4, 2),
                  chunk_size=2, reduction=jnp.add, backend="native")


@adv_only
def test_adv_in_chunk_reduction_matches_explicit_sum():
    """``reduction`` folds inside the scan, so the stack is never materialized."""
    xs = jnp.arange(64.0).reshape(16, 4)
    fun = lambda a: jnp.outer(a, a)  # noqa: E731
    reduced = chunk_map(fun, xs, chunk_size=4, backend="adv",
                        reduction=jnp.add, chunk_reduction=lambda y: jnp.sum(y, 0))
    stacked = jnp.sum(chunk_map(fun, xs, chunk_size=4, backend="native"), axis=0)
    np.testing.assert_allclose(np.asarray(reduced), np.asarray(stacked),
                               rtol=1e-6, atol=1e-6)


@adv_only
def test_adv_reduction_lowers_peak_memory():
    """The point of the optional backend: never build the stacked result."""
    xs = jnp.arange(256.0).reshape(64, 4)
    fun = lambda a: jnp.outer(a, a)  # noqa: E731

    def temp(fn):
        return jax.jit(fn).lower(xs).compile().memory_analysis().temp_size_in_bytes

    stacked = temp(lambda v: jnp.sum(chunk_map(fun, v, chunk_size=8,
                                               backend="native"), axis=0))
    folded = temp(lambda v: chunk_map(fun, v, chunk_size=8, backend="adv",
                                      reduction=jnp.add,
                                      chunk_reduction=lambda y: jnp.sum(y, 0)))
    assert folded < stacked


def test_missing_optional_dependency_message(monkeypatch):
    """Absent the extra, the error must name the install command."""
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "adv_jax_math":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"solvax\[adv\]"):
        chunk_map(lambda a: a, jnp.arange(4.0), chunk_size=2, backend="adv")


# ------------------------------------------------------- resolution branches ---
def test_available_backends_without_the_extra(monkeypatch):
    """The advertised backend list must degrade to native, not raise."""
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "adv_jax_math":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert available_backends() == ("native",)


@adv_only
def test_auto_prefers_the_optional_backend_when_present():
    from solvax.autodiff import _resolve_backend

    assert _resolve_backend("auto") == "adv"


def test_auto_falls_back_to_native_without_the_extra(monkeypatch):
    import builtins

    from solvax.autodiff import _resolve_backend

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "adv_jax_math":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert _resolve_backend("auto") == "native"


@pytest.mark.parametrize("dim,expected", [(0, 1), (1, 1)])
def test_auto_chunk_size_degenerate_dimensions(dim, expected):
    from solvax.autodiff import auto_chunk_size

    assert auto_chunk_size(dim) == expected


def test_auto_chunk_size_without_a_memory_budget():
    from solvax.autodiff import auto_chunk_size

    assert auto_chunk_size(100) == 10          # ceil(sqrt(100))


def test_auto_chunk_size_respects_a_memory_budget():
    from solvax.autodiff import auto_chunk_size

    got = auto_chunk_size(1000, output_size=64, max_memory_bytes=1 << 20)
    assert 1 <= got <= 1000


def test_chunk_size_string_must_be_auto():
    with pytest.raises(ValueError, match="must be 'auto'"):
        chunked_jacfwd(_f, chunk_size="sometimes")(jnp.arange(1.0, 5.0))


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match=">= 1"):
        chunked_jacfwd(_f, chunk_size=0)(jnp.arange(1.0, 5.0))


def test_chunked_jacobian_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        chunked_jacobian(_f, mode="sideways")(jnp.arange(1.0, 5.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_auto_chunk_size_is_the_default_and_is_correct(backend):
    """``chunk_size='auto'`` is the default path and must match ``jax.jacfwd``."""
    x = jnp.arange(1.0, 17.0)
    got = chunked_jacfwd(_f, backend=backend)(x)          # chunk_size='auto'
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(jax.jacfwd(_f)(x)), rtol=1e-5, atol=1e-6
    )
