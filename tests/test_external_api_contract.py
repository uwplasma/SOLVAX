"""Pin the parts of JAX, and of our own surface, that we depend on.

Future-proofing a research library is not a matter of pinning versions --- that
only postpones the break and hides it. What makes an upgrade safe is knowing
*which* external behaviours the library leans on, so that a version which
changes one of them fails here, loudly and in one place, rather than deep
inside a solve or silently in a benchmark number.

Each test below names one such dependency and why it matters. If one starts
failing after a JAX upgrade, the fix belongs at that seam, and the failure
message should say enough to find it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx
from solvax.direct import _residual_from_state
from solvax.native import _is_tracer


def test_tracer_detection_survives_jax_core_moving():
    """``native`` refuses to run under a trace; that guard must keep working.

    It used to test ``isinstance(x, jax.core.Tracer)`` directly. ``jax.core``
    already resolves into ``jax._src`` and is on its way to private, so the
    check is now written to degrade rather than raise ``AttributeError`` from
    inside the guard. This asserts the behaviour, not the mechanism.
    """
    assert not _is_tracer(jnp.ones(3))
    assert not _is_tracer(np.ones(3))
    assert not _is_tracer(2.0)

    seen = []
    jax.jit(lambda x: (seen.append(_is_tracer(x)), x)[1])(jnp.ones(3))
    jax.vmap(lambda x: (seen.append(_is_tracer(x)), x)[1])(jnp.ones((2, 3)))
    jax.grad(lambda x: (seen.append(_is_tracer(x)), jnp.sum(x))[1])(jnp.ones(3))
    assert seen == [True, True, True], (
        "tracer detection broke under a transform; solvax.native's guard would "
        "let SciPy's SuperLU be called on a tracer instead of raising"
    )


def test_custom_vjp_still_refuses_forward_mode():
    """The exact-window rule is reverse-only, and callers are told so.

    Two production integrations now document this restriction and one of them
    pins it in its own suite. If JAX ever grows a forward-mode fallback for
    ``custom_vjp``, this test fails and those docs become wrong --- which is
    the point of asserting it here rather than assuming it.
    """
    n_blocks, m, keep = 8, 2, 2

    def block_fn(p, j):
        eye = jnp.eye(m)
        return (eye * 0.2 * (j > 0), eye * (4.0 + p[0]), eye * 0.2 * (j < n_blocks - 1))

    rhs = jnp.zeros((keep, m)).at[0].set(1.0)

    def loss(p):
        return jnp.sum(
            sx.block_thomas_truncated_fn(
                block_fn, n_blocks, rhs, keep_lowest=keep, params=p, adjoint_window=2
            )
        )

    params = jnp.array([0.5])
    assert jnp.any(jax.grad(loss)(params) != 0.0)
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jacfwd(loss)(params)


def test_compiled_memory_analysis_is_available_and_sane():
    """Every memory number in the paper comes from this API.

    ``lower().compile().memory_analysis().temp_size_in_bytes`` is how the
    benchmarks measure reverse-mode state. It is not part of JAX's documented
    stable surface, so a rename would silently invalidate a benchmark rather
    than fail it. This makes that a test failure.
    """
    n_blocks, m, keep = 16, 3, 2

    def block_fn(p, j):
        eye = jnp.eye(m)
        return (eye * 0.3 * (j > 0), eye * (4.0 + p[0]), eye * 0.3 * (j < n_blocks - 1))

    rhs = jnp.zeros((keep, m)).at[0].set(1.0)

    def state(window):
        fn = jax.grad(
            lambda p: jnp.sum(
                sx.block_thomas_truncated_fn(
                    block_fn, n_blocks, rhs, keep_lowest=keep,
                    params=p, adjoint_window=window,
                )
            )
        )
        analysis = jax.jit(fn).lower(jnp.array([0.5])).compile().memory_analysis()
        return analysis.temp_size_in_bytes

    narrow, wide = state(1), state(n_blocks)
    assert narrow > 0 and wide > 0
    assert narrow < wide, (
        "a narrower window should compile to less reverse-mode temporary state; "
        f"got {narrow} >= {wide}. Either the memory model changed meaning or "
        "the window stopped bounding what it claims to bound"
    )


def test_lu_pivots_to_permutation_still_lives_where_the_residual_expects():
    """The residual diagnostic reconstructs P from LAPACK pivots.

    ``jax.lax.linalg.lu_pivots_to_permutation`` is a low-level entry point. If
    it moves, the residual would either raise or --- worse --- be computed
    against the wrong permutation and look plausible.
    """
    assert hasattr(jax.lax.linalg, "lu_pivots_to_permutation")

    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(5, 5)) + 5.0 * np.eye(5))
    lu, piv = jax.scipy.linalg.lu_factor(a)
    perm = jax.lax.linalg.lu_pivots_to_permutation(piv, 5)
    lower = jnp.tril(lu, -1) + jnp.eye(5)
    upper = jnp.triu(lu)
    assert jnp.allclose(lower @ upper, a[perm], atol=1e-10), (
        "P L U no longer reconstructs A; the residual diagnostic's permutation "
        "convention is wrong"
    )


def test_custom_root_still_backs_the_implicit_solve():
    """``implicit.root_solve`` is ``jax.lax.custom_root``; DKX leans on it."""
    assert hasattr(jax.lax, "custom_root")

    def f(x):
        return x**2 - 2.0

    def solver(g, x0):
        del g
        return jnp.sqrt(2.0) * jnp.ones_like(x0)

    def tangent_solve(g, y):
        return y / jax.grad(g)(jnp.sqrt(2.0))

    root = sx.implicit.root_solve(
        f, jnp.asarray(1.0), solver, tangent_solve=tangent_solve
    )
    assert jnp.allclose(root, jnp.sqrt(2.0))


def test_exported_surface_is_stable():
    """A name disappearing from ``__all__`` breaks a downstream import.

    Two solvers in this organisation import from here. This is the list they
    are entitled to rely on; removing an entry is a breaking change and should
    require editing this test, not merely deleting a line.
    """
    required = {
        "block_thomas",
        "block_thomas_factor",
        "block_thomas_factor_fn",
        "block_thomas_solve",
        "block_thomas_truncated",
        "block_thomas_truncated_fn",
        "block_thomas_truncated_fn_with_residual",
        "block_thomas_checkpointed_fn",
        "BlockTridiagFactors",
        "LocalizationWindow",
        "localization_crossover_window",
        "localization_profile_fn",
        "check_localized_gradient",
        "chunked_jacobian",
        "gcrot",
        "gmres",
        "pcg",
        "newton_krylov",
    }
    missing = sorted(required - set(sx.__all__))
    assert not missing, f"public names disappeared from solvax.__all__: {missing}"
    for name in required:
        assert hasattr(sx, name), f"{name} is in __all__ but not importable"


def test_deprecated_alias_still_warns_and_has_a_stated_removal():
    """``suggest_adjoint_window`` is scheduled for removal in 0.12.0.

    Downstream code may still call it. It must keep working *and* keep warning
    until that release; a silent alias is worse than either.
    """
    n_blocks, m = 8, 2

    def block_fn(j):
        eye = jnp.eye(m)
        return (eye * 0.2 * (j > 0), eye * 5.0, eye * 0.2 * (j < n_blocks - 1))

    with pytest.warns(DeprecationWarning, match="localization_crossover_window"):
        window = sx.suggest_adjoint_window(block_fn, n_blocks, keep_lowest=2)
    assert int(window) == sx.localization_crossover_window(
        block_fn, n_blocks, keep_lowest=2
    ).window


def test_residual_helper_is_shared_by_both_paths():
    """One implementation, so the diagnostic cannot mean two things."""
    n_blocks, m, keep = 12, 2, 2

    def block_fn(p, j):
        eye = jnp.eye(m)
        return (eye * 0.25 * (j > 0), eye * (5.0 + p[0]), eye * 0.25 * (j < n_blocks - 1))

    params = jnp.array([0.4])
    rhs = jnp.zeros((keep, m, 2)).at[0, :, 0].set(1.0)

    _, taped = sx.block_thomas_truncated_fn_with_residual(
        lambda j: block_fn(params, j), n_blocks, rhs, keep, residual_rhs_index=0
    )
    _, exact = sx.block_thomas_truncated_fn_with_residual(
        block_fn, n_blocks, rhs, keep,
        params=params, adjoint_window=n_blocks, residual_rhs_index=0,
    )
    assert jnp.array_equal(taped, exact)
    assert callable(_residual_from_state)
