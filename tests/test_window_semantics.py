"""One meaning for ``adjoint_window`` across both public entry points.

SOLVAX exposes finite-window reverse mode twice: on stored bands
(``block_thomas_truncated``) and on a generated chain
(``block_thomas_truncated_fn(..., params=...)``). Until 0.9.x these had
different mathematical content behind the same keyword -- the array path closed
the chain at the window with a leading principal subsystem, the generated path
used the exact-window construction -- while the documentation described the
stronger one. These tests exist so that cannot silently return.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    LocalizationWindow,
    block_thomas_truncated,
    block_thomas_truncated_fn,
    block_thomas_truncated_fn_with_residual,
    check_localized_gradient,
    localization_crossover_window,
)
from solvax.direct import _block_thomas_truncated_bounded

N, M, KEEP = 14, 3, 2


def _bands(seed: int = 0):
    rng = np.random.default_rng(seed)
    lower = jnp.asarray(rng.standard_normal((N, M, M)) * 0.3)
    diag = jnp.asarray(
        np.stack([np.eye(M) * 3 + rng.standard_normal((M, M)) * 0.2 for _ in range(N)])
    )
    upper = jnp.asarray(rng.standard_normal((N, M, M)) * 0.3)
    rhs = jnp.asarray(rng.standard_normal((KEEP, M)))
    return lower, diag, upper, rhs


def _band_generator(params, k):
    lower, diag, upper = params
    return lower[k], diag[k], upper[k]


def _dense_reference(lower, diag, upper, rhs):
    """Gradient of the head functional through a dense solve of the full chain."""

    def loss(lo, di, up):
        a = jnp.zeros((N * M, N * M))
        for j in range(N):
            a = a.at[j * M : (j + 1) * M, j * M : (j + 1) * M].set(di[j])
            if j:
                a = a.at[j * M : (j + 1) * M, (j - 1) * M : j * M].set(lo[j])
            if j < N - 1:
                a = a.at[j * M : (j + 1) * M, (j + 1) * M : (j + 2) * M].set(up[j])
        b = jnp.zeros(N * M).at[: KEEP * M].set(rhs.reshape(-1))
        return jnp.sum(jnp.linalg.solve(a, b)[: KEEP * M] ** 2)

    return jax.grad(loss, argnums=(0, 1, 2))(lower, diag, upper)


@pytest.mark.parametrize("window", [0, 1, 3, 6, N])
def test_array_and_generated_windows_agree_bitwise(window):
    """The same keyword must mean the same algorithm on both entry points."""
    lower, diag, upper, rhs = _bands()

    def array_loss(lo, di, up):
        return jnp.sum(
            block_thomas_truncated(lo, di, up, rhs, KEEP, adjoint_window=window) ** 2
        )

    def generated_loss(lo, di, up):
        return jnp.sum(
            block_thomas_truncated_fn(
                _band_generator, N, rhs, KEEP,
                params=(lo, di, up), adjoint_window=window,
            )
            ** 2
        )

    ga = jax.grad(array_loss, argnums=(0, 1, 2))(lower, diag, upper)
    gg = jax.grad(generated_loss, argnums=(0, 1, 2))(lower, diag, upper)
    for a, g in zip(ga, gg, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(g))


def test_full_window_is_exact_on_both_paths():
    lower, diag, upper, rhs = _bands()
    ref = _dense_reference(lower, diag, upper, rhs)

    def array_loss(lo, di, up):
        return jnp.sum(
            block_thomas_truncated(lo, di, up, rhs, KEEP, adjoint_window=N) ** 2
        )

    got = jax.grad(array_loss, argnums=(0, 1, 2))(lower, diag, upper)
    for a, r in zip(got, ref, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(r), rtol=1e-9, atol=1e-11)


def test_array_window_is_at_least_as_accurate_as_the_old_closure():
    """The superseded leading-principal closure must not be reachable by default.

    It is kept for the ablation reported in the manuscript, so it is compared
    here rather than deleted; the shipped path must never be worse.
    """
    lower, diag, upper, rhs = _bands()
    ref = _dense_reference(lower, diag, upper, rhs)
    window = 3

    def new_loss(lo, di, up):
        return jnp.sum(
            block_thomas_truncated(lo, di, up, rhs, KEEP, adjoint_window=window) ** 2
        )

    def old_loss(lo, di, up):
        return jnp.sum(
            _block_thomas_truncated_bounded(lo, di, up, rhs, KEEP, window) ** 2
        )

    err_new = max(
        float(np.abs(np.asarray(a) - np.asarray(r)).max())
        for a, r in zip(jax.grad(new_loss, argnums=(0, 1, 2))(lower, diag, upper),
                        ref, strict=True)
    )
    err_old = max(
        float(np.abs(np.asarray(a) - np.asarray(r)).max())
        for a, r in zip(jax.grad(old_loss, argnums=(0, 1, 2))(lower, diag, upper),
                        ref, strict=True)
    )
    assert err_new <= err_old


def test_forward_solution_does_not_depend_on_the_window():
    """``adjoint_window`` selects a reverse rule; the primal must be untouched."""
    lower, diag, upper, rhs = _bands()
    base = block_thomas_truncated(lower, diag, upper, rhs, KEEP)
    for window in (0, 2, 5, N):
        got = block_thomas_truncated(lower, diag, upper, rhs, KEEP, adjoint_window=window)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(base))


# ------------------------------------------------------- window diagnostic ---
def test_crossover_window_returns_a_diagnostic_not_a_certificate():
    off = jnp.eye(M) * 0.2
    base = jnp.eye(M) * 3.0

    def block_fn(k):
        del k
        return off, base, off.T

    out = localization_crossover_window(block_fn, N, KEEP)
    assert isinstance(out, LocalizationWindow)
    assert out.certified is False
    assert out.status == "heuristic"
    assert out.primal_profile.shape == (N,)
    assert int(out) == out.window          # usable directly as adjoint_window


def test_non_localizing_chain_returns_the_full_window():
    """No crossing means no useful window, and the honest answer is exactness.

    With a zero super-diagonal the Schur recursion degenerates to
    ``Delta_k = D_k``, so ``T_k = -D_k^{-1} L_k`` is constant and the envelope
    never falls below one for ``L = 2 D``.
    """
    lower = jnp.eye(M) * 2.0
    base = jnp.eye(M)
    zero = jnp.zeros((M, M))

    def block_fn(k):
        del k
        return lower, base, zero

    out = localization_crossover_window(block_fn, N, KEEP)
    assert np.all(out.primal_profile[1:] >= 1.0)
    assert out.localized is False
    assert out.status == "full-window"
    assert out.window == N - KEEP


def test_deprecated_alias_warns_and_returns_int():
    off = jnp.eye(M) * 0.2
    base = jnp.eye(M) * 3.0

    def block_fn(k):
        del k
        return off, base, off.T

    from solvax import suggest_adjoint_window

    with pytest.deprecated_call():
        value = suggest_adjoint_window(block_fn, N, KEEP)
    assert isinstance(value, int)


def test_nested_window_check_detects_an_inadequate_window():
    lower, diag, upper, rhs = _bands()

    def gradient_fn(window):
        return jax.grad(
            lambda lo: jnp.sum(
                block_thomas_truncated(lo, diag, upper, rhs, KEEP,
                                       adjoint_window=window) ** 2
            )
        )(lower)

    tight = check_localized_gradient(gradient_fn, 0, increment=2, rtol=1e-8)
    loose = check_localized_gradient(gradient_fn, 8, increment=2, rtol=1e-8)
    assert tight["passed"] is False
    assert loose["passed"] is True
    assert loose["relative_difference"] < tight["relative_difference"]
    assert "adjoint_window" in tight["recommendation"]


def test_localization_window_passes_directly_to_both_entry_points():
    """The advisor's record is documented as usable as ``adjoint_window``.

    It defines ``__index__`` precisely so it can be handed straight back to the
    solver. Both entry points used to compare it against a bound before
    coercing it, which raised ``TypeError`` on the documented call; only the
    coercion itself was covered, not the call it exists for.
    """
    n_blocks, m, keep = 12, 3, 2

    def block_fn(p, j):
        eye = jnp.eye(m)
        return (
            eye * 0.3 * (j > 0),
            eye * (4.0 + p[0] * (1.0 + j)),
            eye * 0.3 * (j < n_blocks - 1),
        )

    params = jnp.array([0.5])
    rhs = jnp.ones((keep, m))
    advice = localization_crossover_window(
        lambda k: block_fn(params, k), n_blocks, keep_lowest=keep
    )

    generated = block_thomas_truncated_fn(
        block_fn, n_blocks, rhs, keep_lowest=keep,
        params=params, adjoint_window=advice,
    )
    as_int = block_thomas_truncated_fn(
        block_fn, n_blocks, rhs, keep_lowest=keep,
        params=params, adjoint_window=advice.window,
    )
    assert jnp.array_equal(generated, as_int)

    bands = [
        jnp.stack([block_fn(params, j)[i] for j in range(n_blocks)])
        for i in range(3)
    ]
    stored = block_thomas_truncated(
        *bands, rhs, keep_lowest=keep, adjoint_window=advice
    )
    assert jnp.array_equal(stored, as_int)

    # And through jit, where the window is a static argument.
    jitted = jax.jit(
        lambda p: block_thomas_truncated_fn(
            block_fn, n_blocks, rhs, keep_lowest=keep,
            params=p, adjoint_window=advice,
        )
    )
    assert jnp.allclose(jitted(params), as_int)

    # A gradient through the documented call must be nonzero and match the
    # integer form exactly.
    def loss(p, window):
        return jnp.sum(
            block_thomas_truncated_fn(
                block_fn, n_blocks, rhs, keep_lowest=keep,
                params=p, adjoint_window=window,
            ) ** 2
        )

    g_record = jax.grad(loss)(params, advice)
    g_int = jax.grad(loss)(params, advice.window)
    assert jnp.array_equal(g_record, g_int)
    assert jnp.any(g_record != 0.0)


def test_non_integral_window_is_rejected():
    n_blocks, m, keep = 6, 2, 1
    bands = [jnp.tile(jnp.eye(m) * c, (n_blocks, 1, 1)) for c in (0.2, 3.0, 0.2)]
    with pytest.raises(TypeError, match="__index__"):
        block_thomas_truncated(
            *bands, jnp.ones((keep, m)), keep_lowest=keep, adjoint_window=2.5
        )


def _residual_chain(n_blocks, m):
    def block_fn(p, j):
        eye = jnp.eye(m)
        return (
            eye * 0.3 * (j > 0),
            eye * (4.0 + p[0] * (1.0 + j)),
            eye * 0.3 * (j < n_blocks - 1),
        )

    return block_fn


def test_residual_entry_point_matches_taped_path_at_full_window():
    """The residual diagnostic must mean the same thing on both paths.

    ``block_thomas_truncated_fn_with_residual`` gained a ``params`` path so the
    diagnostic survives when the solve is differentiated by the exact-window
    rule. At full window that rule is exact, so both the solution and the
    residual must come back bitwise identical to the taped path.
    """
    n_blocks, m, keep = 16, 3, 2
    block_fn = _residual_chain(n_blocks, m)
    params = jnp.array([0.6])
    rhs = jnp.zeros((keep, m, 2)).at[1, :, 0].set(1.0)

    x_taped, r_taped = block_thomas_truncated_fn_with_residual(
        lambda j: block_fn(params, j), n_blocks, rhs, keep, residual_rhs_index=0
    )
    x_exact, r_exact = block_thomas_truncated_fn_with_residual(
        block_fn, n_blocks, rhs, keep,
        params=params, adjoint_window=n_blocks, residual_rhs_index=0,
    )
    assert jnp.array_equal(x_taped, x_exact)
    assert jnp.array_equal(r_taped, r_exact)


def test_residual_entry_point_gradient_matches_at_full_window():
    n_blocks, m, keep = 16, 3, 2
    block_fn = _residual_chain(n_blocks, m)
    params = jnp.array([0.6])
    rhs = jnp.zeros((keep, m, 2)).at[1, :, 0].set(1.0)

    def taped_loss(p):
        x, _ = block_thomas_truncated_fn_with_residual(
            lambda j: block_fn(p, j), n_blocks, rhs, keep, residual_rhs_index=0
        )
        return jnp.sum(x**2)

    def exact_loss(p):
        x, _ = block_thomas_truncated_fn_with_residual(
            block_fn, n_blocks, rhs, keep,
            params=p, adjoint_window=n_blocks, residual_rhs_index=0,
        )
        return jnp.sum(x**2)

    g_taped = jax.grad(taped_loss)(params)
    g_exact = jax.grad(exact_loss)(params)
    assert jnp.allclose(g_taped, g_exact, rtol=1e-12, atol=1e-14)
    assert jnp.any(g_exact != 0.0)


def test_residual_carries_no_derivative():
    """The residual is a diagnostic; the reverse rule ignores its cotangent.

    That is only sound because the wrapper detaches it. If the detachment were
    removed, this would silently return a wrong number rather than zero.
    """
    n_blocks, m, keep = 12, 2, 2
    block_fn = _residual_chain(n_blocks, m)
    params = jnp.array([0.5])
    rhs = jnp.zeros((keep, m, 2)).at[1, :, 0].set(1.0)

    def residual_only(p):
        _, r = block_thomas_truncated_fn_with_residual(
            block_fn, n_blocks, rhs, keep,
            params=p, adjoint_window=4, residual_rhs_index=0,
        )
        return r

    assert jnp.array_equal(jax.grad(residual_only)(params), jnp.zeros_like(params))


def test_residual_entry_point_accepts_localization_window():
    n_blocks, m, keep = 12, 2, 2
    block_fn = _residual_chain(n_blocks, m)
    params = jnp.array([0.5])
    rhs = jnp.zeros((keep, m, 2)).at[1, :, 0].set(1.0)
    advice = localization_crossover_window(
        lambda j: block_fn(params, j), n_blocks, keep_lowest=keep
    )
    x_record, r_record = block_thomas_truncated_fn_with_residual(
        block_fn, n_blocks, rhs, keep,
        params=params, adjoint_window=advice, residual_rhs_index=0,
    )
    x_int, r_int = block_thomas_truncated_fn_with_residual(
        block_fn, n_blocks, rhs, keep,
        params=params, adjoint_window=advice.window, residual_rhs_index=0,
    )
    assert jnp.array_equal(x_record, x_int)
    assert jnp.array_equal(r_record, r_int)


def test_residual_entry_point_requires_a_window_with_params():
    n_blocks, m, keep = 8, 2, 2
    block_fn = _residual_chain(n_blocks, m)
    rhs = jnp.zeros((keep, m, 2)).at[1, :, 0].set(1.0)
    with pytest.raises(ValueError, match="adjoint_window"):
        block_thomas_truncated_fn_with_residual(
            block_fn, n_blocks, rhs, keep, params=jnp.array([0.5])
        )
