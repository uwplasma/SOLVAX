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
