"""Randomised property checks for the exact-window rule.

The existing suite pins the theorems on chains chosen so an analytic reference
exists. That is the right way to test a claim, but it leaves a gap: every case
was chosen by the same person who wrote the code, so a systematic blind spot
would be invisible. These tests sweep pseudo-random chains across shapes,
windows, conditioning and arithmetic, and check each result against dense
linear algebra rather than against another SOLVAX path.

Everything here is deterministic: seeds are fixed, so a failure is
reproducible from the parameters in its name.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx

jax.config.update("jax_enable_x64", True)


def random_chain(seed, n_blocks, m, dominance=4.0, complex_valued=False):
    """A pseudo-random admissible chain and its dense assembly."""
    rng = np.random.default_rng(seed)

    def draw(shape):
        real = rng.normal(size=shape)
        if not complex_valued:
            return real
        return real + 1j * rng.normal(size=shape)

    lower = draw((n_blocks, m, m))
    diag = draw((n_blocks, m, m))
    upper = draw((n_blocks, m, m))
    # Make it block diagonally dominant so every Schur complement exists.
    for j in range(n_blocks):
        diag[j] += dominance * m * np.eye(m)
    lower[0] = 0.0
    upper[-1] = 0.0
    dtype = jnp.complex128 if complex_valued else jnp.float64
    return (
        jnp.asarray(lower, dtype=dtype),
        jnp.asarray(diag, dtype=dtype),
        jnp.asarray(upper, dtype=dtype),
    )


def dense_matrix(lower, diag, upper):
    n_blocks, m, _ = diag.shape
    a = np.zeros((n_blocks * m, n_blocks * m), dtype=np.asarray(diag).dtype)
    for j in range(n_blocks):
        a[j * m : (j + 1) * m, j * m : (j + 1) * m] = np.asarray(diag[j])
        if j > 0:
            a[j * m : (j + 1) * m, (j - 1) * m : j * m] = np.asarray(lower[j])
        if j < n_blocks - 1:
            a[j * m : (j + 1) * m, (j + 1) * m : (j + 2) * m] = np.asarray(upper[j])
    return a


@pytest.mark.parametrize(
    "seed,n_blocks,m,keep",
    [
        (s, n, m, k)
        for s, (n, m, k) in enumerate(
            itertools.product([3, 7, 16], [1, 2, 5], [1, 2])
        )
        if k <= n
    ],
)
def test_selected_head_matches_a_dense_solve(seed, n_blocks, m, keep):
    """Forward exactness across shapes, including the degenerate ones.

    ``m = 1`` (scalar chain), ``n_blocks = keep`` (nothing eliminated), and
    ``keep = 1`` are the corners where an off-by-one in the sweep hides.
    """
    lower, diag, upper = random_chain(seed, n_blocks, m)
    rhs = jnp.asarray(
        np.random.default_rng(seed + 1000).normal(size=(keep, m))
    )

    got = sx.block_thomas_truncated(lower, diag, upper, rhs, keep_lowest=keep)

    a = dense_matrix(lower, diag, upper)
    b = np.zeros(n_blocks * m)
    b[: keep * m] = np.asarray(rhs).ravel()
    want = np.linalg.solve(a, b).reshape(n_blocks, m)[:keep]
    assert np.allclose(np.asarray(got), want, rtol=1e-9, atol=1e-11), (
        f"selected head differs from a dense solve at "
        f"n_blocks={n_blocks} m={m} keep={keep}"
    )


@pytest.mark.parametrize("seed,n_blocks,m", [(0, 8, 2), (1, 12, 3), (2, 20, 2)])
def test_full_window_gradient_matches_a_dense_gradient(seed, n_blocks, m):
    """The full window is exact, checked against differentiating a dense solve.

    This is the strongest available reference: it shares no code path with the
    elimination, so agreement is not self-consistency.
    """
    keep = 2
    lower, diag, upper = random_chain(seed, n_blocks, m)
    rhs = jnp.asarray(np.random.default_rng(seed + 7).normal(size=(keep, m)))
    weights = jnp.asarray(np.random.default_rng(seed + 9).normal(size=(keep, m)))

    def block_fn(scale, j):
        return (lower[j] * scale, diag[j] * scale, upper[j] * scale)

    def windowed(scale):
        return jnp.sum(
            weights
            * sx.block_thomas_truncated_fn(
                block_fn, n_blocks, rhs, keep_lowest=keep,
                params=scale, adjoint_window=n_blocks,
            )
        )

    def dense(scale):
        blocks = [block_fn(scale, j) for j in range(n_blocks)]
        rows = []
        for j, (low, dia, up) in enumerate(blocks):
            row = [jnp.zeros((m, m))] * n_blocks
            row[j] = dia
            if j > 0:
                row[j - 1] = low
            if j < n_blocks - 1:
                row[j + 1] = up
            rows.append(jnp.concatenate(row, axis=1))
        a = jnp.concatenate(rows, axis=0)
        b = jnp.concatenate([rhs.ravel(), jnp.zeros((n_blocks - keep) * m)])
        return jnp.sum(weights * jnp.linalg.solve(a, b).reshape(n_blocks, m)[:keep])

    one = jnp.asarray(1.0)
    assert jnp.allclose(windowed(one), dense(one), rtol=1e-9)
    assert jnp.allclose(
        jax.grad(windowed)(one), jax.grad(dense)(one), rtol=1e-7
    ), "the full-window gradient disagrees with differentiating a dense solve"


@pytest.mark.parametrize("seed,n_blocks", [(0, 24), (1, 32)])
def test_window_error_decreases_monotonically_in_practice(seed, n_blocks):
    """Widening the window must not make the gradient worse.

    Not a theorem --- the bound decreases, the realized error need not be
    monotone --- but on a dominant chain a non-monotone sequence would signal
    an indexing error rather than genuine cancellation, so this asserts the
    weaker, testable thing: the error at the full window is the smallest, and
    a wide window beats a zero window by orders of magnitude.
    """
    m, keep = 3, 2
    lower, diag, upper = random_chain(seed, n_blocks, m, dominance=6.0)
    rhs = jnp.asarray(np.random.default_rng(seed + 3).normal(size=(keep, m)))

    # The parameter must enter the operator in a way the retained head does not
    # already determine. A *uniform* scale does not: see
    # test_uniform_scaling_is_exact_at_every_window for why, and why using one
    # here would have made this test vacuous.
    def block_fn(shift, j):
        return (lower[j], diag[j] + shift * jnp.eye(m) * (1.0 + j), upper[j])

    def grad_at(window):
        return float(
            jax.grad(
                lambda s: jnp.sum(
                    sx.block_thomas_truncated_fn(
                        block_fn, n_blocks, rhs, keep_lowest=keep,
                        params=s, adjoint_window=window,
                    ) ** 2
                )
            )(jnp.asarray(1.0))
        )

    exact = grad_at(n_blocks)
    errors = [abs(grad_at(w) - exact) / abs(exact) for w in (0, 2, 4, 8)]
    assert errors[0] > errors[-1], "widening the window did not help at all"
    assert errors[-1] < 1e-6, f"a window of 8 left {errors[-1]:.2e} relative error"


@pytest.mark.parametrize("seed", [0, 1])
def test_complex_chains_agree_with_a_dense_solve(seed):
    """Complex arithmetic is a separate code path and a separate convention."""
    n_blocks, m, keep = 10, 3, 2
    lower, diag, upper = random_chain(seed, n_blocks, m, complex_valued=True)
    rng = np.random.default_rng(seed + 11)
    rhs = jnp.asarray(rng.normal(size=(keep, m)) + 1j * rng.normal(size=(keep, m)))

    got = sx.block_thomas_truncated(lower, diag, upper, rhs, keep_lowest=keep)
    a = dense_matrix(lower, diag, upper)
    b = np.zeros(n_blocks * m, dtype=complex)
    b[: keep * m] = np.asarray(rhs).ravel()
    want = np.linalg.solve(a, b).reshape(n_blocks, m)[:keep]
    assert np.allclose(np.asarray(got), want, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("dominance", [1.2, 2.0, 8.0])
def test_forward_solve_holds_as_conditioning_worsens(dominance):
    """Weaker dominance means a worse-conditioned chain, not a wrong one."""
    n_blocks, m, keep = 24, 3, 2
    lower, diag, upper = random_chain(5, n_blocks, m, dominance=dominance)
    rhs = jnp.asarray(np.random.default_rng(13).normal(size=(keep, m)))

    got = sx.block_thomas_truncated(lower, diag, upper, rhs, keep_lowest=keep)
    a = dense_matrix(lower, diag, upper)
    b = np.zeros(n_blocks * m)
    b[: keep * m] = np.asarray(rhs).ravel()
    want = np.linalg.solve(a, b).reshape(n_blocks, m)[:keep]

    scale = max(np.linalg.norm(want), 1e-30)
    error = np.linalg.norm(np.asarray(got) - want) / scale
    conditioning = np.linalg.cond(a)
    assert error < 1e-10 * max(conditioning, 1.0), (
        f"forward error {error:.2e} is large even for cond={conditioning:.2e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multiple_right_hand_sides_match_solving_them_one_at_a_time(seed):
    """The multi-column path must agree column-by-column with the single one."""
    n_blocks, m, keep, n_rhs = 12, 3, 2, 4
    lower, diag, upper = random_chain(seed, n_blocks, m)
    rhs = jnp.asarray(
        np.random.default_rng(seed + 5).normal(size=(keep, m, n_rhs))
    )

    together = sx.block_thomas_truncated(lower, diag, upper, rhs, keep_lowest=keep)
    for column in range(n_rhs):
        alone = sx.block_thomas_truncated(
            lower, diag, upper, rhs[..., column], keep_lowest=keep
        )
        assert jnp.allclose(together[..., column], alone, rtol=1e-11, atol=1e-13), (
            f"column {column} of a batched solve differs from solving it alone"
        )


@pytest.mark.parametrize("seed,n_blocks", [(0, 16), (1, 24)])
def test_uniform_scaling_is_exact_at_every_window(seed, n_blocks):
    """A third family where the omitted tail vanishes for structural reasons.

    If every block is scaled by the same parameter, ``A(s) = s A_0``, then
    ``x(s) = x(1)/s`` exactly, so a functional of the retained head is a
    function of ``s`` and of the retained blocks alone. Theorem 1 returns those
    blocks exactly at any window, so the parameter gradient is exact at any
    window --- including ``w = 0``.

    This is the same phenomenon as the upper-triangular family in the paper:
    the identity says the error is a sum of named omitted terms, and here that
    sum is structurally zero. A decay-based analysis would predict an error.
    We check both the exactness and the closed form it implies.
    """
    m, keep = 3, 2
    lower, diag, upper = random_chain(seed, n_blocks, m)
    rhs = jnp.asarray(np.random.default_rng(seed + 21).normal(size=(keep, m)))

    def block_fn(scale, j):
        return (lower[j] * scale, diag[j] * scale, upper[j] * scale)

    def loss(scale, window):
        return jnp.sum(
            sx.block_thomas_truncated_fn(
                block_fn, n_blocks, rhs, keep_lowest=keep,
                params=scale, adjoint_window=window,
            ) ** 2
        )

    one = jnp.asarray(1.0)
    exact = jax.grad(lambda s: loss(s, n_blocks))(one)
    for window in (0, 1, 3):
        got = jax.grad(lambda s, w=window: loss(s, w))(one)
        assert jnp.allclose(got, exact, rtol=0.0, atol=0.0), (
            f"uniform scaling should be exact at window {window}"
        )

    # J(s) = J(1)/s^2, so dJ/ds at s=1 is exactly -2 J(1).
    assert jnp.allclose(exact, -2.0 * loss(one, n_blocks), rtol=1e-12)


# --------------------------------------------------------------------------
# Generator invocation count
#
# The manuscript's complexity model rests on a claim about how often the row
# generator is called: once per row for a forward-only solve, and a small
# constant multiple of ``N`` plus one pass over the retained rows for a
# differentiated one. That is the honest price of not taping -- rows are rebuilt
# rather than stored -- and it is quoted in print, so a refactor that adds a
# sweep must fail here.
#
# What is pinned is the *shape*, not one number. The multiplier counts passes
# the framework's own transformations schedule, and it can differ across JAX
# releases; the test therefore checks that
#
#     (differentiated - W - K) / N
#
# is the same integer at every shape, rather than hard-coding the 4 measured on
# the version this was written against. That invariant is what the cost model
# uses and it survives an upgrade; a change in the multiplier itself is
# reported, so it can be checked against the manuscript rather than silently
# absorbed.
#
# Counting has to happen at run time. A plain Python counter in the generator
# counts *tracings*, of which there are a handful regardless of ``N``;
# ``jax.debug.callback`` fires once per scan iteration, which is what the cost
# model is about.
# --------------------------------------------------------------------------


def _generator_calls(n_blocks: int, keep_lowest: int, window: int) -> tuple[int, int]:
    m = 3
    eye = jnp.eye(m)
    calls = {"n": 0}

    def bump(_):
        calls["n"] += 1

    def block_fn(params, j):
        jax.debug.callback(bump, j)
        return (
            eye * 0.2 * (j > 0),
            eye * (4.0 + params[0] * (1.0 + j)),
            eye * 0.2 * (j < n_blocks - 1),
        )

    rhs_low = jnp.ones((keep_lowest, m))
    p = jnp.array([0.7])

    def count(fn):
        calls["n"] = 0
        jax.block_until_ready(fn())
        return calls["n"]

    forward = count(
        lambda: sx.block_thomas_truncated_fn(
            block_fn, n_blocks, rhs_low, keep_lowest, params=p, adjoint_window=window
        )
    )
    differentiated = count(
        lambda: jax.grad(
            lambda q: jnp.sum(
                sx.block_thomas_truncated_fn(
                    block_fn, n_blocks, rhs_low, keep_lowest,
                    params=q, adjoint_window=window,
                )
                ** 2
            )
        )(p)
    )
    return forward, differentiated


def test_generator_call_count_matches_the_published_model() -> None:
    shapes = [(24, 2, 6), (24, 2, 12), (48, 2, 6), (48, 2, 24), (96, 2, 6), (24, 4, 6)]
    multipliers = {}
    for n_blocks, keep_lowest, window in shapes:
        forward, differentiated = _generator_calls(n_blocks, keep_lowest, window)
        assert forward == n_blocks, (
            f"N={n_blocks}: forward-only solve made {forward} generator calls, "
            f"expected one per row"
        )
        excess = differentiated - window - keep_lowest
        assert excess % n_blocks == 0, (
            f"N={n_blocks} W={window} K={keep_lowest}: {differentiated} calls is "
            f"not N*k + W + K for integer k; the cost model in the manuscript "
            f"assumes it is"
        )
        multipliers[(n_blocks, keep_lowest, window)] = excess // n_blocks

    distinct = set(multipliers.values())
    assert len(distinct) == 1, (
        f"the per-row multiplier is not constant across shapes: {multipliers}. "
        f"The manuscript quotes a single constant."
    )
    (k,) = distinct
    assert 2 <= k <= 6, (
        f"per-row multiplier is {k}; the manuscript describes a small constant "
        f"multiple of N, and {k} is not one. Either the implementation changed "
        f"or the model needs rewriting."
    )
