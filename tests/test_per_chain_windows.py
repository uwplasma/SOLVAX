"""Per-chain adjoint windows for a batch of independent chains.

The exact tail identity is per chain already, so nothing here is a theory
change. What is being tested is that the two ways of giving each chain its own
window -- a traced mask under one static bound, and a segmented layout that
actually shrinks the retained state -- agree with the uniform path wherever
they select the same window.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx

jax.config.update("jax_enable_x64", True)

M = 4


def _chain(n_blocks, collisionality, seed=5):
    rng = np.random.default_rng(seed)
    lower = jnp.asarray(rng.standard_normal((n_blocks, M, M)) * 0.5)
    upper = jnp.asarray(rng.standard_normal((n_blocks, M, M)) * 0.5)

    def block_fn(params, j):
        jf = jnp.asarray(j).astype(jnp.float64)
        diag = (1.0 + collisionality * jf**2) * jnp.eye(M) * (1.0 + params[0])
        diag = diag + 0.1 * params[1] * jnp.ones((M, M))
        return lower[j] * (1.0 + params[1]), diag, upper[j] * (1.0 + params[1])

    return block_fn


def _grad(block_fn, n, keep, params, rhs, cotangent, window, chain_window=None):
    def head(p):
        return sx.block_thomas_truncated_fn(
            block_fn, n, rhs, keep_lowest=keep, params=p,
            adjoint_window=int(window), chain_window=chain_window,
        )

    _, pullback = jax.vjp(head, params)
    return pullback(cotangent)[0]


def _data(keep, seed=7):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.standard_normal((keep, M))),
        jnp.asarray(rng.standard_normal((keep, M))),
    )


# ------------------------------------------------------- the masked kernel ----


@pytest.mark.parametrize("window", [0, 1, 3, 6])
def test_a_saturated_mask_reproduces_the_uniform_gradient_exactly(window):
    # Not "close to": the masked kernel differs from the uniform one only by a
    # multiply by ones, so anything but bit-equality would mean the two paths
    # had diverged.
    n, keep = 30, 3
    block_fn = _chain(n, 0.01)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _data(keep)
    uniform = _grad(block_fn, n, keep, params, rhs, cotangent, window)
    masked = _grad(
        block_fn, n, keep, params, rhs, cotangent, window, chain_window=window
    )
    assert jnp.array_equal(uniform, masked)


@pytest.mark.parametrize("narrow", [0, 1, 3])
def test_a_narrow_mask_under_a_wide_bound_equals_the_narrow_uniform_window(narrow):
    n, keep = 30, 3
    block_fn = _chain(n, 0.01)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _data(keep)
    narrow_uniform = _grad(block_fn, n, keep, params, rhs, cotangent, narrow)
    wide_masked = _grad(
        block_fn, n, keep, params, rhs, cotangent, 8, chain_window=narrow
    )
    assert jnp.array_equal(narrow_uniform, wide_masked)


def test_the_source_cotangent_stays_exact_under_a_mask():
    # bar b = P_K lambda does not depend on the window, so the mask must not
    # touch it. Masking it would introduce an error the uniform path is free of.
    n, keep = 30, 3
    block_fn = _chain(n, 0.01)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _data(keep)

    def rhs_grad(window, chain_window):
        def head(b):
            return sx.block_thomas_truncated_fn(
                block_fn, n, b, keep_lowest=keep, params=params,
                adjoint_window=window, chain_window=chain_window,
            )

        return jax.vjp(head, rhs)[1](cotangent)[0]

    exact = rhs_grad(n, None)
    for window, chain_window in ((8, 0), (8, 2), (8, 8), (2, 1)):
        assert jnp.allclose(rhs_grad(window, chain_window), exact, atol=1e-12)


def test_the_mask_is_batchable_and_each_chain_gets_its_own_window():
    # The point of the mask: adjoint_window is static and so is shared by the
    # whole batch, while chain_window is data.
    n, keep = 40, 3
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _data(keep)
    collisionalities = jnp.asarray([0.003, 0.01, 0.03, 0.1])
    per_chain = jnp.asarray([12, 8, 5, 3])

    chains = [_chain(n, float(nu)) for nu in collisionalities]

    batched = jnp.stack([
        _grad(chains[i], n, keep, params, rhs, cotangent, 12,
              chain_window=jnp.int32(per_chain[i]))
        for i in range(len(chains))
    ])
    reference = jnp.stack([
        _grad(chains[i], n, keep, params, rhs, cotangent, int(per_chain[i]))
        for i in range(len(chains))
    ])
    assert jnp.array_equal(batched, reference)


def test_chain_window_without_params_is_rejected():
    n, keep = 20, 2
    block_fn = _chain(n, 0.01)
    rhs, _ = _data(keep)
    with pytest.raises(ValueError, match="chain_window requires params"):
        sx.block_thomas_truncated_fn(
            lambda j: block_fn(jnp.asarray([0.1, 0.0]), j), n, rhs, keep,
            chain_window=jnp.int32(2),
        )


# ------------------------------------------------------------ the planner ----


def test_one_bucket_is_the_uniform_baseline():
    plan = sx.plan_chain_windows([3, 7, 12, 40], keep_lowest=2, max_buckets=1)
    assert plan.retained_rows == plan.uniform_retained_rows == 4 * (2 + 40)
    assert plan.reduction == 0.0
    assert len(plan.buckets) == 1


def test_more_buckets_never_cost_more_and_approach_the_per_chain_ideal():
    windows = [5, 6, 8, 11, 15, 19, 24, 31, 40, 52, 66, 77]
    costs = [
        sx.plan_chain_windows(windows, 3, max_buckets=b).retained_rows
        for b in range(1, 9)
    ]
    assert costs == sorted(costs, reverse=True)
    ideal = sx.plan_chain_windows(windows, 3, max_buckets=99)
    assert ideal.retained_rows == ideal.ideal_retained_rows
    assert min(costs) >= ideal.ideal_retained_rows


def test_the_partition_covers_every_chain_exactly_once():
    windows = np.array([5, 5, 9, 20, 20, 20, 41, 77])
    plan = sx.plan_chain_windows(windows, 3, max_buckets=3)
    seen = np.concatenate([members for _, members in plan.buckets])
    assert sorted(seen.tolist()) == list(range(windows.size))


def test_every_chain_lands_in_a_bucket_at_least_as_wide_as_it_needs():
    # A chain assigned to a narrower bucket would silently lose accuracy, which
    # is the one failure mode a planner must not have.
    windows = np.array([5, 9, 9, 20, 41, 77, 77, 12])
    for buckets in (1, 2, 3, 4, 8):
        plan = sx.plan_chain_windows(windows, 3, max_buckets=buckets)
        for window, members in plan.buckets:
            assert np.all(windows[members] <= window)


def test_identical_windows_collapse_to_a_single_bucket():
    plan = sx.plan_chain_windows([11] * 7, 4, max_buckets=4)
    assert len(plan.buckets) == 1
    assert plan.retained_rows == 7 * (4 + 11) == plan.ideal_retained_rows


@pytest.mark.parametrize(
    "windows,keep,message",
    [([], 2, "non-empty"), ([[1, 2]], 2, "one-dimensional")],
)
def test_invalid_window_arrays_are_rejected(windows, keep, message):
    with pytest.raises(ValueError, match=message):
        sx.plan_chain_windows(windows, keep)


def test_max_buckets_must_be_positive():
    with pytest.raises(ValueError, match="max_buckets"):
        sx.plan_chain_windows([1, 2, 3], 2, max_buckets=0)


# ------------------------------------------------- the end-to-end statement ----


def test_bucketing_a_collisionality_scan_cuts_retained_state_and_keeps_accuracy():
    """The measurement #57 asks for, run as a test rather than asserted in prose."""
    n, keep = 40, 3
    params = jnp.asarray([0.1, 0.05])
    rhs, cotangent = _data(keep, seed=11)
    collisionalities = np.logspace(-2.8, -0.7, 6)

    windows, chains = [], []
    for nu in collisionalities:
        block_fn = _chain(n, float(nu))
        chains.append(block_fn)
        certificate = sx.certified_adjoint_window(
            block_fn, n, keep, params, rhs, cotangent, rtol=1e-6
        )
        windows.append(certificate.window)

    plan = sx.plan_chain_windows(windows, keep, max_buckets=3)
    # The scan spans the crossover, so the uniform window is dragged out by the
    # least collisional chain and bucketing has something real to save.
    assert max(windows) > min(windows)
    assert plan.retained_rows < plan.uniform_retained_rows
    assert plan.reduction > 0.2

    # And every chain still gets a gradient at least as accurate as its own
    # certified window promised.
    for window, members in plan.buckets:
        for chain in members:
            exact = _grad(chains[chain], n, keep, params, rhs, cotangent, n)
            got = _grad(chains[chain], n, keep, params, rhs, cotangent, window)
            relative = float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact))
            assert relative <= 1e-6
