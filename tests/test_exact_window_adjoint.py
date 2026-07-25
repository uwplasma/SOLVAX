"""Exactness contract of the exact-window localized adjoint.

Each test names the statement it pins. The construction differentiates a
generated block-tridiagonal chain whose rows are produced on demand from a
compact parameter, and the reference is ordinary reverse mode through a dense
assembly of the *same* generator, so the comparison isolates the adjoint rule
rather than the discretization.

Contract under test (``K`` source/output blocks, window ``w``,
``W = min(K+w, N)``, primal halo ``M = min(W+1, N)``):

* the selected forward blocks are exact blocks of the full ``N``-row solution;
* the right-hand-side cotangent is exact and independent of ``w``;
* every retained row cotangent ``j < W`` is exact;
* a parameter supported inside the window is differentiated exactly;
* at ``W = N`` the whole gradient is exact;
* the only error is the omitted tail, which decays geometrically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from solvax.direct import (  # noqa: E402
    _block_thomas_selected_fn_state,
    _retained_row_cotangents,
    block_thomas_truncated_fn,
)

M_BLOCK = 3


def _chain(n_blocks, seed=0, dominance=3.0, complex_=False):
    """Block-dominant generated chain plus a dense assembler for references."""
    rng = np.random.default_rng(seed)

    def draw(*shape):
        a = rng.standard_normal(shape)
        return a + 1j * rng.standard_normal(shape) if complex_ else a

    m = M_BLOCK
    scale = 0.25
    base_l = jnp.asarray(draw(n_blocks, m, m) * scale)
    base_u = jnp.asarray(draw(n_blocks, m, m) * scale)
    base_d = jnp.asarray(draw(n_blocks, m, m) * scale + dominance * m * np.eye(m))

    def block_fn(params, j):
        j = jnp.asarray(j)
        jf = j.astype(base_d.real.dtype)
        # a compact parameter that enters *every* row: the hard case, where the
        # omitted tail is genuinely nonzero
        s = 1.0 + params[0] * jnp.cos(jf) + params[1] * 0.1 * jf
        return base_l[j] * s, base_d[j] * (1.0 + params[0]), base_u[j] * s

    def dense(params):
        idx = jnp.arange(n_blocks)
        lo, di, up = jax.vmap(lambda j: block_fn(params, j))(idx)
        a = jnp.zeros((n_blocks * m, n_blocks * m), dtype=di.dtype)
        for j in range(n_blocks):
            a = a.at[j * m : (j + 1) * m, j * m : (j + 1) * m].add(di[j])
            if j + 1 < n_blocks:
                a = a.at[j * m : (j + 1) * m, (j + 1) * m : (j + 2) * m].add(up[j])
                a = a.at[(j + 1) * m : (j + 2) * m, j * m : (j + 1) * m].add(lo[j + 1])
        return a

    return block_fn, dense


def _sq(y):
    return jnp.real(jnp.sum(y * jnp.conj(y)))


def _reference_grads(block_fn, dense, n_blocks, keep, params, rhs):
    """Exact gradients by ordinary reverse mode through the dense assembly."""

    def loss(p, b):
        a = dense(p)
        full = jnp.zeros((n_blocks * M_BLOCK,), dtype=a.dtype)
        full = full.at[: keep * M_BLOCK].set(b.reshape(-1))
        x = jnp.linalg.solve(a, full).reshape(n_blocks, M_BLOCK)
        return _sq(x[:keep])

    return jax.grad(loss, argnums=(0, 1))(params, rhs)


# --------------------------------------------------------------- forward ----


@pytest.mark.parametrize(
    "n_blocks,source,retain",
    [(1, 1, 1), (6, 1, 1), (6, 2, 2), (6, 2, 3), (6, 2, 6), (6, 6, 6)],
)
def test_selected_head_returns_exact_full_solution_blocks(n_blocks, source, retain):
    """Selected-head forward result: exact blocks of the *full* solve.

    This is the property that distinguishes a full-tail selected-head
    elimination from a leading-principal approximation.
    """
    block_fn, dense = _chain(n_blocks, seed=1)
    params = jnp.asarray([0.11, 0.07])
    rng = np.random.default_rng(2)
    rhs_low = jnp.asarray(rng.standard_normal((source, M_BLOCK)))

    a = dense(params)
    full = jnp.zeros((n_blocks * M_BLOCK,))
    full = full.at[: source * M_BLOCK].set(rhs_low.reshape(-1))
    x_full = jnp.linalg.solve(a, full).reshape(n_blocks, M_BLOCK)

    got, *_ = _block_thomas_selected_fn_state(
        lambda j: block_fn(params, j), n_blocks, rhs_low, source, retain
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(x_full[:retain]), atol=1e-11)


def test_multiple_right_hand_sides_and_vmap():
    """Forward exactness with several columns and under ``vmap``."""
    n_blocks, keep = 7, 2
    block_fn, dense = _chain(n_blocks, seed=3)
    params = jnp.asarray([0.1, 0.05])
    rng = np.random.default_rng(4)
    rhs = jnp.asarray(rng.standard_normal((keep, M_BLOCK, 2)))

    got = block_thomas_truncated_fn(
        block_fn, n_blocks, rhs, keep, params=params, adjoint_window=2
    )
    a = dense(params)
    for col in range(2):
        full = jnp.zeros((n_blocks * M_BLOCK,))
        full = full.at[: keep * M_BLOCK].set(rhs[..., col].reshape(-1))
        ref = jnp.linalg.solve(a, full).reshape(n_blocks, M_BLOCK)[:keep]
        np.testing.assert_allclose(
            np.asarray(got[..., col]), np.asarray(ref), atol=1e-11
        )

    batched = jax.vmap(
        lambda b: block_thomas_truncated_fn(
            block_fn, n_blocks, b, keep, params=params, adjoint_window=2
        )
    )(jnp.stack([rhs[..., 0], rhs[..., 1]]))
    np.testing.assert_allclose(
        np.asarray(batched[0]), np.asarray(got[..., 0]), atol=1e-11
    )


# ------------------------------------------------------- exact rhs adjoint ---


@pytest.mark.parametrize("complex_", [False, True])
def test_rhs_cotangent_is_exact_and_window_independent(complex_):
    """``bar b = P_K lambda`` is exact at every window, including ``w = 0``.

    Only invertibility is used, so this must hold with no decay assumption and
    must not move with ``adjoint_window`` beyond floating-point noise.
    """
    n_blocks, keep = 9, 2
    block_fn, dense = _chain(n_blocks, seed=5, complex_=complex_)
    params = jnp.asarray([0.13 + 0j, 0.04 + 0j] if complex_ else [0.13, 0.04])
    rng = np.random.default_rng(6)
    rhs = rng.standard_normal((keep, M_BLOCK))
    if complex_:
        rhs = rhs + 1j * rng.standard_normal((keep, M_BLOCK))
    rhs = jnp.asarray(rhs)

    _, ref = _reference_grads(block_fn, dense, n_blocks, keep, params, rhs)
    base = None
    for w in (0, 1, 3, n_blocks):
        got = jax.grad(
            lambda b, w=w: _sq(
                block_thomas_truncated_fn(
                    block_fn, n_blocks, b, keep, params=params, adjoint_window=w
                )
            )
        )(rhs)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-10)
        if base is None:
            base = got
        np.testing.assert_allclose(np.asarray(got), np.asarray(base), atol=1e-12)


# --------------------------------------------- exact retained-row cotangents -


def test_retained_row_cotangents_match_full_state_outer_products():
    """Every retained row ``j < W`` is exact, including the halo row ``W-1``.

    The halo row is the most likely off-by-one: its upper-block cotangent needs
    the primal block ``x_W`` that lies beyond the returned head.
    """
    n_blocks, keep, w = 10, 2, 3
    block_fn, dense = _chain(n_blocks, seed=7)
    params = jnp.asarray([0.09, 0.03])
    rng = np.random.default_rng(8)
    rhs = jnp.asarray(rng.standard_normal((keep, M_BLOCK)))

    retained = min(keep + w, n_blocks)
    halo = min(retained + 1, n_blocks)

    a = dense(params)
    full_b = jnp.zeros((n_blocks * M_BLOCK,)).at[: keep * M_BLOCK].set(rhs.reshape(-1))
    x_full = jnp.linalg.solve(a, full_b).reshape(n_blocks, M_BLOCK)
    ct = 2.0 * x_full[:keep]
    full_q = jnp.zeros((n_blocks * M_BLOCK,)).at[: keep * M_BLOCK].set(ct.reshape(-1))
    lam_full = jnp.linalg.solve(a.T, full_q).reshape(n_blocks, M_BLOCK)

    lower_bar, diag_bar, upper_bar = _retained_row_cotangents(
        lam_full[:retained], x_full[:halo]
    )
    for j in range(retained):
        below = x_full[j - 1] if j >= 1 else jnp.zeros((M_BLOCK,))
        above = x_full[j + 1] if j + 1 < n_blocks else jnp.zeros((M_BLOCK,))
        np.testing.assert_allclose(
            np.asarray(lower_bar[j]), -np.outer(lam_full[j], below), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(diag_bar[j]), -np.outer(lam_full[j], x_full[j]), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(upper_bar[j]), -np.outer(lam_full[j], above), atol=1e-10
        )


# ------------------------------------------------------ exact special cases -


@pytest.mark.parametrize("complex_", [False, True])
def test_full_window_gradient_is_exact(complex_):
    """``W = N``: the omitted tail is empty, so the gradient is exact."""
    n_blocks, keep = 8, 2
    block_fn, dense = _chain(n_blocks, seed=9, complex_=complex_)
    params = jnp.asarray([0.12 + 0j, 0.05 + 0j] if complex_ else [0.12, 0.05])
    rng = np.random.default_rng(10)
    rhs = rng.standard_normal((keep, M_BLOCK))
    if complex_:
        rhs = rhs + 1j * rng.standard_normal((keep, M_BLOCK))
    rhs = jnp.asarray(rhs)

    ref_p, ref_b = _reference_grads(block_fn, dense, n_blocks, keep, params, rhs)
    got_p, got_b = jax.grad(
        lambda p, b: _sq(
            block_thomas_truncated_fn(
                block_fn, n_blocks, b, keep, params=p, adjoint_window=n_blocks
            )
        ),
        argnums=(0, 1),
    )(params, rhs)
    np.testing.assert_allclose(np.asarray(got_p), np.asarray(ref_p), atol=1e-10)
    np.testing.assert_allclose(np.asarray(got_b), np.asarray(ref_b), atol=1e-10)


def test_locally_supported_parameter_is_exact_at_any_window():
    """A parameter acting only on retained rows is differentiated exactly.

    Corollary of the exact tail identity: every omitted term pairs with a
    generator derivative that vanishes, so a short window loses nothing.
    """
    n_blocks, keep, w = 10, 2, 1
    retained = keep + w
    _, dense_ignored = _chain(n_blocks, seed=11)
    del dense_ignored
    rng = np.random.default_rng(12)
    m = M_BLOCK
    base_l = jnp.asarray(rng.standard_normal((n_blocks, m, m)) * 0.25)
    base_u = jnp.asarray(rng.standard_normal((n_blocks, m, m)) * 0.25)
    base_d = jnp.asarray(
        rng.standard_normal((n_blocks, m, m)) * 0.25 + 3.0 * m * np.eye(m)
    )

    def block_fn(params, j):
        j = jnp.asarray(j)
        # support confined to row 1 only, which is inside the window
        local = jnp.where(j == 1, params[0], 0.0)
        return base_l[j], base_d[j] * (1.0 + local), base_u[j]

    params = jnp.asarray([0.2])
    rhs = jnp.asarray(rng.standard_normal((keep, m)))

    grads = [
        float(
            jax.grad(
                lambda p, w=w: _sq(
                    block_thomas_truncated_fn(
                        block_fn, n_blocks, rhs, keep, params=p, adjoint_window=w
                    )
                )
            )(params)[0]
        )
        for w in (w, n_blocks)
    ]
    assert retained <= n_blocks
    np.testing.assert_allclose(grads[0], grads[1], rtol=1e-9)


# ------------------------------------------------------------- convergence --


def test_tail_error_decays_geometrically_and_beats_no_window():
    """Omitted-tail error decays geometrically in ``w`` for a shared parameter.

    The theorem bounds an *upper envelope*; the realized vector error can
    cancel between rows, so this checks the envelope and the asymptotic trend
    rather than strict monotonicity at every adjacent window.
    """
    n_blocks, keep = 14, 2
    block_fn, dense = _chain(n_blocks, seed=13)
    params = jnp.asarray([0.11, 0.07])
    rng = np.random.default_rng(14)
    rhs = jnp.asarray(rng.standard_normal((keep, M_BLOCK)))
    ref_p, _ = _reference_grads(block_fn, dense, n_blocks, keep, params, rhs)

    errors = []
    for w in range(0, 7):
        got = jax.grad(
            lambda p, w=w: _sq(
                block_thomas_truncated_fn(
                    block_fn, n_blocks, rhs, keep, params=p, adjoint_window=w
                )
            )
        )(params)
        errors.append(float(jnp.linalg.norm(got - ref_p)))

    assert errors[0] > errors[-1] * 1e6  # many orders gained over the window
    assert errors[-1] < 1e-12  # converged to the exact gradient
    # geometric envelope: each added window block buys at least ~one decade
    assert errors[3] < errors[1] * 1e-2


def test_gradient_is_jit_compatible_and_stable_under_pytree_params():
    """``jit`` transparency and PyTree-structured parameters."""
    n_blocks, keep, w = 9, 2, 2
    rng = np.random.default_rng(15)
    m = M_BLOCK
    base_d = jnp.asarray(
        rng.standard_normal((n_blocks, m, m)) * 0.2 + 3.0 * m * np.eye(m)
    )
    base_l = jnp.asarray(rng.standard_normal((n_blocks, m, m)) * 0.2)
    base_u = jnp.asarray(rng.standard_normal((n_blocks, m, m)) * 0.2)

    def block_fn(params, j):
        j = jnp.asarray(j)
        jf = j.astype(jnp.float64)
        s = 1.0 + params["scale"] * jnp.cos(jf)
        return base_l[j] * s, base_d[j] * (1.0 + params["shift"][0]), base_u[j] * s

    params = {"scale": jnp.asarray(0.1), "shift": jnp.asarray([0.05])}
    rhs = jnp.asarray(rng.standard_normal((keep, m)))

    def loss(p):
        return _sq(
            block_thomas_truncated_fn(
                block_fn, n_blocks, rhs, keep, params=p, adjoint_window=w
            )
        )

    eager = jax.grad(loss)(params)
    compiled = jax.jit(jax.grad(loss))(params)
    np.testing.assert_allclose(
        np.asarray(eager["scale"]), np.asarray(compiled["scale"]), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(eager["shift"]), np.asarray(compiled["shift"]), rtol=1e-12
    )


def test_reverse_memory_is_flat_in_chain_length():
    """Fixed ``(K, w, m)`` and compact parameters: adjoint workspace flat in ``N``.

    The arithmetic still scales with ``N`` -- every row is visited. What must be
    independent of ``N`` is the retained state, which is what the compiled
    temporary size reports here.
    """
    keep, w, m = 2, 2, M_BLOCK
    rng = np.random.default_rng(16)
    temps = []
    for n_blocks in (32, 64, 128, 256):
        base_d = jnp.asarray(
            rng.standard_normal((1, m, m)) * 0.2 + 3.0 * m * np.eye(m)
        )[0]
        base_l = jnp.asarray(rng.standard_normal((1, m, m)) * 0.2)[0]
        base_u = jnp.asarray(rng.standard_normal((1, m, m)) * 0.2)[0]

        def block_fn(params, j, d=base_d, lo=base_l, up=base_u):
            jf = jnp.asarray(j).astype(jnp.float64)
            s = 1.0 + params[0] * jnp.cos(jf)
            return lo * s, d * (1.0 + params[0]), up * s

        rhs = jnp.asarray(rng.standard_normal((keep, m)))

        def loss(p, n=n_blocks, bf=block_fn, b=rhs):
            return _sq(
                block_thomas_truncated_fn(
                    bf, n, b, keep, params=p, adjoint_window=w
                )
            )

        compiled = jax.jit(jax.grad(loss)).lower(jnp.asarray([0.1])).compile()
        temps.append(int(compiled.memory_analysis().temp_size_in_bytes))

    assert max(temps) <= min(temps) * 1.5, f"reverse temp not flat in N: {temps}"
