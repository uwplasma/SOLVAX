"""Operator-coupled block Thomas against the stored-band route."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    block_thomas_factor,
    block_thomas_factor_ops,
    block_thomas_solve,
    block_thomas_solve_ops,
    block_tridiag_relative_residual,
)

jax.config.update("jax_enable_x64", True)

T, Z = 5, 4
M = T * Z


def _stencil(weights, u, transpose):
    """Periodic 5-point ``S = w_c + sum_d diag(w_d) P_d`` on ``(m, r)`` operands."""
    u = u.reshape(T, Z, -1)
    shifts = ((-1, 0), (1, 0), (-1, 1), (1, 1))
    out = weights[0, ..., None] * u
    for w, (step, axis) in zip(weights[1:], shifts, strict=True):
        if transpose:  # (diag(w) P)^T = P^T diag(w)
            out = out + jnp.roll(w[..., None] * u, -step, axis)
        else:
            out = out + w[..., None] * jnp.roll(u, step, axis)
    return out.reshape(M, -1)


def stencil_couple(params, k, z, *, which, transpose):
    """``L_k = a_k S + b_k diag(mu)``, ``U_k = c_k S + d_k diag(mu)``."""
    weights, mu, coef, scale = params
    a, b = (coef[k, 0], coef[k, 1]) if which == "lower" else (coef[k, 2], coef[k, 3])
    return scale * (a * _stencil(weights, z, transpose) + b * mu[:, None] * z)


def dense_couple(params, k, z, *, which, transpose):
    lower, upper = params
    block = (lower if which == "lower" else upper)[k]
    return (jnp.swapaxes(block, -1, -2) if transpose else block) @ z


def make_problem(kind, n_blocks, n_rhs=None, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    diag = jnp.asarray(0.3 * rng.standard_normal((n_blocks, M, M)) + 25.0 * np.eye(M))
    shape = (n_blocks, M) if n_rhs is None else (n_blocks, M, n_rhs)
    rhs = jnp.asarray(rng.standard_normal(shape))
    if kind == "stencil":
        params = (
            jnp.asarray(rng.uniform(0.5, 1.5, (5, T, Z))),
            jnp.asarray(rng.standard_normal(M)),
            jnp.asarray(rng.uniform(-1.0, 1.0, (n_blocks, 4))),
            jnp.asarray(scale),
        )
        couple = stencil_couple
    else:  # finite, nonzero boundary blocks must be ignored by both routes
        params = tuple(jnp.asarray(rng.standard_normal((n_blocks, M, M))) for _ in range(2))
        couple = dense_couple
    eye = jnp.eye(M)
    bands = [
        jnp.stack(
            [couple(params, jnp.int32(k), eye, which=w, transpose=False) for k in range(n_blocks)]
        )
        for w in ("lower", "upper")
    ]
    return couple, params, (bands[0], diag, bands[1]), rhs


def rel(actual, expected):
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


def test_stencil_transpose_action_is_the_transpose():
    couple, params, (lower, _, upper), _ = make_problem("stencil", 3)
    for which, band in (("lower", lower), ("upper", upper)):
        t = couple(params, jnp.int32(1), jnp.eye(M), which=which, transpose=True)
        assert rel(t, band[1].T) <= 1e-14


@pytest.mark.parametrize("kind", ["stencil", "dense"])
@pytest.mark.parametrize("n_rhs", [None, 3])
@pytest.mark.parametrize("n_blocks", [1, 6])
def test_matches_stored_band_route(kind, n_rhs, n_blocks):
    couple, params, bands, rhs = make_problem(kind, n_blocks, n_rhs, seed=1)
    factors = block_thomas_factor_ops(bands[1], couple, params)
    reference = block_thomas_factor(*bands)
    assert factors.blocks.shape == (n_blocks, M, M)
    assert factors.pivots.shape == (n_blocks, M)
    for transpose in (False, True):
        actual = block_thomas_solve_ops(factors, rhs, transpose=transpose)
        expected = block_thomas_solve(reference, rhs, transpose=transpose)
        assert actual.shape == rhs.shape
        assert rel(actual, expected) <= 1e-12
    residual = block_tridiag_relative_residual(*bands, block_thomas_solve_ops(factors, rhs), rhs)
    assert float(jnp.max(residual)) <= 1e-13


def test_callable_diag_matches_array():
    couple, params, bands, rhs = make_problem("stencil", 5, seed=2)
    diag = bands[1]
    from_fn = block_thomas_factor_ops(lambda k: diag[k], couple, params, n_blocks=5)
    from_array = block_thomas_factor_ops(diag, couple, params)
    assert rel(from_fn.blocks, from_array.blocks) <= 1e-14
    with pytest.raises(ValueError, match="n_blocks is required"):
        block_thomas_factor_ops(lambda k: diag[k], couple, params)


def test_jit_vmap_factors_cross_transformation_boundaries():
    """Batched coefficients live in ``params``, so factors leave ``jit(vmap)``."""
    n_blocks, batch = 5, 3
    scales = jnp.asarray([0.5, 1.0, 1.5])
    problems = [make_problem("stencil", n_blocks, 2, seed=3, scale=s) for s in (0.5, 1.0, 1.5)]
    couple, base, (_, diag, _), _ = problems[0]
    rhs = jnp.stack([p[3] for p in problems])

    def factor(scale):
        return block_thomas_factor_ops(diag, couple, (*base[:3], scale))

    factors = jax.jit(jax.vmap(factor))(scales)
    assert factors.blocks.shape == (batch, n_blocks, M, M)
    for transpose in (False, True):
        solve = jax.jit(jax.vmap(lambda f, r, t=transpose: block_thomas_solve_ops(f, r, t)))
        actual = solve(factors, rhs)
        for i, (_, _, bands, _) in enumerate(problems):
            expected = block_thomas_solve(block_thomas_factor(*bands), rhs[i], transpose)
            assert rel(actual[i], expected) <= 1e-12


@pytest.mark.parametrize("n_rhs", [None, 2])
def test_linear_transpose_and_grad_match_dense_route(n_rhs):
    couple, params, bands, rhs = make_problem("stencil", 6, n_rhs, seed=4)
    factors = block_thomas_factor_ops(bands[1], couple, params)
    reference = block_thomas_factor(*bands)
    cotangent = jnp.asarray(np.random.default_rng(5).standard_normal(rhs.shape))

    def solve(r):
        return block_thomas_solve_ops(factors, r)

    transposed = block_thomas_solve_ops(factors, cotangent, transpose=True)
    (eager,) = jax.linear_transpose(solve, rhs)(cotangent)
    (jitted,) = jax.jit(jax.linear_transpose(solve, rhs))(cotangent)
    assert rel(eager, transposed) <= 1e-12
    assert rel(jitted, transposed) <= 1e-12
    assert rel(transposed, block_thomas_solve(reference, cotangent, transpose=True)) <= 1e-12

    for transpose in (False, True):

        def loss(r, route, t=transpose):
            return jnp.sum(jnp.sin(route(r, t)) * cotangent)

        ops_route = lambda r, t: block_thomas_solve_ops(factors, r, t)  # noqa: E731
        band_route = lambda r, t: block_thomas_solve(reference, r, t)  # noqa: E731
        expected = jax.grad(loss)(rhs, band_route)
        assert rel(jax.grad(loss)(rhs, ops_route), expected) <= 1e-12
        assert rel(jax.jit(jax.grad(loss), static_argnums=1)(rhs, ops_route), expected) <= 1e-12


def test_solve_temporaries_stay_below_one_factor_band():
    """Neither vmap nor linear_transpose nor reverse mode may copy the band."""
    couple, params, bands, rhs = make_problem("stencil", 40, seed=8)
    factors = block_thomas_factor_ops(bands[1], couple, params)
    batched = jax.vmap(lambda s: block_thomas_factor_ops(bands[1], couple, (*params[:3], s)))(
        jnp.asarray([1.0, 2.0])
    )
    cases = {
        "vmap": (jax.vmap(block_thomas_solve_ops), (batched, jnp.stack([rhs, -rhs]))),
        "linear_transpose": (
            lambda c: jax.linear_transpose(lambda r: block_thomas_solve_ops(factors, r), rhs)(c)[0],
            (rhs,),
        ),
        "grad": (jax.grad(lambda r: jnp.sum(jnp.sin(block_thomas_solve_ops(factors, r)))), (rhs,)),
    }
    for name, (fn, args) in cases.items():
        try:
            analysis = jax.jit(fn).lower(*args).compile().memory_analysis()
        except AttributeError:
            pytest.skip("this JAX has no compiled memory analysis")
        if analysis is None:
            pytest.skip("backend reports no memory analysis")
        band = (batched if name == "vmap" else factors).blocks.nbytes
        assert analysis.temp_size_in_bytes < 0.5 * band, (name, analysis.temp_size_in_bytes, band)


def test_rejects_bad_arguments():
    couple, params, bands, rhs = make_problem("dense", 3, seed=7)
    with pytest.raises(ValueError, match="n_blocks disagrees"):
        block_thomas_factor_ops(bands[1], couple, params, n_blocks=4)
    factors = block_thomas_factor_ops(bands[1], couple, params)
    with pytest.raises(ValueError, match="leading dimension"):
        block_thomas_solve_ops(factors, rhs[:2])
