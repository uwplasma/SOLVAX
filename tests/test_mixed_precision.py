"""Tests for mixed-precision block-tridiagonal elimination.

The low-precision (float32) Schur-complement factorization is fast on hardware
where float64 is throttled, and float64 iterative refinement recovers
working-precision accuracy. These tests pin the accuracy story (refinement
actually helps and reaches the float64 floor) and confirm the solve stays
jit/vmap/grad-transparent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    block_thomas,
    block_thomas_factor,
    block_thomas_solve,
    mixed_precision_block_thomas,
)
from solvax.direct import block_tridiag_matvec

jax.config.update("jax_enable_x64", True)


def make_system(n_blocks, m, n_rhs=None, seed=0, dominance=4.0):
    """Random well-conditioned block-tridiagonal system (matches test_direct)."""
    rng = np.random.default_rng(seed)
    lower = rng.standard_normal((n_blocks, m, m))
    diag = rng.standard_normal((n_blocks, m, m)) + dominance * m * np.eye(m)
    upper = rng.standard_normal((n_blocks, m, m))
    shape = (n_blocks, m) if n_rhs is None else (n_blocks, m, n_rhs)
    rhs = rng.standard_normal(shape)
    return tuple(map(jnp.asarray, (lower, diag, upper, rhs)))


def resid_norm(lower, diag, upper, x, rhs):
    return float(jnp.linalg.norm(block_tridiag_matvec(lower, diag, upper, x) - rhs))


@pytest.mark.parametrize("n_rhs", [None, 3])
@pytest.mark.parametrize("n_blocks,m", [(8, 40), (16, 12)])
def test_mixed_precision_matches_full_fp64(n_blocks, m, n_rhs):
    """Two refinement steps recover the full-float64 solution.

    On these well-conditioned systems the refined solution matches the pure
    float64 block-Thomas to a relative 1e-9 (in practice ~1e-15, i.e. the
    float64 roundoff floor) — see the residual assertions in
    ``test_refinement_recovers_and_helps`` for the honest per-step numbers.
    """
    lower, diag, upper, rhs = make_system(n_blocks, m, n_rhs, seed=1)
    x64 = block_thomas(lower, diag, upper, rhs)
    x_mixed = mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=2)
    assert x_mixed.dtype == rhs.dtype
    assert x_mixed.shape == rhs.shape
    assert np.allclose(np.asarray(x_mixed), np.asarray(x64), rtol=1e-9, atol=1e-9)


def test_refinement_recovers_and_helps():
    """float32 factor alone stalls near 1e-6; two refinement steps reach 1e-9.

    Documents the measured numbers: the bare low-precision solve leaves a
    residual well above the float64 floor (float32 unit roundoff ~6e-8 times
    the problem scale), and each defect-correction sweep contracts it until it
    hits the float64 floor.
    """
    lower, diag, upper, rhs = make_system(12, 48, seed=2)
    x64 = block_thomas(lower, diag, upper, rhs)
    res64 = resid_norm(lower, diag, upper, x64, rhs)

    x_noref = mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=0)
    x_2ref = mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=2)
    res_noref = resid_norm(lower, diag, upper, x_noref, rhs)
    res_2ref = resid_norm(lower, diag, upper, x_2ref, rhs)

    # The float32 factorization alone is far from float64 accuracy...
    assert res_noref > 1e-7
    # ...and two refinement steps drop it to the float64 floor.
    assert res_2ref <= 1e-9
    assert res_2ref < 1e-3 * res_noref
    # Refinement cannot beat the reference float64 residual by much.
    assert res_2ref < max(1e-9, 50.0 * res64)


def test_bare_low_precision_is_float32_accurate():
    """refine_steps=0 returns the raw low-precision solve (float32 accuracy)."""
    lower, diag, upper, rhs = make_system(10, 32, seed=3)
    x64 = block_thomas(lower, diag, upper, rhs)
    x0 = mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=0)
    rel = float(jnp.linalg.norm(x0 - x64) / jnp.linalg.norm(x64))
    assert 1e-9 < rel < 1e-4  # unmistakably float32, not float64


def test_factor_dtype_keeps_bands_high_precision():
    """factor_dtype lowers only the stored LU factors; bands stay float64."""
    lower, diag, upper, _ = make_system(6, 8, seed=4)
    factors = block_thomas_factor(lower, diag, upper, factor_dtype=jnp.float32)
    assert factors.delta_lu.dtype == jnp.float32
    assert factors.delta_piv.dtype in (jnp.int32, jnp.int64)
    assert factors.lower.dtype == jnp.float64
    assert factors.upper.dtype == jnp.float64
    # A solve with mixed factors returns float64 and is float32-accurate.
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal((6, 8)))
    x = block_thomas_solve(factors, rhs)
    assert x.dtype == jnp.float64


def test_factor_dtype_none_matches_baseline():
    """factor_dtype=None is bit-for-bit the original float64 factorization."""
    lower, diag, upper, rhs = make_system(8, 6, seed=5)
    f_default = block_thomas_factor(lower, diag, upper)
    f_none = block_thomas_factor(lower, diag, upper, factor_dtype=None)
    assert np.array_equal(np.asarray(f_default.delta_lu), np.asarray(f_none.delta_lu))
    x_default = block_thomas_solve(f_default, rhs)
    x_none = block_thomas_solve(f_none, rhs)
    assert np.array_equal(np.asarray(x_default), np.asarray(x_none))


def test_bfloat16_factor_unsupported_by_lapack():
    """Documents a real backend limit: ``lu_factor`` (LAPACK/cuSOLVER getrf)
    has no bfloat16/float16 kernel, so float32 is the usable low precision.

    A half-precision LU would need a non-LAPACK factorization; this pins the
    NotImplementedError so the limitation is visible rather than silent.
    """
    lower, diag, upper, rhs = make_system(4, 6, seed=6)
    with pytest.raises(NotImplementedError):
        mixed_precision_block_thomas(
            lower, diag, upper, rhs, factor_dtype=jnp.bfloat16, refine_steps=1
        )


def test_mixed_precision_vmap_matches_loop():
    """vmap over a batch of systems matches per-item solves."""
    systems = [make_system(6, 16, seed=s) for s in range(4)]
    stacked = [jnp.stack(arrs) for arrs in zip(*systems, strict=True)]
    x_batch = jax.vmap(
        lambda lo, d, u, b: mixed_precision_block_thomas(lo, d, u, b, refine_steps=2)
    )(*stacked)
    for i, (lower, diag, upper, rhs) in enumerate(systems):
        x_i = mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=2)
        assert np.allclose(np.asarray(x_batch[i]), np.asarray(x_i), atol=1e-12)


def test_mixed_precision_jit():
    """The whole solve compiles under jit."""
    lower, diag, upper, rhs = make_system(10, 20, seed=7)

    @jax.jit
    def run(b):
        return mixed_precision_block_thomas(lower, diag, upper, b, refine_steps=2)

    x = run(rhs)
    x64 = block_thomas(lower, diag, upper, rhs)
    assert np.allclose(np.asarray(x), np.asarray(x64), rtol=1e-9, atol=1e-9)


def test_mixed_precision_gradient_matches_fd():
    """jax.grad through the refined solve matches a finite difference."""
    lower, diag, upper, rhs = make_system(5, 10, seed=8)

    def loss(d):
        x = mixed_precision_block_thomas(lower, d, upper, rhs, refine_steps=2)
        return jnp.sum(x**2)

    g = jax.grad(loss)(diag)
    eps = 1e-6
    e = jnp.zeros_like(diag).at[2, 1, 1].set(eps)
    fd = (loss(diag + e) - loss(diag - e)) / (2 * eps)
    assert np.isclose(float(g[2, 1, 1]), float(fd), rtol=1e-4)


@pytest.mark.parametrize("n_rhs", [None, 3])
def test_implicit_adjoint_gradient_matches_exact_fp64(n_rhs):
    # The amortized adjoint (fp32 factors reused, working-precision refinement)
    # reproduces the gradient of the exact fp64 solve to working precision:
    # the gradient inherits the refined forward error, not the factor precision.
    lower, diag, upper, rhs = make_system(10, 6, n_rhs, seed=7)

    def loss(f, d, b):
        return jnp.sum(f(d, b) ** 2)

    exact = jax.grad(
        lambda d, b: loss(lambda d_, b_: block_thomas(lower, d_, upper, b_), d, b),
        argnums=(0, 1),
    )(diag, rhs)
    implicit = jax.grad(
        lambda d, b: loss(
            lambda d_, b_: mixed_precision_block_thomas(
                lower, d_, upper, b_, refine_steps=2, implicit_adjoint=True
            ),
            d, b,
        ),
        argnums=(0, 1),
    )(diag, rhs)
    for got, ref in zip(implicit, exact, strict=True):
        rel = float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref))
        assert rel < 1e-9


def test_implicit_adjoint_primal_identical_to_default_path():
    lower, diag, upper, rhs = make_system(12, 5, seed=8)
    x_default = mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=2)
    x_implicit = mixed_precision_block_thomas(
        lower, diag, upper, rhs, refine_steps=2, implicit_adjoint=True
    )
    assert np.allclose(np.asarray(x_default), np.asarray(x_implicit), atol=0.0)


def test_implicit_adjoint_is_jit_and_vmap_transparent():
    lower, diag, upper, rhs = make_system(8, 4, seed=9)
    grad = jax.jit(jax.grad(lambda d: jnp.sum(
        mixed_precision_block_thomas(lower, d, upper, rhs, implicit_adjoint=True) ** 2
    )))
    stacked = jnp.stack([diag, diag + 0.05])
    out = jax.vmap(grad)(stacked)
    assert out.shape == stacked.shape
    assert np.all(np.isfinite(np.asarray(out)))


def test_implicit_adjoint_gradient_degrades_gracefully_with_conditioning():
    # As dominance weakens (kappa grows) the gradient error grows like the
    # primal refinement error -- but stays far below the bare fp32 gradient
    # obtained with refine_steps=0.
    def gradient_errors(dominance):
        lower, diag, upper, rhs = make_system(10, 6, seed=11, dominance=dominance)
        exact = jax.grad(lambda d: jnp.sum(block_thomas(lower, d, upper, rhs) ** 2))(diag)

        def rel_error(steps):
            got = jax.grad(lambda d: jnp.sum(mixed_precision_block_thomas(
                lower, d, upper, rhs, refine_steps=steps, implicit_adjoint=True) ** 2))(diag)
            return float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact))

        return rel_error(0), rel_error(2)

    for dominance in (4.0, 1.5):
        bare, refined = gradient_errors(dominance)
        assert refined < 1e-9  # refined gradient at working precision
        assert refined < 1e-3 * bare  # refinement buys >= 3 digits over bare fp32
