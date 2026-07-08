"""Tests for solvax.refine: mixed-precision iterative refinement."""

import jax
import jax.numpy as jnp
import numpy as np

from solvax import as_low_precision, iterative_refinement

jax.config.update("jax_enable_x64", True)


def make_conditioned_system(n=60, log_cond=2.5, seed=0):
    """SPD system with prescribed condition number ~10**log_cond."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0.0, log_cond, n)
    a = q @ np.diag(s) @ q.T
    b = rng.standard_normal(n)
    return jnp.asarray(a), jnp.asarray(b)


def test_refinement_recovers_float64_accuracy():
    a, b = make_conditioned_system()
    matvec = lambda v: a @ v  # noqa: E731

    # Inner solve runs entirely in float32 (Cholesky of the f32-cast matrix).
    a32 = a.astype(jnp.float32)
    inner = as_low_precision(
        lambda r: jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(a32), r)
    )

    # The pure float32 solve alone is far from float64 accuracy.
    x_low = inner(b)
    assert x_low.dtype == b.dtype  # cast back up
    res_low = float(jnp.linalg.norm(b - a @ x_low))
    assert res_low > 1e-5

    x, history = iterative_refinement(matvec, b, inner, iterations=4)
    res = float(jnp.linalg.norm(b - a @ x))
    assert res < 1e-12
    assert history.shape == (5,)
    assert np.isclose(float(history[0]), res_low, rtol=1e-6)
    assert float(history[-1]) < 1e-12

    # Residual history decreases monotonically until it stalls at the
    # float64 roundoff floor.
    h = np.asarray(history)
    floor = 1e-12
    for i in range(len(h) - 1):
        if h[i] > floor:
            assert h[i + 1] < h[i]


def test_as_low_precision_casts_down_and_back():
    seen = {}

    def solve(b):
        seen["dtype"] = b.dtype
        return 2.0 * b

    wrapped = as_low_precision(solve, dtype=jnp.float32)
    b = jnp.arange(4, dtype=jnp.float64)
    out = wrapped(b)
    assert seen["dtype"] == jnp.float32
    assert out.dtype == jnp.float64
    assert np.allclose(np.asarray(out), 2.0 * np.arange(4))


def test_refinement_jit_compatible():
    a, b = make_conditioned_system(n=20, log_cond=2.0, seed=1)
    a32 = a.astype(jnp.float32)
    inner = as_low_precision(lambda r: jnp.linalg.solve(a32, r))

    @jax.jit
    def run(b_):
        return iterative_refinement(lambda v: a @ v, b_, inner, iterations=3)

    x, history = run(b)
    assert float(jnp.linalg.norm(b - a @ x)) < 1e-12
    assert history.shape == (4,)
