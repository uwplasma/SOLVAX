"""Certified roots: residual, iterations, status, and a NaN derivative when unconverged.

Plan step S6d. Reference values are analytic: the roots of ``z^2 - p`` are
``+-sqrt(p)`` with ``dz/dp = 1 / (2 z)``; ``x^3 - 2x + 2`` has its real root at
``-1.76929235423863`` (Cardano) and plain Newton from 0 cycles 0, 1, 0, ...;
Newton on ``arctan`` diverges from ``|x0| > 1.3917``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx

jax.config.update("jax_enable_x64", True)


def test_unconverged_solver_gives_nan_gradient():
    """A solver that returns its start unchanged must not yield a finite derivative."""

    def root(p):
        res = sx.certified_root(lambda x: x**2 - p, jnp.asarray(1.0), solver=lambda f, x0: x0)
        return res.root, res.converged

    x, converged = root(2.0)
    assert not bool(converged)
    assert float(x) == 1.0
    assert np.isnan(float(jax.grad(lambda p: root(p)[0])(2.0)))


def test_user_solver_converged_keeps_gradient():
    res = sx.certified_root(
        lambda x: x**2 - 2.0, jnp.asarray(1.0), solver=lambda f, x0: jnp.sqrt(2.0)
    )
    assert bool(res.converged)
    g = jax.grad(
        lambda p: (
            sx.certified_root(
                lambda x: x**2 - p, jnp.asarray(1.0), solver=lambda f, x0: jnp.sqrt(p)
            ).root
        )
    )(2.0)
    np.testing.assert_allclose(g, 0.5 / np.sqrt(2.0), rtol=1e-14)


@pytest.mark.parametrize("p", [2.0 + 1.0j, -3.0 + 0.5j, 0.25 - 4.0j])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_complex_roots_of_z2_minus_p_and_derivative(p, sign):
    """Roots +-sqrt(p) and dz/dp = 1/(2z) to 1e-12 (holomorphic, analytic)."""
    p = jnp.asarray(p)
    z0 = sign * (1.0 + 1.0j)

    def root(p):
        return sx.certified_root(lambda z: z**2 - p, jnp.asarray(z0)).root

    res = sx.certified_root(lambda z: z**2 - p, jnp.asarray(z0))
    assert bool(res.converged)
    s = complex(np.sqrt(complex(p)))
    expect = s if abs(complex(res.root) - s) < abs(complex(res.root) + s) else -s
    np.testing.assert_allclose(complex(res.root), expect, atol=1e-12)
    # Forward-mode derivative of a holomorphic map: tangent 1 gives dz/dp.
    dz = jax.jvp(root, (p,), (jnp.asarray(1.0 + 0.0j),))[1]
    np.testing.assert_allclose(complex(dz), complex(0.5 / res.root), atol=1e-12)


def test_batch_of_1000_brackets_under_one_jit():
    """x^2 = p on [0, 2] for 1000 values of p in (0.01, 3.9): root sqrt(p), slope 1/(2 sqrt(p))."""
    ps = jnp.linspace(0.01, 3.9, 1000)

    @jax.jit
    def roots(ps):
        def one(p):
            return sx.certified_root(lambda x: x**2 - p, jnp.asarray(1.0), bracket=(0.0, 2.0)).root

        return jax.vmap(one)(ps), jax.vmap(jax.grad(one))(ps)

    r, g = roots(ps)
    np.testing.assert_allclose(r, np.sqrt(ps), rtol=1e-13)
    np.testing.assert_allclose(g, 0.5 / np.sqrt(ps), rtol=1e-10)


def test_bracket_rescues_newton_on_arctan_far_start():
    """Plain Newton on arctan diverges from x0 = 10; the bracket keeps it at 0."""
    plain = sx.certified_root(jnp.arctan, jnp.asarray(10.0), max_iter=60, damped=False)
    assert not bool(plain.converged)
    res = sx.certified_root(jnp.arctan, jnp.asarray(10.0), bracket=(-5.0, 20.0))
    assert bool(res.converged)
    assert abs(float(res.root)) < 1e-14


def test_bracket_rescues_newton_cycle_on_cubic():
    """Newton on x^3 - 2x + 2 from 0 cycles 0, 1, 0; the bracket reaches the Cardano root."""
    f = lambda x: x**3 - 2.0 * x + 2.0  # noqa: E731
    plain = sx.certified_root(f, jnp.asarray(0.0), max_iter=60, damped=False)
    assert not bool(plain.converged)
    res = sx.certified_root(f, jnp.asarray(0.0), bracket=(-3.0, 2.0))
    assert bool(res.converged)
    np.testing.assert_allclose(float(res.root), -1.76929235423863, atol=1e-13)


def test_real_root_derivative_against_analytic():
    """x = p^(1/3) from x^3 - p; dx/dp = 1 / (3 p^(2/3))."""

    def root(p):
        return sx.certified_root(lambda x: x**3 - p, jnp.asarray(1.0)).root

    p = 5.0
    np.testing.assert_allclose(root(p), p ** (1 / 3), rtol=1e-14)
    np.testing.assert_allclose(jax.grad(root)(p), 1 / (3 * p ** (2 / 3)), rtol=1e-12)


def test_double_root_converges_with_singular_slope():
    """(x - 1)^2: Newton is linear; |f| <= tol = 400 eps puts the root within sqrt(tol) = 3e-7.

    The slope dx/dp = 1/(2(x-1)) of (x-1)^2 - p is unbounded at the double root.
    """
    res = sx.certified_root(lambda x: (x - 1.0) ** 2, jnp.asarray(3.0))
    assert bool(res.converged)
    assert abs(float(res.root) - 1.0) < 3e-7
    g = jax.grad(lambda p: sx.certified_root(lambda x: (x - 1.0) ** 2 - p, jnp.asarray(3.0)).root)(
        0.0
    )
    assert not np.isfinite(float(g)) or abs(float(g)) > 1e6


def test_float32_bracketed_root():
    res = sx.certified_root(lambda x: x**2 - 2.0, jnp.asarray(1.0, jnp.float32), bracket=(0.0, 2.0))
    assert res.root.dtype == jnp.float32
    assert bool(res.converged)
    np.testing.assert_allclose(float(res.root), np.sqrt(2.0), rtol=4e-7)


def test_traced_tolerance_and_stagnation_status():
    """tol may be traced; x^2 + 1 has no real root, so damped Newton stagnates."""

    @jax.jit
    def run(tol):
        return sx.certified_root(lambda x: x**2 - 2.0, jnp.asarray(1.0), tol=tol)

    res = run(jnp.asarray(1e-3))
    assert bool(res.converged) and float(res.residual) <= 1e-3
    bad = sx.certified_root(lambda x: x**2 + 1.0, jnp.asarray(0.5))
    assert not bool(bad.converged)
    assert int(bad.status) == sx.ROOT_STAGNATED
