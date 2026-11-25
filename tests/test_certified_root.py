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
        res = sx.certified_root(
            lambda x: x**2 - p, jnp.asarray(1.0), solver=lambda f, x0: x0, tol=1e-12
        )
        return res.root, res.converged

    x, converged = root(2.0)
    assert not bool(converged)
    assert float(x) == 1.0
    assert np.isnan(float(jax.grad(lambda p: root(p)[0])(2.0)))


def test_user_solver_converged_keeps_gradient():
    res = sx.certified_root(
        lambda x: x**2 - 2.0, jnp.asarray(1.0), solver=lambda f, x0: jnp.sqrt(2.0), tol=1e-12
    )
    assert bool(res.converged)
    g = jax.grad(
        lambda p: (
            sx.certified_root(
                lambda x: x**2 - p, jnp.asarray(1.0), solver=lambda f, x0: jnp.sqrt(p), tol=1e-12
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


def test_far_start_does_not_certify_a_non_root():
    """x^20 - 1e-6 from 10 on (0, 10): the root is 1e-6^(1/20) = 0.501187...; 1.937 is not one."""
    res = sx.certified_root(lambda x: x**20 - 1e-6, jnp.asarray(10.0), bracket=(0.0, 10.0))
    assert bool(res.converged)
    np.testing.assert_allclose(float(res.root), 1e-6 ** (1 / 20), rtol=1e-13)


def test_bracket_iterations_bounded_by_bisection():
    """rtsafe: on (0, 100) bisection to the float64 spacing of 0.5 needs about 60 halvings."""
    res = sx.certified_root(lambda x: x**20 - 1e-6, jnp.asarray(50.0), bracket=(0.0, 100.0))
    assert bool(res.converged)
    assert int(res.iterations) <= 64
    np.testing.assert_allclose(float(res.root), 1e-6 ** (1 / 20), rtol=1e-13)


def test_start_outside_bracket_iterates_inside():
    """(x - 5)(x + 0.5) from 5 with bracket (-1, 1): the bracketed root is -0.5."""
    res = sx.certified_root(lambda x: (x - 5.0) * (x + 0.5), jnp.asarray(5.0), bracket=(-1.0, 1.0))
    assert bool(res.converged)
    np.testing.assert_allclose(float(res.root), -0.5, atol=1e-14)


def test_zero_slope_start_keeps_a_finite_iterate():
    """x^2 - 2 from 0: f'(0) = 0, the Newton step is infinite."""
    res = sx.certified_root(lambda x: x**2 - 2.0, jnp.asarray(0.0))
    assert np.isfinite(float(res.root))
    assert not bool(res.converged)


def test_user_solver_needs_tol():
    with pytest.raises(ValueError, match="tol"):
        sx.certified_root(lambda x: x - 1.0, jnp.asarray(0.0), solver=lambda f, x0: x0)


@pytest.mark.parametrize("dp", [1.0 + 0.0j, 1.0j])
def test_non_holomorphic_complex_scalar_derivative(dp):
    """f = z conj(z) + z - p is not holomorphic; check dz/dp along dp by central differences."""
    z0 = jnp.asarray(1.0 + 0.5j)
    p0 = jnp.asarray(2.0 + 0.5j)

    def solver(f, z):
        # Newton on the real 2 x 2 system, independent of the code under test.
        for _ in range(60):
            v = jnp.array([z.real, z.imag])
            jac = jax.jacfwd(
                lambda u: jnp.array([f(u[0] + 1j * u[1]).real, f(u[0] + 1j * u[1]).imag])
            )(v)
            r = f(z)
            d = jnp.linalg.solve(jac, jnp.array([r.real, r.imag]))
            z = z - (d[0] + 1j * d[1])
        return z

    def root(p):
        return sx.certified_root(
            lambda z: z * jnp.conj(z) + z - p, z0, solver=solver, tol=1e-12
        ).root

    h = 1e-6
    fd = (root(p0 + h * dp) - root(p0 - h * dp)) / (2 * h)
    tangent = jax.jvp(root, (p0,), (jnp.asarray(dp),))[1]
    np.testing.assert_allclose(complex(tangent), complex(fd), atol=1e-8)


_N = 1200  # above DENSE_TANGENT_LIMIT, so the default tangent is GMRES


def _adv_certified(p, **kw):
    """Certified root of A x + 10 x^3 = 10 p b, A advection-diffusion (Peclet 50, cond ~1e6)."""
    h = 1.0 / (_N + 1)
    one = np.ones(_N - 1)
    d2 = (np.diag(-2 * np.ones(_N)) + np.diag(one, 1) + np.diag(one, -1)) / h**2
    a = jnp.asarray(-d2 + 50.0 * (np.diag(one, 1) - np.diag(one, -1)) / (2 * h))
    b = jnp.linspace(0.5, 1.5, _N)

    def solver(f, x0):
        def step(_, x):
            return x - jnp.linalg.solve(a + 30.0 * jnp.diag(x**2), f(x))

        return jax.lax.fori_loop(0, 8, step, x0)

    f = lambda x: a @ x + 10.0 * x**3 - 10.0 * p * b  # noqa: E731
    return sx.certified_root(f, jnp.zeros(_N), solver=solver, tol=1e-8, **kw), a


def test_certified_vector_root_forwards_the_tangent_preconditioner():
    """Unpreconditioned GMRES tangent is NaN; with M = A^{-1} it matches the dense tangent."""
    p = jnp.asarray(1.0)
    base, a = _adv_certified(p)
    lu = jax.scipy.linalg.lu_factor(a)
    pair = (lambda v: jax.scipy.linalg.lu_solve(lu, v),
            lambda v: jax.scipy.linalg.lu_solve(lu, v, trans=1))
    dense = jax.jvp(lambda q: _adv_certified(q, dense_limit=_N)[0].root, (p,), (1.0,))[1]
    nan = jax.jvp(lambda q: _adv_certified(q)[0].root, (p,), (1.0,))[1]
    assert np.all(np.isnan(np.asarray(nan)))
    for pc in (pair, lambda x: pair):
        res = _adv_certified(p, tangent_precond=pc, tangent_rtol=1e-12)[0]
        assert bool(res.converged) and int(res.status) == sx.ROOT_CONVERGED
        assert float(res.residual) == float(base.residual)
        tangent = jax.jvp(
            lambda q, pc=pc: _adv_certified(q, tangent_precond=pc)[0].root, (p,), (1.0,)
        )[1]
        np.testing.assert_allclose(tangent, dense, rtol=1e-8, atol=1e-12)
        grad = jax.grad(
            lambda q, pc=pc: jnp.sum(_adv_certified(q, tangent_precond=pc)[0].root)
        )(p)
        np.testing.assert_allclose(grad, jnp.sum(dense), rtol=1e-8)


def test_scalar_root_refuses_a_tangent_preconditioner():
    with pytest.raises(ValueError, match="scalar"):
        sx.certified_root(lambda x: x**2 - 2.0, jnp.asarray(1.0), tangent_precond=(None, None))
