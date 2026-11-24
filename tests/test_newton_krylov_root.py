"""Newton-Krylov with backtracking and an implicit derivative (plan step S6e).

Backtracking is Algorithm INB of Eisenstat and Walker, SIAM J. Optim. 4, 393 (1994),
in the form of Pawlowski, Shadid, Simonis and Walker, SIAM Rev. 48, 700 (2006),
eq. (2.5): accept ``||F(x + s)|| <= [1 - t (1 - eta)] ||F(x)||``. Newton on
``arctan`` diverges from ``|x0| > 1.3917`` (analytic). The 1-D Bratu problem
``-u'' = lam exp(u) + c``, ``u(0) = u(1) = 0``, has its lower branch for
``lam < 3.5138``; its derivative is checked against central differences and the
dense implicit derivative ``du/dp = -J^{-1} dF/dp``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx

jax.config.update("jax_enable_x64", True)

N = 1000
H = 1.0 / (N + 1)
_A = (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / H**2
_LU = jax.scipy.linalg.lu_factor(jnp.asarray(_A))


def _laplace_inverse(v):
    return jax.scipy.linalg.lu_solve(_LU, v)


def _bratu(lam, c):
    return lambda u: jnp.asarray(_A) @ u - lam * jnp.exp(u) - c


def _root(lam, c, **kw):
    kw = dict(precond=_laplace_inverse, rtol=0.0, atol=1e-8, linear_rtol=1e-10) | kw
    return sx.newton_krylov_root(_bratu(lam, c), jnp.zeros(N), **kw)


def _loss(p):
    return H * jnp.sum(_root(p[0], p[1]).root ** 2)


def test_grad_through_newton_krylov_raises():
    with pytest.raises(ValueError, match="[Rr]everse-mode"):
        jax.grad(lambda c: jnp.sum(sx.newton_krylov(_bratu(1.0, c), jnp.zeros(N)).x))(0.5)


def test_bratu_gradient_against_finite_differences_and_dense():
    p = jnp.array([1.0, 0.5])
    res = _root(*p)
    assert bool(res.converged) and int(res.status) == sx.ROOT_CONVERGED
    assert float(res.residual) < 1e-8
    grad = jax.grad(_loss)(p)
    eye = jnp.eye(2)
    fwd = jnp.stack([jax.jvp(_loss, (p,), (e,))[1] for e in eye])
    fd = jnp.stack([(_loss(p + 1e-4 * e) - _loss(p - 1e-4 * e)) / 2e-4 for e in eye])
    u = res.root
    jac = jnp.asarray(_A) - p[0] * jnp.diag(jnp.exp(u))
    du = jnp.linalg.solve(jac, jnp.stack([jnp.exp(u), jnp.ones(N)], 1))
    dense = 2 * H * u @ du
    for value in (grad, fwd):
        np.testing.assert_allclose(value, fd, rtol=1e-7)
        np.testing.assert_allclose(value, dense, rtol=1e-9)


def _arctan(x):
    return jnp.arctan(x - jnp.linspace(-1.0, 1.0, x.size))


def test_backtracking_converges_where_plain_newton_diverges():
    x0 = jnp.linspace(-1.0, 1.0, 50) + 2.0
    plain = sx.newton_krylov(_arctan, x0, max_steps=40)
    assert not bool(plain.converged)
    assert not np.isfinite(float(plain.residual_norm)) or float(plain.residual_norm) > 1.0
    damped = sx.newton_krylov(_arctan, x0, max_steps=40, line_search=True)
    assert bool(damped.converged)
    np.testing.assert_allclose(damped.x, jnp.linspace(-1.0, 1.0, 50), atol=1e-10)


def test_full_steps_are_bitwise_unchanged_by_the_line_search():
    f = _bratu(1.0, 0.5)
    kw = dict(precond=_laplace_inverse, rtol=0.0, atol=1e-8, linear_rtol=1e-10)
    plain = sx.newton_krylov(f, jnp.zeros(N), **kw)
    damped = sx.newton_krylov(f, jnp.zeros(N), line_search=True, **kw)
    assert np.array_equal(np.asarray(plain.x), np.asarray(damped.x))
    assert int(plain.newton_iterations) == int(damped.newton_iterations)


def test_unconverged_root_reports_status_and_nan_derivative():
    res = _root(1.0, 0.5, max_steps=1)
    assert not bool(res.converged) and int(res.status) == sx.ROOT_MAX_ITER
    assert np.all(np.isfinite(np.asarray(res.root)))
    g = jax.grad(lambda c: jnp.sum(_root(1.0, c, max_steps=1).root))(0.5)
    assert np.isnan(float(g))
    stalled = sx.newton_krylov_root(_arctan, jnp.full(50, 40.0), line_search=True, max_backtracks=2)
    assert int(stalled.status) == sx.ROOT_STAGNATED


def test_jit_and_vmap():
    lams = jnp.array([0.5, 1.0, 2.0])
    batched = jax.jit(jax.vmap(lambda lam: _root(lam, 0.0).root))(lams)
    for i, lam in enumerate(lams):
        # Batched GMRES sums differently; both roots meet atol 1e-8, |du| ~ 1e-9.
        np.testing.assert_allclose(batched[i], _root(lam, 0.0).root, rtol=1e-9)
    grads = jax.jit(jax.vmap(jax.grad(lambda lam: H * jnp.sum(_root(lam, 0.0).root))))(lams)
    assert np.all(np.isfinite(np.asarray(grads))) and np.all(np.diff(np.asarray(grads)) > 0)


def test_hessian_vector_product_small_bratu():
    n = 20
    h = 1.0 / (n + 1)
    a = jnp.asarray((2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2)

    def loss(p):
        f = lambda u: a @ u - p[0] * jnp.exp(u) - p[1]  # noqa: E731
        return h * jnp.sum(sx.newton_krylov_root(f, jnp.zeros(n), rtol=1e-14).root ** 2)

    p, v = jnp.array([1.0, 0.5]), jnp.array([0.3, -0.7])
    hvp = jax.jvp(jax.grad(loss), (p,), (v,))[1]
    fd = (jax.grad(loss)(p + 1e-4 * v) - jax.grad(loss)(p - 1e-4 * v)) / 2e-4
    np.testing.assert_allclose(hvp, fd, rtol=1e-6)


def test_line_search_and_fixed_work_are_exclusive():
    with pytest.raises(ValueError, match="fixed_work"):
        sx.newton_krylov(_arctan, jnp.zeros(3), line_search=True, fixed_work=True)
