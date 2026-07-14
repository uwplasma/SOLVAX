"""Tests for solvax.implicit: implicit-diff linear solves and root finds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import gmres, linear_solve, newton_krylov, root_solve

jax.config.update("jax_enable_x64", True)


def dense_solver(matvec, b):
    """Materialise the operator column-by-column and solve densely."""
    a = jax.vmap(matvec)(jnp.eye(b.shape[0], dtype=b.dtype)).T
    return jnp.linalg.solve(a, b)


def cg_solver(matvec, b):
    """Plain conjugate gradients (SPD systems), run to tight tolerance."""

    def body(carry):
        x, r, p, rs = carry
        ap = matvec(p)
        alpha = rs / jnp.dot(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = jnp.dot(r, r)
        p = r + (rs_new / rs) * p
        return x, r, p, rs_new

    def cond(carry):
        return carry[3] > 1e-28

    x0 = jnp.zeros_like(b)
    x, _, _, _ = jax.lax.while_loop(cond, body, (x0, b, b, jnp.dot(b, b)))
    return x


def fd_grad(f, x, eps=1e-6):
    """Central finite differences of a scalar function, entry by entry."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (float(f(jnp.asarray(x + e))) - float(f(jnp.asarray(x - e)))) / (2 * eps)
        it.iternext()
    return g


def make_spd_params(n, seed=0):
    rng = np.random.default_rng(seed)
    a0 = jnp.asarray(rng.standard_normal((n, n)))
    b = jnp.asarray(rng.standard_normal(n))
    return a0, b


def spd_from(a0):
    return a0 @ a0.T + a0.shape[0] * jnp.eye(a0.shape[0])


@pytest.mark.parametrize("solver", [dense_solver, cg_solver])
def test_linear_solve_grads_match_fd(solver):
    n = 6
    a0, b = make_spd_params(n)

    def loss(a0_, b_):
        matvec = lambda v: spd_from(a0_) @ v  # noqa: E731
        x = linear_solve(matvec, b_, solver)
        return jnp.sum(jnp.sin(x))

    g_a, g_b = jax.grad(loss, argnums=(0, 1))(a0, b)
    fd_a = fd_grad(lambda a: loss(a, b), a0)
    fd_b = fd_grad(lambda v: loss(a0, v), b)
    assert np.allclose(np.asarray(g_a), fd_a, rtol=1e-5, atol=1e-8)
    assert np.allclose(np.asarray(g_b), fd_b, rtol=1e-5, atol=1e-8)


def test_linear_solve_explicit_transpose_nonsymmetric():
    n = 5
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n))
    b = jnp.asarray(rng.standard_normal(n))

    def loss(a_, b_):
        x = linear_solve(
            lambda v: a_ @ v,
            b_,
            dense_solver,
            transpose_matvec=lambda v: a_.T @ v,
        )
        return jnp.sum(x**3)

    g_a, g_b = jax.grad(loss, argnums=(0, 1))(a, b)
    fd_a = fd_grad(lambda m: loss(m, b), a)
    fd_b = fd_grad(lambda v: loss(a, v), b)
    assert np.allclose(np.asarray(g_a), fd_a, rtol=1e-5, atol=1e-8)
    assert np.allclose(np.asarray(g_b), fd_b, rtol=1e-5, atol=1e-8)


def test_linear_solve_gmres_aux_nonsymmetric_jit_jvp_vjp():
    diffusion = jnp.array(
        [
            [2.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ]
    )
    convection = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, -1.0],
        ]
    )
    rhs = jnp.array([1.0, -2.0, 0.5, 3.0])
    cotangent = jnp.array([0.2, -0.4, 0.7, 1.1])

    def solve_with_diagnostics(operator, linear_rhs):
        solution = gmres(
            operator,
            linear_rhs,
            restart=4,
            rtol=1.0e-13,
            atol=1.0e-13,
            max_restarts=2,
        )
        diagnostics = (
            solution.residual_norm,
            solution.iterations,
            solution.converged,
        )
        return solution.x, diagnostics

    def solve(alpha):
        matrix = jnp.eye(4) + 0.1 * diffusion + alpha * convection
        return linear_solve(
            lambda vector: matrix @ vector,
            rhs,
            solve_with_diagnostics,
            transpose_solver=solve_with_diagnostics,
            has_aux=True,
        )

    alpha = 0.07
    matrix = jnp.eye(4) + 0.1 * diffusion + alpha * convection
    expected = jnp.linalg.solve(matrix, rhs)
    (actual, diagnostics) = jax.jit(solve)(alpha)
    residual_norm, iterations, converged = diagnostics

    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    assert float(residual_norm) < 1.0e-12
    assert 0 < int(iterations) <= 4
    assert bool(converged)

    _, tangent = jax.jvp(lambda value: solve(value)[0], (alpha,), (1.0,))
    expected_tangent = jnp.linalg.solve(matrix, -convection @ expected)
    assert np.allclose(tangent, expected_tangent, rtol=1.0e-11, atol=1.0e-12)

    gradient = jax.grad(lambda value: cotangent @ solve(value)[0])(alpha)
    adjoint = jnp.linalg.solve(matrix.T, cotangent)
    expected_gradient = -adjoint @ (convection @ expected)
    assert float(gradient) == pytest.approx(float(expected_gradient), rel=1.0e-11)


def test_linear_solve_uses_distinct_transpose_solver():
    matrix = jnp.array([[3.0, 1.0], [-2.0, 4.0]])
    rhs = jnp.array([1.0, -0.5])
    cotangent = jnp.array([0.25, 2.0])

    # Deliberately ignores the supplied operator, so it is valid only for the
    # forward system. Reusing it for the transpose would produce a wrong VJP.
    def forward_solver(operator, linear_rhs):
        del operator
        return jnp.linalg.solve(matrix, linear_rhs)

    def transpose_solver(operator, linear_rhs):
        return dense_solver(operator, linear_rhs)

    def objective(linear_rhs):
        solved = linear_solve(
            lambda vector: matrix @ vector,
            linear_rhs,
            forward_solver,
            transpose_solver=transpose_solver,
        )
        return cotangent @ solved

    assert np.allclose(jax.grad(objective)(rhs), jnp.linalg.solve(matrix.T, cotangent))


def test_adjoint_costs_one_extra_transposed_solve():
    # Count *executions* of the solver (a Python-side counter would count
    # traces: custom_linear_solve stages both solve and transpose_solve
    # eagerly), via a host callback that fires each time a solve runs.
    n = 4
    a0, b = make_spd_params(n, seed=2)
    counts = {"solve": 0}

    def _bump():
        counts["solve"] += 1

    def counting_solver(matvec, rhs):
        jax.debug.callback(_bump)
        return dense_solver(matvec, rhs)

    def loss(b_):
        x = linear_solve(lambda v: spd_from(a0) @ v, b_, counting_solver)
        return jnp.sum(x**2)

    loss(b)
    jax.effects_barrier()
    assert counts["solve"] == 1

    counts["solve"] = 0
    jax.grad(loss)(b)
    jax.effects_barrier()
    # Forward solve plus exactly one additional (transposed) solve.
    assert counts["solve"] == 2


def newton_solver(f, x0):
    """Fixed-iteration Newton rootfinder (scalar or small vector)."""
    x = x0
    for _ in range(30):
        fx = f(x)
        if jnp.ndim(x0) == 0:
            x = x - fx / jax.grad(f)(x)
        else:
            x = x - jnp.linalg.solve(jax.jacobian(f)(x), fx)
    return x


def test_root_solve_scalar_sqrt():
    def sqrt_root(p):
        return root_solve(lambda x: x**2 - p, jnp.asarray(1.0), newton_solver)

    p = 2.0
    assert np.isclose(float(sqrt_root(p)), np.sqrt(p), rtol=1e-12)
    g = float(jax.grad(sqrt_root)(p))
    assert np.isclose(g, 1.0 / (2.0 * np.sqrt(p)), rtol=1e-10)


def test_root_solve_vector_matches_fd():
    def root(p):
        def f(x):
            return jnp.array(
                [x[0] ** 2 + p[0] * x[1] - 2.0, x[0] * x[1] - p[1]]
            )

        return root_solve(f, jnp.array([1.0, 1.0]), newton_solver)

    p = jnp.array([0.5, 0.7])

    def loss(p_):
        return jnp.sum(root(p_) ** 2)

    x = root(p)
    assert np.allclose(
        [float(x[0] ** 2 + p[0] * x[1] - 2.0), float(x[0] * x[1] - p[1])],
        0.0,
        atol=1e-12,
    )
    g = jax.grad(loss)(p)
    fd = fd_grad(loss, p)
    assert np.allclose(np.asarray(g), fd, rtol=1e-5, atol=1e-8)


def test_root_solve_custom_tangent_solve():
    def tangent_solve(g, y):
        return jnp.linalg.solve(jax.jacobian(g)(jnp.zeros_like(y)), y)

    def root(p):
        f = lambda x: x**3 - p  # noqa: E731
        return root_solve(
            f, jnp.ones_like(p), newton_solver, tangent_solve=tangent_solve
        )

    p = jnp.array([8.0, 27.0])
    g = jax.jacobian(root)(p)
    expected = np.diag(1.0 / (3.0 * np.cbrt(np.asarray(p)) ** 2))
    assert np.allclose(np.asarray(g), expected, rtol=1e-8)


def test_newton_krylov_scalar_converges_under_jit():
    solution = jax.jit(
        lambda initial: newton_krylov(
            lambda x: x**2 - 2.0,
            initial,
            rtol=1e-12,
            atol=1e-12,
            max_steps=8,
            linear_restart=1,
            linear_rtol=1e-12,
            linear_max_restarts=2,
        )
    )(jnp.array(1.0))

    assert bool(solution.converged)
    assert bool(solution.linear_converged)
    assert float(solution.x) == pytest.approx(np.sqrt(2.0), rel=1e-12)
    assert int(solution.newton_iterations) > 0
    assert int(solution.linear_iterations) >= int(solution.newton_iterations)


def test_newton_krylov_checks_residual_after_last_update():
    solution = newton_krylov(
        lambda x: x - 3.0,
        jnp.array(0.0),
        rtol=0.0,
        atol=1e-12,
        max_steps=1,
        linear_restart=1,
        linear_rtol=1e-12,
        linear_max_restarts=1,
    )

    assert bool(solution.converged)
    assert int(solution.newton_iterations) == 1
    assert float(solution.residual_norm) < 1e-12


def test_newton_krylov_reports_linear_failure():
    solution = newton_krylov(
        lambda x: x - 1.0,
        jnp.array(0.0),
        rtol=0.0,
        atol=1e-12,
        max_steps=2,
        linear_restart=1,
        linear_max_restarts=0,
    )

    assert not bool(solution.converged)
    assert not bool(solution.linear_converged)
    assert int(solution.linear_iterations) == 0


def test_newton_krylov_supports_pytree_preconditioner_and_inner_product():
    target = (jnp.array([2.0, -4.0]), jnp.array(9.0))

    def residual(value):
        return 2.0 * value[0] - target[0], 3.0 * value[1] - target[1]

    precond = lambda value: (value[0] / 2.0, value[1] / 3.0)  # noqa: E731
    inner_product = lambda left, right: (  # noqa: E731
        jnp.vdot(left[0], 2.0 * right[0]) + jnp.vdot(left[1], right[1])
    )
    solution = jax.jit(
        lambda: newton_krylov(
            residual,
            (jnp.zeros(2), jnp.array(0.0)),
            precond=precond,
            inner_product=inner_product,
            rtol=0.0,
            atol=1e-12,
            max_steps=2,
            linear_restart=2,
            linear_rtol=1e-12,
            linear_max_restarts=1,
        )
    )()

    assert bool(solution.converged)
    assert np.asarray(solution.x[0]) == pytest.approx([1.0, -2.0])
    assert float(solution.x[1]) == pytest.approx(3.0)
    assert int(solution.linear_iterations) == 1
