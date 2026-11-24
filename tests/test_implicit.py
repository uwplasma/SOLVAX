"""Tests for solvax.implicit: implicit-diff linear solves and root finds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    gcrot,
    gmres,
    linear_solve,
    newton_krylov,
    recycled_linear_solve,
    root_solve,
    tridiagonal_solve,
)

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


def deflatable_operator(n=200, outliers=8, seed=1):
    """Symmetric operator with a few eigenvalues decades below the bulk, plus a
    symmetric perturbation the parameter scales."""
    rng = np.random.default_rng(seed)
    eigenvalues = np.concatenate(
        [np.linspace(1.0, 2.0, n - outliers), np.logspace(-3, -2, outliers)]
    )
    basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
    base = basis @ np.diag(eigenvalues) @ basis.T
    perturbation = rng.standard_normal((n, n)) / np.sqrt(n)
    perturbation = 0.5 * (perturbation + perturbation.T)
    rhs = jnp.asarray(rng.standard_normal(n))

    def operator(params, x):
        return jnp.asarray(base) @ x + params * (jnp.asarray(perturbation) @ x)

    return operator, rhs


RECYCLE_DIRECTIONS = 12


def recording_solver(record, *, reuse):
    """A GCRO-DR solver that reports its iteration count to the host."""

    def solver(matvec, rhs, recycle):
        solution = gcrot(
            matvec, rhs, m=15, k=RECYCLE_DIRECTIONS, rtol=1e-10, max_restarts=80,
            recycle=recycle if reuse else None, recycle_strategy="harmonic",
        )
        jax.debug.callback(record, solution.iterations)
        return solution.x, solution.recycle

    return solver


def test_recycled_linear_solve_gradient_matches_finite_differences():
    operator, rhs = deflatable_operator(n=80, outliers=4, seed=4)
    solver = recording_solver(lambda value: None, reuse=True)

    def loss(params):
        return jnp.sum(recycled_linear_solve(operator, params, rhs, solver) ** 2)

    gradient = float(jax.grad(loss)(0.05))
    step = 1e-6
    finite = float((loss(0.05 + step) - loss(0.05 - step)) / (2 * step))
    assert gradient == pytest.approx(finite, rel=1e-6)

    # And the primal agrees with the dense solve.
    dense = jax.vmap(lambda v: operator(0.05, v))(jnp.eye(80)).T
    solution = recycled_linear_solve(operator, 0.05, rhs, solver)
    np.testing.assert_allclose(
        solution, jnp.linalg.solve(dense, rhs), rtol=1e-7, atol=1e-9
    )


def test_adjoint_reuse_cuts_the_adjoint_iteration_count():
    """The adjoint system has the same spectrum, so the subspace the forward
    solve paid to find still deflates it."""
    operator, rhs = deflatable_operator()

    def measure(reuse):
        counts = []
        solver = recording_solver(lambda value: counts.append(int(value)), reuse=reuse)

        def loss(params):
            return jnp.sum(recycled_linear_solve(operator, params, rhs, solver) ** 2)

        gradient = float(jax.grad(loss)(0.05))
        jax.effects_barrier()
        forward, adjoint = counts  # exactly two solves: forward and adjoint
        return forward, adjoint, gradient

    cold_forward, cold_adjoint, cold_gradient = measure(False)
    warm_forward, warm_adjoint, warm_gradient = measure(True)

    # Only the adjoint differs; the forward solve is identical work.
    assert warm_forward == cold_forward
    # Warm-starting costs one operator application per recycled direction, so
    # require the saving to beat that too.
    assert warm_adjoint + RECYCLE_DIRECTIONS < cold_adjoint
    assert warm_adjoint < cold_adjoint / 2
    assert warm_gradient == pytest.approx(cold_gradient, rel=1e-8)


def test_recycled_linear_solve_takes_pytree_parameters_and_a_transpose_solver():
    rng = np.random.default_rng(9)
    n = 40
    left = jnp.asarray(np.eye(n) + 0.1 * rng.standard_normal((n, n)) / np.sqrt(n))
    right = jnp.asarray(rng.standard_normal((n, n)) / np.sqrt(n))
    rhs = jnp.asarray(rng.standard_normal(n))

    def operator(params, x):
        return left @ x + params["scale"] * (right @ x) + params["shift"] * x

    def solver(matvec, b, recycle):
        solution = gcrot(matvec, b, m=20, k=5, rtol=1e-12, recycle=recycle)
        return solution.x, solution.recycle

    transposed = []

    def transpose_solver(matvec, b, recycle):
        transposed.append(True)
        solution = gcrot(matvec, b, m=20, k=5, rtol=1e-12, recycle=recycle)
        return solution.x, solution.recycle

    params = {"scale": 0.3, "shift": 0.2}

    def loss(values):
        x = recycled_linear_solve(
            operator, values, rhs, solver, transpose_solver=transpose_solver
        )
        return jnp.sum(x**2)

    gradient = jax.grad(loss)(params)
    assert transposed  # the adjoint went through the supplied transpose solver
    step = 1e-6
    for name in ("scale", "shift"):
        shifted = dict(params)
        shifted[name] = params[name] + step
        plus = loss(shifted)
        shifted[name] = params[name] - step
        finite = float((plus - loss(shifted)) / (2 * step))
        assert float(gradient[name]) == pytest.approx(finite, rel=1e-5)


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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("fixed_work", [False, True])
@pytest.mark.parametrize("after_update", [False, True])
def test_newton_krylov_never_certifies_nonfinite_residuals(
    bad_value, fixed_work, after_update
):
    def residual(x):
        return jnp.where(x >= 1.0, bad_value, x - 2.0) if after_update else jnp.asarray(bad_value)

    result = jax.jit(lambda: newton_krylov(
        residual, jnp.asarray(0.0), max_steps=2, linear_restart=1,
        linear_max_restarts=1, fixed_work=fixed_work,
    ))()
    assert not bool(result.converged)
    assert not np.isfinite(float(result.residual_norm))
    assert int(result.newton_iterations) == int(after_update)


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("fixed_work", [False, True])
def test_newton_krylov_rejects_nonfinite_tolerance(name, value, fixed_work):
    result = jax.jit(lambda: newton_krylov(
        lambda x: x - 1.0, jnp.asarray(0.0), max_steps=2,
        linear_restart=1, linear_max_restarts=1, fixed_work=fixed_work,
        **{name: value},
    ))()
    assert not bool(result.converged)
    assert float(result.residual_norm) == 1.0
    assert int(result.newton_iterations) == 0


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


def test_fixed_work_newton_matches_early_exit_and_differentiates_through_scan():
    kwargs = dict(
        rtol=0.0,
        atol=1.0e-13,
        max_steps=8,
        linear_restart=1,
        linear_rtol=1.0e-13,
        linear_max_restarts=1,
    )
    early = newton_krylov(lambda x: x**2 - 2.0, jnp.asarray(1.0), **kwargs)
    fixed = newton_krylov(
        lambda x: x**2 - 2.0, jnp.asarray(1.0), fixed_work=True, **kwargs
    )
    assert bool(early.converged and fixed.converged)
    assert float(fixed.x) == pytest.approx(float(early.x), rel=1.0e-14)
    assert int(fixed.newton_iterations) == int(early.newton_iterations)

    def fixed_rollout(parameter):
        def step(initial, offset):
            solution = newton_krylov(
                lambda x: x**2 - (parameter + offset),
                initial,
                fixed_work=True,
                **kwargs,
            )
            return solution.x, None

        final, _ = jax.lax.scan(step, jnp.asarray(1.0), jnp.asarray([0.0, 1.0]))
        return final

    parameter = jnp.asarray(3.0)
    gradient = jax.grad(fixed_rollout)(parameter)
    expected = 0.5 / jnp.sqrt(parameter + 1.0)
    assert float(gradient) == pytest.approx(float(expected), rel=1.0e-11)


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


def test_newton_krylov_constant_forcing_is_the_default():
    kwargs = dict(
        rtol=0.0,
        atol=1e-12,
        max_steps=8,
        linear_restart=1,
        linear_rtol=1e-10,
        linear_max_restarts=2,
    )
    default = newton_krylov(lambda x: x**2 - 2.0, jnp.array(1.0), **kwargs)
    explicit = newton_krylov(
        lambda x: x**2 - 2.0,
        jnp.array(1.0),
        forcing="constant",
        **kwargs,
    )

    assert all(
        bool(equal)
        for equal in jax.tree.leaves(
            jax.tree.map(
                lambda left, right: jnp.array_equal(left, right), default, explicit
            )
        )
    )


def test_newton_krylov_eisenstat_walker_reduces_inner_work_quadratically():
    n = 32
    h = 1.0 / (n + 1)
    source = jnp.full(n, 10.0)
    off_diagonal = jnp.full(n, -1.0 / h**2)
    diagonal = jnp.full(n, 2.0 / h**2)

    def residual(u):
        laplacian = (-2.0 * u).at[:-1].add(u[1:]).at[1:].add(u[:-1]) / h**2
        return -laplacian + u**3 - source

    def preconditioner(value):
        return tridiagonal_solve(
            off_diagonal, diagonal, off_diagonal, value, method="thomas"
        )

    common = dict(
        precond=preconditioner,
        rtol=0.0,
        atol=1e-10,
        max_steps=6,
        linear_restart=30,
        linear_max_restarts=10,
    )
    initial = jnp.zeros(n)
    fixed = jax.jit(
        lambda value: newton_krylov(
            residual, value, linear_rtol=1e-10, forcing="constant", **common
        )
    )(initial)
    adaptive = jax.jit(
        lambda value: newton_krylov(
            residual,
            value,
            linear_rtol=0.5,
            forcing="eisenstat_walker",
            **common,
        )
    )(initial)

    assert bool(fixed.converged and fixed.linear_converged)
    assert bool(adaptive.converged and adaptive.linear_converged)
    assert int(adaptive.linear_iterations) < int(fixed.linear_iterations)
    assert int(adaptive.newton_iterations) == int(fixed.newton_iterations)
    assert float(adaptive.residual_norm) <= 1e-10
    assert np.asarray(adaptive.x) == pytest.approx(np.asarray(fixed.x), abs=2e-11)

    terminal_residuals = []
    for max_steps in (2, 3, 4):
        partial = newton_krylov(
            residual,
            initial,
            precond=preconditioner,
            rtol=0.0,
            atol=0.0,
            max_steps=max_steps,
            linear_restart=30,
            linear_rtol=0.5,
            linear_max_restarts=10,
            forcing="eisenstat_walker",
        )
        terminal_residuals.append(float(partial.residual_norm))

    penultimate, terminal = terminal_residuals[-2:]
    assert terminal < penultimate
    assert terminal / penultimate**2 < 1e-2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"forcing": "unknown"}, "forcing must be"),
        (
            {"forcing": "eisenstat_walker", "linear_rtol": 1.0},
            "linear_rtol",
        ),
        (
            {"forcing": "eisenstat_walker", "forcing_gamma": 1.0},
            "forcing_gamma",
        ),
        (
            {"forcing": "eisenstat_walker", "forcing_alpha": 1.0},
            "forcing_alpha",
        ),
        (
            {"forcing": "eisenstat_walker", "forcing_max": 1.0},
            "forcing_max",
        ),
    ],
)
def test_newton_krylov_rejects_invalid_forcing_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        newton_krylov(lambda x: x - 1.0, jnp.array(0.0), **kwargs)


# --- root_solve default tangent: dense only for small systems ---------------
#
# Above ``dense_limit`` unknowns the default tangent of ``root_solve`` is a
# matrix-free GMRES solve, so a large root never materialises its n x n
# Jacobian. The checks are exact identities of the implicit function theorem
# (``J dx = df/dp`` at the root) or central differences, not stored numbers.

_N_LARGE = 4000
_COUPLING = 0.3  # contraction factor: J = I + 0.3 diag(sech^2) roll, cond <= 1.9


def _ring_residual(p):
    b = jnp.linspace(0.5, 1.5, _N_LARGE)

    def f(x):
        return x + _COUPLING * jnp.tanh(jnp.roll(x, 1)) - p * b

    return f, b


def _fixed_point_solver(f, x0):
    # x = x - f(x) is a 0.3-contraction; 80 sweeps reach 1e-40 relative.
    return jax.lax.fori_loop(0, 80, lambda _, x: x - f(x), x0)


def _ring_root(p):
    f, _ = _ring_residual(p)
    return root_solve(f, jnp.zeros(_N_LARGE), _fixed_point_solver)


def _largest_aval(closed_jaxpr):
    """Largest array size of any variable in a jaxpr and all its sub-jaxprs."""

    def sub_jaxprs(value):
        if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
            yield value.jaxpr  # ClosedJaxpr
        elif hasattr(value, "eqns"):  # Jaxpr
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from sub_jaxprs(item)

    def walk(jaxpr):
        sizes = [0]
        for var in list(jaxpr.invars) + list(jaxpr.constvars):
            sizes.append(int(np.prod(var.aval.shape)))
        for eqn in jaxpr.eqns:
            for var in eqn.outvars:
                sizes.append(int(np.prod(getattr(var.aval, "shape", ()))))
            for value in eqn.params.values():
                for sub in sub_jaxprs(value):
                    sizes.append(walk(sub))
        return max(sizes)

    return walk(closed_jaxpr.jaxpr)


def test_root_solve_large_vector_has_no_dense_jacobian():
    p = 0.8
    n2 = _N_LARGE * _N_LARGE
    jvp_fn = lambda p_: jax.jvp(_ring_root, (p_,), (1.0,))  # noqa: E731
    loss = lambda p_: jnp.sum(_ring_root(p_) ** 2)  # noqa: E731
    assert _largest_aval(jax.make_jaxpr(jvp_fn)(p)) < n2
    assert _largest_aval(jax.make_jaxpr(jax.grad(loss))(p)) < n2
    # The jaxpr walk cannot see inside custom_root on every JAX version, so
    # also bound the compiled temporaries: a dense float64 Jacobian is 8 n^2.
    for fn in (jvp_fn, jax.grad(loss)):
        mem = jax.jit(fn).lower(p).compile().memory_analysis()
        assert mem.temp_size_in_bytes < 8 * n2

    x, dx = jax.jit(jvp_fn)(p)
    f, b = _ring_residual(p)
    # Implicit function theorem: (df/dx) dx = -df/dp = b at the root.
    _, jdx = jax.jvp(f, (x,), (dx,))
    assert float(jnp.linalg.norm(jdx - b) / jnp.linalg.norm(b)) < 1e-9

    h = 1e-5
    fd = (loss(p + h) - loss(p - h)) / (2 * h)  # truncation O(h^2) ~ 1e-10
    assert np.isclose(float(jax.jit(jax.grad(loss))(p)), float(fd), rtol=1e-8)


def test_root_solve_small_default_is_the_dense_solve():
    # Below the limit the tangent is bit-identical to the former dense solve.
    def dense(g, y):
        return jnp.linalg.solve(jax.jacobian(g)(jnp.zeros_like(y)), y)

    def root(p, **kw):
        f = lambda x: x**3 + 0.1 * jnp.roll(x, 1) - p  # noqa: E731
        return root_solve(f, jnp.ones_like(p), newton_solver, **kw)

    p = jnp.array([8.0, 27.0, 5.0])
    t = jnp.array([1.0, -2.0, 0.5])
    default = jax.jvp(root, (p,), (t,))[1]
    explicit = jax.jvp(lambda q: root(q, tangent_solve=dense), (p,), (t,))[1]
    assert np.array_equal(np.asarray(default), np.asarray(explicit))
    krylov = jax.jvp(lambda q: root(q, dense_limit=0), (p,), (t,))[1]
    assert np.allclose(np.asarray(krylov), np.asarray(default), rtol=1e-9)


def complex_newton(f, z0):
    z = z0
    for _ in range(30):
        z = z - jnp.linalg.solve(jax.jacfwd(f, holomorphic=True)(z), f(z))
    return z


@pytest.mark.parametrize("dense_limit", [1024, 0])
def test_root_solve_complex_vector_tangent(dense_limit):
    # z_k^2 = p_k: dz_k/dp_k = 1 / (2 z_k), holomorphic, both tangent paths.
    p = jnp.array([2.0 + 1.0j, -1.0 + 3.0j, 4.0 - 0.5j])

    def root(q):
        return root_solve(
            lambda z: z**2 - q, jnp.sqrt(q) * (1 + 1e-3), complex_newton,
            dense_limit=dense_limit,
        )

    t = jnp.array([1.0, 1.0j, 0.5 - 0.5j])
    z, dz = jax.jvp(root, (p,), (t,))
    assert np.allclose(np.asarray(z), np.sqrt(np.asarray(p)), rtol=1e-12)
    assert np.allclose(np.asarray(dz), np.asarray(t / (2 * z)), rtol=1e-9)


def test_root_solve_krylov_tangent_under_vmap_and_jit():
    p = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.7, 0.9]])

    def root(q):
        f = lambda x: x**3 + 0.1 * jnp.roll(x, 1) - q  # noqa: E731
        return root_solve(f, jnp.ones_like(q), newton_solver, dense_limit=0)

    batched = jax.jit(jax.vmap(jax.jacobian(root)))(p)
    for row, jac in zip(p, batched, strict=True):
        ref = jax.jacobian(lambda q: root_solve(
            lambda x: x**3 + 0.1 * jnp.roll(x, 1) - q, jnp.ones_like(q),
            newton_solver))(row)
        assert np.allclose(np.asarray(jac), np.asarray(ref), rtol=1e-9)


def test_root_solve_unconverged_krylov_tangent_is_nan():
    # A singular Jacobian at the root (f = x^2, root 0) has no tangent: the
    # Krylov solve cannot converge and the tangent is NaN, never a number.
    def root(p):
        return root_solve(
            lambda x: x**2 - p, jnp.zeros(3), lambda f, x0: x0, dense_limit=0
        )

    _, dx = jax.jvp(root, (jnp.zeros(3),), (jnp.ones(3),))
    assert np.all(np.isnan(np.asarray(dx)))


@pytest.mark.parametrize("dense_limit", [1024, 0])
def test_root_solve_non_holomorphic_complex_tangent(dense_limit):
    # f(z) = z + 0.2 conj(z) - p is only R-linear. With z = a + ib the root is
    # a = Re p / 1.2, b = Im p / 0.8, so dz/dp . t = Re t / 1.2 + i Im t / 0.8.
    def root(q):
        return root_solve(
            lambda z: z + 0.2 * jnp.conj(z) - q, jnp.zeros_like(q),
            lambda f, z0: q.real / 1.2 + 1j * q.imag / 0.8,
            dense_limit=dense_limit,
        )

    p = jnp.array([1.0 + 2.0j, -0.5 + 0.3j, 2.0 - 1.0j])
    t = jnp.array([1.0j, 1.0, 0.5 - 2.0j])
    _, dz = jax.jvp(root, (p,), (t,))
    expected = t.real / 1.2 + 1j * t.imag / 0.8
    assert np.allclose(np.asarray(dz), np.asarray(expected), rtol=1e-9)
    # Reverse mode: loss = sum |z|^2, dL/dRe p = 2a/1.2, dL/dIm p = 2b/0.8;
    # JAX returns the conjugate-free gradient dL/dRe p - i dL/dIm p.
    grad = jax.grad(lambda q: jnp.sum(jnp.abs(root(q)) ** 2))(p)
    a, b = p.real / 1.2, p.imag / 0.8
    assert np.allclose(np.asarray(grad), np.asarray(2 * a / 1.2 - 2j * b / 0.8),
                       rtol=1e-9)


# Preconditioned tangents. The residual f(x) = A x + 10 x^3 - 10 p b uses a
# central-difference advection-diffusion operator A = -D2 + 50 D1 on 1200 cells:
# nonsymmetric, cond(df/dx) = 7.0e4 at the root, and unpreconditioned GMRES(50)
# does not reach 1e-10 in 40 cycles, so the default tangent is NaN. With
# M = A^{-1} the preconditioned Jacobian is I + A^{-1} 30 diag(x^2), of
# condition 1.002, so a residual below rtol = 1e-10 bounds the relative tangent
# error by cond(J M) rtol ~ 1e-10: the tests allow 1e-9 against the dense one.
# Central differences with h = 1e-4 have truncation O(h^2) = 1e-8 relative.
_N_ADV = 1200


def _advection_diffusion():
    h = 1.0 / (_N_ADV + 1)
    one = np.ones(_N_ADV - 1)
    d2 = (np.diag(-2 * np.ones(_N_ADV)) + np.diag(one, 1) + np.diag(one, -1)) / h**2
    d1 = (np.diag(one, 1) - np.diag(one, -1)) / (2 * h)
    return jnp.asarray(-d2 + 50.0 * d1), jnp.linspace(0.5, 1.5, _N_ADV)


def _lu_pair(matrix):
    lu = jax.scipy.linalg.lu_factor(matrix)
    return (lambda v: jax.scipy.linalg.lu_solve(lu, v),
            lambda v: jax.scipy.linalg.lu_solve(lu, v, trans=1))


def _adv_root(p, aux=False, **kw):
    a, b = _advection_diffusion()

    def solver(f, x0):
        def step(_, x):
            return x - jnp.linalg.solve(a + 30.0 * jnp.diag(x**2), f(x))
        x = jax.lax.fori_loop(0, 8, step, x0)
        return (x, jnp.linalg.norm(f(x))[None]) if aux else x

    return root_solve(lambda x: a @ x + 10.0 * x**3 - 10.0 * p * b,
                      jnp.zeros(_N_ADV), solver, has_aux=aux, **kw)


def _rel(left, right):
    return float(jnp.linalg.norm(left - right) / jnp.linalg.norm(right))


def test_root_solve_preconditioned_tangent_on_an_ill_conditioned_root():
    a, _ = _advection_diffusion()
    pair = _lu_pair(a)
    p = jnp.asarray(1.0)
    dense = jax.jvp(lambda q: _adv_root(q, dense_limit=_N_ADV), (p,), (1.0,))[1]
    assert np.all(np.isnan(np.asarray(jax.jvp(_adv_root, (p,), (1.0,))[1])))

    def loss(q, **kw):
        return jnp.sum(_adv_root(q, **kw) ** 2)

    fd = (loss(p + 1e-4) - loss(p - 1e-4)) / 2e-4
    for precond in (pair, (pair[0], None), lambda x: pair):
        jvp = jax.jit(lambda q, pc=precond: jax.jvp(
            lambda r: _adv_root(r, tangent_precond=pc), (q,), (1.0,))[1])
        assert _rel(jvp(p), dense) < 1e-9
        grad = jax.jit(jax.grad(lambda q, pc=precond: loss(q, tangent_precond=pc)))
        assert np.isclose(float(grad(p)), float(fd), rtol=1e-6)
    batched = jax.vmap(jax.grad(lambda q: loss(q, tangent_precond=pair)))(
        jnp.array([0.5, 1.0]))
    assert np.isclose(float(batched[1]), float(grad(p)), rtol=1e-9)


def test_root_solve_precond_factory_receives_the_root():
    a, _ = _advection_diffusion()
    p = jnp.asarray(1.0)
    dense = jax.jvp(lambda q: _adv_root(q, dense_limit=_N_ADV), (p,), (1.0,))[1]
    # One GMRES iteration in one cycle: only the exact inverse of df/dx at
    # the root solves the tangent; the same factory at x0 = 0 cannot.
    one_step = dict(tangent_restart=1, tangent_max_restarts=1)

    def exact(x):
        return _lu_pair(a + 30.0 * jnp.diag(x**2))

    tangent = jax.jvp(lambda q: _adv_root(q, tangent_precond=exact, **one_step),
                      (p,), (1.0,))[1]
    assert _rel(tangent, dense) < 1e-9
    at_x0 = exact(jnp.zeros(_N_ADV))
    stale = jax.jvp(lambda q: _adv_root(q, tangent_precond=at_x0, **one_step),
                    (p,), (1.0,))[1]
    assert np.all(np.isnan(np.asarray(stale)))


def test_root_solve_tangent_keywords_are_honoured():
    a, _ = _advection_diffusion()
    p = jnp.asarray(1.0)
    dense = jax.jvp(lambda q: _adv_root(q, dense_limit=_N_ADV), (p,), (1.0,))[1]
    loose = jax.jvp(lambda q: _adv_root(q, tangent_precond=_lu_pair(a),
                                        tangent_rtol=1e-3), (p,), (1.0,))[1]
    assert 1e-8 < _rel(loose, dense) < 1e-2

    def root(q, **kw):
        f = lambda x: x**3 + 0.1 * jnp.roll(x, 1) - q  # noqa: E731
        return root_solve(f, jnp.ones_like(q), newton_solver, dense_limit=0, **kw)

    q = jnp.array([8.0, 27.0, 5.0])
    assert np.all(np.isfinite(np.asarray(jax.jvp(root, (q,), (q,))[1])))
    few = jax.jvp(lambda r: root(r, tangent_restart=1, tangent_max_restarts=1),
                  (q,), (q,))[1]
    assert np.all(np.isnan(np.asarray(few)))


@pytest.mark.parametrize("precond", [False, True])
def test_root_solve_has_aux_returns_aux_without_derivative(precond):
    a, _ = _advection_diffusion()
    kw = {"tangent_precond": _lu_pair(a)} if precond else {"dense_limit": _N_ADV}
    p = jnp.asarray(1.0)
    x, aux = jax.jit(lambda q: _adv_root(q, aux=True, **kw))(p)
    assert np.allclose(np.asarray(x), np.asarray(_adv_root(p, **kw)), rtol=1e-12)
    assert aux.shape == (1,) and float(aux[0]) < 1e-8
    _, daux = jax.jvp(lambda q: _adv_root(q, aux=True, **kw)[1], (p,), (1.0,))
    assert float(daux[0]) == 0.0
    grad = jax.grad(lambda q: jnp.sum(_adv_root(q, aux=True, **kw)[1]))(p)
    assert float(grad) == 0.0
    batched = jax.vmap(lambda q: _adv_root(q, aux=True, **kw)[1])(jnp.array([0.5, 1.0]))
    assert batched.shape == (2, 1)


@pytest.mark.parametrize("adjoint", [True, False])
def test_root_solve_complex_preconditioned_tangent(adjoint):
    # f(z) = A z + 0.2 conj(z) - q is only R-linear; M = A^{-1} acts on the
    # complex vector and M_T is its adjoint A^{-H} (or the linear transpose).
    a, _ = _advection_diffusion()
    lu = jax.scipy.linalg.lu_factor(a.astype(jnp.complex128))
    m = lambda v: jax.scipy.linalg.lu_solve(lu, v)  # noqa: E731
    m_h = lambda v: jax.scipy.linalg.lu_solve(lu, v, trans=2)  # noqa: E731
    real = jnp.block([[a + 0.2 * jnp.eye(_N_ADV), jnp.zeros_like(a)],
                      [jnp.zeros_like(a), a - 0.2 * jnp.eye(_N_ADV)]])

    def solve(q):
        u = jnp.linalg.solve(real, jnp.concatenate([q.real, q.imag]))
        return u[:_N_ADV] + 1j * u[_N_ADV:]

    def root(q, **kw):
        return root_solve(lambda z: a @ z + 0.2 * jnp.conj(z) - q,
                          jnp.zeros_like(q), lambda f, z0: solve(q), **kw)

    q = jnp.linspace(0.5, 1.5, _N_ADV) * (1.0 + 0.5j)
    t = jnp.ones(_N_ADV) * (0.3 - 1.0j)
    pair = (m, m_h if adjoint else None)
    tangent = jax.jvp(lambda r: root(r, tangent_precond=pair), (q,), (t,))[1]
    assert _rel(tangent, solve(t)) < 1e-9
    loss = lambda r, **kw: jnp.sum(jnp.abs(root(r, **kw)) ** 2)  # noqa: E731
    grad = jax.grad(lambda r: loss(r, tangent_precond=pair))(q)
    dense = jax.grad(lambda r: loss(r, dense_limit=2 * _N_ADV))(q)
    assert _rel(grad, dense) < 1e-9


# A smaller nonsymmetric root whose operator and preconditioner both depend
# on the differentiated parameter: f = (A0 + p B) x + 0.1 sin(p) x^3 - p c.
_N_PAR = 20


def _parametric(p):
    rng = np.random.default_rng(3)
    a0 = jnp.asarray(4.0 * np.eye(_N_PAR) + rng.standard_normal((_N_PAR, _N_PAR)) / 3)
    b = jnp.asarray(rng.standard_normal((_N_PAR, _N_PAR)) / 10)
    c = jnp.linspace(0.5, 1.5, _N_PAR)
    return a0 + p * b, lambda x: (a0 + p * b) @ x + 0.1 * jnp.sin(p) * x**3 - p * c


def _par_root(p, **kw):
    a, f = _parametric(p)
    return root_solve(f, jnp.zeros(_N_PAR), newton_solver, **kw)


def _par_loss(p, **kw):
    return jnp.sum(_par_root(p, **kw) ** 2)


def test_root_solve_preconditioned_tangent_second_derivatives():
    # GMRES at rtol 1e-12 on cond(J M) near 1 leaves ~1e-12 per solve; the
    # central difference of the gradient with h = 1e-4 has truncation O(h^2).
    a0, _ = _parametric(0.0)
    kw = dict(tangent_precond=_lu_pair(a0), tangent_rtol=1e-12)
    p = jnp.asarray(0.7)
    dense = jax.grad(jax.grad(_par_loss))(p)
    grad = jax.grad(lambda q: _par_loss(q, **kw))
    fd = (grad(p + 1e-4) - grad(p - 1e-4)) / 2e-4
    for hess in (jax.grad(grad), jax.jacfwd(grad), jax.jacrev(grad)):
        assert np.isclose(float(hess(p)), float(dense), rtol=1e-9)
        assert np.isclose(float(hess(p)), float(fd), rtol=1e-6)


def test_root_solve_preconditioner_built_from_the_parameter():
    # M built from p (pair) or from p and the root (factory) must give the
    # derivative of a p-independent M: a converged tangent ignores M.
    a0, _ = _parametric(0.0)
    reference = dict(tangent_precond=_lu_pair(a0), tangent_rtol=1e-12)

    def pair(q):
        return dict(tangent_precond=_lu_pair(_parametric(q)[0]), tangent_rtol=1e-12)

    def factory(q):
        jac = lambda x: jax.jacfwd(_parametric(q)[1])(x)  # noqa: E731
        return dict(tangent_precond=lambda x: _lu_pair(jac(x)), tangent_rtol=1e-12)

    ps = jnp.array([0.3, 0.7])
    ref_grad = jax.vmap(jax.grad(lambda q: _par_loss(q, **reference)))(ps)
    for kw in (pair, factory):
        grad = jax.jit(jax.vmap(jax.grad(lambda q, kw=kw: _par_loss(q, **kw(q)))))(ps)
        assert np.allclose(np.asarray(grad), np.asarray(ref_grad), rtol=1e-9)
        p = ps[1]
        jvp = jax.jvp(lambda q, kw=kw: _par_root(q, **kw(q)), (p,), (1.0,))[1]
        ref = jax.jvp(lambda q: _par_root(q, **reference), (p,), (1.0,))[1]
        assert _rel(jvp, ref) < 1e-9


def test_root_solve_nonlinear_preconditioner_needs_its_transpose():
    a0, _ = _parametric(0.0)
    with pytest.raises(ValueError, match="pass M_T"):
        jax.grad(lambda q: _par_loss(q, tangent_precond=(jnp.tanh, None)))(0.7)
