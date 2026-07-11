from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from solvax import linear_solve
from solvax.pcg import (
    CONVERGED,
    MAX_ITERATIONS,
    NON_POSITIVE_CURVATURE,
    NONFINITE,
    PRECONDITIONER_BREAKDOWN,
    PCGSolution,
    pcg,
    pcg_linear_solve,
    status_name,
)


def test_pcg_matches_dense_spd_solution_and_records_history():
    matrix = jnp.array([[5.0, 1.0], [1.0, 3.0]])
    rhs = jnp.array([2.0, -1.0])
    solution = pcg(lambda x: matrix @ x, rhs, rtol=1.0e-12, max_steps=8)

    assert isinstance(solution, PCGSolution)
    assert solution.converged
    assert int(solution.status) == CONVERGED
    assert status_name(solution.status) == "converged"
    assert jnp.allclose(solution.x, jnp.linalg.solve(matrix, rhs), rtol=1.0e-11)
    assert solution.residual_history.shape == (9,)
    assert jnp.all(solution.residual_history[1:] <= solution.residual_history[:-1] + 1.0e-14)


def test_pcg_supports_pytree_systems_and_jit():
    rhs = {"a": jnp.array([2.0, 4.0]), "b": (jnp.array([9.0]),)}

    def matvec(tree):
        return {"a": 2.0 * tree["a"], "b": (3.0 * tree["b"][0],)}

    solve = jax.jit(lambda value: pcg(matvec, value, max_steps=4))
    solution = solve(rhs)
    assert solution.converged
    assert jnp.allclose(solution.x["a"], jnp.array([1.0, 2.0]))
    assert jnp.allclose(solution.x["b"][0], jnp.array([3.0]))


def test_pcg_supports_integer_rhs_and_complex_hermitian_systems():
    integer = pcg(lambda x: 2.0 * x, jnp.array([2, 4]), max_steps=2)
    assert integer.converged
    assert jnp.issubdtype(integer.x.dtype, jnp.floating)
    assert jnp.allclose(integer.x, jnp.array([1.0, 2.0]))

    matrix = jnp.array([[4.0, 1.0j], [-1.0j, 3.0]])
    rhs = jnp.array([1.0 + 2.0j, -2.0 + 0.5j])
    tolerance = max(1.0e-11, 100.0 * jnp.finfo(matrix.real.dtype).eps)
    complex_solution = pcg(lambda x: matrix @ x, rhs, rtol=tolerance, max_steps=8)
    assert complex_solution.converged
    assert jnp.allclose(complex_solution.x, jnp.linalg.solve(matrix, rhs), rtol=tolerance)
    assert jnp.issubdtype(complex_solution.residual_history.dtype, jnp.floating)


def test_pcg_reports_zero_rhs_iteration_limit_curvature_and_nonfinite():
    zero = pcg(lambda x: x, jnp.zeros(2), max_steps=0)
    assert zero.converged
    assert int(zero.iterations) == 0

    warm_start = pcg(lambda x: 2.0 * x, jnp.array([2.0]), x0=jnp.array([1.0]))
    assert warm_start.converged
    assert int(warm_start.iterations) == 0

    limited = pcg(lambda x: jnp.array([[2.0, 1.0], [1.0, 2.0]]) @ x, jnp.ones(2), max_steps=0)
    assert int(limited.status) == MAX_ITERATIONS
    assert not limited.converged

    indefinite = pcg(lambda x: -x, jnp.ones(2), max_steps=2)
    assert int(indefinite.status) == NON_POSITIVE_CURVATURE

    nonfinite = pcg(lambda x: x * jnp.nan, jnp.ones(2), max_steps=2)
    assert int(nonfinite.status) == NONFINITE

    broken_preconditioner = pcg(
        lambda x: x,
        jnp.ones(2),
        precond=lambda residual: -residual,
        max_steps=2,
    )
    assert int(broken_preconditioner.status) == PRECONDITIONER_BREAKDOWN


def test_pcg_is_scale_invariant_and_jacobi_reduces_iterations():
    diagonal = jnp.logspace(0.0, 4.0, 32)
    rhs = jnp.ones_like(diagonal)
    tolerance = max(1.0e-10, 100.0 * jnp.finfo(diagonal.dtype).eps)
    scale = 1.0e-100 if jax.config.read("jax_enable_x64") else 1.0e-10
    plain = pcg(lambda x: diagonal * x, rhs, rtol=tolerance, max_steps=80)
    jacobi = pcg(
        lambda x: diagonal * x,
        rhs,
        precond=lambda residual: residual / diagonal,
        rtol=tolerance,
        max_steps=80,
    )
    scaled = pcg(
        lambda x: scale * diagonal * x,
        scale * rhs,
        precond=lambda residual: residual / (scale * diagonal),
        rtol=tolerance,
        max_steps=80,
    )
    assert jacobi.converged and scaled.converged
    assert int(jacobi.iterations) < int(plain.iterations)
    assert jnp.allclose(jacobi.x, scaled.x, rtol=tolerance)


def test_pcg_validates_inputs_and_status_names():
    with pytest.raises(ValueError, match="max_steps"):
        pcg(lambda x: x, jnp.ones(1), max_steps=-1)
    with pytest.raises(ValueError, match="tolerances"):
        pcg(lambda x: x, jnp.ones(1), rtol=-1.0)
    with pytest.raises(ValueError, match="array leaf"):
        pcg(lambda x: x, {})
    with pytest.raises(ValueError, match="pytree structure"):
        pcg(lambda x: x, {"x": jnp.ones(1)}, x0=jnp.ones(1))
    with pytest.raises(ValueError, match="Unknown PCG status"):
        status_name(99)


def test_pcg_works_as_implicit_linear_solve_backend():
    rhs = jnp.array([1.0, -2.0])

    def objective(scale):
        def matvec(x):
            return scale * x

        def solver(operator, value):
            return pcg(operator, value, rtol=1.0e-13, max_steps=4).x

        return jnp.sum(linear_solve(matvec, rhs, solver) ** 2)

    scale = 3.0
    expected = -2.0 * jnp.sum(rhs**2) / scale**3
    assert jnp.allclose(jax.grad(objective)(scale), expected, rtol=1.0e-6)


def test_pcg_linear_solve_retains_diagnostics_and_uses_implicit_gradient():
    rhs = jnp.array([1.0, -2.0])

    def solve_and_sum(scale):
        diagonal = jnp.array([scale, scale + 1.0])
        solution = pcg_linear_solve(
            lambda x: diagonal * x,
            rhs,
            precond=lambda residual: residual / diagonal,
            rtol=1.0e-12,
            max_steps=4,
            transpose_rtol=1.0e-13,
            transpose_max_steps=6,
        )
        return jnp.sum(solution.x**2), solution

    objective, solution = solve_and_sum(3.0)
    expected_solution = rhs / jnp.array([3.0, 4.0])
    assert solution.converged
    assert int(solution.iterations) == 1
    assert solution.residual_history.shape == (5,)
    assert jnp.allclose(solution.x, expected_solution)
    assert jnp.allclose(objective, jnp.sum(expected_solution**2))

    gradient = jax.grad(lambda scale: solve_and_sum(scale)[0])(3.0)
    expected_gradient = -2.0 * (rhs[0] ** 2 / 3.0**3 + rhs[1] ** 2 / 4.0**3)
    assert jnp.allclose(gradient, expected_gradient, rtol=1.0e-6)
