from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from solvax.fixed_point import aitken_fixed_point, aitken_relaxation
from solvax.implicit import root_solve


def test_aitken_fixed_point_accelerates_slow_affine_vector_map():
    target = jnp.asarray([1.0, -2.0, 0.5])

    def mapping(x):
        return 0.98 * x + 0.02 * target

    solution = aitken_fixed_point(mapping, jnp.zeros_like(target), max_steps=20, rtol=1.0e-6)
    assert solution.converged
    assert solution.iterations < 10
    assert solution.x == pytest.approx(target, rel=5.0e-6, abs=5.0e-6)
    assert solution.residual_norm <= 1.0e-6


def test_aitken_fixed_point_is_jittable_and_vmappable():
    def solve(value):
        result = aitken_fixed_point(
            lambda x: 0.9 * x + 0.1 * value,
            jnp.asarray(0.0),
            max_steps=12,
            rtol=1.0e-6,
        )
        return result.x, result.converged

    values = jnp.asarray([1.0, 2.0, 3.0])
    roots, converged = jax.jit(jax.vmap(solve))(values)
    assert roots == pytest.approx(values, rel=1.0e-6, abs=1.0e-6)
    assert jnp.all(converged)


def test_aitken_fixed_point_reports_nonconvergence_without_nan():
    solution = aitken_fixed_point(
        lambda x: x + 1.0,
        jnp.asarray([0.0, 0.0]),
        max_steps=3,
        min_relaxation=0.1,
        max_relaxation=1.0,
    )
    assert not solution.converged
    assert solution.iterations == 3
    assert jnp.all(jnp.isfinite(solution.x))
    assert jnp.isfinite(solution.residual_norm)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_steps": -1}, "max_steps"),
        ({"rtol": -1.0}, "non-negative"),
        ({"atol": -1.0}, "non-negative"),
        ({"min_relaxation": 0.0}, "relaxation bounds"),
        ({"min_relaxation": 2.0, "max_relaxation": 1.0}, "relaxation bounds"),
    ],
)
def test_aitken_fixed_point_rejects_invalid_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        aitken_fixed_point(lambda x: x, jnp.asarray(0.0), **kwargs)


def test_aitken_fixed_point_handles_zero_step_and_initial_root():
    zero_step = aitken_fixed_point(lambda x: 0.5 * x, jnp.asarray(2.0), max_steps=0)
    assert not zero_step.converged
    assert zero_step.iterations == 0
    exact = aitken_fixed_point(lambda x: x, jnp.asarray([2.0]), max_steps=4)
    assert exact.converged
    assert exact.iterations == 0
    assert exact.residual_norm == 0.0


def test_aitken_primal_supports_implicit_root_gradient():
    def solved(parameter):
        def residual(x):
            return parameter * x - 1.0

        def solver(function, initial):
            return aitken_fixed_point(
                lambda x: x - 0.25 * function(x),
                initial,
                rtol=1.0e-6,
                max_steps=20,
            ).x

        return root_solve(residual, jnp.asarray(0.0), solver)

    assert solved(2.0) == pytest.approx(0.5, rel=1.0e-6)
    assert jax.grad(solved)(2.0) == pytest.approx(-0.25, rel=1.0e-6)


def test_incremental_aitken_relaxation_is_jittable_and_safeguarded():
    previous = jnp.asarray([1.0, -2.0])
    current = 0.9 * previous
    omega = jax.jit(aitken_relaxation)(previous, current, 1.0)
    assert omega == pytest.approx(10.0, rel=1.0e-5)
    unchanged = aitken_relaxation(previous, previous, 0.7)
    assert unchanged == pytest.approx(0.7)
    with pytest.raises(ValueError, match="identical shapes"):
        aitken_relaxation(previous, jnp.ones((3,)))
