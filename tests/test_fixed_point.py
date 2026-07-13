from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax.fixed_point import (
    affine_fixed_point_gmres,
    aitken_fixed_point,
    aitken_relaxation,
    anderson_mixing,
)
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


def test_complex_aitken_and_anderson_converge_with_real_safeguards():
    target = jnp.asarray([1.0 + 2.0j, -0.5 + 0.3j])
    solution = jax.jit(
        lambda: aitken_fixed_point(
            lambda x: 0.95 * x + 0.05 * target,
            jnp.zeros_like(target),
            max_steps=20,
            rtol=1.0e-10,
        )
    )()
    assert solution.converged
    assert not jnp.issubdtype(solution.relaxation.dtype, jnp.complexfloating)
    assert solution.x == pytest.approx(target, rel=2.0e-6, abs=2.0e-6)

    iterates = jnp.stack([jnp.zeros_like(target), 0.2 * target])
    residuals = jnp.stack([0.2 * target, 0.16 * target])
    candidate = jax.jit(anderson_mixing)(iterates, residuals)
    assert jnp.all(jnp.isfinite(candidate))
    assert jnp.issubdtype(candidate.dtype, jnp.complexfloating)


def test_complex_anderson_uses_hermitian_residual_gram_matrix():
    iterates = jnp.asarray(
        [[0.2 + 0.1j, -0.3j], [0.5 - 0.2j, 0.1 + 0.4j], [-0.1j, 0.3]]
    )
    residuals = jnp.asarray(
        [[0.3 + 0.2j, -0.1j], [-0.2 + 0.1j, 0.4j], [0.1 - 0.3j, 0.2]]
    )
    regularization = 1.0e-6
    result = anderson_mixing(
        iterates, residuals, regularization=regularization, damping=1.0
    )

    flat = np.asarray(residuals)
    gram = np.conj(flat) @ flat.T
    scale = np.trace(gram).real / len(flat)
    system = gram + (regularization + np.finfo(float).eps) * scale * np.eye(len(flat))
    weights = np.linalg.solve(system, np.ones(len(flat), dtype=complex))
    weights /= np.sum(weights)
    expected = weights @ np.asarray(iterates + residuals)
    assert np.asarray(result) == pytest.approx(expected, rel=1.0e-6, abs=1.0e-6)


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


def test_anderson_mixing_accelerates_a_multimode_affine_map():
    diagonal = jnp.asarray([0.2, 0.9, 0.99])
    target = jnp.asarray([1.0, -2.0, 0.5])

    def mapping(x):
        return diagonal * x + (1.0 - diagonal) * target

    x = jnp.zeros_like(target)
    iterates = []
    residuals = []
    for _ in range(8):
        residual = mapping(x) - x
        iterates.append(x)
        residuals.append(residual)
        x = anderson_mixing(jnp.stack(iterates[-4:]), jnp.stack(residuals[-4:]))

    assert x == pytest.approx(target, rel=2.0e-5, abs=2.0e-5)


def test_affine_fixed_point_gmres_solves_a_slow_multimode_map():
    diagonal = jnp.asarray([0.2, 0.95, 0.999])
    target = jnp.asarray([1.0, -2.0, 0.5])

    solution = affine_fixed_point_gmres(
        lambda x: diagonal * x + (1.0 - diagonal) * target,
        jnp.zeros_like(target),
        restart=3,
        rtol=1.0e-10,
        max_restarts=2,
    )

    assert solution.converged
    assert solution.iterations <= 3
    assert solution.x == pytest.approx(target, rel=2.0e-5, abs=2.0e-5)


def test_affine_fixed_point_gmres_supports_pytrees_and_custom_inner_products():
    target = {"flow": jnp.asarray([1.0, -1.0]), "potential": jnp.asarray(0.5)}

    def mapping(state):
        return {
            "flow": 0.9 * state["flow"] + 0.1 * target["flow"],
            "potential": 0.5 * state["potential"] + 0.5 * target["potential"],
        }

    def inner(left, right):
        return jnp.vdot(left["flow"], right["flow"]) + 0.25 * jnp.vdot(
            left["potential"], right["potential"]
        )

    solution = jax.jit(
        lambda: affine_fixed_point_gmres(
            mapping,
            jax.tree.map(jnp.zeros_like, target),
            inner_product=inner,
            restart=2,
            rtol=1.0e-10,
        )
    )()

    assert solution.converged
    assert solution.x["flow"] == pytest.approx(
        target["flow"], rel=2.0e-5, abs=2.0e-5
    )
    assert solution.x["potential"] == pytest.approx(
        target["potential"], rel=2.0e-5, abs=2.0e-5
    )


def test_affine_fixed_point_gmres_rejects_structure_changes():
    with pytest.raises(ValueError, match="preserve"):
        affine_fixed_point_gmres(
            lambda x: {"changed": x}, jnp.asarray([0.0]), max_restarts=0
        )


def test_anderson_mixing_is_jittable_safeguarded_and_validated():
    iterates = jnp.asarray([[0.0, 0.0], [0.5, -0.5]])
    residuals = jnp.asarray([[1.0, -1.0], [0.5, -0.5]])
    candidate = jax.jit(anderson_mixing)(iterates, residuals)
    assert jnp.all(jnp.isfinite(candidate))
    assert anderson_mixing(iterates[-1:], residuals[-1:]) == pytest.approx([1.0, -1.0])
    with pytest.raises(ValueError, match="identical shapes"):
        anderson_mixing(iterates, residuals[:1])
    with pytest.raises(ValueError, match="at least one"):
        anderson_mixing(jnp.empty((0, 2)), jnp.empty((0, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        anderson_mixing(iterates, residuals, regularization=-1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        anderson_mixing(iterates, residuals, damping=1.1)
    with pytest.raises(ValueError, match="condition_limit"):
        anderson_mixing(iterates, residuals, condition_limit=0.5)


def test_condition_filtered_anderson_handles_dependent_histories():
    iterates = jnp.asarray([[0.0, 0.0], [0.5, -0.5], [0.75, -0.75]])
    residuals = jnp.asarray([[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]])

    candidate = jax.jit(
        lambda: anderson_mixing(
            iterates, residuals, regularization=0.0, condition_limit=1.0e4
        )
    )()

    assert jnp.all(jnp.isfinite(candidate))
    assert candidate == pytest.approx([1.0, -1.0], rel=1.0e-6, abs=1.0e-6)


def test_anderson_falls_back_when_affine_weights_are_degenerate():
    iterates = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    residuals = jnp.zeros_like(iterates)

    candidate = anderson_mixing(
        iterates, residuals, regularization=0.0, condition_limit=1.0e4
    )

    assert candidate == pytest.approx(iterates[-1])
