"""Tests for solver-independent implicit eigenpair differentiation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from solvax import eigenpair_reverse

jax.config.update("jax_enable_x64", True)


def _problem():
    eigenvalue = jnp.asarray(0.5 + 0.2j, dtype=jnp.complex128)
    other = jnp.asarray(-1.0 + 3.0j, dtype=jnp.complex128)
    start = jnp.ones(2, dtype=jnp.complex128)

    def build(parameter):
        matrix = jnp.asarray(
            [[eigenvalue + 0.3 * parameter, 0.0], [parameter, other]],
            dtype=jnp.complex128,
        )
        return lambda vector: matrix @ vector

    def primal_solver(parameter, _apply, _start):
        value = eigenvalue + 0.3 * parameter
        right = jnp.asarray(
            [1.0, parameter / (value - other)],
            dtype=jnp.complex128,
        )
        return value, right

    def left_solver(_parameter, _apply, _start, _value):
        return jnp.asarray([1.0, 0.0], dtype=jnp.complex128)

    return start, build, primal_solver, left_solver


def test_reverse_eigenpair_matches_phase_invariant_finite_difference() -> None:
    """The generic bordered pullback covers eigenvalue and eigenvector terms."""

    start, build, primal_solver, left_solver = _problem()
    weight = jnp.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=jnp.complex128,
    )

    def objective(parameter):
        value, vector = eigenpair_reverse(
            parameter,
            build,
            start,
            primal_solver=primal_solver,
            left_solver=left_solver,
            sensitivity_rtol=1.0e-12,
        )
        normalized = vector / jnp.linalg.norm(vector)
        return jnp.real(value) + 0.1 * jnp.real(jnp.vdot(normalized, weight @ normalized))

    analytic = float(jax.grad(objective)(0.0))
    step = 1.0e-5
    finite_difference = float((objective(step) - objective(-step)) / (2.0 * step))
    assert analytic == pytest.approx(
        finite_difference,
        rel=1.0e-9,
        abs=1.0e-10,
    )


def test_reverse_eigenpair_accepts_transpose_tangent_solver() -> None:
    """An application may replace generic GMRES for the one pullback solve."""

    start, build, primal_solver, left_solver = _problem()
    calls = {"transpose": 0}

    def transpose_solver(
        _parameter,
        _value,
        _right,
        _left,
        matvec,
        rhs,
    ):
        calls["transpose"] += 1
        identity = jnp.eye(rhs.size, dtype=rhs.dtype)
        matrix = jnp.stack(
            tuple(matvec(identity[:, index]) for index in range(rhs.size)),
            axis=1,
        )
        return jnp.linalg.solve(matrix, rhs)

    def objective(parameter):
        value, vector = eigenpair_reverse(
            parameter,
            build,
            start,
            primal_solver=primal_solver,
            left_solver=left_solver,
            transpose_tangent_solver=transpose_solver,
        )
        return jnp.real(value) + 0.1 * jnp.real(vector[1])

    analytic = float(jax.grad(objective)(0.0))
    step = 1.0e-5
    finite_difference = float((objective(step) - objective(-step)) / (2.0 * step))
    assert analytic == pytest.approx(finite_difference, rel=1.0e-9)
    assert calls["transpose"] == 1


def test_reverse_eigenpair_rejects_exceptional_point_condition() -> None:
    """A nearly biorthogonal left/right pair must not emit a huge gradient."""

    start = jnp.ones(2, dtype=jnp.complex128)

    def objective(parameter):
        value, _vector = eigenpair_reverse(
            parameter,
            lambda p: (
                lambda vector: (
                    jnp.asarray(
                        [[1.0 + p, 0.0], [0.0, -1.0]],
                        dtype=jnp.complex128,
                    )
                    @ vector
                )
            ),
            start,
            primal_solver=lambda p, _apply, _start: (
                1.0 + p,
                jnp.asarray([1.0, 0.0], dtype=jnp.complex128),
            ),
            left_solver=lambda *_args: jnp.asarray(
                [1.0e-8, 1.0],
                dtype=jnp.complex128,
            ),
            condition_limit=100.0,
        )
        return jnp.real(value)

    with pytest.raises(ValueError, match="ill-conditioned"):
        jax.grad(objective)(0.0)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("sensitivity_rtol", 0.0, "sensitivity_rtol"),
        ("sensitivity_restart", 0, "iteration limits"),
        ("sensitivity_max_restarts", 0, "iteration limits"),
        ("condition_limit", 1.0, "condition_limit"),
    ],
)
def test_reverse_eigenpair_rejects_invalid_controls(
    keyword: str,
    value: float,
    message: str,
) -> None:
    start, build, primal_solver, left_solver = _problem()
    options = {
        "primal_solver": primal_solver,
        "left_solver": left_solver,
        keyword: value,
    }
    with pytest.raises(ValueError, match=message):
        eigenpair_reverse(0.0, build, start, **options)
