"""Matrix-free nonlinear least-squares and implicit derivative tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    LeastSquaresConfig,
    gauss_newton_least_squares,
    implicit_least_squares,
    least_squares_stationarity,
)

jax.config.update("jax_enable_x64", True)


def test_overdetermined_linear_least_squares_is_jittable() -> None:
    matrix = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    right_hand_side = jnp.asarray([1.0, -2.0, 0.5])
    expected = np.linalg.lstsq(
        np.asarray(matrix), np.asarray(right_hand_side), rcond=None
    )[0]
    config = LeastSquaresConfig(
        rtol=1.0e-10,
        max_steps=12,
        initial_damping=1.0e-4,
        linear_rtol=1.0e-12,
        linear_max_steps=20,
    )
    solve = jax.jit(
        lambda initial: gauss_newton_least_squares(
            lambda value: matrix @ value - right_hand_side,
            initial,
            config=config,
        )
    )
    result = solve(jnp.zeros((2,)))
    assert result.converged
    assert result.accepted_steps > 0
    np.testing.assert_allclose(result.x, expected, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(
        least_squares_stationarity(
            lambda value: matrix @ value - right_hand_side, result.x
        ),
        0.0,
        atol=1.0e-9,
    )


def test_nonlinear_least_squares_rejects_then_recovers() -> None:
    target = jnp.asarray([1.0, 4.0, 9.0])
    abscissa = jnp.asarray([1.0, 2.0, 3.0])
    config = LeastSquaresConfig(
        rtol=1.0e-8,
        max_steps=30,
        initial_damping=1.0e-6,
        linear_rtol=1.0e-10,
        linear_max_steps=20,
    )
    result = gauss_newton_least_squares(
        lambda value: value[0] * abscissa ** value[1] - target,
        jnp.asarray([4.0, 0.25]),
        config=config,
        admissible=lambda value: jnp.all(value > 0.0),
    )
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=2.0e-6, atol=2.0e-6)


def test_implicit_least_squares_uses_stationary_point_derivative() -> None:
    config = LeastSquaresConfig(
        rtol=1.0e-12,
        max_steps=15,
        initial_damping=1.0e-4,
        linear_rtol=1.0e-12,
        linear_max_steps=10,
    )

    def solution(parameter):
        return implicit_least_squares(
            lambda value: jnp.stack(
                (value[0] - parameter, 2.0 * value[0] - parameter)
            ),
            jnp.zeros((1,)),
            config=config,
        )[0]

    np.testing.assert_allclose(solution(2.0), 1.2, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(jax.grad(solution)(2.0), 0.6, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("rtol", float("inf"), "finite"),
        ("rtol", -1.0, "nonnegative"),
        ("initial_damping", 0.0, "minimum_damping"),
        ("damping_decrease", 1.0, "damping_decrease"),
        ("damping_increase", 1.0, "damping_increase"),
        ("acceptance_ratio", 0.5, "acceptance_ratio"),
        ("good_ratio", 0.1, "poor_ratio"),
        ("linear_rtol", -1.0, "linear tolerances"),
        ("linear_max_steps", 0, "iteration limits"),
    ],
)
def test_least_squares_config_rejects_invalid_controls(
    keyword: str, value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LeastSquaresConfig(**{keyword: value})


def test_least_squares_rejects_nonvector_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        gauss_newton_least_squares(lambda value: value, jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="residual"):
        gauss_newton_least_squares(
            lambda value: jnp.zeros((2, 2)), jnp.zeros((2,))
        )
