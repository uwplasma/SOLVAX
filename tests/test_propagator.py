"""Tests for stability-limited, residual-driven propagator policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    adaptive_eigenpair,
    estimate_rk4_timestep,
    exponential_action,
    exponential_eigenpairs,
    plan_chebyshev_action,
    plan_exponential_action,
    propagator_eigenpairs,
)

jax.config.update("jax_enable_x64", True)


def test_rk4_timestep_keeps_the_full_known_spectrum_stable() -> None:
    """A conservative Arnoldi sketch must not amplify omitted peripheral modes."""

    frequencies = np.linspace(-24.0, 24.0, 18)
    eigenvalues = jnp.asarray(-0.2 + 1j * frequencies)
    apply = jax.jit(lambda vector: eigenvalues * vector)
    estimate = estimate_rk4_timestep(
        apply,
        jnp.ones_like(eigenvalues),
        dimension=12,
        safety=0.7,
    )

    z = estimate.dt * np.asarray(eigenvalues)
    amplification = np.abs(1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    assert estimate.dt > 0.0
    assert estimate.probe_count == 2
    assert estimate.operator_applications == 24
    assert np.max(amplification) <= 1.0


def test_rk4_timestep_broadband_probe_catches_invariant_seed_blindness() -> None:
    """A recycled eigenmode must not hide a stability-limiting peripheral mode."""

    frequencies = np.linspace(-80.0, 80.0, 18)
    eigenvalues = jnp.asarray(-0.2 + 1j * frequencies)
    apply = jax.jit(lambda vector: eigenvalues * vector)
    invariant_seed = jnp.zeros_like(eigenvalues).at[8].set(1.0)
    estimate = estimate_rk4_timestep(
        apply,
        invariant_seed,
        dimension=12,
        safety=0.7,
    )

    z = estimate.dt * np.asarray(eigenvalues)
    amplification = np.abs(1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    assert estimate.spectral_radius > 70.0
    assert np.max(amplification) <= 1.0


def test_adaptive_eigenpair_stops_on_the_original_residual() -> None:
    """The horizon must stop early once the continuous eigenpair is certified."""

    eigenvalues = jnp.asarray([0.3 + 0.2j, 0.1 - 0.4j, -0.5 + 2.0j])
    weights = jnp.exp(20.0 * eigenvalues)
    apply = jax.jit(lambda vector: eigenvalues * vector)

    def restart_once(vector):
        filtered = weights * vector
        filtered = filtered / jnp.linalg.norm(filtered)
        return jnp.vdot(filtered, apply(filtered)), filtered

    solution = adaptive_eigenpair(
        apply,
        restart_once,
        jnp.ones_like(eigenvalues),
        tol=1.0e-6,
        max_restarts=6,
        filter_dt=0.05,
        filter_steps=400,
        applications_per_restart=1,
    )

    assert solution.converged
    assert solution.stable
    assert solution.restarts < 6
    assert complex(np.asarray(solution.eigenvalue)) == pytest.approx(
        np.asarray(eigenvalues[0]).item()
    )
    assert float(np.asarray(solution.residual)) < 1.0e-6


def test_adaptive_eigenpair_rejects_numerical_rk4_growth() -> None:
    """A small residual may not certify a mode made dominant by unstable RK4."""

    eigenvalues = jnp.asarray([0.2 + 0.1j, -0.1 + 3.0j])
    apply = jax.jit(lambda vector: eigenvalues * vector)

    def restart_once(_vector):
        selected = jnp.asarray([0.0, 1.0], dtype=eigenvalues.dtype)
        return eigenvalues[1], selected

    solution = adaptive_eigenpair(
        apply,
        restart_once,
        jnp.ones_like(eigenvalues),
        tol=1.0e-12,
        max_restarts=4,
        filter_dt=1.0,
        filter_steps=1,
        applications_per_restart=1,
    )

    assert not solution.converged
    assert not solution.stable
    assert solution.restarts == 1


def test_propagator_eigenpairs_certifies_a_leading_cluster() -> None:
    """One compiled filtered subspace must expose tied rightmost candidates."""

    eigenvalues = jnp.asarray(
        [
            0.30 + 0.2j,
            0.29 - 0.4j,
            -0.2 + 2.0j,
            -0.3 - 3.0j,
            -0.4 + 4.0j,
            -0.5 - 5.0j,
            -0.6 + 6.0j,
            -0.7 - 7.0j,
            -0.8 + 8.0j,
            -0.9 - 9.0j,
        ],
        dtype=jnp.complex128,
    )
    apply = jax.jit(lambda vector: eigenvalues * vector)
    solution = propagator_eigenpairs(
        apply,
        jnp.ones_like(eigenvalues),
        dt=0.02,
        steps=500,
        krylov_dim=8,
        candidates=2,
        tol=1.0e-9,
    )

    assert np.all(np.asarray(solution.converged))
    assert float(np.max(np.asarray(solution.residuals))) < 1.0e-9
    np.testing.assert_allclose(
        np.sort_complex(np.asarray(solution.eigenvalues)),
        np.sort_complex(np.asarray(eigenvalues[:2])),
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_exponential_eigenpairs_bypasses_the_explicit_stability_limit() -> None:
    """Nested Krylov exponentials must certify stiff leading modes."""

    eigenvalues = jnp.asarray(
        [
            0.30 + 0.2j,
            0.29 - 0.4j,
            -0.2 + 20.0j,
            -0.3 - 30.0j,
            -0.4 + 40.0j,
            -0.5 - 50.0j,
            -0.6 + 60.0j,
            -0.7 - 70.0j,
            -0.8 + 80.0j,
            -0.9 - 90.0j,
            -1.0 + 100.0j,
            -1.1 - 110.0j,
        ],
        dtype=jnp.complex128,
    )
    solution = exponential_eigenpairs(
        jax.jit(lambda vector: eigenvalues * vector),
        jnp.ones_like(eigenvalues),
        horizon=10.0,
        inner_krylov_dim=12,
        outer_krylov_dim=8,
        candidates=2,
        tol=1.0e-8,
        restarts=2,
    )

    assert np.all(np.asarray(solution.converged))
    assert solution.operator_applications == 194
    np.testing.assert_allclose(
        np.sort_complex(np.asarray(solution.eigenvalues)),
        np.sort_complex(np.asarray(eigenvalues[:2])),
        rtol=1.0e-8,
        atol=1.0e-9,
    )


@pytest.mark.parametrize("tolerance", [1e-4, 1e-7, 1e-10])
@pytest.mark.parametrize("family", ["jordan", "skew", "diffusion", "triangular"])
def test_taylor_action_matches_independent_dense_and_sparse(family, tolerance):
    """General vectors, RHS blocks and nonnormal limits need independent oracles."""
    from scipy.linalg import expm
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import expm_multiply

    rng = np.random.default_rng(46)
    n = 9
    matrix = {
        "jordan": -0.7 * np.eye(n) + 3.2 * np.diag(np.ones(n - 1), 1),
        "skew": 1j * np.diag(np.linspace(-4, 5, n)),
        "diffusion": -2 * np.eye(n) + np.diag(np.ones(n - 1), 1)
        + np.diag(np.ones(n - 1), -1),
        "triangular": np.diag(np.linspace(-4, -0.2, n))
        + np.triu(rng.normal(size=(n, n)), 1),
    }[family].astype(np.complex128)
    vector = 3 * rng.normal(size=(n, 2)) + 2j * rng.normal(size=(n, 2))
    time = 0.4
    shift = np.trace(matrix) / n
    # Frobenius norm is an independent, conservative induced-2-norm bound.
    plan = plan_exponential_action(
        horizon=time, norm_bound=np.linalg.norm(matrix - shift * np.eye(n)),
        shift=shift, tolerance=tolerance,
    )
    result = jax.jit(lambda a, v: exponential_action(lambda w: a @ w, v, plan))(
        jnp.asarray(matrix), jnp.asarray(vector)
    )
    expected = expm(time * matrix) @ vector
    np.testing.assert_allclose(expm_multiply(time * csc_matrix(matrix), vector), expected,
                               rtol=2e-13, atol=2e-13)
    error = np.linalg.norm(np.asarray(result.value) - expected)
    assert error <= float(result.truncation_bound) + 2e-13 * np.linalg.norm(expected)
    assert result.valid
    assert plan.error_bound <= tolerance
    assert result.operator_applications == plan.degree * plan.substeps


def test_taylor_action_zero_identity_and_scalar_limits():
    vector = jnp.array([2.0, -3.0])
    for horizon, rho, shift in [(0.0, 3.0, 0.0), (0.4, 0.0, 0.0), (1.2, 0.0, -2.0)]:
        plan = plan_exponential_action(horizon=horizon, norm_bound=rho, shift=shift)
        got = exponential_action(lambda v, shift=shift: shift * v, vector, plan)
        np.testing.assert_allclose(got.value, np.exp(horizon * shift) * vector, atol=0)
        assert got.valid
        zero = exponential_action(lambda v, shift=shift: shift * v, jnp.zeros_like(vector), plan)
        np.testing.assert_array_equal(zero.value, np.zeros(2))
        assert float(zero.truncation_bound) == 0.0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_taylor_action_frechet_time_input_and_complex_adjoint(checkpoint):
    """Dense block Frechet and realified Jacobians validate all operand derivatives."""
    from scipy.linalg import expm, expm_frechet

    matrix = np.array([[-1 + 0.2j, 2 + 1j], [0.1j, -0.8 - 0.3j]])
    direction = np.array([[0.2j, -0.3], [0.1, 0.4 - 0.2j]])
    vector = np.array([2 - 0.2j, -3 + 0.4j])
    dv = np.array([0.1 + 0.4j, -0.3j])
    time, dt = 0.4, -0.1
    plan = plan_exponential_action(horizon=0.5, norm_bound=4.0, tolerance=1e-12)

    def action(a, v, t):
        return exponential_action(lambda w: a @ w, v, plan, horizon=t,
                                  checkpoint=checkpoint).value

    args = (jnp.asarray(matrix), jnp.asarray(vector), jnp.asarray(time))
    value, tangent = jax.jit(lambda a, v, t: jax.jvp(
        action, (a, v, t), (jnp.asarray(direction), jnp.asarray(dv), jnp.asarray(dt))
    ))(*args)
    exponential = expm(time * matrix)
    expected = expm_frechet(time * matrix, time * direction, compute_expm=False) @ vector
    expected += exponential @ dv + dt * matrix @ exponential @ vector
    np.testing.assert_allclose(value, exponential @ vector, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, expected, rtol=2e-11, atol=2e-11)
    for h in (1e-3, 5e-4, 2.5e-4):
        fd = (action(args[0] + h * direction, args[1] + h * dv, time + h * dt)
              - action(args[0] - h * direction, args[1] - h * dv, time - h * dt)) / (2 * h)
        np.testing.assert_allclose(fd, expected, rtol=2 * h**2, atol=2 * h**2)

    # Realification tests the complete small input Jacobian under the real pairing.
    def real_action(x):
        z = x[:2] + 1j * x[2:]
        y = action(args[0], z, args[2])
        return jnp.concatenate((y.real, y.imag))

    real_vector = jnp.concatenate((args[1].real, args[1].imag))
    expected_jacobian = np.block([[exponential.real, -exponential.imag],
                                 [exponential.imag, exponential.real]])
    np.testing.assert_allclose(jax.jacfwd(real_action)(real_vector), expected_jacobian,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(jax.jacrev(real_action)(real_vector), expected_jacobian,
                               rtol=2e-12, atol=2e-12)
    w = jnp.array([0.3 - 0.1j, -0.4 + 0.7j])
    _, pullback = jax.vjp(lambda v: action(args[0], v, args[2]), args[1])
    adjoint = jnp.conj(pullback(jnp.conj(w))[0])
    np.testing.assert_allclose(adjoint, exponential.conj().T @ w, atol=2e-12)
    assert float(jnp.real(jnp.vdot(w, action(args[0], jnp.asarray(dv), time)))) == \
        pytest.approx(float(jnp.real(jnp.vdot(adjoint, dv))), abs=2e-12)
    # Operator-parameter reverse mode must transpose the same fixed recurrence.
    def objective(p):
        return jnp.real(jnp.vdot(w, action(args[0] + p * direction, args[1], time)))
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(0.0),
                               np.real(np.vdot(w, expm_frechet(time * matrix,
                               time * direction, compute_expm=False) @ vector)), atol=2e-12)


def test_taylor_action_batching_and_plan_interval_maximum():
    """A strongly negative shift has an interior maximum of the global tail bound."""
    plan = plan_exponential_action(horizon=20, norm_bound=1, shift=-20, tolerance=1e-9)
    times = jnp.array([0.0, 0.02, 0.1, 0.5, 20.0])
    vectors = jnp.array([[2.0, -3.0], [0.1, 0.2], [1.0, 1.0], [0, 0], [-0.5, 0.7]])
    result = jax.jit(jax.vmap(lambda v, t: exponential_action(lambda w: -19.0 * w,
                                                            v, plan, horizon=t)))(vectors, times)
    expected = np.exp(-19 * np.asarray(times))[:, None] * vectors
    errors = np.linalg.norm(np.asarray(result.value - expected), axis=1)
    assert np.all(errors <= np.asarray(result.truncation_bound) + 2e-15)
    assert np.all(result.valid)
    tiny = plan_exponential_action(horizon=1e-200, norm_bound=1e-200)
    assert 0.0 < tiny.error_bound < 1e-300
    with pytest.raises(ValueError, match="exhausted"):
        plan_exponential_action(horizon=1e300, norm_bound=1e300)
    identity = plan_exponential_action(horizon=1, norm_bound=0)
    huge = exponential_action(lambda v: jnp.zeros_like(v), jnp.array([1e200, -1e200]),
                              identity)
    assert huge.valid and float(huge.truncation_bound) == 0.0


@pytest.mark.parametrize("kwargs, message", [
    ({"horizon": -1}, "horizon"), ({"horizon": np.inf}, "horizon"),
    ({"norm_bound": -1}, "norm_bound"), ({"norm_bound": np.nan}, "norm_bound"),
    ({"tolerance": 0}, "tolerance"), ({"tolerance": 1}, "tolerance"),
    ({"shift": complex(np.nan, 0)}, "shift"), ({"max_degree": 0}, "positive"),
    ({"max_substeps": 0}, "positive"),
    ({"max_degree": 1, "max_substeps": 1}, "exhausted"),
])
def test_taylor_planner_rejects_invalid_inputs(kwargs, message):
    inputs = {"horizon": 1, "norm_bound": 2}
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        plan_exponential_action(**inputs)


def test_taylor_action_invalid_operands_and_status():
    plan = plan_exponential_action(horizon=1, norm_bound=2)
    with pytest.raises(TypeError, match="floating"):
        exponential_action(lambda v: v, jnp.ones(2, dtype=jnp.int32), plan)
    with pytest.raises(ValueError, match="shape and dtype"):
        exponential_action(lambda v: v[:1], jnp.ones(2), plan)
    with pytest.raises(ValueError, match="shape and dtype"):
        exponential_action(lambda v: v.astype(jnp.float32), jnp.ones(2), plan)
    with pytest.raises(ValueError, match="real scalar"):
        exponential_action(lambda v: v, jnp.ones(2), plan, horizon=1j)
    complex_plan = plan_exponential_action(horizon=1, norm_bound=2, shift=1j)
    with pytest.raises(TypeError, match="complex shift"):
        exponential_action(lambda v: v, jnp.ones(2), complex_plan)
    for t in (-1.0, 1.1, np.nan):
        result = jax.jit(lambda t: exponential_action(lambda v: v, jnp.ones(2),
                                                      plan, horizon=t))(t)
        assert not result.valid
        assert np.isinf(result.truncation_bound)
    result = exponential_action(lambda v: v * np.inf, jnp.ones(2), plan)
    assert not result.valid
    result = exponential_action(lambda v: v, jnp.array([np.nan, 1.0]), plan)
    assert not result.valid


def test_taylor_optional_mode_filter_preserves_continuous_eigenpairs():
    values = jnp.array([0.3 + 0.2j, 0.29 - 0.4j, -0.2 + 2j, -0.3 - 3j,
                        -0.4 + 4j, -0.5 - 5j, -0.6 + 6j, -0.7 - 7j,
                        -0.8 + 8j, -0.9 - 9j], dtype=jnp.complex128)
    plan = plan_exponential_action(horizon=10, norm_bound=10, tolerance=1e-11)
    result = exponential_eigenpairs(lambda v: values * v, jnp.ones_like(values),
                                   horizon=10, action_plan=plan, outer_krylov_dim=8,
                                   candidates=2, restarts=1, tol=1e-8)
    np.testing.assert_allclose(np.sort_complex(result.eigenvalues),
                               np.sort_complex(values[:2]), rtol=1e-8, atol=1e-9)
    assert np.all(result.converged)
    assert result.operator_applications == 8 * plan.degree * plan.substeps + 2
    with pytest.raises(ValueError, match="either"):
        exponential_eigenpairs(lambda v: v, values, horizon=5, inner_krylov_dim=4,
                               action_plan=plan, outer_krylov_dim=4)
    with pytest.raises(ValueError, match="interval"):
        exponential_eigenpairs(lambda v: v, values, horizon=11, action_plan=plan,
                               outer_krylov_dim=4)


def test_exponential_mode_breakdown_does_not_certify_zero_vectors():
    """The zero vector is not an eigenvector, even with an exactly zero residual."""
    diagonal = jnp.array([0.3 + 0.2j, 0.29 - 0.4j, -0.2 + 2j, -0.3 - 3j,
                          -0.4 + 4j, -0.5 - 5j, -0.6 + 6j, -0.7 - 7j,
                          -0.8 + 8j, -0.9 - 9j], dtype=jnp.complex128)
    for initial in (jnp.zeros_like(diagonal), jnp.ones_like(diagonal)):
        result = exponential_eigenpairs(lambda v: diagonal * v, initial,
                                       horizon=10, inner_krylov_dim=10,
                                       outer_krylov_dim=8, candidates=2, restarts=2)
        norms = np.linalg.norm(np.asarray(result.eigenvectors), axis=1)
        assert np.all(~np.asarray(result.converged)[norms == 0])
        assert np.all(np.isinf(np.asarray(result.residuals)[norms == 0]))


@pytest.mark.parametrize("tolerance", [1e-4, 1e-7, 1e-10])
@pytest.mark.parametrize("family", ["skew", "nonnormal", "damped"])
def test_chebyshev_action_independent_values_and_input_norm_bound(family, tolerance):
    """Bound a full numerical range, not just eigenvalues or scalar Ritz probes."""
    from scipy.linalg import expm
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import expm_multiply

    rng = np.random.default_rng(149)
    n = 7
    matrices = {
        "skew": 1j * np.diag(np.linspace(-12, 12, n)),
        "nonnormal": 1j * np.diag(np.linspace(-12, 12, n))
        + 2 * np.diag(np.ones(n - 1), 1),
        "damped": -4 * np.eye(n) + 1j * np.diag(np.linspace(-12, 12, n))
        + np.diag(np.ones(n - 1), 1),
    }
    matrix = matrices[family]
    shift = np.trace(matrix) / n
    centered = matrix - shift * np.eye(n)
    hermitian = (centered + centered.conj().T) / 2
    imaginary = (centered - centered.conj().T) / (2j)
    alpha = float(np.max(np.abs(np.linalg.eigvalsh(hermitian)))) * (1 + 1e-12)
    beta = float(np.max(np.abs(np.linalg.eigvalsh(imaginary)))) * (1 + 1e-12)
    plan = plan_chebyshev_action(horizon=0.7, real_halfwidth=alpha,
                                 imag_halfwidth=beta, shift=shift, tolerance=tolerance)
    vector = 3 * rng.normal(size=(n, 2)) + 2j * rng.normal(size=(n, 2))
    expected = expm(0.7 * matrix) @ vector
    np.testing.assert_allclose(expm_multiply(0.7 * csc_matrix(matrix), vector), expected,
                               rtol=2e-13, atol=2e-13)
    action = jax.jit(lambda a, v: exponential_action(lambda w: a @ w, v, plan))
    result = action(jnp.asarray(matrix), jnp.asarray(vector))
    error = np.linalg.norm(np.asarray(result.value) - expected)
    assert result.valid and plan.error_bound <= tolerance
    assert error <= float(result.truncation_bound) + 5e-13 * np.linalg.norm(expected)
    assert result.operator_applications == plan.degree
    np.testing.assert_allclose(action(jnp.asarray(matrix), jnp.zeros_like(vector)).value,
                               np.zeros_like(vector), atol=0)
    # The tail is input-absolute: a nearly annihilated output has no relative promise.
    if family == "damped":
        assert np.linalg.norm(expected) < np.linalg.norm(vector)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_chebyshev_operator_input_jvp_vjp_and_zero_start_adjoint(checkpoint):
    from scipy.linalg import expm, expm_frechet

    matrix = np.array([[-0.2 + 2j, 1.3], [0.1j, -0.3 - 3j]])
    direction = np.array([[0.2j, -0.3], [0.1, 0.4 - 0.2j]])
    vector = np.array([2 - 0.2j, -3 + 0.4j])
    dv = np.array([0.1 + 0.4j, -0.3j])
    plan = plan_chebyshev_action(horizon=0.4, real_halfwidth=2,
                                 imag_halfwidth=5, tolerance=1e-12)

    def action(a, v):
        return exponential_action(lambda w: a @ w, v, plan, checkpoint=checkpoint).value

    args = (jnp.asarray(matrix), jnp.asarray(vector))
    exponential = expm(0.4 * matrix)
    derivative = expm_frechet(0.4 * matrix, 0.4 * direction, compute_expm=False) @ vector
    value, tangent = jax.jit(lambda a, v: jax.jvp(
        action, (a, v), (jnp.asarray(direction), jnp.asarray(dv))))(*args)
    np.testing.assert_allclose(value, exponential @ vector, atol=2e-12)
    np.testing.assert_allclose(tangent, derivative + exponential @ dv, atol=2e-12)
    w = jnp.array([0.3 - 0.1j, -0.4 + 0.7j])
    for start in (args[1], jnp.zeros_like(args[1])):
        _, pullback = jax.vjp(lambda v: action(args[0], v), start)
        adjoint = jnp.conj(pullback(jnp.conj(w))[0])
        np.testing.assert_allclose(adjoint, exponential.conj().T @ w, atol=2e-12)
        np.testing.assert_allclose(jnp.real(jnp.vdot(w, action(args[0], jnp.asarray(dv)))),
                                   jnp.real(jnp.vdot(adjoint, dv)), atol=2e-12)

    def real_action(x):
        y = action(args[0], x[:2] + 1j * x[2:])
        return jnp.concatenate((y.real, y.imag))

    expected = np.block([[exponential.real, -exponential.imag],
                         [exponential.imag, exponential.real]])
    start = jnp.concatenate((args[1].real, args[1].imag))
    np.testing.assert_allclose(jax.jacfwd(real_action)(start), expected, atol=2e-12)
    np.testing.assert_allclose(jax.jacrev(real_action)(start), expected, atol=2e-12)
    def objective(p):
        return jnp.real(jnp.vdot(w, action(args[0] + p * direction, args[1])))
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(0.),
                               np.real(np.vdot(w, derivative)), atol=2e-12)
    errors = []
    for h in (1e-3, 5e-4, 2.5e-4):
        fd = (action(args[0] + h * direction, args[1])
              - action(args[0] - h * direction, args[1])) / (2 * h)
        errors.append(np.linalg.norm(np.asarray(fd) - derivative))
    assert errors[1] < 0.3 * errors[0] and errors[2] < 0.3 * errors[1]


def test_chebyshev_batch_float32_long_horizon_and_dynamic_range():
    plan = plan_chebyshev_action(horizon=30, real_halfwidth=0,
                                 imag_halfwidth=5, tolerance=1e-10, shift=-3)
    diagonal = jnp.array([-3 - 4j, -3 + 2j])
    vectors = jnp.array([[2 + 1j, -3j], [0j, 0j], [-0.5j, 0.7 + 2j]])
    result = jax.jit(jax.vmap(lambda v: exponential_action(lambda w: diagonal * w,
                                                          v, plan)))(vectors)
    expected = np.exp(30 * np.asarray(diagonal)) * vectors
    # Use the documented absolute input norm scale, not a near-zero output scale.
    assert np.all(np.linalg.norm(result.value - expected, axis=1)
                  <= np.asarray(result.truncation_bound) + 1e-15)
    assert np.all(result.valid)
    scalar_plan = plan_chebyshev_action(horizon=0.2, real_halfwidth=0,
                                        imag_halfwidth=2, tolerance=1e-6)
    scalar = jnp.asarray(1j, dtype=jnp.complex64)
    got = exponential_action(lambda v: scalar * v, jnp.asarray(2 + 3j, jnp.complex64),
                              scalar_plan)
    assert got.value.dtype == jnp.complex64
    np.testing.assert_allclose(got.value, np.exp(0.2j) * (2 + 3j), rtol=2e-6)
    zero_plan = plan_chebyshev_action(horizon=0, real_halfwidth=0, imag_halfwidth=2)
    huge = jnp.array([1e200 + 1e200j, -1e200j])
    identity = exponential_action(lambda v: 1j * v, huge, zero_plan)
    np.testing.assert_array_equal(identity.value, huge)
    assert identity.valid and identity.truncation_bound == 0
    tiny = plan_chebyshev_action(horizon=1e-200, real_halfwidth=0,
                                 imag_halfwidth=1e-200)
    assert 0 < tiny.error_bound < 1e-300
    shifted = plan_chebyshev_action(horizon=0.3, real_halfwidth=0,
                                    imag_halfwidth=2, shift=-0.2 + 4j,
                                    tolerance=1e-12)
    value = exponential_action(lambda v: (-0.2 + 5j) * v,
                               jnp.asarray(2 + 3j), shifted)
    np.testing.assert_allclose(value.value, np.exp(0.3 * (-0.2 + 5j)) * (2 + 3j),
                               atol=2e-12)
    zero_operator = exponential_action(jnp.zeros_like, vectors[0], scalar_plan)
    np.testing.assert_allclose(zero_operator.value, vectors[0], atol=1e-6)


def test_chebyshev_oscillatory_coefficients_and_long_time_independent_quadrature():
    """A converged real integral qualifies Bessel coefficients independently."""
    plan = plan_chebyshev_action(horizon=4, real_halfwidth=0,
                                 imag_halfwidth=20, tolerance=1e-12)
    argument = 4 * plan.focus
    indices = np.arange(plan.degree + 1)
    references = []
    for count in (256, 512):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        theta = (nodes + 1) * np.pi / 2
        bessel = np.cos(indices[:, None] * theta - argument * np.sin(theta)) @ weights / 2
        coefficients = 2 * 1j**indices * bessel
        coefficients[0] /= 2
        references.append(coefficients)
    np.testing.assert_allclose(references[0], references[1], atol=8e-14, rtol=0)
    np.testing.assert_allclose(plan.coefficients, references[1], atol=8e-14, rtol=0)
    diagonal = jnp.asarray(1j * np.linspace(-20, 20, 9))
    vector = jnp.asarray(np.arange(1, 10) + 1j * np.arange(9))
    result = exponential_action(lambda v: diagonal * v, vector, plan)
    np.testing.assert_allclose(result.value, np.exp(4 * np.asarray(diagonal)) * vector,
                               atol=5e-12, rtol=0)
    assert result.valid


@pytest.mark.parametrize("kwargs, message", [
    ({"horizon": -1}, "horizon"), ({"horizon": np.nan}, "horizon"),
    ({"real_halfwidth": -1}, "halfwidth"), ({"imag_halfwidth": 0}, "halfwidth"),
    ({"real_halfwidth": 3}, "halfwidth"), ({"imag_halfwidth": np.inf}, "halfwidth"),
    ({"tolerance": 0}, "tolerance"), ({"tolerance": 1}, "tolerance"),
    ({"shift": 1j * np.inf}, "shift"), ({"max_degree": 0}, "positive"),
    ({"max_degree": 1}, "exhausted"),
    ({"horizon": 1e300, "imag_halfwidth": 1e300}, "representable|exhausted"),
    ({"imag_halfwidth": 1e-200, "shift": 1000}, "coefficients"),
])
def test_chebyshev_planner_rejects_invalid_bounds_and_budget(kwargs, message):
    inputs = {"horizon": 1, "real_halfwidth": 0, "imag_halfwidth": 2}
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        plan_chebyshev_action(**inputs)


def test_chebyshev_contract_failures_and_mode_integration():
    plan = plan_chebyshev_action(horizon=0.4, real_halfwidth=0.5,
                                 imag_halfwidth=10, tolerance=1e-12)
    vector = jnp.ones(10, dtype=jnp.complex128)
    with pytest.raises(TypeError, match="complex vector"):
        exponential_action(lambda v: v, jnp.ones(2), plan)
    with pytest.raises(ValueError, match="fixed horizon"):
        exponential_action(lambda v: v, vector, plan, horizon=0.4)
    with pytest.raises(ValueError, match="shape and dtype"):
        exponential_action(lambda v: v[:1], vector, plan)
    with pytest.raises(ValueError, match="shape and dtype"):
        exponential_action(lambda v: v.astype(jnp.complex64), vector, plan)
    assert not exponential_action(lambda v: v * np.nan, vector, plan).valid
    diagonal = jnp.array([0.3 + 0.2j, 0.29 - 0.4j, -0.2 + 2j, -0.3 - 3j,
                          -0.4 + 4j, -0.5 - 5j, -0.6 + 6j, -0.7 - 7j,
                          -0.8 + 8j, -0.9 - 9j])
    # The shifted real extent is verified independently over all diagonal entries.
    mode_plan = plan_chebyshev_action(horizon=10, real_halfwidth=0.7,
                                      imag_halfwidth=10, shift=-0.3, tolerance=1e-12)
    result = exponential_eigenpairs(lambda v: diagonal * v, vector, horizon=10,
                                    action_plan=mode_plan, outer_krylov_dim=8,
                                    candidates=2, tol=1e-8)
    np.testing.assert_allclose(np.sort_complex(result.eigenvalues),
                               np.sort_complex(diagonal[:2]), rtol=1e-8, atol=1e-9)
    assert np.all(result.converged)
    assert result.operator_applications == 8 * mode_plan.degree + 2
    with pytest.raises(ValueError, match="fixed horizon"):
        exponential_eigenpairs(lambda v: diagonal * v, vector, horizon=5,
                                action_plan=mode_plan, outer_krylov_dim=4)
