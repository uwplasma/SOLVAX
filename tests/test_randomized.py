"""Tests for solvax.randomized: Nystrom preconditioning for SPD systems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    nystrom_preconditioner,
    nystrom_preconditioner_adaptive,
    pcg,
    pcg_linear_solve,
)

jax.config.update("jax_enable_x64", True)


def decay_spectrum_system(n=300, head=30, seed=0):
    """SPD operator with a decaying head and a flat tail: Nystrom's regime."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.concatenate([100.0 * 0.5 ** np.arange(head), 1e-2 * np.ones(n - head)])
    matrix = jnp.asarray((q * lam) @ q.T)
    rhs = jnp.asarray(rng.standard_normal(n))
    return matrix, rhs


def test_nystrom_cuts_pcg_iterations():
    matrix, rhs = decay_spectrum_system()
    n, mu = rhs.shape[0], 1e-2
    system = lambda v: matrix @ v + mu * v  # noqa: E731
    plain = pcg(system, rhs, rtol=1e-10, max_steps=1000)
    precond = nystrom_preconditioner(
        lambda v: matrix @ v, n, 50, jax.random.PRNGKey(0), mu=mu
    )
    accelerated = pcg(system, rhs, precond=precond, rtol=1e-10, max_steps=1000)
    assert bool(plain.converged) and bool(accelerated.converged)
    assert int(accelerated.iterations) <= int(plain.iterations) // 2


def test_nystrom_action_is_symmetric_positive_definite():
    matrix, _ = decay_spectrum_system(n=120, head=15, seed=1)
    precond = nystrom_preconditioner(
        lambda v: matrix @ v, 120, 25, jax.random.PRNGKey(1), mu=1e-3
    )
    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.standard_normal(120))
    v = jnp.asarray(rng.standard_normal(120))
    assert np.isclose(float(u @ precond(v)), float(v @ precond(u)), rtol=1e-12)
    assert float(v @ precond(v)) > 0.0


def test_nystrom_is_deterministic_under_fixed_key():
    matrix, rhs = decay_spectrum_system(n=100, head=10, seed=3)
    build = lambda: nystrom_preconditioner(  # noqa: E731
        lambda v: matrix @ v, 100, 20, jax.random.PRNGKey(7), mu=1e-3
    )
    assert np.allclose(np.asarray(build()(rhs)), np.asarray(build()(rhs)), atol=0.0)


def test_nystrom_gradient_matches_finite_differences():
    matrix, rhs = decay_spectrum_system(n=150, head=15, seed=4)
    n, mu = 150, 1e-2

    def loss(scale):
        precond = nystrom_preconditioner(
            lambda v: scale * (matrix @ v), n, 30, jax.random.PRNGKey(0), mu=mu
        )
        solution = pcg_linear_solve(
            lambda v: scale * (matrix @ v) + mu * v, rhs,
            precond=precond, rtol=1e-11, max_steps=600,
        )
        return jnp.sum(solution.x ** 2)

    gradient = jax.grad(loss)(1.0)
    eps = 1e-6
    finite = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    assert np.isclose(float(gradient), float(finite), rtol=1e-4)


def test_nystrom_rejects_bad_rank():
    with pytest.raises(ValueError, match="rank"):
        nystrom_preconditioner(lambda v: v, 10, 0, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="rank"):
        nystrom_preconditioner(lambda v: v, 10, 11, jax.random.PRNGKey(0))


@pytest.mark.parametrize(
    ("label", "mu"),
    [("zero-operator-mu0", 0.0), ("zero-operator-mu-positive", 1.0e-8)],
)
def test_nystrom_survives_a_zero_operator(label: str, mu: float) -> None:
    """The sketch-proportional shift vanishes when the sketch does.

    ``nu`` is proportional to ``||Y||``, so a zero operator gets no shift, the
    core Cholesky is singular, and the triangular solve divides by it. Every
    downstream value used to be NaN -- and a positive ``mu`` did not help,
    because the failure happens before ``mu`` is ever used.
    """
    n = 8
    operator = jnp.zeros((n, n))
    precond = nystrom_preconditioner(
        lambda v: operator @ v, n, 3, jax.random.PRNGKey(0), mu=mu
    )
    out = precond(jnp.ones(n))
    assert jnp.all(jnp.isfinite(out)), label
    # An operator with no range should leave the vector alone.
    assert jnp.allclose(out, jnp.ones(n), atol=1e-10)


def test_nystrom_rank_deficient_operator_has_no_zero_over_zero() -> None:
    """With ``mu = 0`` a null direction gives ``(0 + 0) / (0 + 0)``.

    The null-space limit is the identity on that direction, not an
    indeterminate: a direction the operator does not see should pass through.
    """
    n = 8
    factor = jax.random.normal(jax.random.PRNGKey(1), (n, 2))
    operator = factor @ factor.T          # psd, rank 2 < sketch rank 3
    precond = nystrom_preconditioner(
        lambda v: operator @ v, n, 3, jax.random.PRNGKey(0), mu=0.0
    )
    out = precond(jnp.ones(n))
    assert jnp.all(jnp.isfinite(out))


def test_nystrom_reports_how_much_spectrum_the_sketch_spans() -> None:
    """The posterior read on whether the rank was adequate.

    A sketch sitting on a flat plateau of the spectrum has a span near one and
    is preconditioning almost nothing; one that reaches into the decaying tail
    has a small span. Without this a caller cannot tell the two apart.
    """
    n = 64
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))

    def build(decay):
        spectrum = np.array([decay(i) for i in range(n)])
        return jnp.asarray(basis @ np.diag(spectrum) @ basis.T)

    fast = build(lambda i: 10.0 ** (-0.15 * i))
    slow = build(lambda i: 1.0 / (1.0 + 0.02 * i))
    key = jax.random.PRNGKey(0)

    fast_span = float(
        nystrom_preconditioner(lambda v: fast @ v, n, 32, key, mu=1e-8).spectrum_span
    )
    slow_span = float(
        nystrom_preconditioner(lambda v: slow @ v, n, 32, key, mu=1e-8).spectrum_span
    )
    assert fast_span < 1e-3, "a decaying spectrum should report a small span"
    assert slow_span > 0.1, "a flat spectrum should report a large span"


def test_nystrom_adaptive_stops_early_on_a_decaying_spectrum() -> None:
    """Growth must stop when the sketch is adequate, and cap when it is not."""
    n = 64
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))

    def build(decay):
        return jnp.asarray(
            basis @ np.diag(np.array([decay(i) for i in range(n)])) @ basis.T
        )

    key = jax.random.PRNGKey(0)
    _, fast_rank = nystrom_preconditioner_adaptive(
        lambda v: build(lambda i: 10.0 ** (-0.15 * i)) @ v, n, key, mu=1e-8
    )
    precond, slow_rank = nystrom_preconditioner_adaptive(
        lambda v: build(lambda i: 1.0 / (1.0 + 0.02 * i)) @ v, n, key, mu=1e-8
    )
    assert fast_rank < n, "a decaying spectrum should not need the full rank"
    assert slow_rank == n, "a flat spectrum should grow to the cap"
    # And it reports honestly rather than pretending the cap was enough.
    assert float(precond.spectrum_span) > 0.1
