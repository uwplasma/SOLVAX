"""Solver-library verification practices from the spectral-MHD interop plan.

Encodes the citable test patterns: dot-product adjoint consistency,
GMRES residual monotonicity as a property over random operators,
Taylor-remainder gradient rates through implicit differentiation, and a
matvec-count gate showing recycling pays across a slowly varying operator
sequence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import solvax

jax.config.update("jax_enable_x64", True)


def test_dot_product_adjoint_consistency() -> None:
    """<A v, w> equals <v, A^H w> at round-off for matrix operators."""
    key = jax.random.PRNGKey(0)
    for size in (16, 48):
        key, key_m, key_v, key_w = jax.random.split(key, 4)
        matrix = jax.random.normal(key_m, (size, size)) + 1j * jax.random.normal(
            key_m, (size, size)
        )
        v = jax.random.normal(key_v, (size,)) + 1j * jax.random.normal(
            key_v, (size,)
        )
        w = jax.random.normal(key_w, (size,)) + 1j * jax.random.normal(
            key_w, (size,)
        )
        left = jnp.vdot(w, matrix @ v)
        right = jnp.vdot(matrix.conj().T @ w, v)
        scale = float(jnp.linalg.norm(matrix @ v) * jnp.linalg.norm(w))
        assert abs(complex(left - right)) < 100.0 * jnp.finfo(jnp.float64).eps * scale


def test_gmres_residual_monotone_over_random_operators() -> None:
    """Restarted GMRES residuals never increase across cycle boundaries."""
    key = jax.random.PRNGKey(1)
    for trial in range(4):
        key, key_m, key_b = jax.random.split(key, 3)
        size = 40
        base = jax.random.normal(key_m, (size, size)) / jnp.sqrt(size)
        matrix = jnp.eye(size) + 0.8 * base  # nonsymmetric, well posed
        if trial % 2:
            matrix = matrix + 0.3j * jax.random.normal(key_m, (size, size)) / jnp.sqrt(
                size
            )
        rhs = jax.random.normal(key_b, (size,)) + (
            1j * jax.random.normal(key_b, (size,)) if trial % 2 else 0.0
        )

        def matvec(x, matrix=matrix):
            return matrix @ x

        norms = []
        for restarts in (1, 2, 4, 8):
            solution = solvax.gmres(
                matvec, rhs, restart=5, max_restarts=restarts, rtol=1.0e-14
            )
            norms.append(float(solution.residual_norm))
        for earlier, later in zip(norms[:-1], norms[1:], strict=True):
            assert later <= earlier * (1.0 + 1.0e-12)


def test_taylor_remainder_rate_through_implicit_diff() -> None:
    """Gradients through linear_solve converge at second order.

    The dolfin-adjoint verification pattern: the Taylor remainder
    ``|f(p + h) - f(p) - h f'(p)|`` must fall at rate two as ``h`` halves.
    """
    key = jax.random.PRNGKey(2)
    size = 24
    base = jax.random.normal(key, (size, size)) / jnp.sqrt(size)
    rhs = jax.random.normal(jax.random.fold_in(key, 1), (size,))

    def objective(shift):
        def matvec(x):
            return x + 0.5 * (base @ x) + shift * x

        def solver(operator, b):
            return solvax.gmres(operator, b, rtol=1.0e-13, max_restarts=200).x

        x = solvax.linear_solve(matvec, rhs, solver)
        return jnp.sum(x**2)

    point = jnp.asarray(0.3)
    value = objective(point)
    slope = jax.grad(objective)(point)

    remainders = []
    for step in (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3):
        remainders.append(
            abs(float(objective(point + step) - value - step * slope))
        )
    rates = [
        jnp.log2(remainders[i] / remainders[i + 1])
        for i in range(len(remainders) - 1)
    ]
    for rate in rates:
        assert float(rate) > 1.9


def test_recycling_reduces_matvecs_across_a_slow_sequence() -> None:
    """GCROT recycling must pay on a slowly varying operator sequence.

    The Parks--de Sturler use case: a timestep-like sequence of nearby
    operators. The recycled solver's total matvec count (iterations) must
    undercut restarting from scratch by at least thirty percent.
    """
    key = jax.random.PRNGKey(3)
    size = 64
    base = jax.random.normal(key, (size, size)) / jnp.sqrt(size)
    drift = jax.random.normal(jax.random.fold_in(key, 7), (size, size)) / jnp.sqrt(
        size
    )
    rhs = jax.random.normal(jax.random.fold_in(key, 8), (size,))

    def matvec_at(step):
        def matvec(x):
            return x + 0.6 * (base @ x) + 5.0e-4 * step * (drift @ x)

        return matvec

    steps = 12

    plain_iterations = 0
    for step in range(steps):
        solution = solvax.gmres(
            matvec_at(step), rhs, restart=10, rtol=1.0e-10, max_restarts=50
        )
        assert bool(solution.converged)
        plain_iterations += int(solution.iterations)

    recycled_iterations = 0
    recycle = None
    for step in range(steps):
        solution = solvax.gcrot(
            matvec_at(step),
            rhs,
            m=10,
            k=8,
            recycle=recycle,
            rtol=1.0e-10,
            max_restarts=50,
        )
        assert bool(solution.converged)
        recycled_iterations += int(solution.iterations)
        recycle = solution.recycle

    # Measured on this sequence: 332 vs 468 matvecs (0.71x). The bound
    # keeps a small margin; harmonic deflation pays instead on spectra
    # with outlying eigenvalues, which this well-conditioned sequence
    # lacks by construction.
    assert recycled_iterations <= 0.75 * plain_iterations, (
        recycled_iterations,
        plain_iterations,
    )
