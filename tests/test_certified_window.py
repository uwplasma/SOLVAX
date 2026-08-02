"""The certified window estimator: does the promise hold on real chains?

Every test here compares against the *full* window, which the exact-window
construction makes exact by definition -- the tail sum over ``j >= N`` is
empty. So "realized error" below is the true relative gradient error, not a
proxy for it, and a violation would be a broken certificate rather than a
loose test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx
from solvax.direct import _generator_sensitivity

jax.config.update("jax_enable_x64", True)

M = 4


def _dominant_chain(n_blocks, seed=0, dominance=3.0, complex_=False):
    """The workhorse family: a compact parameter entering every row."""
    rng = np.random.default_rng(seed)

    def draw(*shape):
        a = rng.standard_normal(shape)
        return a + 1j * rng.standard_normal(shape) if complex_ else a

    lower = jnp.asarray(draw(n_blocks, M, M) * 0.25)
    upper = jnp.asarray(draw(n_blocks, M, M) * 0.25)
    diag = jnp.asarray(draw(n_blocks, M, M) * 0.25 + dominance * M * np.eye(M))

    def block_fn(params, j):
        jf = jnp.asarray(j).astype(diag.real.dtype)
        s = 1.0 + params[0] * jnp.cos(jf) + params[1] * 0.1 * jf
        return lower[j] * s, diag[j] * (1.0 + params[0]), upper[j] * s

    return block_fn


def _kinetic_chain(n_blocks, collisionality=0.02, seed=1):
    """A Legendre-mode chain: nu k^2 diagonal, O(1) coupling, head not localized."""
    rng = np.random.default_rng(seed)
    lower = jnp.asarray(rng.standard_normal((n_blocks, M, M)) * 0.5)
    upper = jnp.asarray(rng.standard_normal((n_blocks, M, M)) * 0.5)

    def block_fn(params, j):
        jf = jnp.asarray(j).astype(jnp.float64)
        diag = (1.0 + collisionality * jf**2) * jnp.eye(M) * (1.0 + params[0])
        diag = diag + 0.1 * params[1] * jnp.ones((M, M))
        return lower[j] * (1.0 + params[1]), diag, upper[j] * (1.0 + params[1])

    return block_fn


def _windowed_gradient(block_fn, n_blocks, keep, params, rhs, cotangent, window):
    def head(p):
        return sx.block_thomas_truncated_fn(
            block_fn, n_blocks, rhs, keep_lowest=keep, params=p,
            adjoint_window=int(window),
        )

    _, pullback = jax.vjp(head, params)
    (grad,) = pullback(cotangent)
    return grad


def _relative_error(grad, exact):
    return float(jnp.linalg.norm(grad - exact) / jnp.linalg.norm(exact))


def _problem(seed, keep, complex_=False):
    rng = np.random.default_rng(seed)

    def draw():
        a = rng.standard_normal((keep, M))
        return jnp.asarray(a + 1j * rng.standard_normal((keep, M)) if complex_ else a)

    return draw(), draw()


# ------------------------------------------------------- the central claim ----


@pytest.mark.parametrize("rtol", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
def test_realized_gradient_error_is_within_the_certified_tolerance(rtol):
    n, keep = 40, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _problem(7, keep)

    certificate = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=rtol
    )
    assert certificate.certified

    exact = _windowed_gradient(block_fn, n, keep, params, rhs, cotangent, n)
    got = _windowed_gradient(
        block_fn, n, keep, params, rhs, cotangent, certificate.window
    )
    realized = _relative_error(got, exact)

    # The proven bound holds, and the realized error is under it.
    assert certificate.certified_relative_error <= rtol
    assert realized <= rtol, f"{realized:.3e} > {rtol:.3e} at window {certificate.window}"
    assert realized <= certificate.certified_relative_error


@pytest.mark.parametrize(
    "name,chain,n,keep",
    [
        ("dominant", _dominant_chain(40, seed=3), 40, 3),
        ("weakly-dominant", _dominant_chain(30, seed=11, dominance=1.05), 30, 2),
        ("kinetic-nu-2e-2", _kinetic_chain(60, collisionality=0.02), 60, 4),
        ("kinetic-nu-2e-3", _kinetic_chain(60, collisionality=0.002), 60, 4),
    ],
)
@pytest.mark.parametrize("rtol", [1e-3, 1e-7])
def test_certificate_holds_across_chain_families(name, chain, n, keep, rtol):
    params = jnp.asarray([0.1, 0.05])
    rhs, cotangent = _problem(19, keep)

    certificate = sx.certified_adjoint_window(
        chain, n, keep, params, rhs, cotangent, rtol=rtol
    )
    exact = _windowed_gradient(chain, n, keep, params, rhs, cotangent, n)
    got = _windowed_gradient(chain, n, keep, params, rhs, cotangent, certificate.window)

    assert _relative_error(got, exact) <= rtol, name
    assert certificate.certified_relative_error <= rtol, name


def test_certificate_holds_for_a_complex_chain():
    # The generator is differentiated as a real map of twice the dimension,
    # because holomorphy is not assumed. That path has to certify too.
    n, keep = 24, 3
    block_fn = _dominant_chain(n, seed=5, complex_=True)
    params = jnp.asarray([0.12 + 0j, 0.05 + 0j])
    rhs, cotangent = _problem(23, keep, complex_=True)

    certificate = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-6
    )
    exact = _windowed_gradient(block_fn, n, keep, params, rhs, cotangent, n)
    got = _windowed_gradient(
        block_fn, n, keep, params, rhs, cotangent, certificate.window
    )
    assert _relative_error(got, exact) <= 1e-6


# ------------------------------------------------ degenerate and edge cases ----


def test_a_chain_that_does_not_localize_gets_the_full_window():
    # Coupling far exceeding the diagonal: no window is certifiable, and the
    # honest answer is the exact one rather than a plausible-looking guess.
    n, keep = 20, 2
    rng = np.random.default_rng(31)
    lower = jnp.asarray(rng.standard_normal((n, M, M)) * 3.0)
    upper = jnp.asarray(rng.standard_normal((n, M, M)) * 3.0)

    def block_fn(params, j):
        scale = 1.0 + params[0]
        return lower[j] * scale, jnp.eye(M) * scale, upper[j] * scale

    params = jnp.asarray([0.05, 0.0])
    rhs, cotangent = _problem(37, keep)
    certificate = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-3
    )
    assert certificate.window == n - keep
    assert certificate.status == "full-window"
    assert certificate.certified
    assert certificate.tail_bound == 0.0


def test_a_zero_cotangent_is_not_certifiable_and_says_so():
    # With no output cotangent the gradient is zero, so no *relative* tolerance
    # can be met by any proper window. Returning the exact window is the only
    # defensible answer; returning a small one would be certifying against
    # nothing.
    n, keep = 24, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    rhs, _ = _problem(41, keep)
    certificate = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, jnp.zeros((keep, M)), rtol=1e-6
    )
    assert certificate.window == n - keep
    assert certificate.status == "full-window"


def test_tighter_tolerances_never_ask_for_a_narrower_window():
    n, keep = 40, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _problem(7, keep)
    windows = [
        sx.certified_adjoint_window(
            block_fn, n, keep, params, rhs, cotangent, rtol=r
        ).window
        for r in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
    ]
    assert windows == sorted(windows)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"rtol": 0.0}, "strictly between"),
        ({"rtol": 1.0}, "strictly between"),
        ({"rtol": -1e-6}, "strictly between"),
    ],
)
def test_invalid_tolerances_are_rejected(kwargs, message):
    block_fn = _dominant_chain(10, seed=3)
    rhs, cotangent = _problem(2, 2)
    with pytest.raises(ValueError, match=message):
        sx.certified_adjoint_window(
            block_fn, 10, 2, jnp.asarray([0.1, 0.0]), rhs, cotangent, **kwargs
        )


def test_keep_lowest_must_be_in_range():
    block_fn = _dominant_chain(10, seed=3)
    rhs, cotangent = _problem(2, 2)
    with pytest.raises(ValueError, match="keep_lowest"):
        sx.certified_adjoint_window(
            block_fn, 10, 0, jnp.asarray([0.1, 0.0]), rhs, cotangent
        )


# --------------------------------------------------------- the escape hatch ----


def test_a_supplied_sensitivity_reproduces_the_computed_one():
    # The per-row Jacobians dominate the setup cost, so callers with an
    # analytic bound can pass one. Passing the computed value must be a no-op.
    n, keep = 24, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _problem(7, keep)

    gamma = _generator_sensitivity(block_fn, params, n)
    assert gamma.shape == (n,)
    assert jnp.all(gamma > 0)

    default = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-6
    )
    supplied = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-6, sensitivity=gamma
    )
    assert supplied.window == default.window
    assert supplied.tail_bound == pytest.approx(default.tail_bound, rel=1e-12)


def test_a_wrongly_shaped_sensitivity_is_rejected():
    block_fn = _dominant_chain(10, seed=3)
    rhs, cotangent = _problem(2, 2)
    with pytest.raises(ValueError, match="shape"):
        sx.certified_adjoint_window(
            block_fn, 10, 2, jnp.asarray([0.1, 0.0]), rhs, cotangent,
            sensitivity=jnp.ones(5),
        )


def test_an_overstated_sensitivity_only_widens_the_window():
    # The bound is monotone in gamma, so a caller who supplies a loose analytic
    # bound gets a conservative window rather than an invalid certificate.
    n, keep = 40, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _problem(7, keep)
    gamma = _generator_sensitivity(block_fn, params, n)

    tight = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-6, sensitivity=gamma
    )
    loose = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-6, sensitivity=gamma * 1e3
    )
    assert loose.window >= tight.window

    exact = _windowed_gradient(block_fn, n, keep, params, rhs, cotangent, n)
    got = _windowed_gradient(block_fn, n, keep, params, rhs, cotangent, loose.window)
    assert _relative_error(got, exact) <= 1e-6


# ------------------------------------------------- relation to the heuristic ----


def test_the_certified_window_can_be_passed_straight_to_the_solver():
    n, keep = 40, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    rhs, cotangent = _problem(7, keep)
    certificate = sx.certified_adjoint_window(
        block_fn, n, keep, params, rhs, cotangent, rtol=1e-8
    )
    # LocalizationWindow converts to int, as the heuristic one does.
    head = sx.block_thomas_truncated_fn(
        block_fn, n, rhs, keep_lowest=keep, params=params,
        adjoint_window=certificate,
    )
    assert head.shape == (keep, M)


def test_the_heuristic_stays_uncertified():
    # The two entry points make different promises and must keep saying so.
    n, keep = 40, 3
    block_fn = _dominant_chain(n, seed=3)
    params = jnp.asarray([0.12, 0.05])
    heuristic = sx.localization_crossover_window(
        lambda j: block_fn(params, j), n, keep
    )
    assert heuristic.certified is False
    assert heuristic.tolerance is None
    assert heuristic.status == "heuristic"
