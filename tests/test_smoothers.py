"""Tests for solvax.smoothers: measured smoothing factors of each relaxation.

Every quantitative assertion here is a *measurement* — the smoothing factor
returned by ``smoothing_factor``, i.e. the asymptotic reduction of the error
components the coarsening cannot represent — checked against the local
Fourier analysis value of the model operator where one exists.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    alternating_smoother,
    block_jacobi_smoother,
    high_frequency_mask,
    jacobi_smoother,
    plane_smoother,
    relaxation,
    smoothing_factor,
    tridiagonal_smoother,
    upwind_smoother,
)

jax.config.update("jax_enable_x64", True)

# The power iteration approaches the true rate from below; these counts are
# enough for three-digit agreement with the analytic factors below.
STEPS = 120
KEY = jax.random.PRNGKey(0)


def advection_diffusion(shape, *, diffusion, velocity=None, reaction=0.0):
    """Periodic ``sigma u - eps grad^2 u + c.grad u`` with first-order upwinding.

    Returns ``(matvec, bands)`` where ``bands[axis]`` is the
    ``(lower, diagonal, upper)`` triple of that axis' lines; the diagonal is
    the *full* operator diagonal, as line relaxation requires.
    """
    ndim = len(shape)
    velocity = (0.0,) * ndim if velocity is None else velocity
    diagonal = jnp.full(shape, float(reaction))
    lowers, uppers = [], []
    for axis in range(ndim):
        step = 1.0 / shape[axis]
        eps = float(diffusion[axis])
        wind = jnp.broadcast_to(jnp.asarray(velocity[axis], float), shape)
        diagonal = diagonal + 2.0 * eps / step**2 + jnp.abs(wind) / step
        lowers.append(-eps / step**2 - jnp.maximum(wind, 0.0) / step)
        uppers.append(-eps / step**2 - jnp.maximum(-wind, 0.0) / step)

    def matvec(u):
        out = diagonal * u
        for axis in range(ndim):
            out = out + lowers[axis] * jnp.roll(u, 1, axis)
            out = out + uppers[axis] * jnp.roll(u, -1, axis)
        return out

    return matvec, tuple((lowers[a], diagonal, uppers[a]) for a in range(ndim))


def factor(smooth, matvec, shape, coarsen=None, steps=STEPS):
    return float(
        smoothing_factor(smooth, matvec, shape, key=KEY, coarsen=coarsen, steps=steps)
    )


def test_damped_jacobi_matches_its_local_fourier_analysis_factor():
    """Damped Jacobi on the 2-D Laplacian: mu = 2/3, and mu = 1 undamped."""
    shape = (32, 32)
    matvec, bands = advection_diffusion(shape, diffusion=(1.0, 1.0))
    diagonal = bands[0][1]

    damped = factor(jacobi_smoother(diagonal), matvec, shape)
    assert damped == pytest.approx(2.0 / 3.0, abs=0.02)

    # omega = 1 leaves the checkerboard mode with amplification -1: Jacobi
    # without damping is not a smoother at all.
    undamped = factor(jacobi_smoother(diagonal, omega=1.0), matvec, shape)
    assert undamped > 0.97

    # The 2-D optimum omega = 4/5 gives 3/5.
    optimal = factor(jacobi_smoother(diagonal, omega=0.8), matvec, shape)
    assert optimal == pytest.approx(0.6, abs=0.02)
    assert optimal < damped


def test_anisotropy_needs_semicoarsening_and_a_line_smoother():
    """Full coarsening leaves modes no smoother can damp; semicoarsening the
    strongly coupled axis makes line relaxation almost exact."""
    shape = (32, 32)
    strong, weak = 1.0, 0.01
    matvec, bands = advection_diffusion(shape, diffusion=(strong, weak))
    diagonal = bands[0][1]
    line_strong = tridiagonal_smoother(*bands[0], axis=0, periodic=True)
    line_weak = tridiagonal_smoother(*bands[1], axis=1, periodic=True)

    # Standard (full) coarsening: every smoother stalls, because the modes
    # oscillating along the weakly coupled axis are cheap for the operator.
    assert factor(jacobi_smoother(diagonal), matvec, shape) > 0.95
    assert factor(line_strong, matvec, shape) > 0.95
    assert factor(alternating_smoother([line_strong, line_weak]), matvec, shape) > 0.9

    # Semicoarsening the strong axis only: those modes are now the coarse
    # grid's job, and what is left is exactly what line relaxation removes.
    semi = (True, False)
    jacobi_factor = factor(jacobi_smoother(diagonal), matvec, shape, semi)
    line_factor = factor(line_strong, matvec, shape, semi)
    assert jacobi_factor == pytest.approx(1.0 / 3.0, abs=0.02)
    # Local Fourier analysis of line relaxation along the strong axis gives
    # mu = weak / (strong + weak) for this operator.
    assert line_factor == pytest.approx(weak / (strong + weak), rel=0.05)
    assert line_factor < jacobi_factor / 10.0


def test_alternating_line_relaxation_improves_on_a_single_direction():
    shape = (32, 32)
    matvec, bands = advection_diffusion(shape, diffusion=(1.0, 0.05))
    semi = (True, False)
    line_strong = tridiagonal_smoother(*bands[0], axis=0, periodic=True)
    line_weak = tridiagonal_smoother(*bands[1], axis=1, periodic=True)
    both = alternating_smoother([line_strong, line_weak])
    assert factor(both, matvec, shape, semi) < factor(line_strong, matvec, shape, semi)


@pytest.mark.parametrize("peclet", (1.0, 10.0, 100.0))
def test_upwind_ordering_beats_every_fixed_ordering(peclet):
    """Ordering the sweep with the flow is what makes relaxation work for a
    streaming operator; a fixed ordering cannot follow a sign-varying wind."""
    shape = (32, 16)
    diffusion = 1.0
    speed = diffusion * peclet * shape[0]  # mesh Peclet = |c| h / eps
    wind = jnp.where(jnp.arange(shape[0])[:, None] < shape[0] // 2, speed, -speed)
    wind = jnp.broadcast_to(wind, shape)
    matvec, bands = advection_diffusion(
        shape, diffusion=(diffusion, diffusion), velocity=(wind, 0.0)
    )
    semi = (True, False)
    orderings = {
        name: factor(
            upwind_smoother(wind, *bands[0], axis=0, order=name, periodic=True),
            matvec,
            shape,
            semi,
        )
        for name in ("upwind", "downwind", "forward", "backward")
    }
    assert orderings["upwind"] < min(
        orderings[name] for name in ("downwind", "forward", "backward")
    ), orderings
    # The advantage is a *ratio* that grows with the mesh Peclet number.
    advantage = orderings["forward"] / orderings["upwind"]
    assert advantage > (1.2 if peclet <= 1.0 else 5.0), orderings


def test_upwind_advantage_grows_with_the_peclet_number():
    shape = (32, 16)
    diffusion = 1.0
    advantages = []
    for peclet in (1.0, 10.0, 100.0):
        speed = diffusion * peclet * shape[0]
        wind = jnp.full(shape, -speed)  # against the increasing-index ordering
        matvec, bands = advection_diffusion(
            shape, diffusion=(diffusion, diffusion), velocity=(wind, 0.0)
        )
        semi = (True, False)
        upwind = factor(
            upwind_smoother(wind, *bands[0], axis=0, periodic=True), matvec, shape, semi
        )
        naive = factor(
            upwind_smoother(wind, *bands[0], axis=0, order="forward", periodic=True),
            matvec,
            shape,
            semi,
        )
        advantages.append(naive / upwind)
    assert advantages == sorted(advantages), advantages
    assert advantages[-1] > 20.0 * advantages[0], advantages


def test_upwind_sweep_approaches_the_exact_line_solve_at_high_peclet():
    """One triangular sweep costs half a tridiagonal solve and, once the
    operator is advection dominated, smooths just as well."""
    shape = (32, 16)
    speed = 1.0 * 100.0 * shape[0]
    wind = jnp.full(shape, -speed)
    matvec, bands = advection_diffusion(shape, diffusion=(1.0, 1.0), velocity=(wind, 0.0))
    semi = (True, False)
    upwind = factor(
        upwind_smoother(wind, *bands[0], axis=0, periodic=True), matvec, shape, semi
    )
    exact_line = factor(
        tridiagonal_smoother(*bands[0], axis=0, periodic=True), matvec, shape, semi
    )
    assert upwind < 0.05
    assert upwind < 4.0 * exact_line  # same order, at half the arithmetic


def test_upwind_orderings_reduce_to_fixed_sweeps_for_a_one_signed_wind():
    shape = (16, 4)
    wind = jnp.full(shape, 3.0)
    _, bands = advection_diffusion(shape, diffusion=(1.0, 1.0), velocity=(wind, 0.0))
    rng = np.random.default_rng(0)
    residual = jnp.asarray(rng.standard_normal(shape))
    identity = lambda value: jnp.zeros_like(value)  # noqa: E731
    apply = lambda smooth: smooth(identity, jnp.zeros(shape), residual)  # noqa: E731
    positive = apply(upwind_smoother(wind, *bands[0], axis=0))
    forward = apply(upwind_smoother(wind, *bands[0], axis=0, order="forward"))
    backward = apply(upwind_smoother(wind, *bands[0], axis=0, order="backward"))
    downwind = apply(upwind_smoother(wind, *bands[0], axis=0, order="downwind"))
    np.testing.assert_allclose(positive, forward, atol=1e-14)
    np.testing.assert_allclose(downwind, backward, atol=1e-14)
    assert float(jnp.linalg.norm(positive - backward)) > 1e-6


def plane_operator(shape, diagonal, first, second, axes, periodic):
    """Dense-free five-point plane matvec used to check the plane solve."""

    def matvec(u):
        out = diagonal * u
        for (lower, upper), axis, wrap in zip(
            (first, second), axes, periodic, strict=True
        ):
            left = jnp.roll(u, 1, axis)
            right = jnp.roll(u, -1, axis)
            if not wrap:
                index = (slice(None),) * (axis % u.ndim)
                left = left.at[(*index, 0)].set(0.0)
                right = right.at[(*index, -1)].set(0.0)
            out = out + lower * left + upper * right
        return out

    return matvec


@pytest.mark.parametrize(
    "periodic", ((False, False), (True, False), (False, True), (True, True))
)
def test_plane_smoother_is_an_exact_plane_solve(periodic):
    shape = (6, 5, 2)
    rng = np.random.default_rng(3)
    bands = [-jnp.asarray(rng.uniform(0.1, 0.5, shape)) for _ in range(4)]
    diagonal = jnp.asarray(rng.uniform(4.0, 5.0, shape))
    first, second = (bands[0], bands[1]), (bands[2], bands[3])
    matvec = plane_operator(shape, diagonal, first, second, (0, 1), periodic)
    smooth = plane_smoother(diagonal, first, second, axes=(0, 1), periodic=periodic)

    rhs = jnp.asarray(rng.standard_normal(shape))
    solution = smooth(matvec, jnp.zeros(shape), rhs)
    residual = jnp.linalg.norm(matvec(solution) - rhs) / jnp.linalg.norm(rhs)
    assert float(residual) < 1e-13


def test_plane_smoother_handles_arbitrary_axis_placement():
    shape = (3, 6, 5)
    rng = np.random.default_rng(4)
    bands = [-jnp.asarray(rng.uniform(0.1, 0.5, shape)) for _ in range(4)]
    diagonal = jnp.asarray(rng.uniform(4.0, 5.0, shape))
    first, second = (bands[0], bands[1]), (bands[2], bands[3])
    periodic = (True, False)
    matvec = plane_operator(shape, diagonal, first, second, (2, 1), periodic)
    smooth = plane_smoother(diagonal, first, second, axes=(2, 1), periodic=periodic)
    rhs = jnp.asarray(rng.standard_normal(shape))
    solution = smooth(matvec, jnp.zeros(shape), rhs)
    assert float(jnp.linalg.norm(matvec(solution) - rhs) / jnp.linalg.norm(rhs)) < 1e-13


def test_plane_relaxation_matches_its_local_fourier_analysis_factor():
    """With both plane axes semicoarsened, only the out-of-plane coupling
    survives: mu = weak / (strong + weak)."""
    shape = (16, 16, 16)  # equal spacing, so the analytic factor is coefficient-only
    strong, weak = 1.0, 0.02
    matvec, bands = advection_diffusion(shape, diffusion=(strong, strong, weak))
    diagonal = bands[0][1]
    smooth = plane_smoother(
        diagonal,
        (bands[0][0], bands[0][2]),
        (bands[1][0], bands[1][2]),
        axes=(0, 1),
        periodic=(True, True),
    )
    semi = (True, True, False)
    plane_factor = factor(smooth, matvec, shape, semi)
    assert plane_factor == pytest.approx(weak / (strong + weak), rel=0.1)
    assert plane_factor < factor(jacobi_smoother(diagonal), matvec, shape, semi) / 10.0


def test_block_jacobi_smoother_inverts_shared_and_per_point_blocks():
    shape = (4, 3)
    rng = np.random.default_rng(5)
    shared = jnp.asarray(np.eye(3) * 4.0 + rng.standard_normal((3, 3)) * 0.2)
    matvec = lambda u: u @ shared.T  # noqa: E731
    rhs = jnp.asarray(rng.standard_normal(shape))
    smooth = block_jacobi_smoother(shared, axis=-1)
    solution = smooth(matvec, jnp.zeros(shape), rhs)
    np.testing.assert_allclose(matvec(solution), rhs, atol=1e-12)

    blocks = jnp.asarray(
        np.stack([np.eye(3) * (4.0 + k) + rng.standard_normal((3, 3)) * 0.1
                  for k in range(4)])
    )
    per_point = lambda u: jnp.einsum("nij,nj->ni", blocks, u)  # noqa: E731
    smooth = block_jacobi_smoother(blocks, axis=1)
    solution = smooth(per_point, jnp.zeros(shape), rhs)
    np.testing.assert_allclose(per_point(solution), rhs, atol=1e-12)


def test_block_jacobi_smoother_supports_an_interior_block_axis():
    shape = (2, 3, 4)  # blocks span axis 1, batched over axes 0 and 2
    rng = np.random.default_rng(6)
    blocks = jnp.asarray(
        np.stack([np.eye(3) * 5.0 + rng.standard_normal((3, 3)) * 0.1
                  for _ in range(8)]).reshape(2, 4, 3, 3)
    )
    matvec = lambda u: jnp.einsum(  # noqa: E731
        "abij,ajb->aib", blocks, u
    )
    rhs = jnp.asarray(rng.standard_normal(shape))
    smooth = block_jacobi_smoother(blocks, axis=1)
    solution = smooth(matvec, jnp.zeros(shape), rhs)
    np.testing.assert_allclose(matvec(solution), rhs, atol=1e-12)


@pytest.mark.parametrize("periodic", (False, True))
def test_tridiagonal_smoother_matches_a_dense_line_solve(periodic):
    shape = (5, 3)
    rng = np.random.default_rng(7)
    lower = jnp.asarray(-rng.uniform(0.2, 0.6, shape))
    upper = jnp.asarray(-rng.uniform(0.2, 0.6, shape))
    diagonal = jnp.asarray(rng.uniform(3.0, 4.0, shape))
    smooth = tridiagonal_smoother(lower, diagonal, upper, axis=0, periodic=periodic)
    rhs = jnp.asarray(rng.standard_normal(shape))
    solution = smooth(lambda u: jnp.zeros_like(u), jnp.zeros(shape), rhs)

    for column in range(shape[1]):
        dense = np.diag(np.asarray(diagonal[:, column]))
        for row in range(shape[0] - 1):
            dense[row + 1, row] = lower[row + 1, column]
            dense[row, row + 1] = upper[row, column]
        if periodic:
            dense[0, -1] = lower[0, column]
            dense[-1, 0] = upper[-1, column]
        expected = np.linalg.solve(dense, np.asarray(rhs[:, column]))
        np.testing.assert_allclose(solution[:, column], expected, atol=1e-12)


def test_relaxation_applies_damping_and_repeated_sweeps():
    matvec = lambda u: 2.0 * u  # noqa: E731
    solve = lambda r: 0.5 * r  # noqa: E731
    rhs = jnp.asarray([1.0, -2.0])
    once = relaxation(solve, omega=0.5)(matvec, jnp.zeros(2), rhs)
    np.testing.assert_allclose(once, 0.25 * rhs, atol=1e-15)
    # Two sweeps of an exactly-inverting solve with omega = 1/2 leave the
    # error at (1/2)^2 of its initial value.
    twice = relaxation(solve, omega=0.5, sweeps=2)(matvec, jnp.zeros(2), rhs)
    np.testing.assert_allclose(twice, 0.375 * rhs, atol=1e-15)


def test_alternating_smoother_applies_its_components_in_order():
    order = []

    def record(tag):
        def smooth(matvec, x, b):
            order.append(tag)
            return x + tag * b

        return smooth

    composed = alternating_smoother([record(1.0), record(2.0)])
    result = composed(lambda u: u, jnp.zeros(1), jnp.ones(1))
    assert order == [1.0, 2.0]
    np.testing.assert_allclose(result, 3.0, atol=1e-15)


def test_high_frequency_mask_marks_the_unrepresentable_modes():
    mask = high_frequency_mask((8, 8))
    assert not bool(mask[0, 0])  # the constant is the coarse grid's job
    assert bool(mask[2, 0]) and bool(mask[0, 2])  # theta = pi/2 is the boundary
    assert not bool(mask[1, 1])
    assert bool(mask[4, 4])  # the checkerboard is always high frequency
    assert bool(mask[6, 0])  # folding is symmetric in k -> n - k

    # Semicoarsening: a mode oscillating only along an uncoarsened axis stays
    # the smoother's problem only if it is high along a coarsened one.
    semi = high_frequency_mask((8, 8), (True, False))
    assert bool(semi[2, 0])
    assert not bool(semi[0, 4])


def test_smoothers_are_jit_and_gradient_transparent():
    shape = (8, 4)
    rng = np.random.default_rng(8)
    rhs = jnp.asarray(rng.standard_normal(shape))

    def sweep(scale):
        matvec, bands = advection_diffusion(
            shape, diffusion=(1.0, 0.1), reaction=1.0
        )
        lower, diagonal, upper = bands[0]
        smooth = tridiagonal_smoother(
            scale * lower, diagonal, scale * upper, axis=0, periodic=True
        )
        return smooth(matvec, jnp.zeros(shape), rhs)

    np.testing.assert_allclose(jax.jit(sweep)(1.0), sweep(1.0), rtol=1e-12)
    objective = lambda scale: jnp.sum(sweep(scale) ** 2)  # noqa: E731
    gradient = float(jax.jit(jax.grad(objective))(1.0))
    step = 1e-6
    finite = float((objective(1.0 + step) - objective(1.0 - step)) / (2 * step))
    assert gradient == pytest.approx(finite, rel=1e-5)


def test_smoother_validation():
    ones = jnp.ones((4, 3))
    with pytest.raises(ValueError, match="sweeps must be"):
        relaxation(lambda r: r, sweeps=0)
    with pytest.raises(ValueError, match="blocks must have shape"):
        block_jacobi_smoother(jnp.ones((3, 4)))
    with pytest.raises(ValueError, match="share the iterate shape"):
        tridiagonal_smoother(jnp.ones((4, 2)), ones, ones)
    with pytest.raises(ValueError, match="unknown order"):
        upwind_smoother(ones, ones, ones, ones, order="sideways")
    with pytest.raises(ValueError, match="wind must share"):
        upwind_smoother(jnp.ones(4), ones, ones, ones)
    with pytest.raises(ValueError, match="two distinct grid axes"):
        plane_smoother(ones, (ones, ones), (ones, ones), axes=(0, 0))
    with pytest.raises(ValueError, match="every band must share"):
        plane_smoother(ones, (ones, jnp.ones(4)), (ones, ones), axes=(0, 1))
    with pytest.raises(ValueError, match="at least three points"):
        plane_smoother(
            jnp.ones((4, 2)),
            (jnp.ones((4, 2)),) * 2,
            (jnp.ones((4, 2)),) * 2,
            axes=(0, 1),
            periodic=(False, True),
        )
    with pytest.raises(ValueError, match="at least one sweep"):
        alternating_smoother([])
    with pytest.raises(ValueError, match="one entry per axis"):
        high_frequency_mask((4, 4), (True,))
    with pytest.raises(ValueError, match="at least one axis"):
        high_frequency_mask((4, 4), (False, False))
    with pytest.raises(ValueError, match="steps must be"):
        smoothing_factor(lambda m, x, b: x, lambda u: u, (4,), key=KEY, steps=0)
