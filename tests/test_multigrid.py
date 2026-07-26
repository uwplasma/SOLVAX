"""End-to-end geometric multigrid: transfers, smoothers and cycles together.

The evidence pinned here is the property that makes multigrid worth building
at all: a convergence rate per cycle that does not degrade as the grid is
refined, so the cost of a solve stays proportional to the number of unknowns.
The hierarchies are *semicoarsened* (only the strongly coupled axis is
coarsened) and their coarse operators are *rediscretized* rather than formed
as Galerkin products.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    dense_coarse_solve,
    gmres,
    jacobi_smoother,
    multigrid,
    semicoarsening_hierarchy,
    tridiagonal_smoother,
    upwind_smoother,
)

jax.config.update("jax_enable_x64", True)


def advection_diffusion(shape, *, diffusion, velocity=None, reaction=0.0):
    """Periodic ``sigma u - eps grad^2 u + c.grad u`` with first-order upwinding."""
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


def build_cycle(shape, *, diffusion, speed=0.0, reaction=1.0, smoother="line", **options):
    """Semicoarsened hierarchy over axis 0 with a rediscretized coarse operator."""
    ny = shape[1]

    def level(grid_shape):
        wind = jnp.full(grid_shape, speed)
        matvec, bands = advection_diffusion(
            grid_shape, diffusion=diffusion, velocity=(wind, 0.0), reaction=reaction
        )
        if smoother == "point":
            sweep = jacobi_smoother(bands[0][1])
        elif smoother == "line":
            sweep = tridiagonal_smoother(*bands[0], axis=0, periodic=True)
        else:  # an ordered sweep, either following the wind or ignoring it
            sweep = upwind_smoother(
                wind, *bands[0], axis=0, order=smoother, periodic=True
            )
        return matvec, sweep

    hierarchy = semicoarsening_hierarchy(
        shape, (True, False), level, levels=4, boundary="periodic", min_size=4
    )
    coarse_matvec, _ = level(hierarchy.shapes[-1])
    cycle = multigrid(
        hierarchy.levels, dense_coarse_solve(coarse_matvec, hierarchy.shapes[-1]), **options
    )
    assert all(grid[1] == ny for grid in hierarchy.shapes)  # semicoarsening
    return hierarchy.levels[0].matvec, cycle, hierarchy


def cycle_rate(matvec, cycle, shape, *, cycles=6, seed=0):
    """Average residual reduction factor per cycle."""
    rhs = jnp.asarray(np.random.default_rng(seed).standard_normal(shape))
    step = jax.jit(lambda x: x + cycle(rhs - matvec(x)))
    x = jnp.zeros(shape)
    initial = float(jnp.linalg.norm(rhs))
    for _ in range(cycles):
        x = step(x)
    final = float(jnp.linalg.norm(rhs - matvec(x)))
    return (final / initial) ** (1.0 / cycles)


def test_semicoarsening_hierarchy_rediscretizes_every_fine_level():
    seen = []

    def level(grid_shape):
        seen.append(grid_shape)
        matvec, bands = advection_diffusion(grid_shape, diffusion=(1.0, 1.0), reaction=1.0)
        return matvec, bands[0][1]

    hierarchy = semicoarsening_hierarchy(
        (32, 8), (True, False), level, levels=3, boundary="periodic", min_size=4
    )
    assert hierarchy.shapes == ((32, 8), (16, 8), (8, 8), (4, 8))
    # The operator is rebuilt on each fine grid — never restricted from the
    # fine one — and the coarsest grid is left to the caller's exact solve.
    assert seen == [(32, 8), (16, 8), (8, 8)]
    assert len(hierarchy.levels) == 3
    for level_data, grid in zip(hierarchy.levels, hierarchy.shapes, strict=False):
        assert level_data.restrict(jnp.ones(grid)).shape[0] == grid[0] // 2


@pytest.mark.parametrize("nx", (32, 64, 128))
def test_v_cycle_rate_is_grid_independent_for_anisotropic_diffusion(nx):
    """Line relaxation along the semicoarsened axis: the rate is flat in h."""
    shape = (nx, 16)
    matvec, cycle, _ = build_cycle(shape, diffusion=(1.0, 0.01))
    rate = cycle_rate(matvec, cycle, shape)
    assert rate < 0.05, rate


def test_v_cycle_rate_does_not_degrade_over_an_eightfold_refinement():
    rates = []
    for nx in (32, 64, 128, 256):
        shape = (nx, 16)
        matvec, cycle, _ = build_cycle(shape, diffusion=(1.0, 0.01))
        rates.append(cycle_rate(matvec, cycle, shape))
    # h-independence: the rate on the finest grid is within a small constant
    # factor of the coarsest, rather than approaching one.
    assert max(rates) < 2.0 * min(rates), rates
    assert max(rates) < 0.05, rates


def test_multigrid_preconditioned_gmres_is_bounded_while_plain_gmres_is_not():
    """The cycle is used directly on the two-dimensional state, with no
    ravel/unravel around the operator."""
    shape = (64, 16)
    matvec, cycle, _ = build_cycle(shape, diffusion=(1.0, 0.01))
    rhs = jnp.asarray(np.random.default_rng(1).standard_normal(shape))

    options = dict(restart=30, rtol=1e-10, max_restarts=20)
    preconditioned = gmres(matvec, rhs, precond=cycle, **options)
    plain = gmres(matvec, rhs, **options)

    assert bool(preconditioned.converged)
    assert preconditioned.x.shape == shape
    assert int(preconditioned.iterations) <= 5
    assert not bool(plain.converged)
    assert int(plain.iterations) > 50 * int(preconditioned.iterations)


def test_streaming_operator_needs_the_upwind_ordering():
    """At high mesh Peclet number the same cycle with a wind-agnostic sweep
    ordering is dramatically worse."""
    shape = (64, 16)
    speed = -1.0e3  # against the increasing-index ordering
    settings = dict(diffusion=(1.0e-2, 1.0e-2), speed=speed)
    matvec, upwind_cycle, _ = build_cycle(shape, smoother="upwind", **settings)
    _, naive_cycle, _ = build_cycle(shape, smoother="forward", **settings)

    upwind_rate = cycle_rate(matvec, upwind_cycle, shape)
    naive_rate = cycle_rate(matvec, naive_cycle, shape)
    assert upwind_rate < 0.1, upwind_rate
    assert naive_rate > 5.0 * upwind_rate, (upwind_rate, naive_rate)


@pytest.mark.parametrize("nx", (64, 128))
def test_upwind_cycle_is_grid_independent_for_a_streaming_operator(nx):
    shape = (nx, 16)
    matvec, cycle, _ = build_cycle(
        shape, diffusion=(1.0e-3, 1.0e-3), speed=-1.0, smoother="upwind"
    )
    assert cycle_rate(matvec, cycle, shape) < 0.1


def test_w_and_f_cycles_converge_at_least_as_well_as_the_v_cycle():
    shape = (64, 16)
    rates = {}
    for cycle_shape in ("v", "f", "w"):
        matvec, cycle, _ = build_cycle(shape, diffusion=(1.0, 0.01), cycle=cycle_shape)
        rates[cycle_shape] = cycle_rate(matvec, cycle, shape)
    assert rates["f"] <= rates["v"] * 1.05
    assert rates["w"] <= rates["v"] * 1.05


def test_point_relaxation_is_a_weaker_smoother_than_line_relaxation():
    shape = (64, 16)
    matvec, line_cycle, _ = build_cycle(shape, diffusion=(1.0, 0.01))
    _, point_cycle, _ = build_cycle(shape, diffusion=(1.0, 0.01), smoother="point")
    assert cycle_rate(matvec, line_cycle, shape) < cycle_rate(matvec, point_cycle, shape)


def test_multigrid_cycle_is_jit_and_gradient_transparent():
    shape = (32, 8)
    matvec, cycle, _ = build_cycle(shape, diffusion=(1.0, 0.05))
    rhs = jnp.asarray(np.random.default_rng(2).standard_normal(shape))
    np.testing.assert_allclose(jax.jit(cycle)(rhs), cycle(rhs), rtol=1e-12)

    objective = lambda scale: jnp.sum(cycle(scale * rhs) ** 2)  # noqa: E731
    value, gradient = jax.value_and_grad(objective)(1.0)
    assert float(gradient) == pytest.approx(2.0 * float(value), rel=1e-10)
