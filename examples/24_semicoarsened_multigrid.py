"""Semicoarsened geometric multigrid for an anisotropic streaming operator.

The operator here is strongly coupled along the first axis (fast transport)
and weakly along the second, which is exactly the situation where standard
full coarsening fails: the modes oscillating along the weak axis are cheap for
the operator, so no local relaxation damps them, yet the coarse grid cannot
represent them either. The fix is to coarsen only the strong axis and to relax
along it — with the sweep ordered by the wind, so the error is chased
downstream instead of back in.

Everything is assembled from `solvax.transfer` (separable per-axis transfers),
`solvax.smoothers` (line relaxation ordered by the advection field) and
`solvax.precond.multigrid` (the cycle). Coarse operators are rediscretized on
each grid rather than formed as Galerkin products. The result is a
convergence rate per cycle that does not degrade under refinement.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

DIFFUSION = (1.0e-3, 1.0e-2)  # weak diffusion; axis 0 is dominated by advection
SPEED = -200.0  # advection along axis 0, against the increasing-index sweep
REACTION = 1.0
NY = 16  # the axis that is never coarsened


def discretize(shape):
    """Periodic 5-point operator with first-order upwind advection."""
    diagonal = jnp.full(shape, REACTION)
    lowers, uppers = [], []
    for axis, eps in enumerate(DIFFUSION):
        step = 1.0 / shape[axis]
        wind = SPEED if axis == 0 else 0.0
        diagonal = diagonal + 2.0 * eps / step**2 + abs(wind) / step
        lowers.append(jnp.full(shape, -eps / step**2 - max(wind, 0.0) / step))
        uppers.append(jnp.full(shape, -eps / step**2 - max(-wind, 0.0) / step))

    def matvec(u):
        out = diagonal * u
        for axis in range(len(shape)):
            out = out + lowers[axis] * jnp.roll(u, 1, axis)
            out = out + uppers[axis] * jnp.roll(u, -1, axis)
        return out

    return matvec, (lowers[0], diagonal, uppers[0])


def level(shape):
    """Operator plus its upwind-ordered line smoother on one grid."""
    matvec, bands = discretize(shape)
    wind = jnp.full(shape, SPEED)
    return matvec, sx.upwind_smoother(wind, *bands, axis=0, periodic=True)


def exact_inverse(matvec, shape):
    size = int(np.prod(shape))
    columns = jax.vmap(matvec)(jnp.eye(size).reshape(size, *shape))
    inverse = jnp.linalg.inv(columns.reshape(size, size).T)
    return lambda b: (inverse @ b.reshape(-1)).reshape(shape)


print(f"{'grid':>12}  {'levels':>6}  {'rate/cycle':>10}  {'GMRES+MG':>8}  {'GMRES':>6}")
for nx in (32, 64, 128, 256):
    shape = (nx, NY)

    # Coarsen axis 0 only; axis 1 stays at full resolution on every level.
    hierarchy = sx.semicoarsening_hierarchy(
        shape, (True, False), level, levels=4, boundary="periodic", min_size=4
    )
    coarse_matvec, _ = level(hierarchy.shapes[-1])
    cycle = sx.multigrid(
        hierarchy.levels,
        exact_inverse(coarse_matvec, hierarchy.shapes[-1]),
        cycle="v",
        pre_smooth=1,
        post_smooth=1,
    )

    matvec = hierarchy.levels[0].matvec
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(shape))

    # Standalone: average residual reduction per cycle.
    step = jax.jit(lambda x, matvec=matvec, cycle=cycle: x + cycle(rhs - matvec(x)))
    x = jnp.zeros(shape)
    for _ in range(6):
        x = step(x)
    rate = (float(jnp.linalg.norm(rhs - matvec(x)) / jnp.linalg.norm(rhs))) ** (1 / 6)

    # As a right preconditioner, acting on the 2-D state directly.
    options = dict(restart=30, rtol=1e-10, max_restarts=20)
    accelerated = sx.gmres(matvec, rhs, precond=cycle, **options)
    plain = sx.gmres(matvec, rhs, **options)
    plain_iters = f"{int(plain.iterations)}" + ("" if plain.converged else "+")
    print(
        f"{str(shape):>12}  {len(hierarchy.levels):>6}  {rate:>10.4f}  "
        f"{int(accelerated.iterations):>8}  {plain_iters:>6}"
    )

# Why the ordering matters: measure the smoothing factor of the same line
# relaxation swept with and against the wind.
shape = (64, NY)
_, bands = discretize(shape)
matvec, _ = discretize(shape)
wind = jnp.full(shape, SPEED)
key = jax.random.PRNGKey(0)
print("\nsmoothing factor of one line sweep (semicoarsening axis 0):")
for order in ("upwind", "forward"):
    smooth = sx.upwind_smoother(wind, *bands, axis=0, order=order, periodic=True)
    factor = sx.smoothing_factor(
        smooth, matvec, shape, key=key, coarsen=(True, False), steps=120
    )
    print(f"  {order:>8}: mu = {float(factor):.4f}")
