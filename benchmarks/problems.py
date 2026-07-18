"""Research-grade sweep problems for solver benchmarks.

Deterministic generators for the standard hard families used to characterize
iterative solvers and preconditioners, each parameterized by the quantity that
makes it hard:

- ``convection_diffusion``: 2-D convection-diffusion, upwind convection; the
  cell Peclet number sweeps the operator from SPD-like to strongly nonsymmetric
  (Elman, Silvester & Wathen 2014).
- ``helmholtz``: 2-D indefinite Helmholtz ``-lap u - k^2 u``; the wavenumber
  sweeps the spectrum through zero, the classically hard regime for Krylov
  methods (Ernst & Gander 2012).
- ``anisotropic_diffusion``: ``-u_xx - eps u_yy``; the anisotropy ratio defeats
  point preconditioners and motivates line/additive ones (Trottenberg,
  Oosterlee & Schueller 2001).
- ``poisson``: the SPD baseline (mesh-independence sanity).
- ``kinetic_block_tridiagonal``: dense-block streaming+collision structure of
  spectral kinetic equations, the shape SOLVAX's block solvers target.

Each generator returns a :class:`Problem` with a pure-JAX matvec on flat
``(n,)`` vectors, a deterministic right-hand side, and ``dense()`` for
small-size verification. All stencils are homogeneous-Dirichlet 5-point finite
differences on the unit square with ``grid x grid`` interior points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Problem(NamedTuple):
    """A benchmark linear system ``A x = b``.

    Attributes:
        name: family identifier, e.g. ``"helmholtz"``.
        params: the sweep parameters that generated it.
        matvec: pure-JAX operator ``v -> A v`` on flat ``(n,)`` vectors.
        rhs: deterministic right-hand side, shape ``(n,)``.
        diagonal: the operator diagonal (for Jacobi-type preconditioning).
        dense: callable returning the assembled dense matrix (verification
            only; O(n^2) memory).
        spd: whether ``A`` is symmetric positive definite (selects PCG; the
            indefinite and nonsymmetric families take FGMRES).
    """

    name: str
    params: dict
    matvec: Callable[[jax.Array], jax.Array]
    rhs: jax.Array
    diagonal: jax.Array
    dense: Callable[[], np.ndarray]
    spd: bool


def _grid_rhs(grid: int) -> jax.Array:
    x = (jnp.arange(grid) + 1.0) / (grid + 1)
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    return (jnp.sin(jnp.pi * xx) * yy * (1.0 - yy)).reshape(-1)


def _stencil_matvec(grid, center, west, east, south, north):
    """5-point stencil application with homogeneous Dirichlet boundaries."""

    def matvec(v):
        u = v.reshape(grid, grid)
        out = center * u
        out = out.at[1:, :].add(west * u[:-1, :])
        out = out.at[:-1, :].add(east * u[1:, :])
        out = out.at[:, 1:].add(south * u[:, :-1])
        out = out.at[:, :-1].add(north * u[:, 1:])
        return out.reshape(-1)

    return matvec


def _stencil_dense(grid, center, west, east, south, north) -> np.ndarray:
    n = grid * grid
    dense = np.zeros((n, n))
    for i in range(grid):
        for j in range(grid):
            row = i * grid + j
            dense[row, row] = center
            if i > 0:
                dense[row, row - grid] = west
            if i < grid - 1:
                dense[row, row + grid] = east
            if j > 0:
                dense[row, row - 1] = south
            if j < grid - 1:
                dense[row, row + 1] = north
    return dense


def _stencil_problem(name, params, grid, center, west, east, south, north, spd):
    return Problem(
        name=name,
        params={"grid": grid, **params},
        matvec=_stencil_matvec(grid, center, west, east, south, north),
        rhs=_grid_rhs(grid),
        diagonal=jnp.full(grid * grid, center),
        dense=lambda: _stencil_dense(grid, center, west, east, south, north),
        spd=spd,
    )


def poisson(grid: int) -> Problem:
    """SPD 5-point Laplacian, the mesh-independence baseline."""
    h2 = (grid + 1.0) ** 2
    return _stencil_problem(
        "poisson", {}, grid, 4.0 * h2, -h2, -h2, -h2, -h2, spd=True
    )


def convection_diffusion(grid: int, peclet: float) -> Problem:
    """Convection-diffusion with upwind convection along x.

    ``-lap u + beta u_x`` with ``beta = 2 * peclet * (grid + 1)``, so ``peclet``
    is the cell Peclet number ``beta h / 2``: below 1 the operator is nearly
    symmetric, far above 1 convection dominates and the operator is strongly
    nonsymmetric with boundary layers.
    """
    h = 1.0 / (grid + 1.0)
    h2 = 1.0 / (h * h)
    beta = 2.0 * peclet / h
    # First-order upwind for beta > 0: u_x ~ (u_i - u_{i-1}) / h.
    center = 4.0 * h2 + beta / h
    west = -h2 - beta / h
    return _stencil_problem(
        "convection_diffusion", {"peclet": peclet}, grid,
        center, west, -h2, -h2, -h2, spd=False,
    )


def helmholtz(grid: int, wavenumber: float) -> Problem:
    """Indefinite Helmholtz ``-lap u - k^2 u``: symmetric but indefinite once
    ``k`` exceeds the smallest Laplacian eigenvalue, so it takes the FGMRES
    path rather than PCG."""
    h2 = (grid + 1.0) ** 2
    return _stencil_problem(
        "helmholtz", {"wavenumber": wavenumber}, grid,
        4.0 * h2 - wavenumber**2, -h2, -h2, -h2, -h2, spd=False,
    )


def anisotropic_diffusion(grid: int, epsilon: float) -> Problem:
    """``-u_xx - eps u_yy``: strong coupling along x only as ``eps -> 0``."""
    h2 = (grid + 1.0) ** 2
    return _stencil_problem(
        "anisotropic_diffusion", {"epsilon": epsilon}, grid,
        2.0 * h2 * (1.0 + epsilon), -h2, -h2, -epsilon * h2, -epsilon * h2,
        spd=True,
    )


def kinetic_block_tridiagonal(n_blocks: int, block_size: int, coupling: float = 0.4):
    """Streaming+collision block-tridiagonal bands ``(lower, diag, upper, rhs)``.

    Dense diagonal blocks (collision-like, diagonally dominant) with
    nearest-neighbor streaming coupling of strength ``coupling`` — the
    spectral-kinetic structure SOLVAX's block-Thomas family targets. Returns
    band arrays for the ``block_thomas*`` and truncated/mixed-precision paths.
    """
    rng = np.random.default_rng(7)
    eye = np.eye(block_size)
    neighbor = np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    diag = jnp.asarray(
        rng.standard_normal((n_blocks, block_size, block_size)) * 0.3
        + 2.0 * block_size * eye
    )
    stream = coupling * block_size * neighbor
    lower = jnp.asarray(
        np.broadcast_to(stream, (n_blocks, block_size, block_size)).copy()
    )
    upper = jnp.asarray(
        np.broadcast_to(-stream, (n_blocks, block_size, block_size)).copy()
    )
    rhs = jnp.asarray(rng.standard_normal((n_blocks, block_size)))
    return lower, diag, upper, rhs


FAMILIES = {
    "poisson": poisson,
    "convection_diffusion": convection_diffusion,
    "helmholtz": helmholtz,
    "anisotropic_diffusion": anisotropic_diffusion,
}
