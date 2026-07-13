"""Spectral Fourier--Helmholtz elliptic solve.

Solves a separable elliptic problem of Helmholtz type on a periodic ``z`` axis
and a bounded ``x`` axis by Fourier-transforming in ``z`` (turning the periodic
Laplacian into a per-mode ``-k_z^2`` multiplier) and solving the remaining
tridiagonal system in ``x`` for every Fourier mode at once. This is the
``lap phi = rhs`` inversion used by reduced drift-plane / vorticity models,
where the operator is ``d/dx(g11 d/dx) + g33 d^2/dz^2`` with metric weights
``g11(x)``, ``g33(x)``.

All routines are pure JAX (``jit``/``grad``/``vmap`` transparent). Build the
operator once for a fixed geometry with :func:`build_fourier_helmholtz_operator`
and reuse it across right-hand sides with :func:`solve_fourier_helmholtz`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from solvax.tridiagonal import tridiagonal_solve

__all__ = [
    "FourierHelmholtzOperator",
    "build_fourier_helmholtz_operator",
    "solve_fourier_helmholtz",
]


@dataclass(frozen=True)
class FourierHelmholtzOperator:
    """Per-mode complex tridiagonal factors of the Fourier--Helmholtz operator."""

    lower_diagonals: jax.Array
    diagonals: jax.Array
    upper_diagonals: jax.Array
    rhs_scale: jax.Array
    nz: int
    zlength: float


def build_fourier_helmholtz_operator(
    *,
    dx: jax.Array,
    dz: jax.Array,
    g11: jax.Array,
    g33: jax.Array,
    rhs_scale: jax.Array,
    nz: int,
) -> FourierHelmholtzOperator:
    """Assemble the per-mode tridiagonal operator for a fixed ``(g11, g33)`` geometry.

    ``dx``, ``g11``, ``g33``, ``rhs_scale`` are length-``nx`` arrays along the
    bounded ``x`` axis; ``dz`` sets the periodic ``z`` spacing and ``nz`` its
    length. The ``x`` boundaries use a reflected (homogeneous-Neumann-like)
    closure consistent with the reduced drift-plane potential solve.
    """

    dx = jnp.asarray(dx, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    g11 = jnp.asarray(g11, dtype=jnp.float64)
    g33 = jnp.asarray(g33, dtype=jnp.float64)
    rhs_scale = jnp.asarray(rhs_scale, dtype=jnp.float64)

    zlength = float(dz[0]) * float(nz)
    x_coef = g11 / (dx * dx)
    modes = nz // 2 + 1
    wave_numbers = (2.0 * jnp.pi * jnp.arange(modes, dtype=jnp.float64)) / zlength
    diagonals = -2.0 * x_coef[None, :] - jnp.square(wave_numbers)[:, None] * g33[None, :]
    diagonals = diagonals.at[:, 0].add(-x_coef[0])
    diagonals = diagonals.at[:, -1].add(-x_coef[-1])

    lower_diagonals = jnp.zeros_like(diagonals, dtype=jnp.complex128)
    upper_diagonals = jnp.zeros_like(diagonals, dtype=jnp.complex128)
    lower_diagonals = lower_diagonals.at[:, 1:].set(x_coef[1:][None, :].astype(jnp.complex128))
    upper_diagonals = upper_diagonals.at[:, :-1].set(x_coef[:-1][None, :].astype(jnp.complex128))

    return FourierHelmholtzOperator(
        lower_diagonals=lower_diagonals,
        diagonals=diagonals.astype(jnp.complex128),
        upper_diagonals=upper_diagonals,
        rhs_scale=rhs_scale,
        nz=int(nz),
        zlength=zlength,
    )


def solve_fourier_helmholtz(
    rhs: jax.Array,
    *,
    operator: FourierHelmholtzOperator,
    method: str = "thomas",
) -> jax.Array:
    """Solve ``operator @ solution = rhs`` for a real ``(nx, nz)`` right-hand side.

    Each Fourier mode is a complex tridiagonal system in the bounded ``x`` axis;
    all modes are solved in one call via :func:`solvax.tridiagonal.tridiagonal_solve`
    with the ``x`` axis leading and the mode index batched. ``method`` selects the
    tridiagonal backend and defaults to ``"thomas"`` (pure ``lax.scan``), which is
    complex-safe on every supported JAX version; pass ``"auto"``/``"lax"`` to use
    the fused kernel where the JAX build supports complex tridiagonal solves.
    """

    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    rhs_hat = jnp.fft.rfft(rhs * operator.rhs_scale[:, None], axis=-1)
    # The band arrays are stored as (mode, x); the solver wants the tridiagonal
    # (x) axis leading with the mode index as a batched trailing column.
    lower = jnp.swapaxes(operator.lower_diagonals, 0, 1)
    diag = jnp.swapaxes(operator.diagonals, 0, 1)
    upper = jnp.swapaxes(operator.upper_diagonals, 0, 1)
    interior_hat = tridiagonal_solve(lower, diag, upper, rhs_hat, method=method)
    return jnp.fft.irfft(interior_hat, n=operator.nz, axis=-1)
