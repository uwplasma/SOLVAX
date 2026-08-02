"""Spectral Fourier--Helmholtz elliptic solve.

Solves a separable elliptic problem of Helmholtz type on a periodic ``z`` axis
and a bounded ``x`` axis by Fourier-transforming in ``z`` (turning the periodic
Laplacian into a per-mode ``-k_z^2`` multiplier) and solving the remaining
tridiagonal system in ``x`` for every Fourier mode at once. This is the
``lap phi = rhs`` inversion used by reduced drift-plane / vorticity models,
where the operator is ``d/dx(g11 d/dx) + g33 d^2/dz^2`` with metric weights
``g11(x)``, ``g33(x)`` and homogeneous Dirichlet conditions at the bounded
cell faces.

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
    length. The bounded cell faces use homogeneous Dirichlet conditions. An
    odd-reflected ghost value, ``phi_ghost = -phi_boundary_cell``, places
    ``phi = 0`` halfway between the ghost and boundary-cell centers and gives
    the endpoint stencil ``(-3 phi_0 + phi_1) / dx^2`` for constant
    coefficients.
    """

    # Infer the working precision from the caller instead of hard-casting to
    # float64. The previous casts silently upgraded an x64-disabled program:
    # the geometry arrived as float32, came back as float64, and the caller got
    # a result in a precision it had deliberately not asked for -- or, with x64
    # off entirely, JAX quietly truncated the request back to float32 and the
    # `float64` in the signature meant nothing. Promoting the inputs against
    # each other keeps mixed float32/float64 geometry working and leaves the
    # decision where it belongs.
    real_dtype = jnp.result_type(
        jnp.asarray(dx), jnp.asarray(dz), jnp.asarray(g11),
        jnp.asarray(g33), jnp.asarray(rhs_scale),
    )
    if not jnp.issubdtype(real_dtype, jnp.floating):
        raise TypeError(
            f"the Fourier-Helmholtz geometry must be real; got {real_dtype}"
        )
    complex_dtype = jnp.result_type(real_dtype, jnp.complex64)

    dx = jnp.asarray(dx, dtype=real_dtype)
    dz = jnp.asarray(dz, dtype=real_dtype)
    g11 = jnp.asarray(g11, dtype=real_dtype)
    g33 = jnp.asarray(g33, dtype=real_dtype)
    rhs_scale = jnp.asarray(rhs_scale, dtype=real_dtype)

    zlength = float(dz[0]) * float(nz)
    x_coef = g11 / (dx * dx)
    modes = nz // 2 + 1
    wave_numbers = (2.0 * jnp.pi * jnp.arange(modes, dtype=real_dtype)) / zlength
    diagonals = -2.0 * x_coef[None, :] - jnp.square(wave_numbers)[:, None] * g33[None, :]
    diagonals = diagonals.at[:, 0].add(-x_coef[0])
    diagonals = diagonals.at[:, -1].add(-x_coef[-1])

    lower_diagonals = jnp.zeros_like(diagonals, dtype=complex_dtype)
    upper_diagonals = jnp.zeros_like(diagonals, dtype=complex_dtype)
    lower_diagonals = lower_diagonals.at[:, 1:].set(x_coef[1:][None, :].astype(complex_dtype))
    upper_diagonals = upper_diagonals.at[:, :-1].set(x_coef[:-1][None, :].astype(complex_dtype))

    return FourierHelmholtzOperator(
        lower_diagonals=lower_diagonals,
        diagonals=diagonals.astype(complex_dtype),
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

    # Match the operator's precision rather than forcing float64 here too.
    rhs = jnp.asarray(rhs, dtype=operator.rhs_scale.dtype)
    rhs_hat = jnp.fft.rfft(rhs * operator.rhs_scale[:, None], axis=-1)
    # The band arrays are stored as (mode, x); the solver wants the tridiagonal
    # (x) axis leading with the mode index as a batched trailing column.
    lower = jnp.swapaxes(operator.lower_diagonals, 0, 1)
    diag = jnp.swapaxes(operator.diagonals, 0, 1)
    upper = jnp.swapaxes(operator.upper_diagonals, 0, 1)
    interior_hat = tridiagonal_solve(lower, diag, upper, rhs_hat, method=method)
    return jnp.fft.irfft(interior_hat, n=operator.nz, axis=-1)
