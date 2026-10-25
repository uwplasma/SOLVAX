"""Separable (fast-diagonalization) inverses and the variable-coefficient flux-form operator.

On a rectilinear cell-centred grid the finite-volume Laplacian is a Kronecker sum of
one-dimensional operators ``M_a = W_a^{-1} K_a``: ``K_a`` is the symmetric conductance matrix,
``W_a`` the cell widths. ``K_a v = lambda W_a v`` is solved once on the host, so ``shift -
coefficient * laplacian`` is inverted by three contractions and a divide (:class:`FastDiagonal`).
Each axis is homogeneous Dirichlet (the face lies half a cell from the centre), Neumann or
periodic.

:class:`FluxFormOperator` is ``A x = V (s x - c div(G grad x))`` for cell arrays ``k`` and ``s``
and a scalar ``c``, with ``V`` the cell volumes. The flux per unit area through a face is ``(x2 -
x1) / (d1/k1 + d2/k2)``, ``d`` the half widths of its two cells (a series resistance, so what
leaves one cell enters the next); a boundary face takes its cell's ``k``. Face area and cell
volume enter once, in the divergence. ``A`` is symmetric positive definite, solved by
:func:`solvax.pcg_linear_solve` preconditioned with the :class:`FastDiagonal` inverse at the
volume-mean coefficients.

References
----------
- R. E. Lynch, J. R. Rice & D. H. Thomas, "Direct solution of partial difference equations by
  tensor product methods", Numer. Math. 6, 185 (1964) -- fast diagonalization.
- S. V. Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), section 4.2.3 -- the
  harmonic-mean face conductivity.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from solvax.pcg import pcg_linear_solve

__all__ = [
    "FastDiagonal",
    "FluxFormOperator",
    "SpdSolveInfo",
    "fast_diagonal",
    "flux_form_operator",
]

_KINDS = ("dirichlet", "neumann", "periodic")


class SpdSolveInfo(NamedTuple):
    """Status of a symmetric positive-definite solve. A solve with ``converged`` False returns NaN.

    Attributes:
        converged: traced boolean: PCG reached ``rtol``, everything is finite and, for a singular
            operator, the right-hand side is compatible.
        iterations: PCG iterations.
        residual: final relative residual norm of the gauge-fixed system.
        status: the :func:`solvax.pcg` status code.
        defect: for a singular operator the volume mean of the right-hand side,
            ``sum(rhs) / sum(V)``, which the gauge would otherwise absorb; zero otherwise.
    """

    converged: jax.Array
    iterations: jax.Array
    residual: jax.Array
    status: jax.Array
    defect: jax.Array


def _axis_modes(widths, kind: str):
    """``(Q, Q^{-1}, lambda)`` with ``W^{-1} K = Q diag(lambda) Q^{-1}`` for one axis."""
    w = np.asarray(widths, dtype=np.float64)
    if kind not in _KINDS:
        raise ValueError(f"axis kind must be one of {_KINDS}, got {kind!r}")
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w) & (w > 0.0)):
        raise ValueError("widths must be one 1-D array of positive finite cell widths per axis")
    n, i = w.size, np.arange(w.size - 1)
    c = 2.0 / (w[:-1] + w[1:])
    k = np.zeros((n, n))
    k[i, i] += c
    k[i + 1, i + 1] += c
    k[i, i + 1] -= c
    k[i + 1, i] -= c
    if kind == "dirichlet":
        k[0, 0] += 2.0 / w[0]
        k[-1, -1] += 2.0 / w[-1]
    elif kind == "periodic":
        wrap = 2.0 / (w[0] + w[-1])
        k[0, 0] += wrap
        k[-1, -1] += wrap
        k[0, -1] -= wrap
        k[-1, 0] -= wrap
    root = np.sqrt(w)
    lam, vec = np.linalg.eigh(k / np.outer(root, root))
    lam = np.maximum(lam, 0.0)
    if kind != "dirichlet":
        lam[0] = 0.0  # the constants: exactly zero, not rounding
    return vec / root[:, None], vec.T * root[None, :], lam


def _volumes(widths) -> np.ndarray:
    return functools.reduce(np.multiply.outer, [np.asarray(w, np.float64) for w in widths])


def _contract(matrices, x: jax.Array, transpose: bool) -> jax.Array:
    for axis, m in enumerate(matrices):
        m = jnp.asarray(m, jnp.real(x).dtype)
        x = jnp.moveaxis(jnp.tensordot(m.T if transpose else m, x, axes=([1], [axis])), 0, axis)
    return x


def _is_zero(value) -> bool:
    try:
        return bool(jnp.all(jnp.asarray(value) == 0.0))
    except jax.errors.ConcretizationTypeError:
        return False


class FastDiagonal(eqx.Module):
    """The separable operator ``shift - coefficient * laplacian``, applied and inverted spectrally.

    Cell-space convention: ``solve(r)`` returns ``x`` with ``(shift - coefficient * laplacian) x
    = r``, ``laplacian`` the finite-volume one (per unit volume), so the operator is symmetric
    only in the volume-weighted inner product and ``transpose=True`` solves the plain transpose.
    With ``shift == 0`` and no Dirichlet axis the constants are its null space (``singular``):
    the solve is the pseudo-inverse, which drops the volume-mean part of ``r`` and returns zero
    volume mean. ``shift`` and ``coefficient`` are leaves and differentiable (the solve is a
    direct formula in them); the widths are geometry and are not differentiated. Build with
    :func:`fast_diagonal`.
    """

    basis: tuple[jax.Array, ...]
    inverse: tuple[jax.Array, ...]
    eigenvalues: tuple[jax.Array, ...]
    volumes: jax.Array
    shift: jax.Array
    coefficient: jax.Array
    singular: bool = eqx.field(static=True)

    def symbol(self) -> jax.Array:
        """The eigenvalue of every mode, on the grid shape."""
        total: Any = 0.0
        for axis, lam in enumerate(self.eigenvalues):
            total = total + lam.reshape([-1 if i == axis else 1 for i in range(len(self.basis))])
        return self.shift + self.coefficient * total

    def matvec(self, x: jax.Array) -> jax.Array:
        """Apply the operator spectrally."""
        return _contract(self.basis, self.symbol() * _contract(self.inverse, x, False), False)

    def solve(self, rhs: jax.Array, *, transpose: bool = False) -> jax.Array:
        """Return ``x`` with ``operator x = rhs`` (``operator^T x = rhs`` if ``transpose``)."""
        first, last = (self.basis, self.inverse) if transpose else (self.inverse, self.basis)
        symbol, null = self.symbol(), (0,) * len(self.basis)
        if self.singular:
            symbol = symbol.at[null].set(1.0)
        modes = _contract(first, rhs, transpose) / symbol
        if self.singular:
            modes = modes.at[null].set(0.0)
        return _contract(last, modes, transpose)

    @property
    def scale(self) -> jax.Array:
        """The mean eigenvalue over the non-null modes: the operator's typical size."""
        return jnp.sum(self.symbol()) / (self.volumes.size - self.singular)


def fast_diagonal(widths, kinds, *, shift=0.0, coefficient=1.0) -> FastDiagonal:
    """Factor ``shift - coefficient * laplacian`` on cells of ``widths`` (one 1-D array per axis).

    ``kinds`` gives each axis ``"dirichlet"``, ``"neumann"`` or ``"periodic"``, all homogeneous
    (boundary data belong in the right-hand side). The widths must be concrete. ``shift == 0``,
    concretely, on a grid with no Dirichlet axis is the singular case of :class:`FastDiagonal`.
    """
    if len(widths) != len(kinds):
        raise ValueError("widths and kinds need one entry per axis")
    modes = [_axis_modes(w, k) for w, k in zip(widths, kinds, strict=True)]
    real = jnp.result_type(float)
    return FastDiagonal(
        tuple(jnp.asarray(m[0], real) for m in modes),
        tuple(jnp.asarray(m[1], real) for m in modes),
        tuple(jnp.asarray(m[2], real) for m in modes),
        jnp.asarray(_volumes(widths), real),
        jnp.asarray(shift, real),
        jnp.asarray(coefficient, real),
        singular=_is_zero(shift) and "dirichlet" not in kinds,
    )


def _spd_solve(
    operator,
    precondition,
    rhs,
    volumes,
    scale,
    singular,
    valid,
    *,
    x0,
    rtol,
    max_steps,
    defect_rtol,
):
    """PCG on a symmetric ``V``-space operator with gauge, compatibility and failure handling.

    Singular: the rank-one term ``alpha V (V . x) / sum V`` makes the operator positive definite,
    the preconditioner gains the matching inverse on the constants, and the volume mean is
    removed afterwards.
    """
    eps = jnp.finfo(rhs.dtype).eps
    rtol = (1.0e-11 if rhs.dtype == jnp.float64 else 1.0e-5) if rtol is None else rtol
    defect_rtol = jnp.sqrt(eps) if defect_rtol is None else defect_rtol
    total = jnp.sum(volumes)
    if singular:
        alpha, base, inner = lax.stop_gradient(scale), operator, precondition

        def operator(x):
            return base(x) + alpha * volumes * jnp.sum(volumes * x) / total

        def precondition(r):
            return inner(r) + jnp.sum(r) / (alpha * total)

    result = pcg_linear_solve(
        operator, rhs, x0=x0, precond=precondition, rtol=rtol, max_steps=max_steps
    )
    x, ok = result.x, valid & result.converged & jnp.isfinite(result.residual_norm)
    defect = jnp.sum(rhs) / total if singular else jnp.zeros((), rhs.dtype)
    if singular:
        x = x - jnp.sum(volumes * x) / total
        ok = ok & (jnp.abs(jnp.sum(rhs)) <= defect_rtol * jnp.sum(jnp.abs(rhs)))
    # NaN value and NaN derivatives in every input, closed-over leaves included: the factor's
    # derivative is NaN and the adjoint solve carries it (not a silent zero gradient).
    x = x * jnp.where(ok, 1.0, jnp.nan)
    return x, SpdSolveInfo(
        ok, result.iterations, result.relative_residual_norm, result.status, defect
    )


def _slab(x, axis, start, stop):
    return lax.slice_in_dim(x, start, stop, axis=axis)


def _pad(x, axis, kind, odd=False):
    """Ghost cells on both sides of ``axis``: wrapped, or the edge cell (negated if ``odd``)."""
    n = x.shape[axis]
    first, last = _slab(x, axis, 0, 1), _slab(x, axis, n - 1, n)
    low, high = (last, first) if kind == "periodic" else (first, last)
    if odd:
        low, high = -low, -high
    return jnp.concatenate([low, x, high], axis=axis)


class FluxFormOperator(eqx.Module):
    """``A x = V (shift x - coefficient div(G grad x))`` on a rectilinear grid (module docstring).

    Leaves: ``conductivity`` and ``shift`` (cell arrays), ``coefficient`` (scalar), the widths and
    the preconditioner. They may be traced, and :meth:`with_coefficients` swaps the first three
    without changing the structure, so the compiled program is reused. ``A`` is symmetric, so its
    transpose solve is the solve itself; :meth:`solve` differentiates implicitly
    (:func:`solvax.pcg_linear_solve`) in ``rhs``, ``conductivity``, ``shift`` and ``coefficient``.
    The preconditioner stays at the volume means given at construction and carries no derivative:
    it only steers the iteration.

    ``shift == 0`` (concretely) with no Dirichlet axis is the singular variant, the constants its
    null space: :meth:`solve` solves the compatible part, returns zero volume mean and reports the
    removed defect ``sum(rhs) / sum(V)`` in its info.
    """

    widths: tuple[jax.Array, ...]
    conductivity: jax.Array
    shift: jax.Array
    coefficient: jax.Array
    preconditioner: FastDiagonal
    kinds: tuple[str, ...] = eqx.field(static=True)

    def with_coefficients(
        self, conductivity=None, shift=None, coefficient=None
    ) -> FluxFormOperator:
        """The same operator at other coefficient values (traced arrays allowed, no recompile)."""
        grid = self.conductivity.shape
        new: dict[str, Any] = {
            name: jnp.broadcast_to(
                jnp.asarray(value, self.conductivity.dtype), grid if name != "coefficient" else ()
            )
            for name, value in (
                ("conductivity", conductivity),
                ("shift", shift),
                ("coefficient", coefficient),
            )
            if value is not None
        }
        return dataclasses.replace(self, **new)

    def _along(self, axis):
        return self.widths[axis].reshape([-1 if i == axis else 1 for i in range(len(self.widths))])

    def conductances(self) -> tuple[jax.Array, ...]:
        """Per axis ``1 / (d1/k1 + d2/k2)`` on its ``n + 1`` faces: flux per unit area per jump."""
        out = []
        for axis, kind in enumerate(self.kinds):
            k, half = _pad(self.conductivity, axis, kind), _pad(self._along(axis), axis, kind) / 2.0
            n = self.conductivity.shape[axis]
            out.append(
                1.0
                / (
                    _slab(half, axis, 0, n + 1) / _slab(k, axis, 0, n + 1)
                    + _slab(half, axis, 1, n + 2) / _slab(k, axis, 1, n + 2)
                )
            )
        return tuple(out)

    def _apply(self, x, conductances):
        volumes = self.preconditioner.volumes
        out = 0.0 if self.preconditioner.singular else volumes * self.shift * x
        for axis, kind in enumerate(self.kinds):
            n = x.shape[axis]
            xp = _pad(x, axis, kind, odd=kind == "dirichlet")
            flux = conductances[axis] * (_slab(xp, axis, 1, n + 2) - _slab(xp, axis, 0, n + 1))
            out = out - self.coefficient * volumes / self._along(axis) * (
                _slab(flux, axis, 1, n + 1) - _slab(flux, axis, 0, n)
            )
        return out

    def apply(self, x: jax.Array) -> jax.Array:
        """``A x``."""
        return self._apply(x, self.conductances())

    def lift(self, data) -> jax.Array:
        """What non-homogeneous boundary data add to the rhs: solve ``A x = rhs + lift(data)``.

        ``data[axis] = (lower, upper)`` or ``None``, each broadcastable to the boundary slab: the
        value of ``x`` on a Dirichlet face, or the gradient ``dx/d(axis)`` on a Neumann face (the
        flux is that gradient times the boundary cell's ``k``). A periodic axis takes none.
        """
        out = jnp.zeros_like(self.conductivity)
        for axis, (kind, pair, c) in enumerate(
            zip(self.kinds, data, self.conductances(), strict=True)
        ):
            if pair is None:
                continue
            if kind == "periodic":
                raise ValueError("a periodic axis has no boundary data")
            n, k, (lo, hi) = self.conductivity.shape[axis], self.conductivity, pair
            if kind == "dirichlet":
                low, high = -2.0 * _slab(c, axis, 0, 1) * lo, 2.0 * _slab(c, axis, n, n + 1) * hi
            else:
                low, high = _slab(k, axis, 0, 1) * lo, _slab(k, axis, n - 1, n) * hi
            extra = jnp.concatenate(
                [low, jnp.zeros_like(_slab(c, axis, 0, n - 1)), high], axis=axis
            )
            step = _slab(extra, axis, 1, n + 1) - _slab(extra, axis, 0, n)
            out = out + self.coefficient * self.preconditioner.volumes / self._along(axis) * step
        return out

    def solve(
        self,
        rhs: jax.Array,
        *,
        rtol: float | None = None,
        max_steps: int = 200,
        x0: jax.Array | None = None,
        defect_rtol: float | None = None,
    ) -> tuple[jax.Array, SpdSolveInfo]:
        """Return ``x`` with ``A x = rhs`` (``rhs`` includes the cell volume) and a status.

        PCG from ``x0`` (zero by default; the tangent and adjoint solves start from zero) to
        ``rtol`` of ``rhs`` (default 1e-11 in float64, 1e-5 otherwise). Unconverged, a
        nonpositive coefficient, a nonfinite value or, singular, a right-hand side whose volume
        mean exceeds ``defect_rtol`` of its absolute sum (default sqrt(eps)): NaN value and NaN
        derivatives in every input.
        """
        pre, g = lax.stop_gradient(self.preconditioner), self.conductances()
        volumes = pre.volumes
        valid = (
            jnp.all(self.conductivity > 0.0) & (self.coefficient > 0.0) & jnp.all(self.shift >= 0.0)
        )
        return _spd_solve(
            lambda x: self._apply(x, g),
            lambda r: pre.solve(r / volumes),
            rhs,
            volumes,
            pre.scale,
            pre.singular,
            valid,
            x0=x0,
            rtol=rtol,
            max_steps=max_steps,
            defect_rtol=defect_rtol,
        )


def flux_form_operator(widths, kinds, conductivity, shift, coefficient) -> FluxFormOperator:
    """Build :class:`FluxFormOperator` on cells of ``widths`` (concrete, one 1-D array per axis).

    ``conductivity`` and ``shift`` are cell arrays or scalars, ``coefficient`` a scalar. The
    preconditioner is the separable inverse at their volume means. ``shift == 0`` selects the
    singular variant on a grid with no Dirichlet axis.
    """
    real = jnp.result_type(float)
    shape = tuple(len(w) for w in widths)
    k = jnp.broadcast_to(jnp.asarray(conductivity, real), shape)
    s = jnp.broadcast_to(jnp.asarray(shift, real), shape)
    volumes = jnp.asarray(_volumes(widths), real)

    def mean(a):
        return jnp.sum(volumes * a) / jnp.sum(volumes)

    c = jnp.asarray(coefficient, real)
    pre = fast_diagonal(
        widths, kinds, shift=0.0 if _is_zero(s) else mean(s), coefficient=c * mean(k)
    )
    widths = tuple(jnp.asarray(w, real) for w in widths)
    return FluxFormOperator(widths, k, s, c, lax.stop_gradient(pre), tuple(kinds))
