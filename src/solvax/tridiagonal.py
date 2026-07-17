"""Backend-aware batched tridiagonal solve (Thomas / cuSPARSE fast paths).

A scalar tridiagonal system ``lower[j] x[j-1] + diag[j] x[j] + upper[j]
x[j+1] = rhs[j]`` is the banded special case ``lower_bw = upper_bw = 1`` of
:mod:`solvax.banded`, but it is common and structured enough to deserve a
dedicated, hardware-aware fast path — this module specializes it. The system
lives on the **leading** axis; every trailing axis of ``rhs`` (extra columns,
stacked fields, batch dimensions) is solved simultaneously in one call, which
is exactly the layout that maps a stack of independent tridiagonal systems
onto the vendor batched kernels without an outer :func:`jax.vmap`.

Two backends, selected per *lowering platform* at trace time:

- **Thomas** (``method="thomas"``): the classic two-sweep elimination

      c'[0] = upper[0] / diag[0],           d'[0] = rhs[0] / diag[0],
      c'[j] = upper[j] / (diag[j] - lower[j] c'[j-1]),
      d'[j] = (rhs[j] - lower[j] d'[j-1]) / (diag[j] - lower[j] c'[j-1]),
      x[n-1] = d'[n-1],   x[j] = d'[j] - c'[j] x[j+1],

  implemented as two ``jax.lax.scan`` sweeps over the radial axis with a
  ``eps = 1e-12`` guard on vanishing pivots. Fully jit/vmap/grad-transparent
  and, because the arithmetic is fixed, **bitwise reproducible** — the CPU
  path.

- **Fused** (``method="lax"``): XLA's batched
  :func:`jax.lax.linalg.tridiagonal_solve` (cuSPARSE ``gtsv2`` on CUDA,
  LAPACK ``gtsv`` on CPU). On a GPU the ``n`` sequential scan steps of Thomas
  serialize into ``n`` kernel launches (latency-bound, independent of how many
  columns ride along), so the single fused kernel is dramatically faster
  there. Numerically equivalent to Thomas (same solution to roundoff) but not
  bit-identical.

``method="auto"`` (default) uses :func:`jax.lax.platform_dependent` to pick
Thomas when the code lowers for CPU (bit parity, honouring a
``JAX_PLATFORMS=cpu`` / :func:`jax.default_device` pin even on an accelerator
host) and the fused kernel everywhere else. Systems with fewer than three rows
always use Thomas — the cuSPARSE kernel requires ``n >= 3``.

This solver is a drop-in preconditioner core: hand ``lambda r:
tridiagonal_solve(lower, diag, upper, r)`` to :func:`solvax.precond.
coarse_operator` or use it as a line solve inside
:func:`solvax.precond.line_smoother`.

References
----------
- L. H. Thomas, *Elliptic Problems in Linear Difference Equations over a
  Network*, Watson Sci. Comput. Lab. Report, Columbia University (1949) —
  the tridiagonal elimination algorithm.
- G. H. Golub & C. F. Van Loan, *Matrix Computations*, 4th ed., section 4.3
  (tridiagonal / banded Gaussian elimination).
- NVIDIA cuSPARSE ``gtsv2`` batched tridiagonal solver, exposed in JAX as
  :func:`jax.lax.linalg.tridiagonal_solve`.
- W. H. Press et al., *Numerical Recipes*, 3rd ed., section 2.7.3 — cyclic
  tridiagonal systems as a rank-one correction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

try:  # jax >= 0.4.16: per-lowering-platform branch selection.
    from jax.lax import platform_dependent as _platform_dependent
except ImportError:  # pragma: no cover - very old jax
    _platform_dependent = None

#: Guard substituted for a vanishing pivot in both backends.
_PIVOT_EPS = 1.0e-12


def tridiagonal_solve(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
    *,
    method: str = "auto",
) -> jax.Array:
    r"""Solve a batched scalar tridiagonal system along the leading axis.

    Solves ``lower[j] x[j-1] + diag[j] x[j] + upper[j] x[j+1] = rhs[j]`` for
    ``j = 0 .. n-1`` (with ``lower[0]`` and ``upper[-1]`` ignored), for every
    trailing column of ``rhs`` at once. The band arrays ``lower``, ``diag``,
    ``upper`` share the *system* shape (leading axis ``n`` plus any batch
    columns); ``rhs`` may carry **extra trailing axes** beyond that shape —
    stacked right-hand sides / fields — which are all solved in a single pass.
    Real bands with a complex ``rhs`` use independent real and imaginary solves,
    preserving real storage and fused accelerator kernels. Genuinely complex
    bands use the portable Thomas implementation.

    Args:
        lower: sub-diagonal, couples row ``j`` to ``j-1``; ``lower[0]``
            ignored. Shape broadcastable to ``diag``.
        diag: main diagonal; its shape ``(n, *columns)`` defines the system
            extent.
        upper: super-diagonal, couples row ``j`` to ``j+1``; ``upper[-1]``
            ignored. Shape broadcastable to ``diag``.
        rhs: right-hand side, shape ``diag.shape`` or ``diag.shape + fields``.
        method: backend selection.

            - ``"thomas"``: two-sweep ``lax.scan`` Thomas (bit-reproducible
              CPU path).
            - ``"lax"``: fused :func:`jax.lax.linalg.tridiagonal_solve`
              (cuSPARSE on GPU).
            - ``"auto"`` (default): Thomas when lowering for CPU, fused
              otherwise, chosen with :func:`jax.lax.platform_dependent`;
              systems with ``n < 3`` always use Thomas.

            Complex coefficient matrices use Thomas for every method because
            the fused JAX primitive is real-only on supported JAX releases.

    Returns:
        The solution with the same shape as ``rhs``.

    Raises:
        ValueError: if ``method`` is not one of ``"auto"``, ``"thomas"``,
            ``"lax"``.
    """
    lower, diag, upper, rhs = map(jnp.asarray, (lower, diag, upper, rhs))
    n_rows = int(rhs.shape[0])
    if n_rows == 0:
        return rhs
    if method not in ("auto", "thomas", "lax"):
        raise ValueError(f"unknown method {method!r}; expected 'auto', 'thomas' or 'lax'")

    band_dtype = jnp.result_type(lower, diag, upper)
    bands_are_complex = jnp.issubdtype(band_dtype, jnp.complexfloating)
    rhs_is_complex = jnp.issubdtype(rhs.dtype, jnp.complexfloating)
    if rhs_is_complex and not bands_are_complex:
        # A real matrix acts independently on the real and imaginary parts.
        dtype = jnp.result_type(lower, diag, upper, rhs.real)
        bands = tuple(value.astype(dtype) for value in (lower, diag, upper))

        def solve(value):
            return tridiagonal_solve(*bands, value.astype(dtype), method=method)

        return lax.complex(solve(rhs.real), solve(rhs.imag))

    dtype = jnp.result_type(lower, diag, upper, rhs)
    lower = lower.astype(dtype)
    diag = diag.astype(dtype)
    upper = upper.astype(dtype)
    rhs = rhs.astype(dtype)
    if bands_are_complex or method == "thomas" or n_rows < 3:
        return _thomas_solve(lower, diag, upper, rhs)
    if method == "lax":
        return _lax_solve(lower, diag, upper, rhs)
    if _platform_dependent is not None:
        return _platform_dependent(lower, diag, upper, rhs, cpu=_thomas_solve, default=_lax_solve)
    if jax.default_backend() == "cpu":  # pragma: no cover - old-jax fallback
        return _thomas_solve(lower, diag, upper, rhs)
    return _lax_solve(lower, diag, upper, rhs)  # pragma: no cover


def cyclic_tridiagonal_solve(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    rhs: jax.Array,
    *,
    method: str = "auto",
) -> jax.Array:
    """Solve periodic tridiagonal systems along the leading axis.

    ``lower[0]`` couples row zero to the last unknown and ``upper[-1]``
    couples the last row to unknown zero. A Sherman--Morrison correction turns
    the periodic system into one ordinary tridiagonal solve with two stacked
    right-hand sides, retaining the backend selected by
    :func:`tridiagonal_solve`.

    The system arrays have shape ``(n, *columns)``. ``rhs`` may append extra
    trailing field axes, which are solved simultaneously.
    """
    lower = jnp.broadcast_to(jnp.asarray(lower), jnp.shape(diag))
    diag = jnp.asarray(diag)
    upper = jnp.broadcast_to(jnp.asarray(upper), diag.shape)
    rhs = jnp.asarray(rhs)
    if diag.shape[0] < 3:
        raise ValueError("cyclic tridiagonal systems require at least three rows")
    if rhs.ndim < diag.ndim or rhs.shape[: diag.ndim] != diag.shape:
        raise ValueError("rhs must begin with the cyclic tridiagonal system shape")

    # Numerical Recipes names the bottom-left corner alpha and top-right beta;
    # the public band arrays store those at upper[-1] and lower[0].
    alpha, beta = upper[-1], lower[0]
    eps = jnp.asarray(_PIVOT_EPS, dtype=diag.dtype)
    gamma = jnp.where(jnp.abs(diag[0]) > eps, -diag[0], -jnp.ones_like(diag[0]))
    core_diag = diag.at[0].set(diag[0] - gamma)
    core_diag = core_diag.at[-1].set(diag[-1] - alpha * beta / gamma)
    core_lower = lower.at[0].set(0.0)
    core_upper = upper.at[-1].set(0.0)

    correction_rhs = jnp.zeros_like(diag).at[0].set(gamma).at[-1].set(alpha)
    extra_axes = rhs.ndim - diag.ndim
    correction_rhs = jnp.broadcast_to(
        correction_rhs.reshape(correction_rhs.shape + (1,) * extra_axes), rhs.shape
    )
    solved = tridiagonal_solve(
        core_lower,
        core_diag,
        core_upper,
        jnp.stack((rhs, correction_rhs), axis=-1),
        method=method,
    )
    solution, correction = solved[..., 0], solved[..., 1]
    corner_shape = alpha.shape + (1,) * extra_axes
    beta_rhs = beta.reshape(corner_shape)
    gamma_rhs = gamma.reshape(corner_shape)
    scale = (solution[0] + beta_rhs * solution[-1] / gamma_rhs) / (
        1.0 + correction[0] + beta_rhs * correction[-1] / gamma_rhs
    )
    return solution - correction * scale[None, ...]


def _lax_solve(lower: jax.Array, diag: jax.Array, upper: jax.Array, rhs: jax.Array) -> jax.Array:
    """Differentiable wrapper around the fused tridiagonal primitive."""

    def matvec(value):
        extra = (1,) * (value.ndim - diag.ndim)
        lo = lower.reshape(lower.shape + extra)
        middle = diag.reshape(diag.shape + extra)
        up = upper.reshape(upper.shape + extra)
        result = middle * value
        result = result.at[1:].add(lo[1:] * value[:-1])
        return result.at[:-1].add(up[:-1] * value[1:])

    def solve(_, value):
        return _lax_solve_raw(lower, diag, upper, value)

    transpose_lower = jnp.concatenate((jnp.zeros_like(upper[:1]), upper[:-1]))
    transpose_upper = jnp.concatenate((lower[1:], jnp.zeros_like(lower[:1])))

    def transpose_solve(_, value):
        return _lax_solve_raw(transpose_lower, diag, transpose_upper, value)

    return lax.custom_linear_solve(
        matvec, rhs, solve=solve, transpose_solve=transpose_solve
    )


def _lax_solve_raw(
    lower: jax.Array, diag: jax.Array, upper: jax.Array, rhs: jax.Array
) -> jax.Array:
    """Fused batched solve via :func:`jax.lax.linalg.tridiagonal_solve`.

    Maps the leading-axis / trailing-column layout onto the ``lax.linalg``
    convention (system dimension *last* in the bands, *second-to-last* in the
    right-hand side; columns become batch dimensions and any extra field axes
    flatten into the multiple-RHS axis). ``lower[0]`` / ``upper[-1]`` are
    zeroed (required to be zero by cuSPARSE) and vanishing diagonal entries
    get the same ``eps`` guard as the Thomas path.
    """
    d = jnp.broadcast_to(diag, diag.shape)
    du = jnp.broadcast_to(upper, d.shape)
    dl = jnp.broadcast_to(lower, d.shape)
    dl = dl.at[0].set(0.0)
    du = du.at[-1].set(0.0)
    eps = jnp.asarray(_PIVOT_EPS, dtype=rhs.dtype)
    d = jnp.where(d != 0.0, d, eps)
    tail = rhs.shape[d.ndim :]
    n_fields = int(np.prod(tail, dtype=np.int64)) if tail else 1
    rhs_in = rhs.reshape(d.shape + (n_fields,))
    solution_t = jax.lax.linalg.tridiagonal_solve(
        jnp.moveaxis(dl, 0, -1),
        jnp.moveaxis(d, 0, -1),
        jnp.moveaxis(du, 0, -1),
        jnp.moveaxis(rhs_in, 0, -2),
    )
    return jnp.moveaxis(solution_t, -2, 0).reshape(rhs.shape)


def _thomas_solve(lower: jax.Array, diag: jax.Array, upper: jax.Array, rhs: jax.Array) -> jax.Array:
    """Two-sweep ``lax.scan`` Thomas elimination (bit-reproducible)."""
    if rhs.ndim > diag.ndim:
        expand = (1,) * (rhs.ndim - diag.ndim)
        upper = upper.reshape(upper.shape + expand)
        diag = diag.reshape(diag.shape + expand)
        lower = lower.reshape(lower.shape + expand)
    n_rows = int(rhs.shape[0])
    if n_rows == 0:  # pragma: no cover - guarded upstream by tridiagonal_solve
        return rhs

    eps = jnp.asarray(_PIVOT_EPS, dtype=rhs.dtype)
    diag0 = jnp.where(diag[0] != 0.0, diag[0], eps)
    upper0 = upper[0] / diag0
    x0 = rhs[0] / diag0

    def forward(carry, inputs):
        upper_prev, x_prev = carry
        upper_j, diag_j, lower_j, rhs_j = inputs
        denom = diag_j - upper_prev * lower_j
        denom = jnp.where(denom != 0.0, denom, eps)
        upper_new = upper_j / denom
        x_new = (rhs_j - x_prev * lower_j) / denom
        return (upper_new, x_new), (upper_new, x_new)

    if n_rows == 1:
        return x0[None, ...]
    inputs = (upper[1:], diag[1:], lower[1:], rhs[1:])
    _, (upper_rest, x_rest) = lax.scan(forward, (upper0, x0), inputs)
    upper_norm = jnp.concatenate([upper0[None, ...], upper_rest], axis=0)
    x = jnp.concatenate([x0[None, ...], x_rest], axis=0)

    def backward(x_next, inputs):
        upper_j, x_j = inputs
        x_new = x_j - upper_j * x_next
        return x_new, x_new

    x_last = x[-1]
    _, x_body = lax.scan(backward, x_last, (upper_norm[:-1], x[:-1]), reverse=True)
    return jnp.concatenate([x_body, x_last[None, ...]], axis=0)


def _reusable_tridiagonal_solver(lower, diag, upper):
    """Cache CPU Thomas factors while retaining the fused accelerator path."""
    lower, diag, upper = map(jnp.asarray, (lower, diag, upper))
    band_dtype = jnp.result_type(lower, diag, upper)
    lower, diag, upper = (value.astype(band_dtype)
        for value in (lower, diag, upper))
    eps = jnp.asarray(_PIVOT_EPS, dtype=band_dtype)
    pivot0 = jnp.where(diag[0] != 0.0, diag[0], eps)
    upper0 = upper[0] / pivot0

    def eliminate(previous, values):
        upper_j, diagonal_j, lower_j = values
        pivot = diagonal_j - previous * lower_j
        pivot = jnp.where(pivot != 0.0, pivot, eps)
        normalized = upper_j / pivot
        return normalized, (normalized, pivot)

    _, (upper_rest, pivot_rest) = lax.scan(
        eliminate, upper0, (upper[1:], diag[1:], lower[1:]))
    normalized_upper = jnp.concatenate((upper0[None], upper_rest))
    pivots = jnp.concatenate((pivot0[None], pivot_rest))

    def thomas(rhs):
        first = rhs[0] / pivots[0]

        def forward(previous, values):
            lower_j, pivot_j, rhs_j = values
            result = (rhs_j - previous * lower_j) / pivot_j
            return result, result

        _, rest = lax.scan(forward, first, (lower[1:], pivots[1:], rhs[1:]))
        values = jnp.concatenate((first[None], rest))

        def backward(following, values):
            upper_j, value_j = values
            result = value_j - upper_j * following
            return result, result

        _, body = lax.scan(backward, values[-1],
            (normalized_upper[:-1], values[:-1]), reverse=True)
        return jnp.concatenate((body, values[-1:]))

    def solve(rhs):
        rhs = jnp.asarray(rhs)
        if rhs.dtype != band_dtype:
            return tridiagonal_solve(lower, diag, upper, rhs)
        if jnp.issubdtype(band_dtype, jnp.complexfloating) or rhs.shape[0] < 3:
            return thomas(rhs)
        if _platform_dependent is not None:
            return _platform_dependent(rhs, cpu=thomas,
                default=lambda value: _lax_solve(lower, diag, upper, value))
        if jax.default_backend() == "cpu":  # pragma: no cover - old JAX
            return thomas(rhs)
        return _lax_solve(lower, diag, upper, rhs)  # pragma: no cover

    return solve
