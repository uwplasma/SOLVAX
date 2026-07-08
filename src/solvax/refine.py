"""Mixed-precision iterative refinement (defect correction).

Given an approximate solver ``M ~ A^{-1}`` — typically a factorization or
iterative solve carried out in *low* precision — classic iterative
refinement recovers high-precision accuracy by repeated defect correction:

    r_i     = b - A x_i        (residual, computed in high precision)
    d_i     = M r_i            (correction, cheap low-precision solve)
    x_{i+1} = x_i + d_i

Each sweep contracts the error by roughly ``u_f * kappa(A)``, where ``u_f``
is the unit roundoff of the precision used inside ``M`` and ``kappa(A)`` the
condition number, until the residual stalls at the roundoff floor of the
*residual* precision. Carson & Higham formalised the three-precision
variant — factorization precision ``u_f``, working precision ``u``, and
residual precision ``u_r`` (e.g. float32 / float64 / float64 here, or
float16 / float32 / float64 on tensor-core hardware) — and showed
convergence to working-precision accuracy for ``kappa(A) u_f < 1``. This is
the standard trick for exploiting fast low-precision hardware, and the
recommended fallback when the unpivoted block eliminations in
``solvax.direct`` are used in weakly diagonally dominant regimes.

References
----------
- E. Carson & N. J. Higham, *Accelerating the Solution of Linear Systems by
  Iterative Refinement in Three Precisions*, SIAM J. Sci. Comput. 40(2),
  A817–A847 (2018), DOI 10.1137/17M1140819.
- J. H. Wilkinson, *Rounding Errors in Algebraic Processes*, Prentice-Hall
  (1963) — the original fixed-precision analysis.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def iterative_refinement(
    matvec: Callable,
    b: jax.Array,
    approx_solve: Callable,
    *,
    iterations: int = 3,
    residual_dtype=jnp.float64,
) -> tuple[jax.Array, jax.Array]:
    """Refine an approximate solve of ``matvec(x) = b`` by defect correction.

    Runs ``x_0 = M(b)`` followed by ``iterations`` sweeps of
    ``x_{i+1} = x_i + M(b - A x_i)``, with residuals accumulated in
    ``residual_dtype``. ``approx_solve`` may internally run in float32 (see
    :func:`as_low_precision`); the iterate and residual are kept in
    ``residual_dtype`` so the refined solution reaches high-precision
    accuracy whenever ``kappa(A) * u_low < 1``.

    Args:
        matvec: callable computing ``A @ x`` in (at least) ``residual_dtype``.
        b: right-hand side.
        approx_solve: callable ``approx_solve(r) -> d`` applying an
            approximate inverse of ``A``; may be low precision internally.
        iterations: number of correction sweeps (static Python int).
        residual_dtype: precision for iterates and residual accumulation.

    Returns:
        A pair ``(x, residual_norms)`` where ``x`` is the refined solution in
        ``residual_dtype`` and ``residual_norms`` has shape
        ``(iterations + 1,)`` — the 2-norm of ``b - A x_i`` after the initial
        solve and after each sweep, decreasing until it stalls at the
        ``residual_dtype`` roundoff floor.
    """
    b = jnp.asarray(b, residual_dtype)
    x = jnp.asarray(approx_solve(b), residual_dtype)
    norms = []
    for _ in range(iterations):
        r = b - jnp.asarray(matvec(x), residual_dtype)
        norms.append(jnp.linalg.norm(r))
        x = x + jnp.asarray(approx_solve(r), residual_dtype)
    r = b - jnp.asarray(matvec(x), residual_dtype)
    norms.append(jnp.linalg.norm(r))
    return x, jnp.stack(norms)


def as_low_precision(solve: Callable, dtype=jnp.float32) -> Callable:
    """Wrap a solve callable to run its inputs in a lower precision.

    The returned callable casts every array argument down to ``dtype``,
    calls ``solve``, and casts the result back up to the original dtype of
    the first argument — the usual way to build the low-precision inner
    solve of :func:`iterative_refinement`.

    Args:
        solve: callable taking one or more arrays and returning an array
            (or pytree of arrays).
        dtype: precision to run ``solve`` in.

    Returns:
        A callable with the same signature operating in ``dtype`` internally.
    """

    def wrapped(b, *args, **kwargs):
        out_dtype = jnp.result_type(b)
        low = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, dtype), (b, *args)
        )
        result = solve(*low, **kwargs)
        return jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, out_dtype), result
        )

    return wrapped
