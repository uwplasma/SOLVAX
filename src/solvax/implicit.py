"""Implicit differentiation of linear solves and root finds.

For a parameterised linear system ``A(theta) x = b(theta)``, the implicit
function theorem gives, without ever differentiating through the solver's
internal iterations,

    dx = A^{-1} (db - dA x),

so the VJP of ``x = solve(A, b)`` against a cotangent ``xbar`` is obtained
from a single *transposed* solve,

    A^T lambda = xbar,   bbar = lambda,   Abar = -lambda x^T.

Likewise, for a root ``x*`` of ``f(x, theta) = 0``,

    dx*/dtheta = -(df/dx)^{-1} (df/dtheta),

which requires one linear solve against the Jacobian of ``f`` at the root.
In both cases the forward solver is treated as a black box: it may iterate
to arbitrary tolerance, use stopping criteria, restarts, preconditioning —
none of that is differentiated. The adjoint costs exactly one additional
(transposed / tangent) solve, independent of how many iterations the
forward solve took.

These wrappers are thin layers over :func:`jax.lax.custom_linear_solve` and
:func:`jax.lax.custom_root`, specialised to the "bring your own solver"
pattern used throughout solvax (e.g. Krylov methods from ``solvax.krylov``
or the structured direct solves in ``solvax.direct``).

References
----------
- M. Blondel et al., *Efficient and Modular Implicit Differentiation*,
  NeurIPS 2022, https://arxiv.org/abs/2105.15183.
- C. S. Skene & K. J. Burns, *Fast adjoints for kinetic solvers*,
  https://arxiv.org/abs/2506.14792 — the one-extra-transposed-solve adjoint
  for iterative kinetic-equation solvers.
- JAX documentation for ``jax.lax.custom_linear_solve`` and
  ``jax.lax.custom_root``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def linear_solve(
    matvec: Callable,
    b: jax.Array,
    solver: Callable,
    *,
    transpose_matvec: Callable | None = None,
) -> jax.Array:
    """Differentiable linear solve with a user-supplied black-box solver.

    Solves ``matvec(x) = b`` by calling ``solver(matvec, b)``, and registers
    an implicit-function-theorem VJP via :func:`jax.lax.custom_linear_solve`:
    gradients w.r.t. anything ``matvec`` or ``b`` close over cost exactly one
    additional transposed solve — the solver's internal iterations are never
    differentiated.

    Args:
        matvec: linear callable computing ``A @ x``; parameters of ``A``
            should be closed over so gradients can flow to them.
        b: right-hand side.
        solver: callable ``solver(matvec, b) -> x``, e.g. a lambda around a
            Krylov method or a dense solve. It is treated as a black box
            (never differentiated through), but must be traceable by JAX.
        transpose_matvec: optional callable computing ``A^T @ y`` for the
            adjoint solve. If omitted, the transpose obtained from
            ``jax.linear_transpose`` of ``matvec`` is used.

    Returns:
        The solution ``x``, differentiable w.r.t. the parameters of
        ``matvec`` and ``b``.
    """
    if transpose_matvec is None:
        # custom_linear_solve hands transpose_solve the linear transpose of
        # matvec (computed with jax.linear_transpose); reuse the same solver.
        def _transpose_solve(vecmat, y):
            return solver(vecmat, y)

    else:

        def _transpose_solve(vecmat, y):
            del vecmat  # user-supplied transpose is presumed cheaper/exact
            return solver(transpose_matvec, y)

    return jax.lax.custom_linear_solve(
        matvec, b, solve=solver, transpose_solve=_transpose_solve
    )


def _default_tangent_solve(g: Callable, y: jax.Array) -> jax.Array:
    """Solve the linearised system ``g(x) = y`` for the root tangent.

    ``g`` is the (linear) Jacobian-vector product of ``f`` at the root. For
    scalars a single division suffices; for small vector systems the dense
    Jacobian is materialised and solved with :func:`jnp.linalg.solve`. For
    large systems pass a custom ``tangent_solve`` (e.g. a Krylov solve).
    """
    y = jnp.asarray(y)
    if y.ndim == 0:
        return y / g(jnp.ones_like(y))
    jac = jax.jacobian(g)(jnp.zeros_like(y))
    return jnp.linalg.solve(jac, y)


def root_solve(
    f: Callable,
    x0: jax.Array,
    solver: Callable,
    *,
    tangent_solve: Callable | None = None,
) -> jax.Array:
    """Differentiable root find with a user-supplied black-box rootfinder.

    Finds ``x`` with ``f(x) = 0`` by calling ``solver(f, x0)``, and registers
    the implicit-function-theorem derivative via :func:`jax.lax.custom_root`:
    ``dx/dtheta = -(df/dx)^{-1} df/dtheta`` for any parameters ``theta``
    closed over by ``f``. The rootfinder's internal iterations (e.g. a Newton
    loop with line search) are never differentiated.

    The signature mirrors :func:`jax.lax.custom_root`; parameters of ``f``
    beyond ``x`` should be closed over so gradients can flow to them.

    Args:
        f: function whose root is sought, ``f(x) -> residual`` with the same
            shape as ``x``.
        x0: initial guess passed to ``solver``.
        solver: callable ``solver(f, x0) -> x_root``, e.g. a Newton loop.
            Treated as a black box (never differentiated through), but must
            be traceable by JAX.
        tangent_solve: optional callable ``tangent_solve(g, y) -> x`` solving
            the linearised system ``g(x) = y``, where ``g`` is the Jacobian
            of ``f`` at the root as a linear map. Defaults to a scalar
            division for scalar problems and a dense
            :func:`jnp.linalg.solve` of the materialised Jacobian for small
            vector systems.

    Returns:
        The root ``x``, differentiable w.r.t. the parameters closed over by
        ``f``.
    """
    if tangent_solve is None:
        tangent_solve = _default_tangent_solve
    return jax.lax.custom_root(f, x0, solver, tangent_solve)
