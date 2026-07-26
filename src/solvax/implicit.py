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
- D. A. Knoll & D. E. Keyes, *Jacobian-free Newton--Krylov methods: a
  survey of approaches and applications*, J. Comput. Phys. 193, 357 (2004).
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
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from solvax.krylov import gmres

PyTree = Any
InnerProduct = Callable[[PyTree, PyTree], jax.Array]


class NewtonKrylovSolution(NamedTuple):
    """Result of :func:`newton_krylov`."""

    x: PyTree
    residual_norm: jax.Array
    newton_iterations: jax.Array
    linear_iterations: jax.Array
    converged: jax.Array
    linear_converged: jax.Array


def _tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    products = jax.tree.leaves(jax.tree.map(jnp.vdot, left, right))
    return sum(products[1:], products[0])


def _tree_norm(value: PyTree, inner_product: InnerProduct) -> jax.Array:
    return jnp.sqrt(jnp.maximum(jnp.real(inner_product(value, value)), 0.0))


def newton_krylov(
    residual_fn: Callable[[PyTree], PyTree],
    x0: PyTree,
    *,
    precond: Callable[[PyTree], PyTree] | None = None,
    inner_product: InnerProduct | None = None,
    norm: Callable[[PyTree], jax.Array] | None = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_steps: int = 20,
    linear_restart: int = 30,
    linear_rtol: float = 0.1,
    linear_atol: float = 0.0,
    linear_max_restarts: int = 10,
) -> NewtonKrylovSolution:
    """Solve a nonlinear system with matrix-free Newton--GMRES.

    Jacobian-vector products are obtained from :func:`jax.linearize`; the
    Jacobian is never materialised. The nonlinear iteration stops when

    ``norm(residual) <= max(atol, rtol * norm(initial_residual))``.

    Args:
        residual_fn: nonlinear residual with the same PyTree input/output structure.
        x0: initial iterate.
        precond: optional right preconditioner passed to GMRES.
        inner_product: optional GMRES inner product; useful for weighted or
            distributed PyTrees.
        norm: optional nonlinear residual norm. Defaults to the norm induced by
            ``inner_product``, or the Euclidean PyTree norm when both are omitted.
        rtol: nonlinear relative residual tolerance.
        atol: nonlinear absolute residual tolerance.
        max_steps: maximum Newton updates.
        linear_restart: GMRES Arnoldi cycle size.
        linear_rtol: GMRES relative residual tolerance for each Newton update.
        linear_atol: GMRES absolute residual tolerance.
        linear_max_restarts: maximum GMRES restart cycles per Newton update.

    Returns:
        A :class:`NewtonKrylovSolution` with the final iterate and diagnostics.
    """
    x0 = jax.tree.map(jnp.asarray, x0)
    inner_product = _tree_dot if inner_product is None else inner_product
    norm = (lambda value: _tree_norm(value, inner_product)) if norm is None else norm
    residual0 = residual_fn(x0)
    residual_norm0 = norm(residual0)
    tolerance = jnp.maximum(atol, rtol * residual_norm0)

    def cond_fun(state):
        return ~state[-1]

    def body_fun(state):
        x, _, newton_iterations, linear_iterations, linear_converged, _ = state
        residual, jvp = jax.linearize(residual_fn, x)
        residual_norm = norm(residual)
        update = (
            (residual_norm > tolerance)
            & (newton_iterations < max_steps)
            & linear_converged
        )

        def update_fn(_):
            linear_solution = gmres(
                jvp,
                jax.tree.map(jnp.negative, residual),
                precond=precond,
                inner_product=inner_product,
                restart=linear_restart,
                rtol=linear_rtol,
                atol=linear_atol,
                max_restarts=linear_max_restarts,
            )
            next_x = jax.tree.map(
                lambda value, step: value + step, x, linear_solution.x
            )
            return (
                next_x,
                jnp.asarray(jnp.inf, residual_norm.dtype),
                newton_iterations + 1,
                linear_iterations + linear_solution.iterations,
                linear_converged & linear_solution.converged,
                jnp.array(False),
            )

        def finish_fn(_):
            return (
                x,
                residual_norm,
                newton_iterations,
                linear_iterations,
                linear_converged,
                jnp.array(True),
            )

        return jax.lax.cond(update, update_fn, finish_fn, operand=None)

    finished0 = (residual_norm0 <= tolerance) | (max_steps == 0)
    initial = (
        x0,
        residual_norm0,
        jnp.int32(0),
        jnp.int32(0),
        jnp.array(True),
        finished0,
    )
    x, residual_norm, newton_iterations, linear_iterations, linear_converged, _ = (
        jax.lax.while_loop(cond_fun, body_fun, initial)
    )
    return NewtonKrylovSolution(
        x,
        residual_norm,
        newton_iterations,
        linear_iterations,
        residual_norm <= tolerance,
        linear_converged,
    )


def linear_solve(
    matvec: Callable,
    b: jax.Array,
    solver: Callable,
    *,
    transpose_matvec: Callable | None = None,
    transpose_solver: Callable | None = None,
    has_aux: bool = False,
) -> Any:
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
            Krylov method or a dense solve. With ``has_aux=True``, it must
            instead return ``(x, aux)``. It is treated as a black box (never
            differentiated through), but must be traceable by JAX.
        transpose_matvec: optional callable computing ``A^T @ y`` for the
            adjoint solve. If omitted, the transpose obtained from
            ``jax.linear_transpose`` of ``matvec`` is used.
        transpose_solver: optional solver for the transposed system. Defaults
            to ``solver``. With ``has_aux=True``, it must also return
            ``(x, aux)``.
        has_aux: whether both solver callbacks return auxiliary data alongside
            the solution. The forward auxiliary data is returned with ``x``.

    Returns:
        The solution ``x``, differentiable w.r.t. the parameters of
        ``matvec`` and ``b``. When ``has_aux=True``, returns ``(x, aux)``
        using the forward solver's auxiliary data.
    """
    transpose_solver = solver if transpose_solver is None else transpose_solver
    if transpose_matvec is None:
        # custom_linear_solve hands transpose_solve the linear transpose of
        # matvec (computed with jax.linear_transpose).
        def _transpose_solve(vecmat, y):
            return transpose_solver(vecmat, y)

    else:

        def _transpose_solve(vecmat, y):
            del vecmat  # user-supplied transpose is presumed cheaper/exact
            return transpose_solver(transpose_matvec, y)

    return jax.lax.custom_linear_solve(
        matvec,
        b,
        solve=solver,
        transpose_solve=_transpose_solve,
        has_aux=has_aux,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _recycled_solve(operator, solver, transpose_solver, params, b):
    """Primal of :func:`recycled_linear_solve` (the recycle pair is discarded)."""
    x, _ = solver(lambda value: operator(params, value), b, None)
    return x


def _recycled_solve_fwd(operator, solver, transpose_solver, params, b):
    x, recycle = solver(lambda value: operator(params, value), b, None)
    return x, (params, x, recycle)


def _recycled_solve_bwd(operator, solver, transpose_solver, residuals, cotangent):
    params, x, recycle = residuals

    def transpose(value):
        return jax.linear_transpose(lambda u: operator(params, u), x)(value)[0]

    # The adjoint system has the *same* operator up to transposition, so the
    # near-invariant subspace the forward solve paid to find is still the one
    # that limits convergence here (Parks et al. 2006, section 5).
    adjoint, _ = transpose_solver(transpose, cotangent, recycle)
    _, pullback = jax.vjp(lambda values: operator(values, x), params)
    return jax.tree.map(jnp.negative, pullback(adjoint)[0]), adjoint


_recycled_solve.defvjp(_recycled_solve_fwd, _recycled_solve_bwd)


def recycled_linear_solve(
    operator: Callable,
    params: PyTree,
    b: PyTree,
    solver: Callable,
    *,
    transpose_solver: Callable | None = None,
) -> PyTree:
    """Implicit-differentiation solve that recycles the forward Krylov space.

    Same implicit-function-theorem adjoint as :func:`linear_solve` — one
    transposed solve, nothing differentiated through the iteration — but the
    Krylov *recycle space* built during the forward solve is handed to the
    adjoint solve instead of being thrown away. The adjoint system
    ``A^T lambda = xbar`` has the same spectrum as the forward one, so the
    directions that limited forward convergence limit the adjoint too; a
    solve warm-started with them skips rediscovering the same subspace
    (Parks et al. 2006, section 5, recycling across a sequence of related
    systems).

    The parameters are explicit rather than closed over, because the
    forward's recycle pair has to reach the backward pass through the
    residual of a custom VJP. That is the only difference in usage::

        def operator(params, x):
            return params["diffusion"] * laplacian(x) + params["shift"] * x

        def solver(matvec, rhs, recycle):
            solution = gcrot(matvec, rhs, m=30, k=10, recycle=recycle)
            return solution.x, solution.recycle

        x = recycled_linear_solve(operator, params, b, solver)

    Args:
        operator: linear in its second argument, ``operator(params, x) -> A x``.
            Gradients flow to ``params`` and ``b``.
        params: pytree the operator depends on.
        b: right-hand side.
        solver: ``solver(matvec, b, recycle) -> (x, recycle)``. It is called
            with ``recycle=None`` for the forward solve and must return the
            recycle pair to hand on; a solver that cannot recycle may return
            ``None`` and still be correct, merely without the saving.
        transpose_solver: ``solver(matvec_T, y, recycle) -> (y, recycle)`` for
            the adjoint system. Defaults to ``solver``. It receives the
            forward pair, whose second element is ignored.

    Returns:
        The solution ``x``, differentiable with respect to ``params`` and
        ``b``. ``operator``, ``solver`` and ``transpose_solver`` are static:
        pass stable callables, not freshly built closures, to avoid
        retracing.
    """
    transpose_solver = solver if transpose_solver is None else transpose_solver
    return _recycled_solve(operator, solver, transpose_solver, params, b)


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
