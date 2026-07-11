"""Matrix-free preconditioned conjugate gradients on JAX pytrees."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

PyTree = Any
MatVec = Callable[[PyTree], PyTree]
RUNNING = 0
CONVERGED = 1
MAX_ITERATIONS = 2
NON_POSITIVE_CURVATURE = 3
NONFINITE = 4
PRECONDITIONER_BREAKDOWN = 5

STATUS_NAMES = {
    RUNNING: "running",
    CONVERGED: "converged",
    MAX_ITERATIONS: "max_iterations",
    NON_POSITIVE_CURVATURE: "non_positive_curvature",
    NONFINITE: "nonfinite",
    PRECONDITIONER_BREAKDOWN: "preconditioner_breakdown",
}


class PCGSolution(NamedTuple):
    """Result of :func:`pcg` with fixed-shape, JIT-safe diagnostics."""

    x: PyTree
    residual_norm: jax.Array
    relative_residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    status: jax.Array
    residual_history: jax.Array


class PCGDiagnostics(NamedTuple):
    """Fixed-shape PCG diagnostics carried through an implicit solve."""

    residual_norm: jax.Array
    relative_residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    status: jax.Array
    residual_history: jax.Array


def status_name(status: int | jax.Array) -> str:
    """Return the host-readable name for a materialized PCG status code."""

    code = int(status)
    if code not in STATUS_NAMES:
        raise ValueError(f"Unknown PCG status code {code}")
    return STATUS_NAMES[code]


def _tree_add_scaled(left: PyTree, scale: jax.Array, right: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: x + scale * y, left, right)


def _tree_sub(left: PyTree, right: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    products = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.vdot(x, y), left, right))
    return jnp.real(sum(products[1:], products[0]))


def _tree_norm(value: PyTree) -> jax.Array:
    return jnp.sqrt(jnp.maximum(_tree_dot(value, value), 0.0))


def _identity(value: PyTree) -> PyTree:
    return value


def pcg(
    matvec: MatVec,
    b: PyTree,
    *,
    x0: PyTree | None = None,
    precond: MatVec | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    max_steps: int = 500,
) -> PCGSolution:
    """Solve a symmetric positive-definite pytree system with matrix-free PCG.

    ``matvec`` and ``precond`` operate on a pytree with the same structure as
    ``b``. The residual history always has shape ``(max_steps + 1,)``; entries
    after termination repeat the final norm, which keeps the result compatible
    with ``jit`` and ``vmap`` without dynamic allocation.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("PCG tolerances must be non-negative")
    leaves = jax.tree.leaves(b)
    if not leaves:
        raise ValueError("PCG right-hand side must contain at least one array leaf")
    dtype = jnp.result_type(*[jnp.asarray(leaf).dtype for leaf in leaves], jnp.float32)
    scalar_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
    b = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype=dtype), b)
    if x0 is None:
        x0 = jax.tree.map(jnp.zeros_like, b)
    else:
        if jax.tree.structure(x0) != jax.tree.structure(b):
            raise ValueError("x0 and b must have identical pytree structure")
        x0 = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype=dtype), x0)
    precond = _identity if precond is None else precond

    b_norm = _tree_norm(b)
    tolerance = jnp.maximum(
        jnp.asarray(atol, scalar_dtype),
        jnp.asarray(rtol, scalar_dtype) * b_norm,
    )
    residual0 = _tree_sub(b, matvec(x0))
    residual_norm0 = _tree_norm(residual0)
    z0 = precond(residual0)
    rho0 = _tree_dot(residual0, z0)
    tiny = jnp.asarray(jnp.finfo(scalar_dtype).tiny, dtype=scalar_dtype)
    finite0 = jnp.isfinite(residual_norm0) & jnp.isfinite(rho0)
    status0 = jnp.where(
        ~finite0,
        NONFINITE,
        jnp.where(
            residual_norm0 <= tolerance,
            CONVERGED,
            jnp.where(rho0 > 0.0, RUNNING, PRECONDITIONER_BREAKDOWN),
        ),
    ).astype(jnp.int32)
    history0 = jnp.full((max_steps + 1,), residual_norm0, dtype=scalar_dtype)

    def cond_fun(state):
        count, _, _, _, _, _, _, status, _ = state
        return (count < max_steps) & (status == RUNNING)

    def body_fun(state):
        count, x, residual, z, direction, rho, _, _, history = state
        applied = matvec(direction)
        curvature = _tree_dot(direction, applied)
        curvature_finite = jnp.isfinite(curvature)
        curvature_ok = curvature > 0.0
        safe_curvature = jnp.where(curvature_ok, curvature, 1.0)
        alpha = jnp.where(curvature_ok, rho / safe_curvature, 0.0)
        x_next = _tree_add_scaled(x, alpha, direction)
        residual_next = _tree_add_scaled(residual, -alpha, applied)
        residual_norm = _tree_norm(residual_next)
        z_next = precond(residual_next)
        rho_next = _tree_dot(residual_next, z_next)
        finite = (
            curvature_finite
            & jnp.isfinite(alpha)
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(rho_next)
        )
        rho_ok = rho_next > 0.0
        safe_rho = jnp.where(rho > 0.0, rho, 1.0)
        beta = jnp.where(rho_ok, rho_next / safe_rho, 0.0)
        direction_next = _tree_add_scaled(z_next, beta, direction)
        status = jnp.where(
            ~finite,
            NONFINITE,
            jnp.where(
                ~curvature_ok,
                NON_POSITIVE_CURVATURE,
                jnp.where(
                    residual_norm <= tolerance,
                    CONVERGED,
                    jnp.where(rho_ok, RUNNING, PRECONDITIONER_BREAKDOWN),
                ),
            ),
        ).astype(jnp.int32)
        history = history.at[count + 1].set(residual_norm)
        return (
            count + 1,
            x_next,
            residual_next,
            z_next,
            direction_next,
            rho_next,
            residual_norm,
            status,
            history,
        )

    initial = (
        jnp.int32(0),
        x0,
        residual0,
        z0,
        z0,
        rho0,
        residual_norm0,
        status0,
        history0,
    )
    iterations, x, _, _, _, _, residual_norm, status, history = lax.while_loop(
        cond_fun, body_fun, initial
    )
    status = jnp.where(status == RUNNING, MAX_ITERATIONS, status).astype(jnp.int32)
    history = jnp.where(
        jnp.arange(max_steps + 1) <= iterations,
        history,
        residual_norm,
    )
    relative = residual_norm / jnp.maximum(b_norm, tiny)
    return PCGSolution(
        x=x,
        residual_norm=residual_norm,
        relative_residual_norm=relative,
        iterations=iterations,
        converged=status == CONVERGED,
        status=status,
        residual_history=history,
    )


def _diagnostics(solution: PCGSolution) -> PCGDiagnostics:
    return PCGDiagnostics(*solution[1:])


def pcg_linear_solve(
    matvec: MatVec,
    b: PyTree,
    *,
    x0: PyTree | None = None,
    precond: MatVec | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    max_steps: int = 500,
    transpose_precond: MatVec | None = None,
    transpose_rtol: float | None = None,
    transpose_atol: float | None = None,
    transpose_max_steps: int | None = None,
) -> PCGSolution:
    """Implicitly differentiable PCG with retained forward diagnostics.

    The primal and transpose solves use :func:`jax.lax.custom_linear_solve`, so
    derivatives do not trace through iteration-count branches. The operator is
    assumed Hermitian positive definite; the transpose preconditioner defaults
    to the primal preconditioner, while transpose tolerances may be controlled
    independently.
    """

    adjoint_precond = precond if transpose_precond is None else transpose_precond
    adjoint_rtol = rtol if transpose_rtol is None else transpose_rtol
    adjoint_atol = atol if transpose_atol is None else transpose_atol
    adjoint_steps = max_steps if transpose_max_steps is None else transpose_max_steps

    def solve(operator: MatVec, value: PyTree):
        solution = pcg(
            operator,
            value,
            x0=x0,
            precond=precond,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )
        return solution.x, _diagnostics(solution)

    def transpose_solve(operator: MatVec, value: PyTree):
        solution = pcg(
            operator,
            value,
            precond=adjoint_precond,
            rtol=adjoint_rtol,
            atol=adjoint_atol,
            max_steps=adjoint_steps,
        )
        return solution.x, _diagnostics(solution)

    x, diagnostics = jax.lax.custom_linear_solve(
        matvec,
        b,
        solve=solve,
        transpose_solve=transpose_solve,
        has_aux=True,
    )
    return PCGSolution(x, *diagnostics)


__all__ = [
    "PCGSolution",
    "PCGDiagnostics",
    "RUNNING",
    "CONVERGED",
    "MAX_ITERATIONS",
    "NON_POSITIVE_CURVATURE",
    "NONFINITE",
    "PRECONDITIONER_BREAKDOWN",
    "STATUS_NAMES",
    "pcg",
    "pcg_linear_solve",
    "status_name",
]
