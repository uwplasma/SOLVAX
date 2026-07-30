"""Implicit reverse-mode differentiation for application-solved eigenpairs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["eigenpair_reverse"]


PrimalSolver = Callable[
    [Any, Callable[[jax.Array], jax.Array], jax.Array],
    tuple[jax.Array, jax.Array],
]
LeftSolver = Callable[
    [Any, Callable[[jax.Array], jax.Array], jax.Array, complex],
    jax.Array,
]
TangentPreconditioner = Callable[
    [Any, complex, jax.Array, jax.Array],
    Callable[[jax.Array], jax.Array],
]
TransposeTangentSolver = Callable[
    [
        Any,
        complex,
        jax.Array,
        jax.Array,
        Callable[[jax.Array], jax.Array],
        jax.Array,
    ],
    jax.Array,
]


def eigenpair_reverse(
    theta: Any,
    build: Callable[[Any], Callable[[jax.Array], jax.Array]],
    v0: jax.Array,
    *,
    primal_solver: PrimalSolver,
    left_solver: LeftSolver,
    tangent_preconditioner: TangentPreconditioner | None = None,
    transpose_tangent_solver: TransposeTangentSolver | None = None,
    sensitivity_rtol: float = 1.0e-9,
    sensitivity_restart: int = 40,
    sensitivity_max_restarts: int = 20,
    condition_limit: float = 1.0e8,
) -> tuple[jax.Array, jax.Array]:
    """Differentiate an externally solved simple eigenpair in reverse mode.

    Applications retain control of the primal and adjoint eigensolvers.  SOLVAX
    differentiates their converged eigenpair implicitly, so no Krylov or
    timestepper iteration is recorded on the autodiff tape.

    ``primal_solver(theta, apply, v0)`` returns ``(lambda, right)`` and
    ``left_solver(theta, apply, v0, lambda)`` returns a matching left
    eigenvector.  The pair is normalized internally to ``leftᴴ right = 1``.
    Eigenvalue cotangents use ``leftᴴ (dA) right``.  Eigenvector cotangents use
    Nelson's bordered reduced resolvent and therefore require one transposed
    linear solve.

    ``transpose_tangent_solver`` may replace generic GMRES with an
    application-specific solver.  ``tangent_preconditioner`` supplies an
    approximate inverse for the bordered operator; its linear transpose is
    used automatically in the pullback.

    A large ``||left|| ||right|| / |leftᴴ right|`` signals a nearly defective
    eigenvalue.  The function rejects that case because a single-eigenvector
    derivative is not trustworthy at an exceptional point.
    """

    if sensitivity_rtol <= 0.0:
        raise ValueError("sensitivity_rtol must be positive")
    if sensitivity_restart < 1 or sensitivity_max_restarts < 1:
        raise ValueError("sensitivity iteration limits must be positive")
    if condition_limit <= 1.0:
        raise ValueError("condition_limit must exceed one")
    restart = min(int(v0.size), int(sensitivity_restart))

    def solve_pair(parameters):
        apply = build(parameters)
        value, right = primal_solver(parameters, apply, v0)
        value = jnp.asarray(value)
        right = jnp.asarray(right)
        left = jnp.asarray(
            left_solver(parameters, apply, v0, complex(value)),
        )
        overlap = jnp.vdot(left, right)
        overlap_abs = abs(complex(overlap))
        condition = float(
            jnp.linalg.norm(left) * jnp.linalg.norm(right) / max(overlap_abs, np.finfo(float).tiny)
        )
        if not np.isfinite(condition) or condition > condition_limit:
            raise ValueError(
                "eigenpair sensitivity is ill-conditioned: "
                f"condition number {condition:.3e} exceeds "
                f"{condition_limit:.3e}; differentiate an invariant subspace "
                "or smooth the branch selection"
            )
        return value, right, left / jnp.conj(overlap)

    @jax.custom_vjp
    def pair(parameters):
        value, right, _left = solve_pair(parameters)
        return value, right

    def pair_fwd(parameters):
        value, right, left = solve_pair(parameters)
        return (value, right), (parameters, value, right, left)

    def pair_bwd(residual, cotangents):
        parameters, value, right, left = residual
        value_cotangent, vector_cotangent = cotangents
        apply = build(parameters)

        def bordered(vector):
            return apply(vector) - value * vector + right * jnp.vdot(left, vector)

        preconditioner = (
            None
            if tangent_preconditioner is None
            else tangent_preconditioner(parameters, complex(value), right, left)
        )
        if preconditioner is None:
            transpose_preconditioner = None
        else:
            transpose_action = jax.linear_transpose(
                preconditioner,
                jnp.zeros_like(v0),
            )

            def transpose_preconditioner(vector):
                return transpose_action(vector)[0]

        from solvax.implicit import linear_solve
        from solvax.krylov import gmres

        def gmres_solve(matvec, rhs, *, precond):
            solution = gmres(
                matvec,
                rhs,
                precond=precond,
                restart=restart,
                rtol=sensitivity_rtol,
                max_restarts=sensitivity_max_restarts,
            )
            return jax.tree.map(
                lambda leaf: jnp.where(
                    solution.converged,
                    leaf,
                    jnp.full_like(leaf, jnp.nan),
                ),
                solution.x,
            )

        def solve(matvec, rhs):
            return gmres_solve(matvec, rhs, precond=preconditioner)

        def transpose_solve(matvec, rhs):
            if transpose_tangent_solver is not None:
                return transpose_tangent_solver(
                    parameters,
                    complex(value),
                    right,
                    left,
                    matvec,
                    rhs,
                )
            return gmres_solve(
                matvec,
                rhs,
                precond=transpose_preconditioner,
            )

        def tangent_from_operator_image(image):
            value_tangent = jnp.vdot(left, image)
            vector_tangent = linear_solve(
                bordered,
                value_tangent * right - image,
                solve,
                transpose_solver=transpose_solve,
            )
            return value_tangent, vector_tangent

        _zero, image_pullback = jax.vjp(
            tangent_from_operator_image,
            jnp.zeros_like(right),
        )
        (image_cotangent,) = image_pullback((value_cotangent, vector_cotangent))
        _image, parameter_pullback = jax.vjp(
            lambda p: build(p)(right),
            parameters,
        )
        (parameter_cotangent,) = parameter_pullback(image_cotangent)
        return (parameter_cotangent,)

    pair.defvjp(pair_fwd, pair_bwd)
    return pair(theta)
