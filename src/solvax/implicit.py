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
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp

from solvax.krylov import gmres
from solvax.nonlinear import eisenstat_walker_forcing

PyTree = Any
InnerProduct = Callable[[PyTree, PyTree], jax.Array]
ForcingPolicy = Literal["constant", "eisenstat_walker"]


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
    squared = jnp.maximum(jnp.real(inner_product(value, value)), 0.0)
    # Preserve NaN instead of turning a failed residual into an exact root.
    # The guarded zero branch still has a finite derivative at the origin.
    nonzero = squared != 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, squared, 1.0)), 0.0)


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
    forcing: ForcingPolicy = "constant",
    forcing_gamma: float = 0.9,
    forcing_alpha: float = 2.0,
    forcing_max: float = 0.9,
    fixed_work: bool = False,
) -> NewtonKrylovSolution:
    """Solve a nonlinear system with matrix-free Newton--GMRES.

    Jacobian-vector products are obtained from :func:`jax.linearize`; the
    Jacobian is never materialised. The nonlinear iteration stops when

    ``norm(residual) <= max(atol, rtol * norm(initial_residual))``.

    Nonfinite residual norms or stopping thresholds terminate with ``converged=False``.
    Such a result must not be used as a certified implicit differentiation root.

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
        forcing: inner-tolerance policy. ``"constant"`` uses ``linear_rtol``
            for every Newton update. ``"eisenstat_walker"`` uses
            ``linear_rtol`` for the first update and safeguarded
            Eisenstat--Walker choice 2 thereafter.
        forcing_gamma: residual-ratio scale for Eisenstat--Walker choice 2.
        forcing_alpha: residual-ratio exponent for Eisenstat--Walker choice 2.
        forcing_max: strict upper bound on the adaptive forcing term.
        fixed_work: execute all ``max_steps`` Newton slots and all configured
            inner GMRES slots, masking updates after convergence. This uses
            reverse-mode-compatible ``scan`` control flow for bounded-cost
            embedding in an outer scan. The default early-exit path is
            unchanged.

    Returns:
        A :class:`NewtonKrylovSolution` with the final iterate and diagnostics.

    References:
        S. C. Eisenstat and H. F. Walker, "Choosing the Forcing Terms in an
        Inexact Newton Method", SIAM J. Sci. Comput. 17, 16--32 (1996),
        https://doi.org/10.1137/0917003.
    """
    if forcing not in ("constant", "eisenstat_walker"):
        raise ValueError(
            "forcing must be 'constant' or 'eisenstat_walker', "
            f"got {forcing!r}"
        )
    if forcing == "eisenstat_walker":
        if not 0.0 <= linear_rtol < 1.0:
            raise ValueError("linear_rtol must lie in [0, 1) for adaptive forcing")
        if not 0.0 <= forcing_gamma < 1.0:
            raise ValueError("forcing_gamma must lie in [0, 1)")
        if not 1.0 < forcing_alpha <= 2.0:
            raise ValueError("forcing_alpha must lie in (1, 2]")
        if not 0.0 < forcing_max < 1.0:
            raise ValueError("forcing_max must lie in (0, 1)")

    x0 = jax.tree.map(jnp.asarray, x0)
    inner_product = _tree_dot if inner_product is None else inner_product
    norm = (lambda value: _tree_norm(value, inner_product)) if norm is None else norm
    residual0 = residual_fn(x0)
    residual_norm0 = norm(residual0)
    tolerance = jnp.maximum(atol, rtol * residual_norm0)

    def cond_fun(state):
        return ~state[-1]

    def body_fun(state):
        (
            x,
            previous_residual_norm,
            previous_forcing,
            newton_iterations,
            linear_iterations,
            linear_converged,
            _,
        ) = state
        residual, jvp = jax.linearize(residual_fn, x)
        residual_norm = norm(residual)
        update = (
            (residual_norm > tolerance)
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(tolerance)
            & (newton_iterations < max_steps)
            & linear_converged
        )

        def update_fn(_):
            if forcing == "eisenstat_walker":
                adaptive_forcing = eisenstat_walker_forcing(
                    residual_norm,
                    previous_residual_norm,
                    previous_forcing,
                    gamma=forcing_gamma,
                    power=forcing_alpha,
                    eta_min=0.0,
                    eta_max=forcing_max,
                )
                current_forcing = jnp.where(
                    newton_iterations == 0,
                    jnp.asarray(linear_rtol, residual_norm.dtype),
                    adaptive_forcing,
                )
            else:
                current_forcing = jnp.asarray(linear_rtol, residual_norm.dtype)

            linear_solution = gmres(
                jvp,
                jax.tree.map(jnp.negative, residual),
                precond=precond,
                inner_product=inner_product,
                restart=linear_restart,
                rtol=current_forcing,
                atol=linear_atol,
                max_restarts=linear_max_restarts,
                fixed_work=fixed_work,
            )
            next_x = jax.tree.map(
                lambda value, step: value + step, x, linear_solution.x
            )
            return (
                next_x,
                residual_norm,
                current_forcing,
                newton_iterations + 1,
                linear_iterations + linear_solution.iterations,
                linear_converged & linear_solution.converged,
                jnp.array(False),
            )

        def finish_fn(_):
            return (
                x,
                residual_norm,
                previous_forcing,
                newton_iterations,
                linear_iterations,
                linear_converged,
                jnp.array(True),
            )

        return jax.lax.cond(update, update_fn, finish_fn, operand=None)

    finished0 = (
        (residual_norm0 <= tolerance) | (max_steps == 0)
        | ~jnp.isfinite(residual_norm0) | ~jnp.isfinite(tolerance)
    )
    initial = (
        x0,
        residual_norm0,
        jnp.asarray(linear_rtol, residual_norm0.dtype),
        jnp.int32(0),
        jnp.int32(0),
        jnp.array(True),
        finished0,
    )
    if fixed_work:
        def scan_body(state, _):
            return body_fun(state), None

        final, _ = jax.lax.scan(scan_body, initial, xs=None, length=max_steps)
    else:
        final = jax.lax.while_loop(cond_fun, body_fun, initial)
    (
        x,
        residual_norm,
        _,
        newton_iterations,
        linear_iterations,
        linear_converged,
        _,
    ) = final
    if fixed_work:
        residual_norm = norm(residual_fn(x))
    return NewtonKrylovSolution(
        x,
        residual_norm,
        newton_iterations,
        linear_iterations,
        jnp.isfinite(residual_norm) & jnp.isfinite(tolerance) & (residual_norm <= tolerance),
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


DENSE_TANGENT_LIMIT = 1024
"""Largest vector size whose default root tangent forms the dense Jacobian."""


def _default_tangent_solve(
    g: Callable,
    y: jax.Array,
    dense_limit: int = DENSE_TANGENT_LIMIT,
    *,
    precond: tuple[Callable | None, Callable | None] | None = None,
    rtol: float | None = None,
    restart: int | None = None,
    max_restarts: int = 40,
) -> jax.Array:
    """Solve the linearised system ``g(x) = y`` for the root tangent.

    ``g`` is the (linear) Jacobian-vector product of ``f`` at the root. For
    scalars a single division suffices. Up to ``dense_limit`` unknowns, and
    without ``precond``, the dense Jacobian is materialised and solved with
    :func:`jnp.linalg.solve`; otherwise restarted GMRES on ``g`` runs
    matrix-free (memory ``O(n m)`` for a cycle of ``m``) to a relative
    residual of ``rtol`` (default ``max(1e-10, 100 eps)``), right
    preconditioned by ``M`` and its transposed system by ``M_T`` from
    ``precond = (M, M_T)``, and the tangent is NaN if it does not converge.
    ``M_T=None`` with a given ``M`` uses ``jax.linear_transpose(M)``. A
    complex vector is solved as the real system of twice the size, so ``g``
    (and ``M``) need only be real-linear (a residual with ``conj`` is
    handled); a complex scalar uses the division and requires a holomorphic
    residual.
    """
    y = jnp.asarray(y)
    if y.ndim == 0:
        return y / g(jnp.ones_like(y))
    options = dict(rtol=rtol, restart=restart, max_restarts=max_restarts)
    if jnp.iscomplexobj(y):
        # Solve the real 2n system, valid for any R-linear g (holomorphic or not).
        n = y.size

        def split(w: jax.Array) -> jax.Array:
            return jnp.concatenate([w.real.ravel(), w.imag.ravel()])

        def real(fn: Callable | None) -> Callable | None:
            if fn is None:
                return None
            return lambda u: split(fn((u[:n] + 1j * u[n:]).reshape(y.shape)))

        pair = None if precond is None else (real(precond[0]), real(precond[1]))
        v = _default_tangent_solve(
            real(g), split(y), 2 * dense_limit, precond=pair, **options
        )
        return (v[:n] + 1j * v[n:]).reshape(y.shape)
    if precond is None and y.size <= dense_limit:
        jac = jax.jacobian(g)(jnp.zeros_like(y))
        return jnp.linalg.solve(jac, y)
    if rtol is None:
        rtol = max(1e-10, 100 * float(jnp.finfo(y.dtype).eps))
    m, m_t = (None, None) if precond is None else precond
    if m is not None and m_t is None:

        def m_t(v: jax.Array) -> jax.Array:
            try:
                return jax.linear_transpose(m, v)(v)[0]
            except NotImplementedError as error:
                raise ValueError(
                    "tangent_precond: M is not linear, so M_T=None cannot "
                    "transpose it; pass M_T"
                ) from error

    def krylov(pc: Callable | None, matvec: Callable, b: jax.Array) -> jax.Array:
        sol = gmres(
            matvec, b, precond=pc, rtol=rtol, max_restarts=max_restarts,
            restart=min(50, b.size) if restart is None else restart,
        )
        return jnp.where(sol.converged, sol.x, jnp.nan)

    return jax.lax.custom_linear_solve(
        g, y, partial(krylov, m), transpose_solve=partial(krylov, m_t)
    )


def root_solve(
    f: Callable,
    x0: jax.Array,
    solver: Callable,
    *,
    tangent_solve: Callable | None = None,
    dense_limit: int = DENSE_TANGENT_LIMIT,
    tangent_precond: tuple | Callable | None = None,
    tangent_rtol: float | None = None,
    tangent_restart: int | None = None,
    tangent_max_restarts: int = 40,
    has_aux: bool = False,
) -> Any:
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
            be traceable by JAX. With ``has_aux=True`` it returns
            ``(x_root, aux)``.
        tangent_solve: optional callable ``tangent_solve(g, y) -> x`` solving
            the linearised system ``g(x) = y``, where ``g`` is the Jacobian
            of ``f`` at the root as a linear map. Defaults to a scalar
            division for scalar problems, a dense :func:`jnp.linalg.solve`
            of the materialised Jacobian up to ``dense_limit`` unknowns, and
            matrix-free restarted GMRES above it, whose tangent is NaN when
            GMRES does not reach a relative residual of ``tangent_rtol``
            (default ``max(1e-10, 100 eps)``). The limit is not a cost crossover; it keeps
            the dense solve, which never gives NaN on a nonsingular
            Jacobian, where it is affordable: unpreconditioned GMRES is
            cheaper on a well-conditioned Jacobian but returns NaN on a hard one (even a non-normal
            Jacobian of condition 1e2), where the remedy is
            ``dense_limit=n``, ``tangent_precond`` or a ``tangent_solve``. Complex
            vector roots may be non-holomorphic (``conj`` in ``f``): they
            are solved as the real system of size ``2n``. Complex scalar
            roots require a holomorphic ``f``.
        dense_limit: largest number of unknowns for which the default
            tangent forms the dense ``n x n`` Jacobian (static). Ignored
            when ``tangent_solve`` or ``tangent_precond`` is given.
        tangent_precond: right preconditioner of the default GMRES tangent,
            which it then selects at every size: a pair ``(M, M_T)`` of
            callables ``v -> M^{-1} v`` and ``v -> M^{-T} v`` (``M_T`` for
            the transposed system of reverse mode), or a factory
            ``x_root -> (M, M_T)`` called once with the root, so a
            preconditioner built from the root costs no second solve.
            ``M_T=None`` takes ``jax.linear_transpose`` of ``M``, which
            must then be linear. ``M`` acts on arrays shaped like ``x``;
            for a complex ``x`` it must be real-linear and ``M_T`` is its
            adjoint (conjugate transpose), since the tangent is the real
            ``2n`` system. Exclusive with ``tangent_solve``.
        tangent_rtol: GMRES relative residual of the default tangent.
        tangent_restart: GMRES cycle (default ``min(50, size)``).
        tangent_max_restarts: GMRES cycles before the tangent is NaN.
        has_aux: ``solver`` returns ``(x_root, aux)`` and so does this
            function; ``aux`` (iterations, residual, status) carries a zero
            derivative. Before JAX 0.7 an integer or boolean ``aux`` fails
            (in ``jax.lax.custom_root``): give it a floating dtype.

    Returns:
        The root ``x``, differentiable w.r.t. the parameters closed over by
        ``f``, or ``(x, aux)`` with ``has_aux=True``.
    """
    if tangent_solve is not None and tangent_precond is not None:
        raise ValueError("pass tangent_solve or tangent_precond, not both")
    options = dict(
        rtol=tangent_rtol, restart=tangent_restart,
        max_restarts=tangent_max_restarts,
    )
    if tangent_precond is None:
        if tangent_solve is None:
            tangent_solve = partial(
                _default_tangent_solve, dense_limit=dense_limit, **options
            )
        return jax.lax.custom_root(f, x0, solver, tangent_solve, has_aux=has_aux)
    factory = (
        tangent_precond if callable(tangent_precond)
        else lambda _: tuple(tangent_precond)
    )
    # Solve once for the root, build the pair there, then let custom_root
    # differentiate (to any order) with a no-op solver returning that root.
    # The pair enters custom_root's tangent as a constant, so derivatives of
    # M (and of values it closes over) are not taken: a converged tangent
    # does not depend on its preconditioner.
    found = jax.lax.stop_gradient(
        jax.lax.custom_root(f, x0, solver, lambda g, y: y, has_aux=has_aux)
    )
    precond = factory(found[0] if has_aux else found)
    tangent = partial(_default_tangent_solve, precond=precond, **options)
    return jax.lax.custom_root(f, x0, lambda *_: found, tangent, has_aux=has_aux)


ROOT_CONVERGED = 0
"""Status of :func:`certified_root`: the residual reached ``tol``."""
ROOT_MAX_ITER = 1
"""Status of :func:`certified_root`: ``max_iter`` steps without reaching ``tol``."""
ROOT_STAGNATED = 2
"""Status of :func:`certified_root`: no step lowers the residual (or no sign change)."""


class RootResult(NamedTuple):
    """Result of :func:`certified_root`.

    ``root`` is differentiable; its derivative is NaN when ``converged`` is
    false. ``residual`` is ``max |f(root)|``. ``iterations`` is ``-1`` for a
    user ``solver``, which does not report it.
    """

    root: jax.Array
    residual: jax.Array
    iterations: jax.Array
    converged: jax.Array
    status: jax.Array


@jax.custom_jvp
def _nan_unless(x: jax.Array, ok: jax.Array) -> jax.Array:
    return x


@_nan_unless.defjvp
def _nan_unless_jvp(primals, tangents):
    x, ok = primals
    return x, tangents[0] * jnp.where(ok, 1.0, jnp.nan).astype(x.dtype)


def _value_and_slope(f, x):
    return jax.jvp(f, (x,), (jnp.ones_like(x),))


def _scalar_newton(f, x0, bracket, accept, max_iter, damped):
    """Newton for a scalar root: rtsafe bisection inside ``bracket``, else halving on ``|f|``.

    ``accept(x, fx, dfx)`` is the stopping test. Inside a bracket a Newton
    step is replaced by bisection when it leaves the sign change or is not
    at most half the previous step (Press et al., Numerical Recipes, rtsafe),
    so the iteration count is bounded by that of bisection.
    """

    def step(x, fx, dfx, lo, hi, dxold):
        dx = fx / dfx
        xn = x - dx
        if bracket is not None:
            a, b = jnp.minimum(lo, hi), jnp.maximum(lo, hi)
            newton = jnp.isfinite(xn) & (xn > a) & (xn < b)
            newton = newton & (2.0 * jnp.abs(dx) <= jnp.abs(dxold))
            xn = jnp.where(newton, xn, 0.5 * (lo + hi))
            fn, dfn = _value_and_slope(f, xn)
            return xn, fn, dfn, jnp.abs(xn - x), True
        if damped:

            def halve(c):
                t = c[0] * 0.5
                return t, f(x - t * dx)

            def worse(c):
                t, fn = c
                return (t > 2.0**-30) & ~(jnp.abs(fn) < jnp.abs(fx))

            one = jnp.ones((), jnp.abs(x).dtype)
            t, fn = jax.lax.while_loop(worse, halve, (one, f(xn)))
            xn = x - t * dx
        fn, dfn = _value_and_slope(f, xn)
        ok = jnp.isfinite(xn) & jnp.isfinite(fn) & (jnp.abs(fn) < jnp.abs(fx))
        if not damped:
            ok = jnp.isfinite(xn) & jnp.isfinite(fn)
        # Keep the last finite iterate when the step fails (f' = 0, overflow).
        keep = ~ok
        xn = jnp.where(keep, x, xn)
        return xn, jnp.where(keep, fx, fn), jnp.where(keep, dfx, dfn), dxold, ok

    def body(c):
        x, fx, dfx, lo, hi, dxold, it, _ = c
        xn, fn, dfn, dxold, ok = step(x, fx, dfx, lo, hi, dxold)
        if bracket is not None:
            same = jnp.sign(fn) == jnp.sign(f(lo))
            lo, hi = jnp.where(same, xn, lo), jnp.where(same, hi, xn)
        status = jnp.where(accept(xn, fn, dfn), ROOT_CONVERGED, jnp.where(ok, -1, ROOT_STAGNATED))
        return xn, fn, dfn, lo, hi, dxold, it + 1, status

    def cond(c):
        return (c[7] < 0) & (c[6] < max_iter)

    status0 = jnp.asarray(-1)
    dxold = jnp.asarray(jnp.inf, jnp.abs(x0).dtype)
    lo = hi = x0
    if bracket is not None:
        lo, hi = (jnp.asarray(b, x0.dtype) for b in bracket)
        status0 = jnp.where(jnp.sign(f(lo)) == jnp.sign(f(hi)), ROOT_STAGNATED, status0)
        out_of = (x0 < jnp.minimum(lo, hi)) | (x0 > jnp.maximum(lo, hi))
        x0 = jnp.where(out_of, 0.5 * (lo + hi), x0)
        dxold = jnp.abs(hi - lo)
    fx0, dfx0 = _value_and_slope(f, x0)
    status0 = jnp.where((status0 < 0) & accept(x0, fx0, dfx0), ROOT_CONVERGED, status0)
    c = (x0, fx0, dfx0, lo, hi, dxold, jnp.zeros((), jnp.int32), status0)
    x, fx, dfx, lo, hi, dxold, it, status = jax.lax.while_loop(cond, body, c)
    # One more Newton step, kept if it lowers |f|: the stopping test alone
    # leaves the root at tol / |f'| when Newton has just entered its basin.
    xp, fp, _, _, _ = step(x, fx, dfx, lo, hi, jnp.asarray(jnp.inf, dxold.dtype))
    x = jnp.where(jnp.isfinite(fp) & (jnp.abs(fp) < jnp.abs(fx)), xp, x)
    # Float aux: JAX before 0.7 gives custom_root's integer aux an int tangent.
    status = jnp.where(status < 0, ROOT_MAX_ITER, status)
    return x, jnp.stack([it, status]).astype(jnp.abs(x).dtype)


def _complex_scalar_tangent(g, y, dense_limit=DENSE_TANGENT_LIMIT):
    """Complex scalar tangent as the real 2 x 2 system (valid without holomorphy)."""
    return _default_tangent_solve(lambda u: g(u[0])[None], y[None], dense_limit)[0]


def certified_root(
    f: Callable,
    x0: jax.Array,
    *,
    solver: Callable | None = None,
    bracket: tuple | None = None,
    tol: jax.Array | float | None = None,
    max_iter: int = 100,
    damped: bool = True,
    tangent_solve: Callable | None = None,
    dense_limit: int = DENSE_TANGENT_LIMIT,
) -> RootResult:
    """Root of ``f`` with its residual and status; the derivative is NaN unless converged.

    The root is differentiated by the implicit rule of :func:`root_solve`.
    Unlike :func:`root_solve`, a solver that stops short (for instance one
    that returns its start) is reported, and the derivative of an
    unconverged root is NaN.

    Certification is by residual. A function with no root can be reported
    converged where ``|f|`` falls below the tolerance (``exp(z)`` far to the
    left). A double root converges to about the square root of the
    tolerance and its slope is unbounded. Arrays only, no pytrees. Under
    ``vmap`` the batch iterates until its slowest member stops.

    Args:
        f: residual, ``f(x)`` of the shape of ``x``. The built-in Newton
            iteration on a complex scalar assumes a holomorphic ``f``; the
            derivative of the root is correct for any ``f`` (it is solved as
            the real 2 x 2 system).
        x0: start.
        solver: optional ``solver(f, x0) -> x`` for arrays of any shape; it
            then needs an explicit ``tol``. When omitted, ``x0`` must be a
            scalar and a built-in Newton iteration runs: inside ``bracket``
            (real roots), bisection replaces a Newton step that leaves the
            current sign change or is not at most half the previous step, so
            it needs no more steps than bisection; without ``bracket`` (real
            or complex), the step is halved until ``|f|`` decreases, or taken
            whole when ``damped=False``. Usable under ``jit`` and ``vmap``.
        bracket: optional ``(a, b)`` with ``f(a) f(b) < 0`` (real scalars only).
            Without a sign change the status is ``ROOT_STAGNATED``. A start
            outside it is moved to its midpoint.
        tol: absolute bound on ``max |f(root)|``; may be a traced value.
            Required with ``solver``. For the built-in solver the default
            certifies the Newton correction at the returned root,
            ``|f| <= 100 eps max(1, |x|) |f'|``; where ``f'`` is zero or not
            finite the root is converged only if ``f(x) == 0``.
        max_iter: built-in iteration limit (static).
        damped: halve Newton steps that do not lower ``|f|`` (no bracket).
        tangent_solve, dense_limit: as in :func:`root_solve`.

    Returns:
        :class:`RootResult` with ``status`` one of ``ROOT_CONVERGED``,
        ``ROOT_MAX_ITER``, ``ROOT_STAGNATED``. ``residual`` is differentiable:
        its derivative is that of ``|f|`` in the parameters at the fixed root.
    """
    x0 = jnp.asarray(x0)
    if bracket is not None and (jnp.iscomplexobj(x0) or x0.ndim):
        raise ValueError("bracket needs a real scalar start")
    if solver is not None and tol is None:
        raise ValueError("certified_root with solver= needs an explicit tol on max |f(root)|")
    if solver is None and x0.ndim:
        raise ValueError("the built-in solver is scalar; pass solver= for a vector root")
    if tol is None:
        eps = 100 * jnp.finfo(x0.dtype).eps

        def accept(x, fx, dfx):
            scale = eps * jnp.maximum(1.0, jnp.abs(x)) * jnp.abs(dfx)
            return (fx == 0) | (jnp.isfinite(dfx) & (jnp.abs(fx) <= scale))

    else:
        tol = jax.lax.stop_gradient(tol)

        def accept(x, fx, dfx):
            return jnp.max(jnp.abs(fx)) <= tol

    if solver is None:

        def solve(g, x):
            return _scalar_newton(g, x, bracket, accept, max_iter, damped)

    else:

        def solve(g, x):
            return solver(g, x), jnp.full((2,), -1.0, jnp.abs(x0).dtype)

    if tangent_solve is None:
        tangent_solve = partial(_default_tangent_solve, dense_limit=dense_limit)
        if jnp.iscomplexobj(x0) and x0.ndim == 0:
            tangent_solve = partial(_complex_scalar_tangent, dense_limit=dense_limit)
    x, aux = jax.lax.custom_root(f, x0, solve, tangent_solve, has_aux=True)
    iterations, status = jax.lax.stop_gradient(aux).astype(jnp.int32)
    xs = jax.lax.stop_gradient(x)
    fr = f(xs)
    if solver is None and tol is None:
        dfr = _value_and_slope(f, xs)[1]
    else:
        dfr = jnp.zeros_like(fr)
    residual = jnp.max(jnp.abs(fr))
    converged = accept(xs, fr, dfr) & jnp.all(jnp.isfinite(xs))
    status = jnp.where(converged, ROOT_CONVERGED, jnp.where(status < 0, ROOT_STAGNATED, status))
    status = jnp.where(~converged & (status == ROOT_CONVERGED), ROOT_STAGNATED, status)
    status = status.astype(jnp.int32)
    return RootResult(_nan_unless(x, converged), residual, iterations, converged, status)
