"""Eigenpairs: implicit reverse-mode differentiation of application-solved pairs, and a
targeted generalized eigensolver for Hermitian block-tridiagonal pencils."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve, solve_triangular

from solvax.direct import (
    block_thomas_factor,
    block_tridiag_matvec,
    block_tridiag_relative_residual,
)

__all__ = ["EigenpairResult", "eigenpair_reverse", "generalized_eigenpairs"]


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

    ``condition_limit`` gates the eigenvalue condition number
    ``||left|| ||right|| / |left^H right|``, which is computed from the pair the
    application supplies. That is the only pair available, and it bounds what
    the gate can see: near an exceptional point the application's own solver
    loses accuracy first, and the degraded vectors it returns can make the
    condition number look acceptable. Measured on a perturbed Jordan block at a
    separation of ``1e-18``, the true condition number is ``5e8`` -- above the
    default limit -- while the gate saw ``4.5e7``, accepted, and the pipeline
    returned a derivative of ``0.955`` where the analytic value is ``0.5``. A
    condition number within an order of magnitude of the limit therefore
    deserves independent confirmation rather than trust.

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
        apply = build(parameters)
        value, right = primal_solver(parameters, apply, v0)
        return jnp.asarray(value), jnp.asarray(right)

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


class EigenpairResult(NamedTuple):
    """Result of :func:`generalized_eigenpairs`. Only ``values`` and ``vectors`` carry derivatives.

    Attributes:
        values: ``(count,)`` real eigenvalues nearest the target, ascending.
        vectors: ``(n_blocks, m, count)`` eigenvectors, ``G``-normalized
            (``y^H G y = 1``), each defined up to a phase.
        residuals: ``(count,)`` relative residuals
            ``||F y - lam G y|| / ((||F|| + |lam| ||G||) ||y||)``, with ``||.||`` the
            infinity norm (largest absolute row sum) of a band.
        orthogonality: ``max |Y^H G Y - I|``.
        iterations: shift-invert applications performed.
        converged: ``max(residuals) <= tol``; False if anything is not finite.
        gap: ``|lam - target|`` of the nearest excluded Ritz value minus that of the
            farthest selected one. The iteration also converges that excluded pair to ``tol``,
            so the gap is a converged eigenvalue distance, but it only sees the subspace: an
            eigenvalue the iteration never captured is not counted. Zero means the selection
            may split a degenerate or tied set; ``inf`` if every Ritz value was selected.
        fallback: True if block Thomas failed its probe solve and the dense pivoted
            path ran.
    """

    values: jax.Array
    vectors: jax.Array
    residuals: jax.Array
    orthogonality: jax.Array
    iterations: jax.Array
    converged: jax.Array
    gap: jax.Array
    fallback: jax.Array


def _hermitian(band):
    """``(lower, diag, upper)`` with ``lower`` rebuilt as ``upper^H`` and ``diag`` Hermitian."""
    _, diag, upper = band
    upper = upper.at[-1].set(0)
    adjoint = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))  # noqa: E731
    return jnp.roll(adjoint(upper), 1, axis=0), (diag + adjoint(diag)) / 2, upper


def _shift(F, G, sigma):
    return tuple(f - sigma * g for f, g in zip(F, G, strict=True))


def _norm(band):
    """Infinity norm: the largest absolute row sum of the band."""
    return sum(jnp.abs(a).sum(-1) for a in band).max()


def _dense(band):
    lower, diag, upper = band
    n, m = diag.shape[:2]
    i = jnp.arange(n)
    full = jnp.zeros((n, n, m, m), diag.dtype).at[i, i].set(diag)
    full = full.at[i[:-1], i[1:]].set(upper[:-1]).at[i[1:], i[:-1]].set(lower[1:])
    return jnp.swapaxes(full, 1, 2).reshape(n * m, n * m)


def _ritz(Fp, Gp):
    """Eigenpairs of the Hermitian pencil ``(Fp, Gp)`` with ``Gp`` positive definite."""
    L = jnp.linalg.cholesky((Gp + jnp.conj(Gp.T)) / 2)
    M = solve_triangular(L, jnp.conj(solve_triangular(L, Fp, lower=True).T), lower=True)
    theta, W = jnp.linalg.eigh((M + jnp.conj(M.T)) / 2)
    return theta, solve_triangular(L, W, lower=True, trans="C")


def _thomas(factors, b):
    """Solve from :func:`block_thomas_factor` factors with two scans, so compile time does not
    grow with ``n_blocks``. ``b`` is ``(n_blocks, m)`` or ``(n_blocks, m, k)``.

    ``b`` rides in the scan carry: JAX before 0.10 cannot transpose a scan whose linear inputs
    are ``xs``, and the tangent rule of ``_attach`` applies this solve to a linear right-hand side.
    """
    lu, piv, lower, upper = factors
    read = lambda a, k: jax.lax.dynamic_index_in_dim(a, k, 0, keepdims=False)  # noqa: E731
    write = lambda a, v, k: jax.lax.dynamic_update_index_in_dim(a, v, k, 0)  # noqa: E731

    def sweep(step, band, carry, **kw):
        def body(carry, xs):
            v, buf = carry
            k, block, lu_k, piv_k = xs
            v = step(read(buf, k), block @ v, lambda r: lu_solve((lu_k, piv_k), r))
            return (v, write(buf, v, k)), None

        xs = (jnp.arange(b.shape[0]), band, lu, piv)
        return jax.lax.scan(body, (jnp.zeros_like(b[0]), carry), xs, **kw)[0][1]

    # z_k = Delta_k^{-1} (b_k - U_k z_{k+1}), then x_k = z_k - Delta_k^{-1} L_k x_{k-1};
    # the last U_k and the first L_k meet a zero carry.
    z = sweep(lambda r, c, solve: solve(r - c), upper, b, reverse=True)
    return sweep(lambda r, c, solve: r - solve(c), lower, z)


def _shifted_solver(band, dense_limit):
    """``(solve, fallback)`` for a Hermitian band: block Thomas, else dense pivoted LU.

    Block Thomas does not pivot across blocks, so it breaks down on an invertible matrix with a
    singular Schur complement such as ``[[0, 1], [1, 0]]``. A probe solve whose relative
    residual is not below ``sqrt(eps)``, or is not finite, selects the dense path, which is
    compiled in only if ``n_blocks * m <= dense_limit``.
    """
    n, m = band[1].shape[:2]
    size = n * m
    factors = block_thomas_factor(*band)
    thomas = lambda b: _thomas(factors, b)  # noqa: E731
    probe = jnp.sin(1.0 + jnp.arange(size)).reshape(n, m).astype(band[1].dtype)
    residual = block_tridiag_relative_residual(*band, thomas(probe), probe)
    ok = residual <= jnp.sqrt(jnp.finfo(band[1].dtype).eps)
    if size > dense_limit:
        return thomas, jnp.zeros((), bool)
    lu = jax.lax.cond(
        ok,
        lambda: (jnp.zeros((size, size), band[1].dtype), jnp.zeros(size, jnp.int32)),
        lambda: lu_factor(_dense(band)),
    )
    dense = lambda b: lu_solve(lu, b.reshape(size, -1)).reshape(b.shape)  # noqa: E731
    return (lambda b: jax.lax.cond(ok, thomas, dense, b)), ~ok


def _scale(values, nF, nG):
    return (nF + jnp.abs(values) * nG) / nG


def _cluster_tol(values, nF, nG):
    """Ritz values closer than this to a value are degenerate with it."""
    return jnp.sqrt(jnp.finfo(values.dtype).eps) * _scale(values, nF, nG)


def _subspace_iteration(F, G, target, count, tol, max_iter, dense_limit):
    """Shift-invert subspace iteration; returns the result and each value's Ritz separation."""
    n, m = F[1].shape[:2]
    size = n * m
    dtype = jnp.result_type(F[1], G[1])
    real = jnp.zeros((), dtype).real.dtype
    nF, nG = _norm(F), _norm(G)
    target = jnp.asarray(target, real)
    # A shift that is exactly an eigenvalue makes the factorization singular: nudge it.
    sigma = target + 16 * jnp.finfo(real).eps * (nF + jnp.abs(target) * nG) / nG
    solve, fallback = _shifted_solver(_shift(F, G, sigma), dense_limit)
    matvec = lambda A, X: block_tridiag_matvec(*A, X)  # noqa: E731
    width = min(2 * count + 8, size)
    keep = min(count + 1, width)  # the selected pairs and the nearest excluded one
    adjoint = lambda a: jnp.conj(a.T)  # noqa: E731

    def sweep(Y):
        """Orthonormalize ``Y``, Rayleigh-Ritz on ``(F, G)``; return ``G`` times the basis."""
        Q = jnp.linalg.qr(Y.reshape(size, width))[0]
        FQ, GQ = (matvec(A, Q.reshape(n, m, width)).reshape(size, width) for A in (F, G))
        theta, C = _ritz(adjoint(Q) @ FQ, adjoint(Q) @ GQ)
        order = jnp.argsort(jnp.abs(theta - target))[:keep]
        X, GX = Q @ C[:, order], GQ @ C[:, order]
        R = FQ @ C[:, order] - GX * theta[order]
        scale = (nF + jnp.abs(theta[order]) * nG) * jnp.linalg.norm(X, axis=0)
        return GQ @ C, theta, order, jnp.linalg.norm(R, axis=0) / scale, X

    start = jax.random.normal(jax.random.PRNGKey(0), (n, m, width), dtype)
    running = lambda k, res: (k < max_iter) & ~(res.max() <= tol) & jnp.isfinite(res.max())  # noqa: E731
    iterations, _, theta, order, res, X = jax.lax.while_loop(
        lambda s: running(s[0], s[4]),
        lambda s: (s[0] + 1, *sweep(solve(s[1].reshape(n, m, width)))),
        (jnp.zeros((), jnp.int32), *sweep(start)),
    )
    perm = jnp.argsort(theta[order[:count]])  # ascending values among the selected
    sel, res = order[:count][perm], res[:count][perm]
    vectors, values = X[:, :count][:, perm].reshape(n, m, count), theta[sel]
    distance = jnp.sort(jnp.abs(theta - target))
    gap = distance[count] - distance[count - 1] if width > count else jnp.asarray(jnp.inf, real)
    gram = adjoint(X[:, :count][:, perm]) @ matvec(G, vectors).reshape(size, count)
    # Distance from each value to the nearest Ritz value outside its (degenerate) cluster.
    others = jnp.abs(values[:, None] - theta[None, :])
    others = jnp.where(others > _cluster_tol(values, nF, nG)[:, None], others, jnp.inf)
    result = EigenpairResult(
        values, vectors, res, jnp.abs(gram - jnp.eye(count)).max(), iterations,
        res.max() <= tol, gap, fallback,
    )  # fmt: skip
    return result, others.min(axis=1)


@partial(jax.custom_jvp, nondiff_argnums=(5,))
def _attach(F, G, values, vectors, sep, dense_limit):
    """Identity on ``(values, vectors)`` that carries the first-order derivative rule."""
    return values, vectors


def _first_order(F, G, dF, dG, values, vectors, sep, dense_limit):
    """Tangents ``(d values, d vectors)`` of ``_attach`` for the band tangents ``dF, dG``."""
    matvec = lambda A, X: block_tridiag_matvec(*A, X)  # noqa: E731
    inner = lambda a, b: jnp.einsum("kic,kic->c", jnp.conj(a), b)  # noqa: E731
    dGY = matvec(dG, vectors)
    dA = matvec(dF, vectors) - values * dGY
    dvalues = jnp.real(inner(vectors, dA))
    GY = matvec(G, vectors)
    nF, nG = _norm(F), _norm(G)
    scale = _scale(values, nF, nG)
    # ``cluster[j, i]``: eigenpair j is degenerate with i (itself included). ``sep`` is the
    # distance to the nearest Ritz value outside it, so the shifted solves are never singular.
    cluster = jnp.abs(values[:, None] - values[None, :]) <= _cluster_tol(values, nF, nG)[None, :]
    dot = lambda a, b: jnp.einsum("kij,ki->j", jnp.conj(a), b)  # noqa: E731

    def one(i):
        lam, c = values[i], cluster[:, i]
        solve, _ = _shifted_solver(
            _shift(F, G, lam + jnp.minimum(sep[i], scale[i]) / 1e3), dense_limit
        )
        Yc, GYc = vectors * c, GY * c
        project = lambda x: x - Yc @ dot(GYc, x)  # noqa: E731 - G-orthogonal to the cluster
        rhs = GYc @ dot(Yc, dA[..., i]) - dA[..., i]
        x = project(solve(rhs))
        for _ in range(3):  # shift is off by <= sep / 1e3: refine on the exact operator
            x = x + project(solve(rhs - (matvec(F, x) - lam * matvec(G, x))))
        # No rotation inside the cluster: its block of ``Y^H G dY`` is ``-Y^H dG Y / 2``.
        return x - 0.5 * Yc @ dot(vectors, dGY[..., i])

    return dvalues, jnp.stack([one(i) for i in range(values.size)], -1)


@_attach.defjvp
def _attach_jvp(dense_limit, primals, tangents):
    F, G, values, vectors, sep = primals
    dF, dG = tangents[:2]
    # Re-enter ``_attach`` so ``values`` and ``vectors`` are differentiable inside this rule too.
    values, vectors = _attach(F, G, values, vectors, sep, dense_limit)
    return (values, vectors), _first_order(F, G, dF, dG, values, vectors, sep, dense_limit)


@partial(jax.jit, static_argnames=("count", "max_iter", "dense_limit"))
def generalized_eigenpairs(
    F, G, target, count: int = 8, tol: float = 1e-10, max_iter: int = 100, dense_limit: int = 2048
) -> EigenpairResult:
    """The ``count`` eigenpairs of ``F y = lam G y`` nearest ``target`` (block-tridiagonal pencil).

    ``F`` and ``G`` are SOLVAX band triples ``(lower, diag, upper)`` of shape
    ``(n_blocks, m, m)``: real symmetric or complex Hermitian, with ``G`` positive definite.
    Both may depend on the parameters being differentiated. Only ``diag`` and ``upper[:-1]``
    are read. The pencil is made Hermitian by construction: ``lower`` is rebuilt as
    ``lower[k] = upper[k-1]^H`` (the conjugate transpose, not the transpose) and ``diag`` is
    replaced by its Hermitian part.

    The method is shift-invert subspace iteration with Rayleigh-Ritz on ``(F, G)``, over
    ``min(2 count + 8, n_blocks m)`` vectors, factoring ``F - sigma G`` with
    :func:`block_thomas_factor`. ``sigma`` is ``target`` moved by ``16 eps ||F|| / ||G||``, so
    that a target equal to an eigenvalue does not give a singular factor. A probe solve checks
    every factorization (see ``_shifted_solver``): if block Thomas has broken down, a dense
    pivoted LU takes over and ``fallback`` is True. The dense path exists only if
    ``n_blocks * m <= dense_limit``; above that a breakdown gives non-finite residuals and
    ``converged=False``. Under ``vmap`` a ``cond`` runs both branches, so pass
    ``dense_limit=0`` there to avoid always paying for the dense factorization. ``count``,
    ``max_iter`` and ``dense_limit`` are static; everything else may be traced.

    Derivatives are implicit, so the iteration is not differentiated, and are valid only when
    ``converged`` is True. They support ``jax.jvp``, ``jax.grad`` and ``jax.jacfwd/jacrev``.
    The second derivative of ``values`` is tested; that of ``vectors`` is not. ``target``,
    ``tol`` and the diagnostics have zero derivative.

    * ``values``: ``d lam = Re y^H (dF - lam dG) y``. For a degenerate or clustered selected
      set the individual values are not differentiable but their sum is:
      ``d mean(values) = tr(Y^H (dF - Lam dG) Y) / count``, the trace of the perturbation
      projected on the cluster. This needs ``gap > 0``: if an excluded eigenvalue ties a
      selected one, the selection itself jumps.
    * ``vectors``: ``dy`` solves ``(F - lam G) dy = -(dF - lam dG) y + d lam G y`` with
      ``Re(y^H G dy) = -y^H dG y / 2`` and the phase condition ``Im(y^H G dy) = 0``, so only
      phase-invariant functions of ``vectors`` (``y y^H``, ``|y|``) are meaningful. The singular
      system is solved on the ``G``-orthogonal complement of ``y`` at the shift ``lam + sep/1e3``,
      ``sep`` being the distance to the nearest Ritz value outside the cluster of ``y``, followed
      by three refinement sweeps on the exact operator. Ritz values within
      ``sqrt(eps) (||F|| + |lam| ||G||) / ||G||`` of ``lam`` are degenerate with it; for such a
      cluster the tangent is the gauge-fixed subspace tangent, with no rotation inside the
      cluster (its block of ``Y^H G dY`` is ``-Y^H dG Y / 2``) and only the out-of-cluster
      components from the solve. It is finite, never NaN, in forward and reverse mode; only
      subspace-invariant functions of ``vectors`` (the projector) are meaningful there. Value and
      cluster-mean tangents are unaffected. Certify the selection with ``gap``.

    Float64 is the validated precision. ``tol`` is the stopping relative residual.

    Args:
        F: stiffness band triple.
        G: mass band triple, positive definite.
        target: real number to search near.
        count: number of eigenpairs, at most ``n_blocks * m``.
        tol: relative residual to stop at.
        max_iter: maximum shift-invert applications.
        dense_limit: largest ``n_blocks * m`` for which the dense fallback is compiled.
    """
    F, G = _hermitian(F), _hermitian(G)
    if not 1 <= count <= F[1].shape[0] * F[1].shape[1]:
        raise ValueError("count must lie in [1, n_blocks * m]")
    stop = lambda t: jax.tree_util.tree_map(jax.lax.stop_gradient, t)  # noqa: E731
    solved = _subspace_iteration(stop(F), stop(G), target, count, tol, max_iter, dense_limit)
    result, sep = stop(solved)
    values, vectors = _attach(F, G, result.values, result.vectors, sep, dense_limit)
    return result._replace(values=values, vectors=vectors)
