"""Right preconditioners for the Krylov solvers in ``solvax.krylov``.

Every builder in this module returns a callable ``precond(v) -> v``
applying an approximate inverse ``M^{-1}``, suitable for the ``precond=``
argument of :func:`solvax.krylov.gmres` and :func:`solvax.krylov.gcrot`
(right preconditioning: the solver iterates on ``A M^{-1}`` and applies
``x = M^{-1} y`` internally, so the *residual* being minimized is that of
the original system). All factorizations happen once, at build time, and
are closed over — applying the preconditioner is factor-solve only.

The catalogue, roughly in order of increasing structure exploited:

- :func:`jacobi` / :func:`block_jacobi` — (block-)diagonal scaling,
  ``M = diag(A)`` or the block diagonal with batched LU (Saad, ch. 10).
- :func:`coarse_operator` — *the* physics-based pattern: precondition a
  hard operator with an exact/structured solve of a simplified one
  (physics-coarsened, coupling-dropped), e.g. a fluid/moment approximation
  of a kinetic Jacobian (Chen & Chacón) or the "preconditioner matrix"
  handed to LU in production PETSc codes.
- :func:`line_smoother` — alternating-direction block Jacobi: damped line
  solves along different tensor axes of a structured grid, composed as

      x <- x + omega_i * M_i^{-1} (b - A x),

  the classic remedy for anisotropic coupling (Trottenberg et al., ch. 5).
- :func:`additive_preconditioner` — a positive weighted sum of symmetric
  positive component inverse actions, suitable for PCG and additive line or
  Schwarz preconditioning on arrays and arbitrary pytrees.
- :func:`p_multigrid` — a V-cycle over caller-supplied levels
  (pre-smooth, restrict residual, recurse, prolong correction,
  post-smooth), physics-agnostic: all matvecs, transfers, and smoothers
  are injected. Covers h- and p-/spectral coarsening alike.
- :func:`galerkin_deflation` — balance a symmetric smoother around an
  adjoint-transfer Galerkin coarse correction, preserving the symmetry
  required by conjugate-gradient methods.
- :func:`mixed_precision` — run any preconditioner in low precision;
  flexible GMRES tolerates the inexactness and the outer residual is
  still accumulated in working precision (Carson & Higham).
- :func:`kronecker_nkp` / :func:`nearest_kronecker` — inverse of a
  Kronecker product ``A ⊗ B`` at the cost of two small solve sets, with
  the factors extracted automatically from a dense matrix via the
  Van Loan-Pitsianis rearrangement (nearest Kronecker product).

A preconditioner only has to *cluster the spectrum* of ``A M^{-1}`` — an
O(1)-accurate inverse of the dominant physics usually beats an expensive
exact inverse of the wrong terms. Because :func:`solvax.krylov.gmres` is
*flexible* (FGMRES), the callables returned here may themselves be inner
iterations or change between applications.

References
----------
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
  (2003), chapters 9-10 — preconditioned Krylov methods, (block) Jacobi.
- G. Chen & L. Chacón, "An implicit energy-conserving particle-in-cell
  scheme" / moment-based preconditioning of kinetic Jacobians,
  https://arxiv.org/abs/1309.6243 — solve a fluid (physics-coarsened)
  operator exactly to precondition the kinetic one; the same strategy as
  the PETSc ``Pmat`` (preconditioner-matrix) idiom of production codes.
- U. Trottenberg, C. W. Oosterlee & A. Schüller, *Multigrid*, Academic
  Press (2001) — smoothers, line relaxation, V-cycles.
- L. Fischer et al., https://arxiv.org/abs/2110.07663 and M. Thompson et
  al., https://arxiv.org/abs/2108.01751 — p-multigrid / spectral
  coarsening with caller-supplied transfer operators.
- E. Carson & N. J. Higham, "Accelerating the solution of linear systems
  by iterative refinement in three precisions", SIAM J. Sci. Comput.
  40(2), A817 (2018) — low-precision inner solves, high-precision outer.
- C. F. Van Loan & N. Pitsianis, "Approximation with Kronecker products",
  in *Linear Algebra for Large Scale and Real-Time Applications*, Kluwer
  (1993) — the rearrangement turning nearest-Kronecker-product
  approximation into a rank-1 SVD.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

from solvax.refine import as_low_precision
from solvax.tridiagonal import (
    _reusable_tridiagonal_solver,
    cyclic_tridiagonal_solve,
)

MatVec = Callable[[jax.Array], jax.Array]
PytreePreconditioner = Callable[[Any], Any]


class _Jacobi(eqx.Module):
    """PyTree point-Jacobi application, including its array state."""

    inverse_diagonal: jax.Array

    def __call__(self, vector: jax.Array) -> jax.Array:
        return self.inverse_diagonal * vector


class _BlockJacobi(eqx.Module):
    """PyTree block-Jacobi application with reusable batched LU factors."""

    lu: jax.Array
    pivots: jax.Array
    n_blocks: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)

    def __call__(self, vector: jax.Array) -> jax.Array:
        residual = vector.reshape(self.n_blocks, self.block_size)
        batched_solve = jax.vmap(
            lambda lu_k, piv_k, rhs_k: lu_solve((lu_k, piv_k), rhs_k)
        )
        return batched_solve(self.lu, self.pivots, residual).reshape(vector.shape)


def jacobi(diagonal: jax.Array) -> MatVec:
    """Diagonal (point-Jacobi) preconditioner ``M^{-1} = diag(A)^{-1}``.

    The cheapest useful preconditioner: rescales each equation by its
    diagonal entry, which equilibrates row magnitudes and collapses the
    spectrum of diagonally dominant operators toward 1.

    Args:
        diagonal: the diagonal of ``A``, shape ``(n,)``.

    Returns:
        A callable ``precond(v) -> diagonal**-1 * v``.
    """
    return _Jacobi(1.0 / jnp.asarray(diagonal))


def block_jacobi(blocks: jax.Array) -> MatVec:
    """Block-Jacobi preconditioner from the dense diagonal blocks of ``A``.

    Each ``m x m`` diagonal block is LU-factored once (batched, with
    partial pivoting); applying the preconditioner reshapes the vector to
    ``(n_blocks, m)`` and runs one batched ``lu_solve``. Exact for
    block-diagonal ``A`` — with a single block equal to the full matrix
    this is a direct solve.

    Args:
        blocks: diagonal blocks of ``A``, shape ``(n_blocks, m, m)``; the
            preconditioned vectors have length ``n_blocks * m``.

    Returns:
        A callable applying ``blockdiag(blocks)^{-1}`` to flat vectors.
    """
    blocks = jnp.asarray(blocks)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("blocks must have shape (n_blocks, m, m)")
    n_blocks, m, _ = blocks.shape
    lu, piv = jax.vmap(lu_factor)(blocks)
    return _BlockJacobi(lu, piv, n_blocks, m)


def coarse_operator(solve: MatVec) -> MatVec:
    """Precondition with an exact solve of a *simplified* operator.

    This trivial adaptor documents the central physics-based pattern:
    when the full operator ``A`` is too hard to invert (kinetic, dense,
    matrix-free), build a simplified operator ``A_s`` — physics-coarsened
    (e.g. a fluid/moment closure of a kinetic Jacobian, Chen & Chacón,
    https://arxiv.org/abs/1309.6243), or coupling-dropped (keep the
    block-tridiagonal / banded core, discard long-range terms) — factor
    *it* exactly with the structured solvers in :mod:`solvax.direct` or
    :mod:`solvax.banded`, and hand ``v -> A_s^{-1} v`` to the Krylov
    method. The preconditioned operator is ``A A_s^{-1} = I + (A - A_s)
    A_s^{-1}``, so convergence is governed by how much physics ``A_s``
    captures, not by the conditioning of ``A``. This mirrors the
    "preconditioner matrix" (``Pmat``) strategy of production PETSc
    codes, where the LU of a simplified operator preconditions the true
    Jacobian.

    Args:
        solve: any callable ``v -> A_s^{-1} v`` applying the exact (or
            structured) inverse of the simplified operator — e.g. a
            closure over :func:`solvax.direct.block_thomas_factor` /
            :func:`solvax.direct.block_thomas_solve` factors, or the
            banded factors of :mod:`solvax.banded`.

    Returns:
        The callable itself, usable directly as ``precond=``.
    """

    def apply(v: jax.Array) -> jax.Array:
        return solve(v)

    return apply


def line_smoother(
    matvec: MatVec,
    line_solves: Sequence[MatVec],
    *,
    omega: float | Sequence[float] = 0.8,
    sweeps: int = 1,
) -> MatVec:
    """Alternating-direction block-Jacobi (line) smoother.

    Given exact solves along different tensor axes of a structured grid
    vector — e.g. tridiagonal x-line and y-line solves built from
    :func:`solvax.banded.lu_factor_banded` factors, each already closing
    over its axis reshape — compose them as under-relaxed corrections

        x <- x + omega_i * solve_i(r),    r = b - A x,

    starting from ``x = 0``, cycling through the directions ``sweeps``
    times. Line relaxation solves the *strongly coupled* direction
    exactly, which is the standard cure for anisotropic operators where
    point smoothers stall; alternating directions covers anisotropy of
    unknown or mixed orientation (Trottenberg et al., ch. 5).

    Args:
        matvec: the full operator ``v -> A v``, used to refresh the
            residual between line corrections.
        line_solves: callables ``r -> M_i^{-1} r`` on flat vectors, one
            per direction, applied in order.
        omega: under-relaxation weight(s); a scalar is broadcast, or one
            weight per entry of ``line_solves``.
        sweeps: number of passes over all directions (static Python int).

    Returns:
        A callable ``precond(b) -> x`` approximating ``A^{-1} b``.
    """
    line_solves = tuple(line_solves)
    if not line_solves:
        raise ValueError("line_solves must contain at least one solve")
    if isinstance(omega, (int, float)):
        omegas = (float(omega),) * len(line_solves)
    else:
        omegas = tuple(float(w) for w in omega)
        if len(omegas) != len(line_solves):
            raise ValueError("omega must be a scalar or match len(line_solves)")

    n_corrections = sweeps * len(line_solves)

    def apply(b: jax.Array) -> jax.Array:
        x = jnp.zeros_like(b)
        r = b
        step = 0
        for _ in range(sweeps):
            for w, solve in zip(omegas, line_solves, strict=True):
                x = x + w * solve(r)
                step += 1
                if step < n_corrections:  # last residual update is unused
                    r = b - matvec(x)
        return x

    return apply


def additive_preconditioner(
    preconditioners: Sequence[PytreePreconditioner],
    *,
    weights: Sequence[float] | None = None,
) -> PytreePreconditioner:
    """Combine symmetric positive inverse actions without breaking PCG.

    Returns ``sum_i weights[i] * preconditioners[i](residual)``. The default
    is the arithmetic mean, which keeps the action's scale independent of the
    number of components. Positive weights preserve self-adjoint positive
    definiteness when every component has those properties, making this the
    PCG-safe counterpart to multiplicative :func:`line_smoother` for additive
    line, block, or Schwarz preconditioners.

    Inputs and component results may be arrays or matching arbitrary pytrees.
    The combination is pure JAX tree arithmetic, so JIT, differentiation, and
    the input leaves' device placement are preserved. The caller owns the
    component symmetry and positivity; this function validates only weights
    and pytree structure.

    Args:
        preconditioners: nonempty sequence of fixed inverse actions.
        weights: optional finite positive weights, one per action. Defaults to
            equal weights summing to one.

    Returns:
        A callable with the same pytree input and output structure.
    """
    preconditioners = tuple(preconditioners)
    if not preconditioners:
        raise ValueError("preconditioners must contain at least one action")
    if weights is None:
        coefficients = (1.0 / len(preconditioners),) * len(preconditioners)
    else:
        coefficients = tuple(float(weight) for weight in weights)
        if len(coefficients) != len(preconditioners):
            raise ValueError("weights must match len(preconditioners)")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in coefficients):
            raise ValueError("weights must be finite and positive")

    def apply(residual: Any) -> Any:
        structure = jax.tree_util.tree_structure(residual)
        terms = tuple(preconditioner(residual) for preconditioner in preconditioners)
        if any(jax.tree_util.tree_structure(term) != structure for term in terms):
            raise ValueError("preconditioners must preserve the input pytree structure")

        def combine(*leaves: jax.Array) -> jax.Array:
            return sum(
                coefficient * leaf
                for coefficient, leaf in zip(coefficients, leaves, strict=True)
            )

        return jax.tree_util.tree_map(combine, *terms)

    return apply


def additive_tridiagonal_line_preconditioner(
    diagonal: jax.Array,
    directions: Sequence[tuple[int, jax.Array, jax.Array]],
    *,
    periodic_last_axis: tuple[jax.Array, jax.Array] | None = None,
) -> MatVec:
    """Build an additive inverse from batched tridiagonal grid-line solves.

    Each ``(axis, lower, upper)`` entry defines ordinary tridiagonal lines
    along one array axis. Optionally, ``periodic_last_axis`` adds a cyclic
    solve along the final axis. Component inverses are combined with
    :func:`additive_preconditioner`, so their arithmetic mean is fixed,
    symmetric, differentiable, and suitable for PCG when each line operator
    is symmetric positive definite.

    Args:
        diagonal: shared cell diagonal with the same shape as a residual.
        directions: axis and lower/upper bands for each nonperiodic line set.
        periodic_last_axis: optional lower/upper bands for cyclic final-axis
            lines, including their two corner couplings.

    Returns:
        A JIT- and differentiation-transparent additive inverse action.
    """
    diagonal = jnp.asarray(diagonal)
    line_solves = []
    for axis, lower, upper in directions:
        axis %= diagonal.ndim
        permutation = (axis,) + tuple(i for i in range(diagonal.ndim) if i != axis)
        inverse = tuple(permutation.index(i) for i in range(diagonal.ndim))
        solve = _reusable_tridiagonal_solver(*(jnp.transpose(value, permutation)
            for value in (lower, diagonal, upper)))

        def solve_line(residual, solve=solve,
                       permutation=permutation, inverse=inverse):
            with jax.named_scope("solvax.line_preconditioner.tridiagonal_solve"):
                solved = solve(jnp.transpose(residual, permutation))
            return jnp.transpose(solved, inverse)

        line_solves.append(solve_line)
    if periodic_last_axis is not None:
        lower, upper = periodic_last_axis

        def solve_periodic(residual: jax.Array) -> jax.Array:
            with jax.named_scope("solvax.line_preconditioner.cyclic_solve"):
                solved = cyclic_tridiagonal_solve(
                    jnp.moveaxis(lower, -1, 0),
                    jnp.moveaxis(diagonal, -1, 0),
                    jnp.moveaxis(upper, -1, 0),
                    jnp.moveaxis(residual, -1, 0),
                )
            return jnp.moveaxis(solved, 0, -1)

        line_solves.append(solve_periodic)
    return additive_preconditioner(line_solves)


def p_multigrid(
    matvecs: Sequence[MatVec],
    restricts: Sequence[MatVec],
    prolongs: Sequence[MatVec],
    coarse_solve: MatVec,
    *,
    smoothers: Sequence[jax.Array | Callable],
    cycles: int = 1,
) -> MatVec:
    """Multigrid V-cycle preconditioner over caller-supplied levels.

    Levels are ordered finest first; level ``l`` (0 <= l < L) carries a
    fine matvec, a smoother, and transfers to/from level ``l + 1``, and
    the coarsest level ``L`` is handled by ``coarse_solve``. One V-cycle
    on level ``l`` with operator ``A_l``, restriction ``R_l`` and
    prolongation ``P_l`` computes

        x <- S_l(0, b)                       (pre-smooth from zero)
        e <- V-cycle_{l+1}(R_l (b - A_l x))  (coarse-grid correction)
        x <- S_l(x + P_l e, b)               (post-smooth)

    with the recursion bottoming out at ``x = coarse_solve(b)``. The
    recursion is plain Python over the static level list, so the whole
    cycle stays jit-able. This library is physics-agnostic: nothing is
    assumed about the transfers, so the same cycle covers geometric
    h-coarsening and p-/spectral coarsening (lowering polynomial or
    Legendre/Hermite resolution) alike — see Trottenberg et al. for the
    classical theory and https://arxiv.org/abs/2110.07663 (Fischer et
    al.) and https://arxiv.org/abs/2108.01751 (Thompson et al.) for
    p-multigrid with spectral level hierarchies.

    Args:
        matvecs: fine-level operators ``v -> A_l v``, finest first,
            length ``L`` (the coarsest level has no matvec).
        restricts: transfers ``r_l -> r_{l+1}``, length ``L``.
        prolongs: transfers ``e_{l+1} -> e_l``, length ``L``.
        coarse_solve: exact (or strong) solve on the coarsest level,
            ``b -> A_L^{-1} b``.
        smoothers: one per fine level. Either a ``jax.Array`` holding
            ``diag(A_l)`` — giving one damped-Jacobi sweep
            ``x + (2/3) diag^{-1} (b - A_l x)`` — or a callable
            ``smoother(matvec, x, b) -> x`` improving the iterate ``x``.
        cycles: number of V-cycles per application (static Python int);
            cycles after the first act on the residual.

    Returns:
        A callable ``precond(b) -> x`` approximating ``A_0^{-1} b``.
    """
    matvecs = tuple(matvecs)
    restricts = tuple(restricts)
    prolongs = tuple(prolongs)
    smoothers = tuple(smoothers)
    n_fine = len(matvecs)
    if not (len(restricts) == len(prolongs) == len(smoothers) == n_fine):
        raise ValueError(
            "matvecs, restricts, prolongs and smoothers must have equal length"
        )
    if cycles < 1:
        raise ValueError("cycles must be >= 1")

    def _damped_jacobi(diagonal: jax.Array) -> Callable:
        inv_diag = 1.0 / jnp.asarray(diagonal)

        def smooth(matvec_l: MatVec, x: jax.Array, b: jax.Array) -> jax.Array:
            return x + (2.0 / 3.0) * inv_diag * (b - matvec_l(x))

        return smooth

    smooth_fns = tuple(
        s if callable(s) else _damped_jacobi(s) for s in smoothers
    )

    def vcycle(level: int, b: jax.Array) -> jax.Array:
        if level == n_fine:
            return coarse_solve(b)
        matvec_l = matvecs[level]
        smooth = smooth_fns[level]
        x = smooth(matvec_l, jnp.zeros_like(b), b)
        e = vcycle(level + 1, restricts[level](b - matvec_l(x)))
        return smooth(matvec_l, x + prolongs[level](e), b)

    def apply(b: jax.Array) -> jax.Array:
        x = vcycle(0, b)
        for _ in range(cycles - 1):
            x = x + vcycle(0, b - matvecs[0](x))
        return x

    return apply


def galerkin_deflation(
    matvec: MatVec,
    smoother: MatVec,
    prolong: MatVec,
    coarse_solve: MatVec,
    coarse_template: jax.Array,
) -> MatVec:
    """Build a symmetry-preserving Galerkin deflation preconditioner.

    Restriction is the exact transpose of ``prolong``. For symmetric
    ``matvec``, ``smoother``, and ``coarse_solve``, the balanced operator

    ``S + (I - S A) P A_c^-1 P.T (I - A S)``

    is symmetric and therefore suitable for PCG. The caller constructs and
    factors the Galerkin coarse operator ``A_c = P.T A P`` once; applications
    require one smoothing solve, one coarse solve, and one balancing solve.

    Args:
        matvec: fine operator ``v -> A v``.
        smoother: symmetric fine approximate inverse ``v -> S v``.
        prolong: coarse-to-fine linear transfer ``v -> P v``.
        coarse_solve: coarse inverse ``v -> A_c^-1 v``.
        coarse_template: zero-like coarse array used to transpose ``prolong``.

    Returns:
        A callable applying the balanced fine-plus-coarse inverse.
    """
    coarse_template = jnp.asarray(coarse_template)

    def restrict(fine: jax.Array) -> jax.Array:
        return jax.linear_transpose(prolong, coarse_template)(fine)[0]

    def apply(residual: jax.Array) -> jax.Array:
        fine = smoother(residual)
        coarse = prolong(coarse_solve(restrict(residual - matvec(fine))))
        return fine + coarse - smoother(matvec(coarse))

    return apply


def mixed_precision(precond: MatVec, dtype=jnp.float32) -> MatVec:
    """Run any preconditioner in low precision.

    Wraps ``precond`` with :func:`solvax.refine.as_low_precision`: the
    input vector is cast down to ``dtype``, the preconditioner applied,
    and the result cast back to the input's precision. Since a right
    preconditioner only needs to *cluster the spectrum*, low-precision
    application typically changes the iteration count marginally while
    halving memory traffic — and flexible GMRES (:func:`solvax.krylov.
    gmres`) is specifically robust to such inexact, step-dependent
    preconditioning, with residuals still accumulated in working
    precision (Carson & Higham, SIAM J. Sci. Comput. 40, A817 (2018)).

    Args:
        precond: any preconditioner callable ``v -> M^{-1} v``.
        dtype: precision to apply it in (default ``float32``).

    Returns:
        A callable with the same signature, low precision inside.
    """
    return as_low_precision(precond, dtype)


def kronecker_nkp(
    a_factors: tuple[jax.Array, jax.Array],
    b_factors: tuple[jax.Array, jax.Array],
) -> MatVec:
    """Apply ``(A ⊗ B)^{-1}`` from LU factors of the small factors.

    For ``v = vec(V)`` in row-major (C) order with ``V`` of shape
    ``(na, nb)``, the Kronecker identity reads ``(A ⊗ B) vec(V) =
    vec(A V B^T)``, so the inverse is two *small* solve sets instead of
    one ``(na*nb)``-sized one:

        X = A^{-1} V B^{-T},        (A ⊗ B)^{-1} v = vec(X),

    at O(na^2 nb + na nb^2) cost per application. Combine with
    :func:`nearest_kronecker` to build an automatic structural
    preconditioner for operators that are only *approximately* Kronecker
    (separable up to weak coupling): ``M = A ⊗ B`` nearest to the true
    operator clusters the spectrum of ``A M^{-1}`` around 1.

    Args:
        a_factors: ``jax.scipy.linalg.lu_factor`` output for ``A``,
            shape ``(na, na)``.
        b_factors: ``jax.scipy.linalg.lu_factor`` output for ``B``,
            shape ``(nb, nb)``.

    Returns:
        A callable applying ``(A ⊗ B)^{-1}`` to flat ``(na * nb,)``
        vectors.
    """
    a_lu, a_piv = a_factors
    b_lu, b_piv = b_factors
    na = a_lu.shape[0]
    nb = b_lu.shape[0]

    def apply(v: jax.Array) -> jax.Array:
        rhs = v.reshape(na, nb)
        y = lu_solve((a_lu, a_piv), rhs)  # A^{-1} V
        x = lu_solve((b_lu, b_piv), y.T).T  # A^{-1} V B^{-T}
        return x.reshape(v.shape)

    return apply


def nearest_kronecker(
    matrix: jax.Array, na: int, nb: int
) -> tuple[jax.Array, jax.Array]:
    """Nearest-Kronecker-product factors of a dense matrix.

    Finds ``A`` (``na x na``) and ``B`` (``nb x nb``) minimizing
    ``||M - A ⊗ B||_F`` via the Van Loan-Pitsianis rearrangement: the
    permutation ``R(M)[i*na + j, p*nb + q] = M[i*nb + p, j*nb + q]``
    turns every Kronecker product into a rank-1 matrix
    ``vec(A) vec(B)^T``, so the nearest one is the leading singular
    triplet of ``R(M)`` — ``A = sqrt(s_1) unvec(u_1)``,
    ``B = sqrt(s_1) unvec(v_1)`` (Van Loan & Pitsianis 1993). The
    factors are unique up to the inert scaling
    ``(c A) ⊗ (B / c) = A ⊗ B``. Feed the LU of the result to
    :func:`kronecker_nkp` for an automatic structural preconditioner.

    Args:
        matrix: dense matrix of shape ``(na * nb, na * nb)``.
        na: size of the left (outer) factor.
        nb: size of the right (inner) factor.

    Returns:
        The pair ``(A, B)`` with shapes ``(na, na)`` and ``(nb, nb)``.
    """
    matrix = jnp.asarray(matrix)
    if matrix.shape != (na * nb, na * nb):
        raise ValueError(f"matrix must have shape {(na * nb, na * nb)}")
    r = matrix.reshape(na, nb, na, nb).transpose(0, 2, 1, 3)
    r = r.reshape(na * na, nb * nb)
    u, s, vt = jnp.linalg.svd(r, full_matrices=False)
    scale = jnp.sqrt(s[0])
    a = (scale * u[:, 0]).reshape(na, na)
    b = (scale * vt[0]).reshape(nb, nb)
    return a, b
