"""Flexible restarted GMRES and GCROT-style Krylov subspace recycling.

Right-preconditioned flexible GMRES (FGMRES) builds the Arnoldi relation

    A Z_m = V_{m+1} Hbar_m,      Z_m = [M_1^{-1} v_1, ..., M_m^{-1} v_m],

where ``V_{m+1}`` is orthonormal and ``Hbar_m`` is (m+1) x m upper
Hessenberg. Because the preconditioned vectors ``z_j`` are stored
explicitly, the preconditioner may change from step to step (flexible
mode); the correction is ``x += Z_m y`` with ``y`` minimizing
``|| beta e_1 - Hbar_m y ||``, solved incrementally with Givens rotations
so the residual norm is available at every inner step. Orthogonalization
uses classical Gram-Schmidt applied twice (CGS2), which reduces the
sequential inner-product latency of modified Gram-Schmidt on accelerators
while retaining O(eps) loss of orthogonality.

GCROT(m, k)-style recycling maintains an outer pair ``(C, U)`` with
``A U = C`` and ``C^H C = I``. Each outer iteration first minimizes over
the recycled space (``x += U C^H r``, ``r -= C C^H r``), then runs one
FGMRES(m) cycle on the deflated operator ``(I - C C^H) A``, giving

    (I - C C^H) A Z_m = V_{m+1} Hbar_m,     B_m = C^H A Z_m,

so the cycle correction is ``dx = Z_m y - U B_m y`` with
``A dx = V_{m+1} Hbar_m y`` orthogonal to ``C``. In this v0.1 the recycle
space is updated with *one* direction per cycle — the cycle's own optimal
correction ``(dx, A dx)``, normalized and inserted FIFO — rather than the
harmonic Ritz vectors of GCRO-DR. This is a deliberate simplification: it
keeps the update O(nk) and shape-static, and it retains the directions
that restarting would otherwise discard, but it deflates slowly-converging
eigenmodes only indirectly. Recycle pairs may be passed between solves in
a parameter continuation; on entry ``A U`` is recomputed and the pair is
re-orthonormalized (thin QR) so ``A U = C`` holds for the *current*
operator, as in Parks et al.

References
----------
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
  (2003), sections 6.3-6.5 and 9.4 (GMRES, restarting, FGMRES).
- R. B. Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24,
  20 (2002) — GMRES-DR.
- M. L. Parks, E. de Sturler, G. Mackey, D. D. Johnson & S. Maiti,
  "Recycling Krylov subspaces for sequences of linear systems", SIAM J.
  Sci. Comput. 28, 1651 (2006) — GCRO-DR, recycling across a sequence.
- E. de Sturler, "Truncation strategies for optimal Krylov subspace
  methods", SIAM J. Numer. Anal. 36, 864 (1999) — GCROT.
- L. Giraud, J. Langou, M. Rozloznik & J. van den Eshof, "Rounding error
  analysis of the classical Gram-Schmidt orthogonalization process",
  Numer. Math. 101, 87 (2005) — CGS2 stability.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

PyTree = Any
MatVec = Callable[[PyTree], PyTree]
InnerProduct = Callable[[PyTree, PyTree], jax.Array]


class KrylovSolution(NamedTuple):
    """Result of :func:`gmres` or :func:`gcrot`.

    Attributes:
        x: approximate solution with the same structure and leaf shapes as ``b``.
        residual_norm: true residual norm ``||b - A x||`` (recomputed once
            after the iteration, not the Givens estimate).
        iterations: total inner (Arnoldi) iterations across all cycles,
            ``int32``.
        converged: whether ``residual_norm <= max(atol, rtol * ||b||)``.
        recycle: updated recycle pair ``(C, U)`` with fixed shapes
            ``(n, k)`` for :func:`gcrot`, ``None`` for :func:`gmres`.
            Unfilled columns are zero.
        recycle_drift: for a warm-started :func:`gcrot`, the mean principal-
            angle sine between the incoming recycle image space and its
            re-established span under the *current* operator — a direct
            measure of how far the operator has drifted since the pair was
            built (0 for an unchanged operator, up to 1 for an orthogonal
            rotation). ``0.0`` on a cold start, ``None`` for :func:`gmres`.
    """

    x: PyTree
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    recycle: tuple[jax.Array, jax.Array] | None = None
    recycle_drift: jax.Array | None = None


def _identity(v: jax.Array) -> jax.Array:
    return v


def _adjoint(matrix: jax.Array) -> jax.Array:
    """Return the conjugate transpose used by complex Krylov projections."""
    return jnp.conj(matrix).T


def _tree_add_scaled(left: PyTree, scale: jax.Array, right: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: x + scale * y, left, right)


def _tree_sub(left: PyTree, right: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _gmres_matvec(matvec: MatVec, value: PyTree) -> PyTree:
    """Apply the operator under a stable profiler scope."""
    with jax.named_scope("solvax.gmres.matvec"):
        return matvec(value)


def _gmres_precondition(precond: MatVec, value: PyTree) -> PyTree:
    """Apply the preconditioner under a stable profiler scope."""
    with jax.named_scope("solvax.gmres.preconditioner"):
        return precond(value)


def _tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    products = jax.tree.leaves(jax.tree.map(jnp.vdot, left, right))
    return sum(products[1:], products[0])


def _tree_norm(value: PyTree, inner_product: InnerProduct) -> jax.Array:
    return jnp.sqrt(jnp.maximum(jnp.real(inner_product(value, value)), 0.0))


def _tree_basis(value: PyTree, size: int) -> PyTree:
    return jax.tree.map(lambda x: jnp.zeros((size, *x.shape), x.dtype), value)


def _tree_basis_dot(
    basis: PyTree, value: PyTree, inner_product: InnerProduct
) -> jax.Array:
    if inner_product is not _tree_dot:
        return jax.vmap(inner_product, in_axes=(0, None))(basis, value)
    products = jax.tree.leaves(
        jax.tree.map(
            lambda vectors, x: jnp.conj(vectors).reshape(vectors.shape[0], -1)
            @ x.reshape(-1),
            basis,
            value,
        )
    )
    return sum(products[1:], products[0])


def _tree_basis_sum(coefficients: jax.Array, basis: PyTree) -> PyTree:
    return jax.tree.map(lambda vectors: jnp.tensordot(coefficients, vectors, axes=1), basis)


def _tree_basis_set(basis: PyTree, index: jax.Array, value: PyTree) -> PyTree:
    return jax.tree.map(lambda vectors, x: vectors.at[index].set(x), basis, value)


def _tree_basis_get(basis: PyTree, index: jax.Array) -> PyTree:
    return jax.tree.map(lambda vectors: vectors[index], basis)


def _complex_givens(a: jax.Array, b: jax.Array):
    r"""Return a unitary Givens rotation that annihilates ``b``.

    The returned real ``c`` and possibly-complex ``s`` satisfy

    ``[[c, s], [-conj(s), c]] @ [a, b] == [r, 0]``.

    This is the complex ``xLARTG`` convention used by LAPACK. Scaling by
    ``max(abs(a), abs(b))`` avoids avoidable overflow in the norm.
    """
    real_dtype = jnp.real(a).dtype
    scale = jnp.maximum(jnp.abs(a), jnp.abs(b))
    safe_scale = jnp.where(scale > 0, scale, jnp.asarray(1.0, real_dtype))
    rho = safe_scale * jnp.sqrt(
        (jnp.abs(a) / safe_scale) ** 2 + (jnp.abs(b) / safe_scale) ** 2
    )
    abs_a = jnp.abs(a)
    alpha = jnp.where(abs_a > 0, a / abs_a, jnp.asarray(1.0, a.dtype))
    safe_rho = jnp.where(rho > 0, rho, jnp.asarray(1.0, real_dtype))
    c = jnp.where(rho > 0, abs_a / safe_rho, jnp.asarray(1.0, real_dtype))
    s = jnp.where(rho > 0, alpha * jnp.conj(b) / safe_rho, 0.0)
    r = jnp.where(rho > 0, alpha * rho, jnp.asarray(0.0, a.dtype))
    return c, s, r


def _fgmres_cycle(
    matvec: MatVec,
    precond: MatVec,
    r0: jax.Array,
    beta: jax.Array,
    tol: jax.Array,
    m: int,
    C: jax.Array,
    U: jax.Array,
):
    """One flexible Arnoldi cycle of size ``m`` on the deflated operator.

    Builds ``(I - C C^H) A Z = V Hbar`` with CGS2 orthogonalization and an
    incremental Givens-rotation least-squares solve, stopping early (via
    ``lax.while_loop`` over a zero-padded fixed-size basis) once the
    residual estimate drops below ``tol``. Plain FGMRES is the special
    case ``k = 0`` (empty ``C``/``U``).

    Args:
        matvec: the operator ``v -> A v``.
        precond: right preconditioner ``v -> M^{-1} v``.
        r0: current residual, already orthogonal to ``range(C)``.
        beta: ``||r0||``.
        tol: absolute residual tolerance for early exit.
        m: static cycle size.
        C: orthonormal recycle image basis, shape ``(n, k)`` (``k`` may
            be 0); zero columns are inert.
        U: recycle source basis with ``A U = C``, shape ``(n, k)``.

    Returns:
        Tuple ``(dx, adx, k_done, res_est)``: the correction
        ``dx = Z y - U B y``, its image ``adx = A dx`` (reconstructed from
        the Arnoldi relation, no extra matvec), the number of inner steps
        taken (``int32``), and the final least-squares residual norm.
    """
    n = r0.shape[0]
    dtype = r0.dtype
    k = C.shape[1]

    beta_safe = jnp.where(beta > 0, beta, 1.0)
    V = jnp.zeros((m + 1, n), dtype).at[0].set(r0 / beta_safe)
    Z = jnp.zeros((m, n), dtype)
    H = jnp.zeros((m + 1, m), dtype)  # Hessenberg (Arnoldi relation)
    R = jnp.zeros((m, m), dtype)  # Givens-rotated triangular factor
    B = jnp.zeros((k, m), dtype)
    real_dtype = jnp.real(r0).dtype
    cs = jnp.zeros((m,), real_dtype)
    sn = jnp.zeros((m,), dtype)
    g = jnp.zeros((m + 1,), dtype).at[0].set(beta)

    def cond_fun(state):
        j, _, _, _, _, _, _, _, _, res_est = state
        return (j < m) & (res_est > tol)

    def body_fun(state):
        j, V, Z, H, R, B, cs, sn, g, _ = state

        z = _gmres_precondition(precond, V[j])
        w = _gmres_matvec(matvec, z)
        with jax.named_scope("solvax.gmres.arnoldi_reductions"):
            b_j = _adjoint(C) @ w  # project out the recycled image space
            w = w - C @ b_j

            # CGS2: two passes of classical Gram-Schmidt against the padded
            # basis (zero rows beyond j contribute nothing).
            h1 = jnp.conj(V) @ w
            w = w - h1 @ V
            h2 = jnp.conj(V) @ w
            w = w - h2 @ V
            h = h1 + h2
            h_next = jnp.linalg.norm(w)
        V = V.at[j + 1].set(w / jnp.where(h_next > 0, h_next, 1.0))
        h = h.at[j + 1].set(h_next)
        H = H.at[:, j].set(h)  # unrotated column, keeps A Z = C B + V H

        # Apply the accumulated Givens rotations to the new column.
        def apply_rotation(i, hc):
            hi, hi1 = hc[i], hc[i + 1]
            return hc.at[i].set(cs[i] * hi + sn[i] * hi1).at[i + 1].set(
                -jnp.conj(sn[i]) * hi + cs[i] * hi1
            )

        h = lax.fori_loop(0, j, apply_rotation, h)

        # New rotation annihilating h[j + 1]; happy breakdown (rho == 0)
        # degenerates to the identity rotation.
        c_j, s_j, rho = _complex_givens(h[j], h[j + 1])
        h = h.at[j].set(rho).at[j + 1].set(0.0)
        g_j = g[j]
        g = g.at[j].set(c_j * g_j).at[j + 1].set(-jnp.conj(s_j) * g_j)
        res_est = jnp.abs(g[j + 1])

        R = R.at[:, j].set(h[:m])
        B = B.at[:, j].set(b_j)
        Z = Z.at[j].set(z)
        cs = cs.at[j].set(c_j)
        sn = sn.at[j].set(s_j)
        return (j + 1, V, Z, H, R, B, cs, sn, g, res_est)

    init = (jnp.int32(0), V, Z, H, R, B, cs, sn, g, beta)
    j_f, V, Z, H, R, B, cs, sn, g, res_est = lax.while_loop(cond_fun, body_fun, init)

    # Triangular solve on the used leading block; unused columns get a unit
    # diagonal and a zero right-hand side so they contribute y_i = 0.
    used = jnp.arange(m) < j_f
    R = R + jnp.diag(jnp.where(used, 0.0, 1.0).astype(dtype))
    y = solve_triangular(R, jnp.where(used, g[:m], 0.0), lower=False)

    dx = y @ Z - U @ (B @ y)
    adx = (H @ y) @ V
    return dx, adx, j_f, res_est


def _restarted(
    matvec: MatVec,
    b: jax.Array,
    x0: jax.Array,
    precond: MatVec,
    m: int,
    tol: jax.Array,
    max_restarts: int,
    C: jax.Array,
    U: jax.Array,
    fill: jax.Array,
    recycling: bool,
):
    """Outer restart loop shared by :func:`gmres` (k = 0) and :func:`gcrot`.

    The residual is carried by exact recurrences (``r -= C C^H r`` after the
    outer projection, ``r -= A dx`` after each cycle, with ``A dx``
    reconstructed from the Arnoldi relation), so each cycle costs no extra
    matvec; the true residual is recomputed once at the end for honest
    reporting.

    Args:
        matvec, b, x0, precond, m, tol, max_restarts: as in :func:`gmres`.
        C: recycle image basis ``(n, k)``, orthonormal up to zero padding.
        U: recycle source basis ``(n, k)`` with ``A U = C``.
        fill: number of recycle columns filled so far (``int32``).
        recycling: static flag; when False the recycle update is skipped
            entirely (``k`` may be 0).

    Returns:
        ``(x, residual_norm, iterations, converged, C, U, fill)``.
    """
    dtype = b.dtype
    eps = jnp.finfo(dtype).eps
    k = C.shape[1]
    r0 = b - _gmres_matvec(matvec, x0)

    def cond_fun(state):
        _, _, res, _, cycles, _, _, _ = state
        return (res > tol) & (cycles < max_restarts)

    def body_fun(state):
        x, r, _, iters, cycles, C, U, fill = state

        # Minimize over the recycled space first: x += U C^H r, r ⊥ C.
        ctr = _adjoint(C) @ r
        x = x + U @ ctr
        r = r - C @ ctr
        beta = jnp.linalg.norm(r)

        dx, adx, k_done, _ = _fgmres_cycle(matvec, precond, r, beta, tol, m, C, U)
        x = x + dx
        # Recompute the residual exactly at the restart boundary (one extra
        # matvec per cycle): the incremental update r - adx inherits CGS2
        # orthogonality drift, and a stale small estimate would end the loop
        # while the true residual is still large.
        r = b - _gmres_matvec(matvec, x)
        res = jnp.linalg.norm(r)

        if recycling:
            # v0.1 update: keep the cycle's own optimal correction. One
            # projection pass against C for numerical hygiene (adx is
            # orthogonal to C in exact arithmetic), then FIFO insertion.
            proj = _adjoint(C) @ adx
            c_new = adx - C @ proj
            u_new = dx - U @ proj
            nc = jnp.linalg.norm(c_new)
            ok = nc > eps * (1.0 + jnp.linalg.norm(adx))
            nc_safe = jnp.where(ok, nc, 1.0)
            slot = jnp.mod(fill, k)
            C = jnp.where(ok, C.at[:, slot].set(c_new / nc_safe), C)
            U = jnp.where(ok, U.at[:, slot].set(u_new / nc_safe), U)
            fill = fill + ok.astype(fill.dtype)

        return (x, r, res, iters + k_done, cycles + 1, C, U, fill)

    init = (
        x0,
        r0,
        jnp.linalg.norm(r0),
        jnp.int32(0),
        jnp.int32(0),
        C,
        U,
        fill,
    )
    x, _, _, iters, _, C, U, fill = lax.while_loop(cond_fun, body_fun, init)

    res = jnp.linalg.norm(b - _gmres_matvec(matvec, x))
    return x, res, iters, res <= tol, C, U, fill


def _pytree_fgmres_cycle(
    matvec: MatVec,
    precond: MatVec,
    inner_product: InnerProduct,
    residual: PyTree,
    beta: jax.Array,
    tolerance: jax.Array,
    restart: int,
    dtype: jnp.dtype,
):
    """Run one FGMRES cycle without flattening a pytree operand."""
    basis = _tree_basis(residual, restart + 1)
    preconditioned = _tree_basis(residual, restart)
    beta_safe = jnp.where(beta > 0, beta, 1.0)
    basis = _tree_basis_set(
        basis, jnp.int32(0), jax.tree.map(lambda x: x / beta_safe, residual)
    )
    triangular = jnp.zeros((restart, restart), dtype)
    real_dtype = jnp.real(jnp.zeros((), dtype)).dtype
    cosines = jnp.zeros((restart,), real_dtype)
    sines = jnp.zeros((restart,), dtype)
    rotated_rhs = jnp.zeros((restart + 1,), dtype).at[0].set(beta)

    def cond_fun(state):
        index, _, _, _, _, _, _, residual_estimate = state
        return (index < restart) & (residual_estimate > tolerance)

    def body_fun(state):
        index, basis, z_basis, triangular, cosines, sines, rhs, _ = state
        z = _gmres_precondition(precond, _tree_basis_get(basis, index))
        applied = _gmres_matvec(matvec, z)

        with jax.named_scope("solvax.gmres.arnoldi_reductions"):
            first = _tree_basis_dot(basis, applied, inner_product)
            applied = _tree_sub(applied, _tree_basis_sum(first, basis))
            second = _tree_basis_dot(basis, applied, inner_product)
            applied = _tree_sub(applied, _tree_basis_sum(second, basis))
            column = first + second
            next_norm = _tree_norm(applied, inner_product)
        next_vector = jax.tree.map(
            lambda x: x / jnp.where(next_norm > 0, next_norm, 1.0), applied
        )
        basis = _tree_basis_set(basis, index + 1, next_vector)
        column = column.at[index + 1].set(next_norm)

        def apply_rotation(i, values):
            first_value, second_value = values[i], values[i + 1]
            return values.at[i].set(
                cosines[i] * first_value + sines[i] * second_value
            ).at[i + 1].set(
                -jnp.conj(sines[i]) * first_value + cosines[i] * second_value
            )

        column = lax.fori_loop(0, index, apply_rotation, column)
        cosine, sine, diagonal = _complex_givens(column[index], column[index + 1])
        column = column.at[index].set(diagonal).at[index + 1].set(0.0)
        rhs_value = rhs[index]
        rhs = rhs.at[index].set(cosine * rhs_value).at[index + 1].set(
            -jnp.conj(sine) * rhs_value
        )
        residual_estimate = jnp.abs(rhs[index + 1])

        triangular = triangular.at[:, index].set(column[:restart])
        z_basis = _tree_basis_set(z_basis, index, z)
        cosines = cosines.at[index].set(cosine)
        sines = sines.at[index].set(sine)
        return (
            index + 1,
            basis,
            z_basis,
            triangular,
            cosines,
            sines,
            rhs,
            residual_estimate,
        )

    initial = (
        jnp.int32(0),
        basis,
        preconditioned,
        triangular,
        cosines,
        sines,
        rotated_rhs,
        beta,
    )
    used_count, _, z_basis, triangular, _, _, rhs, _ = lax.while_loop(
        cond_fun, body_fun, initial
    )
    used = jnp.arange(restart) < used_count
    triangular = triangular + jnp.diag(jnp.where(used, 0.0, 1.0).astype(dtype))
    coefficients = solve_triangular(
        triangular, jnp.where(used, rhs[:restart], 0.0), lower=False
    )
    correction = _tree_basis_sum(coefficients, z_basis)
    return correction, used_count


def _pytree_gmres(
    matvec: MatVec,
    b: PyTree,
    x0: PyTree,
    precond: MatVec,
    inner_product: InnerProduct,
    restart: int,
    tolerance: jax.Array,
    max_restarts: int,
    dtype: jnp.dtype,
    zero_initial: bool,
):
    """Restarted FGMRES implementation for matching pytree operands."""
    residual = b if zero_initial else _tree_sub(b, _gmres_matvec(matvec, x0))

    def cond_fun(state):
        _, _, residual_norm, _, cycles = state
        return (residual_norm > tolerance) & (cycles < max_restarts)

    def body_fun(state):
        x, residual, _, iterations, cycles = state
        residual_norm = _tree_norm(residual, inner_product)
        correction, used = _pytree_fgmres_cycle(
            matvec, precond, inner_product, residual, residual_norm,
            tolerance, restart, dtype
        )
        x = _tree_add_scaled(x, 1.0, correction)
        residual = _tree_sub(b, _gmres_matvec(matvec, x))
        return (x, residual, _tree_norm(residual, inner_product),
                iterations + used, cycles + 1)

    initial = (
        x0, residual, _tree_norm(residual, inner_product),
        jnp.int32(0), jnp.int32(0),
    )
    x, _, residual_norm, iterations, _ = lax.while_loop(cond_fun, body_fun, initial)
    return KrylovSolution(x, residual_norm, iterations, residual_norm <= tolerance, None)


def gmres(
    matvec: MatVec,
    b: PyTree,
    *,
    x0: PyTree | None = None,
    precond: MatVec | None = None,
    inner_product: InnerProduct | None = None,
    restart: int = 30,
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_restarts: int = 50,
) -> KrylovSolution:
    """Restarted flexible GMRES with right preconditioning.

    Solves ``A x = b`` for a matrix-free operator, stopping when
    ``||b - A x|| <= max(atol, rtol * ||b||)``. Fully jit-able: all loop
    state has fixed shapes (the Krylov basis is zero-padded to the cycle
    size and early convergence exits via ``lax.while_loop``).

    Args:
        matvec: the operator ``v -> A v`` on an array or pytree; must be pure
            JAX (traceable) and preserve the input tree structure.
        b: array or pytree right-hand side. Pytree leaves must have one common
            inexact dtype.
        x0: initial guess (defaults to zeros).
        precond: right preconditioner ``v -> M^{-1} v`` (defaults to the
            identity). May be flexible, i.e. nonlinear or changing between
            inner steps — the update uses the stored preconditioned
            vectors.
        inner_product: optional ``(left, right) -> scalar`` product used for
            PyTree Arnoldi projections and norms. Defaults to the Euclidean
            product. Supplying it also selects the PyTree path for array inputs.
        restart: static Arnoldi cycle size ``m``.
        rtol: relative tolerance on ``||b||``.
        atol: absolute tolerance floor.
        max_restarts: static maximum number of cycles.

    Returns:
        A :class:`KrylovSolution` with ``recycle=None``.
    """
    if (inner_product is not None or jax.tree.structure(b) != jax.tree.structure(0)
            or jnp.ndim(b) == 0):
        b = jax.tree.map(jnp.asarray, b)
        structure = jax.tree.structure(b)
        leaves = jax.tree.leaves(b)
        if not leaves:
            raise ValueError("b must contain at least one array leaf")
        dtype = jnp.result_type(*[leaf.dtype for leaf in leaves])
        if not jnp.issubdtype(dtype, jnp.inexact) or any(
            leaf.dtype != dtype for leaf in leaves
        ):
            raise ValueError("pytree leaves must have one common inexact dtype")
        zero_initial = x0 is None
        if zero_initial:
            x0 = jax.tree.map(jnp.zeros_like, b)
        elif jax.tree.structure(x0) != structure:
            raise ValueError("x0 and b must have identical pytree structure")
        else:
            x0 = jax.tree.map(lambda x: jnp.asarray(x, dtype), x0)
        precond = _identity if precond is None else precond
        inner_product = _tree_dot if inner_product is None else inner_product
        tol = jnp.maximum(atol, rtol * _tree_norm(b, inner_product))
        return _pytree_gmres(
            matvec, b, x0, precond, inner_product, restart, tol,
            max_restarts, dtype, zero_initial
        )

    b = jnp.asarray(b)
    n = b.shape[0]
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    precond = _identity if precond is None else precond
    tol = jnp.maximum(atol, rtol * jnp.linalg.norm(b))

    empty = jnp.zeros((n, 0), b.dtype)
    x, res, iters, converged, _, _, _ = _restarted(
        matvec, b, x0, precond, restart, tol, max_restarts,
        empty, empty, jnp.int32(0), recycling=False,
    )
    return KrylovSolution(x, res, iters, converged, None)


def gcrot(
    matvec: MatVec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    precond: MatVec | None = None,
    m: int = 30,
    k: int = 10,
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_restarts: int = 50,
    recycle: tuple[jax.Array, jax.Array] | None = None,
) -> KrylovSolution:
    """GCROT(m, k)-style FGMRES with Krylov subspace recycling.

    Like :func:`gmres`, but maintains a recycle pair ``(C, U)`` with
    ``A U = C`` that deflates the operator between restarts and can be
    carried across solves in a slowly-varying sequence (parameter
    continuation): pass ``solution.recycle`` of one solve as ``recycle=``
    of the next. On warm start ``A U`` is recomputed for the current
    operator and the pair is re-orthonormalized by thin QR (rank-deficient
    columns — e.g. zero padding from a short previous solve — are dropped),
    so a stale pair is always consistent, merely less effective.

    The recycle space grows by one direction per cycle (the cycle's own
    correction; see the module docstring for why this simplification, and
    Parks et al. 2006 for the harmonic-Ritz alternative), stored FIFO in
    fixed-shape ``(n, k)`` arrays so the whole solve stays jit-able.

    Args:
        matvec: the operator ``v -> A v`` on flat ``(n,)`` arrays.
        b: right-hand side, shape ``(n,)``.
        x0: initial guess (defaults to zeros).
        precond: right preconditioner ``v -> M^{-1} v`` (identity default).
        m: static inner FGMRES cycle size.
        k: static number of recycle directions kept.
        rtol: relative tolerance on ``||b||``.
        atol: absolute tolerance floor.
        max_restarts: static maximum number of outer cycles.
        recycle: optional ``(C, U)`` pair of shape ``(n, k)`` from a
            previous :class:`KrylovSolution` to warm-start deflation.

    Returns:
        A :class:`KrylovSolution` whose ``recycle`` field holds the updated
        ``(C, U)`` pair, zero-padded to shape ``(n, k)``.
    """
    b = jnp.asarray(b)
    n = b.shape[0]
    dtype = b.dtype
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    precond = _identity if precond is None else precond
    tol = jnp.maximum(atol, rtol * jnp.linalg.norm(b))

    if recycle is None:
        C = jnp.zeros((n, k), dtype)
        U = jnp.zeros((n, k), dtype)
        fill = jnp.int32(0)
        drift = jnp.asarray(0.0, jnp.real(jnp.zeros((), dtype)).dtype)
    else:
        C_in, U_in = recycle
        if C_in.shape != (n, k) or U_in.shape != (n, k):
            raise ValueError(
                f"recycle pair must have shape {(n, k)}, got "
                f"{C_in.shape} and {U_in.shape}"
            )
        # Re-establish A U = C for the *current* operator (Parks et al.
        # 2006, section 4): W = A U, thin QR W = Q R, then C <- Q,
        # U <- U R^{-1}. Numerically rank-deficient columns (zero padding
        # from an early-converged previous solve) are masked out and
        # sorted to the back so FIFO insertion refills them first.
        U_in = jnp.asarray(U_in, dtype)
        W = jnp.stack([_gmres_matvec(matvec, U_in[:, i]) for i in range(k)], axis=1)
        Q, R = jnp.linalg.qr(W)
        diag = jnp.abs(jnp.diagonal(R))
        good = diag > n * jnp.finfo(dtype).eps * jnp.max(diag, initial=0.0)
        R_safe = R + jnp.diag(jnp.where(good, 0.0, 1.0).astype(dtype))
        U_new = solve_triangular(R_safe.T, U_in.T, lower=True).T
        order = jnp.argsort(jnp.logical_not(good), stable=True)
        C = (Q * good)[:, order]
        U = (U_new * good)[:, order]
        fill = jnp.sum(good).astype(jnp.int32)
        # Operator-drift diagnostic: the incoming image columns were
        # orthonormal (up to zero padding); after re-establishing A U = C for
        # the current operator, the mean sine of the principal angles between
        # the old filled columns and the new span measures how far the
        # operator moved the recycled space. sin(theta_i) = ||(I - C C^H)
        # c_i^old|| for unit c_i^old.
        C_in = jnp.asarray(recycle[0], dtype)
        filled = jnp.linalg.norm(C_in, axis=0) > 0.5
        residual_cols = C_in - C @ (_adjoint(C) @ C_in)
        sines = jnp.linalg.norm(residual_cols, axis=0)
        count = jnp.maximum(jnp.sum(filled), 1)
        drift = (jnp.sum(jnp.where(filled, sines, 0.0)) / count).real

    x, res, iters, converged, C, U, _ = _restarted(
        matvec, b, x0, precond, m, tol, max_restarts, C, U, fill, recycling=True,
    )
    return KrylovSolution(x, res, iters, converged, (C, U), drift)
