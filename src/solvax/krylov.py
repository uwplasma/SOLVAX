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

            On when to act on it, see :data:`RECYCLE_DRIFT_ADVISORY`. The
            short version, from the calibration recorded there: reuse keeps
            paying much further into the drift range than one might guess, and
            the penalty for keeping a stale pair is small, so this is a
            monitor rather than a trigger.
    """

    x: PyTree
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    recycle: tuple[jax.Array, jax.Array] | None = None
    recycle_drift: jax.Array | None = None


#: Drift above which dropping the recycle pair is worth considering.
#:
#: Calibrated on a continuation of a 200x200 diagonally dominant operator
#: perturbed along a fixed direction, ``m = 20``, ``k = 8``, ``rtol = 1e-10``,
#: comparing a warm-started solve against a cold one at the same parameter:
#:
#: =========  =====================  ============
#: drift      warm / cold iterations  verdict
#: =========  =====================  ============
#: 3.5e-05    12 / 21                 reuse helps
#: 3.5e-03    16 / 21                 reuse helps
#: 3.5e-02    18 / 21                 reuse helps
#: 0.33       28 / 30                 reuse helps
#: 0.58       69 / 70                 reuse helps
#: 0.73       449 / 432               reuse costs 4%
#: 0.87       neither converges       operator is the problem
#: =========  =====================  ============
#:
#: Two things that calibration settles. Reuse pays over almost the whole range
#: -- a pair whose subspace has rotated by a third of a right angle on average
#: still beats starting cold -- so a conservative threshold throws away most of
#: the benefit. And the downside is mild: keeping an obviously stale pair cost
#: four percent, not a factor. Dropping is therefore worth doing above roughly
#: this value and not worth agonizing over below it.
#:
#: This is one operator family. A problem whose spectrum reorganizes rather
#: than shifts can invalidate a recycle pair at much lower drift, so treat the
#: number as a starting point and confirm it by comparing against a cold solve
#: on your own continuation.
RECYCLE_DRIFT_ADVISORY = 0.6


def _identity(v: jax.Array) -> jax.Array:
    return v


def _reshaped(fn: MatVec, shape: tuple[int, ...]) -> MatVec:
    """Run ``fn`` on the caller's array shape while the solver stays flat.

    The reshapes are metadata-only for contiguous arrays, so a
    multidimensional state costs nothing to iterate on and the operator
    never sees a raveled vector.
    """

    def apply(value: jax.Array) -> jax.Array:
        return jnp.reshape(fn(value.reshape(shape)), -1)

    return apply


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
        Tuple ``(dx, adx, k_done, res_est, factorization)``: the correction
        ``dx = Z y - U B y``, its image ``adx = A dx`` (reconstructed from
        the Arnoldi relation, no extra matvec), the number of inner steps
        taken (``int32``), the final least-squares residual norm, and the
        cycle's Arnoldi factorization ``(V, Z, H, B)`` for deflated
        restarting.
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
    return dx, adx, j_f, res_est, (V, Z, H, B)


def _orthonormalize_recycle(
    c_new: jax.Array, u_new: jax.Array, threshold: jax.Array
):
    """Re-establish ``A U = C`` with orthonormal ``C`` by a thin QR.

    Given any consistent pair ``(C_new, U_new)`` with ``A U_new = C_new``,
    the QR factorization ``C_new = Q R`` gives the equivalent pair
    ``(Q, U_new R^{-1})`` spanning the same space with an orthonormal image
    basis. Numerically rank-deficient columns — zero padding from a short
    solve, or duplicated directions — are masked out and sorted to the back
    so FIFO insertion refills them first.

    Args:
        c_new: candidate image basis ``(n, k)``.
        u_new: matching source basis ``(n, k)``.
        threshold: relative floor on ``|diag(R)|`` below which a column is
            treated as rank deficient.

    Returns:
        ``(C, U, fill)`` with ``fill`` the number of retained columns.
    """
    q, r = jnp.linalg.qr(c_new)
    magnitudes = jnp.abs(jnp.diagonal(r))
    good = magnitudes > threshold * jnp.max(magnitudes, initial=0.0)
    r_safe = r + jnp.diag(jnp.where(good, 0.0, 1.0).astype(r.dtype))
    u_scaled = solve_triangular(r_safe.T, u_new.T, lower=True).T
    order = jnp.argsort(jnp.logical_not(good), stable=True)
    return (
        (q * good)[:, order],
        (u_scaled * good)[:, order],
        jnp.sum(good).astype(jnp.int32),
    )


def _harmonic_recycle(
    C: jax.Array,
    U: jax.Array,
    factorization,
    j_f: jax.Array,
    m: int,
    k: int,
    threshold: jax.Array,
):
    r"""Rebuild the recycle pair from the smallest harmonic Ritz pairs.

    This is the deflated restart of GCRO-DR (Parks et al. 2006, section 3).
    Over the *augmented* space ``W = [U, Z]`` — the recycled directions
    together with the cycle's preconditioned Krylov basis — the two stored
    relations ``A U = C`` and ``A Z = C B + V_{m+1} Hbar`` combine into

        A W = Y Gbar,   Y = [C, V_{m+1}],
        Gbar = [[I_k, B], [0, Hbar]].

    Harmonic Ritz pairs ``(theta, W g)`` impose ``A W g - theta W g ⊥
    range(A W)``, i.e. ``(AW)^H (AW) g = theta (AW)^H W g``. Since ``Y`` is
    orthonormal (``V`` is built orthogonal to ``C``), this is the small
    dense generalized problem ``Gbar^H Gbar g = theta Gbar^H Y^H W g``,
    which is solved here as the ordinary eigenproblem

        (Gbar^H Gbar)^{-1} Gbar^H (Y^H W) g = mu g,   mu = 1 / theta,

    keeping the ``k`` largest ``|mu|``. Those approximate the eigenvalues
    nearest the origin — exactly the ones that make a restarted method
    stall, and what the FIFO update reaches only indirectly. Augmenting
    with ``U`` is what lets the space *accumulate* across cycles instead of
    being rebuilt from a Krylov space that the deflation has already
    emptied of those directions.

    The new pair is reconstructed from the stored bases,

        U_new = W G,     C_new = Y (Gbar G) = A U_new,

    so deflated restarting costs no operator applications at all. Columns
    beyond an early exit are padded to the identity to keep every shape
    static; their ``Y^H W`` columns vanish, so they carry ``mu = 0`` and are
    never selected, and the selected vectors are masked back onto the used
    block so the reconstruction stays exact.

    For a real operator the eigenvectors come in conjugate pairs. Rather
    than splitting a pair across the cut (which would need ``k + 1``
    columns), the real span of the selected vectors is taken and truncated
    to its leading ``k`` singular directions, keeping the shapes static.

    Args:
        C: current recycle image basis ``(n, k)``.
        U: current recycle source basis ``(n, k)`` with ``A U = C``.
        factorization: the cycle's ``(V, Z, H, B)``.
        j_f: number of inner steps the cycle actually took.
        m: static cycle size.
        k: static number of recycle directions.
        threshold: relative rank floor for the re-orthonormalization.

    Returns:
        ``(C, U, fill)`` for the next cycle.
    """
    V, Z, H, B = factorization
    dtype = H.dtype
    used = jnp.arange(m) < j_f
    active = jnp.concatenate([jnp.ones((k,), bool), used])

    # Gbar = [[I_k, B], [0, Hbar]], with the unused Hessenberg columns padded
    # to the identity so the pencil keeps full column rank.
    hessenberg = H + jnp.pad(
        jnp.diag(jnp.where(used, 0.0, 1.0).astype(dtype)), ((0, 1), (0, 0))
    )
    top = jnp.concatenate([jnp.eye(k, dtype=dtype), B], axis=1)
    bottom = jnp.concatenate([jnp.zeros((m + 1, k), dtype), hessenberg], axis=1)
    gbar = jnp.concatenate([top, bottom], axis=0)

    # Y^H W, the only quantity that needs the long vectors (four products).
    yw = jnp.concatenate(
        [
            jnp.concatenate([_adjoint(C) @ U, _adjoint(C) @ Z.T], axis=1),
            jnp.concatenate([jnp.conj(V) @ U, jnp.conj(V) @ Z.T], axis=1),
        ],
        axis=0,
    )
    pencil = jnp.linalg.solve(_adjoint(gbar) @ gbar, _adjoint(gbar) @ yw)

    values, vectors = jnp.linalg.eig(pencil)
    basis = vectors[:, jnp.argsort(-jnp.abs(values))[:k]]
    if jnp.issubdtype(dtype, jnp.complexfloating):
        basis = basis.astype(dtype)
    else:
        parts = jnp.concatenate([jnp.real(basis), jnp.imag(basis)], axis=1)
        left, _, _ = jnp.linalg.svd(parts, full_matrices=False)
        basis = left[:, :k].astype(dtype)
    basis = jnp.where(active[:, None], basis, 0.0)

    u_new = U @ basis[:k] + Z.T @ basis[k:]
    image = gbar @ basis
    c_new = C @ image[:k] + V.T @ image[k:]
    return _orthonormalize_recycle(c_new, u_new, threshold)


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
    recycling: str,
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
        recycling: static recycle update: ``"none"`` skips it entirely
            (``k`` may be 0), ``"fifo"`` keeps the cycle's own correction,
            ``"harmonic"`` deflates the smallest harmonic Ritz pairs.

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

        dx, adx, k_done, _, factorization = _fgmres_cycle(
            matvec, precond, r, beta, tol, m, C, U
        )
        x = x + dx
        # Recompute the residual exactly at the restart boundary (one extra
        # matvec per cycle): the incremental update r - adx inherits CGS2
        # orthogonality drift, and a stale small estimate would end the loop
        # while the true residual is still large.
        r = b - _gmres_matvec(matvec, x)
        res = jnp.linalg.norm(r)

        if recycling == "fifo":
            # Keep the cycle's own optimal correction. One projection pass
            # against C for numerical hygiene (adx is orthogonal to C in
            # exact arithmetic), then FIFO insertion.
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
        elif recycling == "harmonic":
            C, U, fill = _harmonic_recycle(
                C, U, factorization, k_done, m, k, C.shape[0] * eps
            )

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

    The iteration is structure-preserving: scalars, arrays of any rank, and
    arbitrary pytrees are all iterated in their own layout, so a
    multidimensional state — species, speed, pitch, and two angles, say —
    never has to be raveled and unraveled around each operator application.
    Flat ``(n,)`` arrays take a dedicated matrix path.

    Args:
        matvec: the operator ``v -> A v`` on an array or pytree; must be pure
            JAX (traceable) and preserve the input structure.
        b: right-hand side: a scalar, an array of any rank, or a pytree.
            Pytree leaves must have one common inexact dtype.
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
            or jnp.ndim(b) != 1):
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
        empty, empty, jnp.int32(0), recycling="none",
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
    recycle_strategy: str = "fifo",
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

    Two strategies decide *what* the recycle space keeps:

    - ``"fifo"`` (default) inserts one direction per cycle — the cycle's own
      optimal correction, normalized — into fixed-shape ``(n, k)`` storage.
      It is cheap, O(nk), and retains what restarting would otherwise throw
      away, but it deflates slowly-converging eigenmodes only indirectly.
    - ``"harmonic"`` is deflated restarting (GCRO-DR, Parks et al. 2006):
      each cycle recomputes the pair from the ``k`` harmonic Ritz pairs of
      smallest magnitude, which approximate the eigenvalues nearest the
      origin — the ones that actually stall a restarted method. The
      eigenproblem is ``m x m`` dense and the new pair is reconstructed from
      the Arnoldi relation, so no extra operator applications are needed.
      Because the eigenproblem is posed over the current cycle's Krylov
      space rather than over the space augmented by ``U``, the previously
      recycled directions influence it only through the deflated operator.
      This strategy calls :func:`jax.numpy.linalg.eig`, which JAX implements
      on CPU, and is not intended to be differentiated through — take
      gradients with :func:`solvax.implicit.linear_solve`, which needs no
      derivative of the solver.

    Like :func:`gmres`, the iteration preserves the operand's shape: an
    array of any rank is handed to ``matvec`` and ``precond`` in its own
    layout, so a multidimensional state needs no ravel/unravel around each
    application. The recycle pair itself is stored as flat ``(n, k)``
    columns, since it is a *basis*, not a state. Multi-leaf pytrees are not
    supported here — use :func:`gmres`, which is pytree-native.

    Args:
        matvec: the operator ``v -> A v``, on arrays shaped like ``b``.
        b: right-hand side, an array of any rank; ``n`` below is its size.
        x0: initial guess (defaults to zeros).
        precond: right preconditioner ``v -> M^{-1} v`` (identity default).
        m: static inner FGMRES cycle size.
        k: static number of recycle directions kept.
        rtol: relative tolerance on ``||b||``.
        atol: absolute tolerance floor.
        max_restarts: static maximum number of outer cycles.
        recycle: optional ``(C, U)`` pair of shape ``(n, k)`` from a
            previous :class:`KrylovSolution` to warm-start deflation.
        recycle_strategy: ``"fifo"`` or ``"harmonic"``, as above.

    Returns:
        A :class:`KrylovSolution` whose ``x`` has the shape of ``b`` and
        whose ``recycle`` field holds the updated ``(C, U)`` pair,
        zero-padded to shape ``(n, k)``.
    """
    if recycle_strategy not in ("fifo", "harmonic"):
        raise ValueError(
            f"unknown recycle_strategy {recycle_strategy!r}; "
            "expected 'fifo' or 'harmonic'"
        )
    if recycle_strategy == "harmonic" and k > m:
        raise ValueError("harmonic recycling needs k <= m")
    if jax.tree.structure(b) != jax.tree.structure(0):
        raise ValueError(
            "gcrot operates on a single array; use gmres for multi-leaf pytrees"
        )
    b = jnp.asarray(b)
    shape = b.shape
    if b.ndim != 1:
        # Keep the caller's layout: the operator and preconditioner always see
        # the state's own shape, and only the Krylov bookkeeping is flat.
        matvec = _reshaped(matvec, shape)
        precond = None if precond is None else _reshaped(precond, shape)
        b = b.reshape(-1)
        x0 = None if x0 is None else jnp.asarray(x0).reshape(-1)
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
        C, U, fill = _orthonormalize_recycle(W, U_in, n * jnp.finfo(dtype).eps)
        # Operator-drift diagnostic: how far the current operator moved the
        # recycled space, as the mean sine of the principal angles between the
        # old filled image columns and the new span.
        #
        # This is deliberately computed from singular values rather than from
        # per-column projected residuals. Principal angles are a property of
        # the two subspaces; the mean of ||(I - C C^H) c_i|| over a basis is
        # not -- mix the columns by a unitary and the subspace is unchanged
        # while that average moves. Measured on random 4-dimensional subspaces
        # of R^40, a unitary remixing shifted the per-column average by 1e-4
        # while the singular-value form held to 1e-16. The cost is one SVD of a
        # k-by-k matrix, with k the recycle dimension.
        C_in = jnp.asarray(recycle[0], dtype)
        filled = jnp.linalg.norm(C_in, axis=0) > 0.5
        # Zero-padded columns must not enter the overlap; they would register
        # as a right angle and inflate the drift.
        masked = jnp.where(filled[None, :], C_in, 0.0)
        # The singular values of the *projected residual* are the sines
        # directly. Going through cosines and sqrt(1 - c^2) instead would
        # cancel catastrophically exactly where this diagnostic is used: for
        # two nearly equal subspaces every cosine is near one, and the
        # subtraction throws away half the digits -- an unchanged subspace
        # measured 1.6e-08 rather than zero. The residual form has no such
        # cancellation and stays basis-independent, because right-multiplying
        # by a unitary leaves singular values alone.
        residual_cols = masked - C @ (_adjoint(C) @ masked)
        sines = jnp.linalg.svd(residual_cols, compute_uv=False)
        count = jnp.maximum(jnp.sum(filled), 1)
        # Only the leading `count` values belong to filled columns; the rest
        # are structural zeros from the padding.
        keep = jnp.arange(sines.shape[0]) < count
        drift = (jnp.sum(jnp.where(keep, sines, 0.0)) / count).real

    x, res, iters, converged, C, U, _ = _restarted(
        matvec, b, x0, precond, m, tol, max_restarts, C, U, fill,
        recycling=recycle_strategy,
    )
    return KrylovSolution(x.reshape(shape), res, iters, converged, (C, U), drift)
