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
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

MatVec = Callable[[jax.Array], jax.Array]


class KrylovSolution(NamedTuple):
    """Result of :func:`gmres` or :func:`gcrot`.

    Attributes:
        x: approximate solution, shape ``(n,)``.
        residual_norm: true residual norm ``||b - A x||`` (recomputed once
            after the iteration, not the Givens estimate).
        iterations: total inner (Arnoldi) iterations across all cycles,
            ``int32``.
        converged: whether ``residual_norm <= max(atol, rtol * ||b||)``.
        recycle: updated recycle pair ``(C, U)`` with fixed shapes
            ``(n, k)`` for :func:`gcrot`, ``None`` for :func:`gmres`.
            Unfilled columns are zero.
    """

    x: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array
    recycle: tuple[jax.Array, jax.Array] | None = None


def _identity(v: jax.Array) -> jax.Array:
    return v


def _adjoint(matrix: jax.Array) -> jax.Array:
    """Return the conjugate transpose used by complex Krylov projections."""
    return jnp.conj(matrix).T


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

        z = precond(V[j])
        w = matvec(z)
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
    r0 = b - matvec(x0)

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
        r = b - matvec(x)
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

    res = jnp.linalg.norm(b - matvec(x))
    return x, res, iters, res <= tol, C, U, fill


def gmres(
    matvec: MatVec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    precond: MatVec | None = None,
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
        matvec: the operator ``v -> A v`` on flat ``(n,)`` arrays; must be
            pure JAX (traceable).
        b: right-hand side, shape ``(n,)``.
        x0: initial guess (defaults to zeros).
        precond: right preconditioner ``v -> M^{-1} v`` (defaults to the
            identity). May be flexible, i.e. nonlinear or changing between
            inner steps — the update uses the stored preconditioned
            vectors.
        restart: static Arnoldi cycle size ``m``.
        rtol: relative tolerance on ``||b||``.
        atol: absolute tolerance floor.
        max_restarts: static maximum number of cycles.

    Returns:
        A :class:`KrylovSolution` with ``recycle=None``.
    """
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


def gmres_cycle(
    matvec: MatVec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    precond: MatVec | None = None,
    restart: int = 30,
    rtol: float = 1e-8,
    atol: float = 0.0,
) -> KrylovSolution:
    """Run one independently compilable flexible-GMRES restart cycle.

    This is the staged counterpart to :func:`gmres`: callers may JIT this
    bounded cycle once and invoke it repeatedly from a fixed Python loop. That
    keeps expensive matrix-free operators behind compiled call boundaries
    instead of lowering every restart into one monolithic executable. Fixed
    outer loops remain JAX-traceable. For reverse-mode differentiation, wrap
    the staged solver with :func:`solvax.linear_solve`, just like
    :func:`gmres`; dynamic Krylov loops are differentiated implicitly rather
    than by reversing through their iterations.

    The arguments match :func:`gmres` except that exactly one restart cycle is
    attempted. Pass the returned ``x`` as ``x0`` to continue.
    """

    b = jnp.asarray(b)
    n = b.shape[0]
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    precond = _identity if precond is None else precond
    tol = jnp.maximum(atol, rtol * jnp.linalg.norm(b))
    empty = jnp.zeros((n, 0), b.dtype)
    x, res, iters, converged, _, _, _ = _restarted(
        matvec,
        b,
        x0,
        precond,
        restart,
        tol,
        1,
        empty,
        empty,
        jnp.int32(0),
        recycling=False,
    )
    return KrylovSolution(x, res, iters, converged, None)


@jax.jit
def _staged_arnoldi_step(j, active, w, z, V, Z, R, cs, sn, g, tol):
    """Update the small Arnoldi state without tracing the matrix actions."""

    m = Z.shape[0]
    basis_mask = jnp.arange(m + 1) <= j
    basis = jnp.where(basis_mask[:, None], V, 0.0)
    h1 = jnp.conj(basis) @ w
    w = w - h1 @ basis
    h2 = jnp.conj(basis) @ w
    w = w - h2 @ basis
    h = h1 + h2
    h_next = jnp.linalg.norm(w)
    next_vector = w / jnp.where(h_next > 0, h_next, 1.0)
    candidate_V = V.at[j + 1].set(next_vector)
    h = h.at[j + 1].set(h_next)

    def rotate(i, column):
        first, second = column[i], column[i + 1]
        return column.at[i].set(cs[i] * first + sn[i] * second).at[i + 1].set(
            -jnp.conj(sn[i]) * first + cs[i] * second
        )

    h = lax.fori_loop(0, j, rotate, h)
    c_j, s_j, diagonal = _complex_givens(h[j], h[j + 1])
    h = h.at[j].set(diagonal).at[j + 1].set(0.0)
    g_j = g[j]
    candidate_g = g.at[j].set(c_j * g_j).at[j + 1].set(
        -jnp.conj(s_j) * g_j
    )
    candidate_R = R.at[:, j].set(h[:m])
    candidate_Z = Z.at[j].set(z)
    candidate_cs = cs.at[j].set(c_j)
    candidate_sn = sn.at[j].set(s_j)
    residual_estimate = jnp.abs(candidate_g[j + 1])

    def choose(candidate, previous):
        return jnp.where(active, candidate, previous)
    return (
        choose(candidate_V, V),
        choose(candidate_Z, Z),
        choose(candidate_R, R),
        choose(candidate_cs, cs),
        choose(candidate_sn, sn),
        choose(candidate_g, g),
        active & (residual_estimate > tol),
    )


def gmres_staged(
    matvec: MatVec,
    b: jax.Array,
    *,
    x0: jax.Array | None = None,
    precond: MatVec | None = None,
    restart: int = 30,
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_restarts: int = 50,
    fixed_cycles: bool = False,
    operator_sharding=None,
) -> KrylovSolution:
    """Host-stage FGMRES around opaque compiled operator call boundaries.

    Unlike :func:`gmres`, this routine does not JIT the Krylov loop. It calls
    ``matvec`` and ``precond`` from a fixed Python Arnoldi loop and JITs only
    the small dense orthogonalization update. This is useful when those actions
    are already compiled multi-device kernels that would be prohibitively
    expensive to inline into a monolithic XLA program.

    Pass ``operator_sharding`` when opaque actions require a different device
    mesh from ``b``. Inputs are placed on that sharding for each action and
    exact outputs return to the right-hand side's sharding for Arnoldi work.

    By default the host checks the true residual once per restart and exits
    early. Set ``fixed_cycles=True`` when tracing the solver through
    :func:`solvax.linear_solve`; all cycles then execute and reverse-mode uses
    implicit differentiation rather than the dynamic Krylov trace.
    """

    b = jnp.asarray(b)
    state_sharding = getattr(b, "sharding", None)

    def place_on_state(value):
        return (
            value
            if state_sharding is None
            else jax.device_put(value, state_sharding)
        )

    def apply_staged(action, value):
        if operator_sharding is not None:
            value = jax.device_put(value, operator_sharding)
        return place_on_state(action(value))

    x = jnp.zeros_like(b) if x0 is None else place_on_state(jnp.asarray(x0))
    precond = _identity if precond is None else precond
    tol = jnp.maximum(atol, rtol * jnp.linalg.norm(b))
    total_iterations = jnp.int32(0)
    residual = b - apply_staged(matvec, x)
    residual_norm = jnp.linalg.norm(residual)

    for _ in range(max_restarts):
        beta = residual_norm
        beta_safe = jnp.where(beta > 0, beta, 1.0)
        n = b.shape[0]
        dtype = b.dtype
        V = jnp.zeros((restart + 1, n), dtype).at[0].set(residual / beta_safe)
        Z = jnp.zeros((restart, n), dtype)
        R = jnp.zeros((restart, restart), dtype)
        real_dtype = jnp.real(b).dtype
        cs = jnp.zeros((restart,), real_dtype)
        sn = jnp.zeros((restart,), dtype)
        g = jnp.zeros((restart + 1,), dtype).at[0].set(beta)
        active = beta > tol
        cycle_iterations = jnp.int32(0)

        for step in range(restart):
            if not fixed_cycles and not bool(active):
                break
            was_active = active
            z = apply_staged(precond, V[step])
            w = apply_staged(matvec, z)
            V, Z, R, cs, sn, g, active = _staged_arnoldi_step(
                jnp.int32(step), active, w, z, V, Z, R, cs, sn, g, tol
            )
            cycle_iterations = cycle_iterations + was_active.astype(jnp.int32)

        used = jnp.arange(restart) < cycle_iterations
        triangular = R + jnp.diag(jnp.where(used, 0.0, 1.0).astype(dtype))
        coefficients = solve_triangular(
            triangular, jnp.where(used, g[:restart], 0.0), lower=False
        )
        x = x + coefficients @ Z
        residual = b - apply_staged(matvec, x)
        residual_norm = jnp.linalg.norm(residual)
        total_iterations = total_iterations + cycle_iterations
        if not fixed_cycles and bool(residual_norm <= tol):
            break

    return KrylovSolution(
        x, residual_norm, total_iterations, residual_norm <= tol, None
    )


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
        W = jnp.stack([matvec(U_in[:, i]) for i in range(k)], axis=1)
        Q, R = jnp.linalg.qr(W)
        diag = jnp.abs(jnp.diagonal(R))
        good = diag > n * jnp.finfo(dtype).eps * jnp.max(diag, initial=0.0)
        R_safe = R + jnp.diag(jnp.where(good, 0.0, 1.0).astype(dtype))
        U_new = solve_triangular(R_safe.T, U_in.T, lower=True).T
        order = jnp.argsort(jnp.logical_not(good), stable=True)
        C = (Q * good)[:, order]
        U = (U_new * good)[:, order]
        fill = jnp.sum(good).astype(jnp.int32)

    x, res, iters, converged, C, U, _ = _restarted(
        matvec, b, x0, precond, m, tol, max_restarts, C, U, fill, recycling=True,
    )
    return KrylovSolution(x, res, iters, converged, (C, U))
