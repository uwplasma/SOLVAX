"""Harmonic Krylov-Schur: interior eigenpairs of large non-Hermitian operators.

Finds eigenvalues of a matrix-free operator near a target ``sigma``, for
spectra where the wanted eigenvalue is *interior* — small in magnitude compared
with the spectral radius. Plain Arnoldi is useless there: its Ritz values
approximate the peripheral spectrum, so it returns the extremal modes no matter
how large the subspace grows.

Two ingredients, neither sufficient alone.

**Krylov-Schur restarting.** Stewart's Krylov decomposition

    A V_m = V_m B_m + v_{m+1} b_m^H

drops Arnoldi's requirement that ``B_m`` be Hessenberg. Because Krylov
decompositions are invariant under similarity transformations of ``B_m``, one
may bring ``B_m`` to Schur form, reorder the wanted Ritz values into the leading
block, and truncate — the result is still a Krylov decomposition. That is the
restart: it discards the unwanted directions and keeps a subspace enriched
toward the target, with no implicit QR sweep and hence none of the forward
instability that makes deflation awkward in implicitly restarted Arnoldi.

**Harmonic extraction.** Standard Rayleigh-Ritz extracts eigenvalues of
``B_m``. Harmonic Ritz values about ``sigma`` instead solve a projected problem
whose values approximate eigenvalues of ``A`` *closest to* ``sigma``: with
``g = (B_m - sigma I)^-H b_m``, they are the eigenvalues of ``B_m + g b_m^H``.
This is an m x m computation with m ~ 20, so it costs nothing beside the
matrix-vector products — the decisive advantage over a shift-and-invert spectral
transformation, which pays a large linear solve per iteration and needs a
preconditioner the operator may not admit.

The two must be combined. Harmonic extraction on a single Arnoldi pass does not
converge for hard interior problems; the restart is what repeatedly filters the
subspace toward ``sigma`` so the extraction has something to extract from.

Design notes:

- The projected problem (``m x m``, ``m ~ 20``) is solved on host with
  ``numpy``/``scipy``, and the basis and matrix-vector products stay in JAX.
  ``jnp.linalg.eig`` has no GPU lowering, so keeping the split explicit avoids
  a silent device-to-host round trip per restart on a large array.
- Orthogonalization is classical Gram-Schmidt applied twice (CGS2), matching
  the rest of solvax: two passes give O(eps) loss of orthogonality while
  avoiding the sequential inner-product latency of modified Gram-Schmidt.
- Convergence is judged on the residual of the *original* problem,
  ``||A x - lambda x|| / |lambda|``. Judging on the projected or transformed
  problem is the classic way an interior eigensolver reports success while
  returning the wrong eigenvalue.
- Shapes are static across restarts so the inner work stays jit-friendly; the
  restart loop itself is Python, since its trip count depends on measured
  residuals.

References
----------
- G. W. Stewart, "A Krylov-Schur algorithm for large eigenproblems",
  SIAM J. Matrix Anal. Appl. 23(3), 601-614 (2001).
- G. W. Stewart, "Addendum to A Krylov-Schur algorithm for large
  eigenproblems", SIAM J. Matrix Anal. Appl. 24(2), 599-601 (2002).
- R. B. Morgan, "Computing interior eigenvalues of large matrices",
  Linear Algebra Appl. 154-156, 289-309 (1991).
- Z. Jia, "The refined harmonic Arnoldi method and an implicitly restarted
  refined algorithm for computing interior eigenpairs of large matrices",
  Applied Numerical Mathematics 42, 489-512 (2002).
- C. C. Paige, B. N. Parlett & H. A. van der Vorst, "Approximate solutions and
  eigenvalue bounds from Krylov subspaces", Numer. Linear Algebra Appl. 2(2),
  115-133 (1995).
- J. E. Roman, M. Kammerer, F. Merz & F. Jenko, "Fast eigenvalue calculations in
  a massively parallel plasma turbulence code", Parallel Computing 36, 339-358
  (2010) — harmonic projection beats shift-and-invert by an order of magnitude
  on matrix-free non-Hermitian operators of this kind.
- J. E. Roman, "Practical implementation of harmonic Krylov-Schur",
  SLEPc Technical Report STR-9.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

__all__ = [
    "EigenSolution",
    "block_harmonic_krylov",
    "eigenpair",
    "eigenvalue",
    "harmonic_krylov_schur",
]


class EigenSolution(NamedTuple):
    """Converged eigenpairs and the diagnostics needed to trust them.

    Attributes
    ----------
    eigenvalues:
        Converged eigenvalues, ordered by ``which``.
    eigenvectors:
        Corresponding eigenvectors, stacked on the leading axis and carrying the
        operator's own shape on the trailing axes.
    residuals:
        ``||A x - lambda x|| / |lambda|`` for each pair, on the *original*
        problem.
    converged:
        Per-pair flag: residual below ``tol``.
    restarts:
        Restart cycles consumed. Equal to ``max_restarts`` when not all pairs
        converged.
    matvecs:
        Operator applications used, the honest cost measure for a matrix-free
        solve.
    orthogonality:
        ``max |V^H V - I|`` over the final basis. A value far above machine
        epsilon means the returned pairs are not trustworthy regardless of their
        residuals.
    """

    eigenvalues: jax.Array
    eigenvectors: jax.Array
    residuals: jax.Array
    converged: jax.Array
    restarts: int
    matvecs: int
    orthogonality: float


def _flatten(x: jax.Array) -> jax.Array:
    return jnp.reshape(x, (-1,))


def _cgs2(basis: jax.Array, w: jax.Array, count: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Orthogonalize ``w`` against ``basis[:count]`` with two CGS passes.

    Returns the orthogonalized vector, its norm, and the accumulated
    coefficients. Two passes are used rather than one because a single classical
    pass loses orthogonality catastrophically when the new vector is nearly
    dependent on the basis, which is exactly the regime a restarted method
    operates in.
    """

    mask = jnp.arange(basis.shape[0]) < count
    coefficients = jnp.where(mask, basis.conj() @ w, 0.0)
    w = w - coefficients @ basis
    correction = jnp.where(mask, basis.conj() @ w, 0.0)
    w = w - correction @ basis
    return w, jnp.linalg.norm(w), coefficients + correction


def _extend(
    apply: Callable[[jax.Array], jax.Array],
    basis: jax.Array,
    projected: jax.Array,
    residual_row: jax.Array,
    start: int,
    stop: int,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Grow a Krylov decomposition from dimension ``start`` to ``stop``.

    On entry ``A V[:start] = V[:start] B + v_start r^H`` holds with ``r`` the
    residual row. On exit the same relation holds at ``stop``. The first step
    absorbs ``r`` into the projected matrix, which is what lets a truncated
    Krylov-Schur decomposition be extended without rebuilding it.
    """

    matvecs = 0
    for j in range(start, stop):
        w = _flatten(apply(basis[j]))
        matvecs += 1
        w, norm, coefficients = _cgs2(basis, w, j + 1)

        projected = projected.at[:, j].set(coefficients)
        if j == start and start > 0:
            # Re-attach the previous residual row: it is the (start, :) coupling
            # of the truncated decomposition, not part of this step's projection.
            projected = projected.at[start, :start].set(residual_row[:start])

        # The subdiagonal entry belongs to the projected matrix: B_m carries
        # h_{j+1,j} for every j+1 < m, and only the final coupling h_{stop,stop-1}
        # lives in the residual row. Omitting it leaves B_m without a
        # subdiagonal, which silently destroys the decomposition.
        if j + 1 < projected.shape[1]:
            projected = projected.at[j + 1, j].set(norm)

        # A breakdown means the subspace is already invariant; leave the basis
        # vector zero and let the restart logic see a zero residual row.
        safe = jnp.where(norm > 0, norm, 1.0)
        basis = basis.at[j + 1].set(jnp.where(norm > 0, w / safe, 0.0))
        residual_row = jnp.zeros_like(residual_row).at[j].set(norm)

    return basis, projected, residual_row, matvecs


def _harmonic_projection(
    projected: np.ndarray, residual_row: np.ndarray, sigma: complex, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the harmonic Rayleigh quotient and its translation vectors.

    With ``A V = V B + v b^H`` and ``g = (B - sigma I)^-H b``, the harmonic Ritz
    values are the eigenvalues of ``B + g b^H`` (Roman et al. 2010, Eq. 21).
    The vector ``g`` is also required by the recovery step of a harmonic
    Krylov-Schur restart; computing only the eigenpairs is not enough.

    Falls back to standard Rayleigh-Ritz when ``B - sigma I`` is numerically
    singular — which happens exactly when ``sigma`` has landed on a Ritz value,
    and where the harmonic correction is both undefined and unnecessary.
    """

    block = np.asarray(projected[:size, :size])
    b = residual_row[:size].conj()
    shifted = block - sigma * np.eye(size)
    g = np.zeros(size, dtype=np.result_type(block, b, complex))

    condition = np.linalg.cond(shifted)
    condition_limit = np.finfo(float).eps ** (-0.5)
    try:
        # STR-9 notes that moderate errors in this solve are generally benign,
        # but once the projected shift loses roughly half the working digits,
        # the rank-one update becomes dominated by roundoff. At that point the
        # untranslated quotient is the stable fallback.
        if condition < condition_limit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                candidate = scipy.linalg.solve(shifted.conj().T, b, check_finite=False)
            if np.all(np.isfinite(candidate)):
                g = candidate
                block = block + np.outer(g, b.conj())
    except scipy.linalg.LinAlgError:
        # An exactly singular projected shift leaves no harmonic translation.
        pass

    return block, g, b


def _harmonic_ritz(
    projected: np.ndarray, residual_row: np.ndarray, sigma: complex, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic Ritz pairs of the projected problem about ``sigma``."""

    block, _g, _b = _harmonic_projection(projected, residual_row, sigma, size)
    values, vectors = scipy.linalg.eig(block)
    return values, vectors


def _refined_harmonic_vectors(
    projected: np.ndarray,
    residual_row: np.ndarray,
    values: np.ndarray,
    wanted: np.ndarray,
    size: int,
) -> np.ndarray:
    """Minimum-residual coefficient vectors for selected harmonic Ritz values.

    For a Krylov decomposition ``A U = U B + u b^H`` with orthonormal
    ``[U, u]``, the vector in ``span(U)`` minimizing
    ``||(A - theta I) U y||`` is the right singular vector associated with the
    smallest singular value of ``[B - theta I; b^H]``.  Harmonic Ritz values
    identify the desired interior modes; this refinement supplies substantially
    more accurate vectors for nonnormal and clustered problems.
    """

    block = np.asarray(projected[:size, :size])
    row = np.asarray(residual_row[:size])
    identity = np.eye(size, dtype=block.dtype)
    vectors = np.empty((size, len(wanted)), dtype=block.dtype)
    for column, index in enumerate(wanted):
        augmented = np.vstack((block - values[index] * identity, row[None, :]))
        _u, _singular_values, vh = scipy.linalg.svd(
            augmented, full_matrices=False, check_finite=False
        )
        vectors[:, column] = vh.conj().T[:, -1]
    return vectors


def _rank_key(values: np.ndarray, which: str, sigma: complex) -> np.ndarray:
    """Ranking key for eigenvalue selection; smaller means more wanted.

    Shared by the ordering and the Schur-selection predicate so that "wanted"
    means exactly the same thing in both places -- a mismatch there silently
    retains the wrong subspace on restart.
    """

    if which == "largest_real":
        return -values.real
    if which == "target":
        return np.abs(values - sigma)
    raise ValueError(f"which must be 'largest_real' or 'target', got {which!r}")


def _order(values: np.ndarray, which: str, sigma: complex) -> np.ndarray:
    """Index order placing the wanted eigenvalues first."""

    return np.argsort(_rank_key(values, which, sigma))


def _harmonic_restart(
    basis: jax.Array,
    projected: jax.Array,
    residual_row: jax.Array,
    *,
    sigma: complex,
    which: str,
    keep: int,
    size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Truncate and recover an orthonormal harmonic Krylov decomposition.

    This is Algorithm 3 of SLEPc Technical Report STR-9.  The Schur vectors
    must come from the harmonic Rayleigh quotient ``B + g b^H``.  Truncating
    Schur vectors of the unmodified ``B`` instead silently reverts the restart
    to standard Krylov-Schur and discards the interior spectral filter.

    After truncating the translated decomposition, the recovery translation
    restores a decomposition of the original operator and constructs a new
    residual vector orthogonal to the retained basis.  Both translations are
    essential: retaining harmonic Schur vectors without recovery does not
    leave a decomposition that Arnoldi can validly extend.
    """

    block = np.asarray(projected[:size, :size])
    row = np.asarray(residual_row[:size])
    harmonic, g, b = _harmonic_projection(block, row, sigma, size)

    spectrum = scipy.linalg.eigvals(harmonic)
    cut = np.sort(_rank_key(spectrum, which, sigma))[keep - 1]
    schur_values, schur_vectors, _selected = scipy.linalg.schur(
        harmonic,
        output="complex",
        sort=lambda value: bool(_rank_key(np.array([value]), which, sigma)[0] <= cut),
    )

    q_host = schur_vectors[:, :keep]
    b_hat = q_host.conj().T @ b
    g_hat = -(q_host.conj().T @ g)
    recovered = schur_values[:keep, :keep] + np.outer(g_hat, b_hat.conj())

    # g_tilde = (I - Q Q^H) g is the discarded component of the translation.
    # Since the old residual vector is normalized and orthogonal to U,
    # ||u - U g_tilde|| = sqrt(1 + ||g_tilde||^2).
    g_tilde = g - q_host @ (q_host.conj().T @ g)
    gamma = float(np.sqrt(1.0 + np.vdot(g_tilde, g_tilde).real))

    dtype = basis.dtype
    q = jnp.asarray(q_host, dtype=dtype)
    discarded_translation = jnp.asarray(g_tilde, dtype=dtype) @ basis[:size]
    new_residual_vector = (basis[size] - discarded_translation) / gamma

    restarted_basis = (
        jnp.zeros_like(basis).at[:keep].set(q.T @ basis[:size]).at[keep].set(new_residual_vector)
    )
    restarted_projected = (
        jnp.zeros_like(projected).at[:keep, :keep].set(jnp.asarray(recovered, dtype=dtype))
    )
    restarted_row = (
        jnp.zeros_like(residual_row).at[:keep].set(jnp.asarray(gamma * b_hat.conj(), dtype=dtype))
    )
    return restarted_basis, restarted_projected, restarted_row


def harmonic_krylov_schur(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    sigma: complex = 0.0,
    k: int = 1,
    m: int = 24,
    tol: float = 1e-10,
    max_restarts: int = 60,
    which: str = "largest_real",
    restart_keep: int | None = None,
    refined: bool = True,
) -> EigenSolution:
    """Compute ``k`` eigenpairs of a matrix-free operator near ``sigma``.

    Parameters
    ----------
    apply:
        Matrix-free operator. Receives and returns arrays shaped like ``v0``, so
        a caller with structured state need not flatten it.
    v0:
        Starting vector; its shape defines the operator's shape. Must be
        nonzero. Complex dtype is required for non-Hermitian problems — a real
        start cannot represent a complex eigenvector.
    sigma:
        Harmonic target. Eigenvalues near it converge first. Accuracy is far
        less sensitive to this than a shift-and-invert shift would be, because
        no linear system is solved with it; a value within order-unity of the
        wanted eigenvalue is enough.
    k:
        Number of eigenpairs wanted.
    m:
        Maximum subspace dimension. Must exceed ``k``; ``m ~ 2k + 20`` is
        usually ample for interior problems and larger values mostly cost
        memory.
    tol:
        Convergence tolerance on ``||A x - lambda x|| / |lambda|``, measured on
        the original problem.
    max_restarts:
        Cap on restart cycles.
    restart_keep:
        Subspace dimension retained across a restart. Defaults to ``m // 2``,
        which is what makes hard interior problems converge; smaller values
        discard the filtering the previous cycles produced.
    refined:
        If true (the default), replace each selected harmonic Ritz vector by
        the minimum-residual vector in the same subspace. This costs a small
        ``(m + 1) x m`` singular-value decomposition per requested eigenpair and
        improves eigenvectors for nonnormal and clustered operators.
    which:
        ``"largest_real"`` orders by decreasing real part — the rightmost
        eigenvalues, i.e. the most unstable modes of a physical operator.
        ``"target"`` orders by distance from ``sigma``.

    Returns
    -------
    EigenSolution
        Eigenpairs plus residuals, convergence flags, and cost/orthogonality
        diagnostics. Non-convergence is reported through ``converged`` rather
        than raised: a caller running a resolution ladder usually wants the
        partial answer and the diagnostics.

    Notes
    -----
    The restart keeps ``restart_keep`` Schur directions and rebuilds to ``m``
    each cycle, so peak memory is ``(m + 1)`` vectors — independent of how many
    restarts are needed. That is the property that makes this affordable where
    a dense eigendecomposition, needing ``O(n^2)``, is not.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if m <= k:
        raise ValueError(f"m must exceed k, got m={m}, k={k}")

    shape = v0.shape
    n = int(np.prod(shape))
    if m >= n:
        raise ValueError(
            f"m={m} must be smaller than the operator dimension {n}; "
            "use a dense eigendecomposition for problems this small"
        )

    dtype = jnp.result_type(v0, jnp.complex64)
    flat = _flatten(jnp.asarray(v0, dtype=dtype))
    norm = jnp.linalg.norm(flat)
    if not np.isfinite(float(norm)) or float(norm) == 0.0:
        raise ValueError("v0 must be finite and nonzero")

    basis = jnp.zeros((m + 1, n), dtype=dtype).at[0].set(flat / norm)
    projected = jnp.zeros((m + 1, m), dtype=dtype)
    residual_row = jnp.zeros((m,), dtype=dtype)

    def apply_flat(vector: jax.Array) -> jax.Array:
        return _flatten(apply(jnp.reshape(vector, shape)))

    matvecs = 0
    keep = 0
    restarts = 0
    active_basis_size = 1
    values = np.zeros(k, dtype=complex)
    vectors = jnp.zeros((k, n), dtype=dtype)
    residuals = np.full(k, np.inf)

    for restart_index in range(1, max_restarts + 1):
        restarts = restart_index
        basis, projected, residual_row, used = _extend(
            apply_flat, basis, projected, residual_row, keep, m
        )
        active_basis_size = m + 1
        matvecs += used

        block = np.asarray(projected[:m, :m])
        row = np.asarray(residual_row)
        ritz_values, ritz_vectors = _harmonic_ritz(block, row, sigma, m)
        order = _order(ritz_values, which, sigma)

        # Ritz vectors lift to the full space through the basis; residuals are
        # measured on the original problem, never on the projected one.
        wanted = order[:k]
        coefficient_vectors = (
            _refined_harmonic_vectors(block, row, ritz_values, wanted, m)
            if refined
            else ritz_vectors[:, wanted]
        )
        lifted = jnp.asarray(coefficient_vectors.T, dtype=dtype) @ basis[:m]
        lifted = lifted / jnp.linalg.norm(lifted, axis=1, keepdims=True)

        values = np.empty(k, dtype=complex)
        residuals = np.empty(k)
        for i in range(k):
            image = apply_flat(lifted[i])
            matvecs += 1
            # Harmonic projection is primarily a vector extraction method.
            # Its eigenvalue can be less accurate than the vector it identifies,
            # so evaluate the original-operator Rayleigh quotient before
            # measuring the residual (STR-9, section 2.2).
            values[i] = complex(jnp.vdot(lifted[i], image))
            residuals[i] = float(
                jnp.linalg.norm(image - values[i] * lifted[i]) / max(abs(values[i]), 1e-300)
            )
        vectors = lifted

        if np.all(residuals < tol):
            break

        # Restart: keep the leading harmonic Schur block, then recover an
        # orthonormal Krylov decomposition of the original operator.
        #
        # How much to retain is the one real tuning decision. Keeping only k+1
        # discards almost the whole subspace each cycle and stalls on hard
        # interior problems -- the filtering has to accumulate across restarts,
        # and it cannot if each restart throws away what the last one learned.
        # Retaining about half the subspace is the standard compromise (SLEPc
        # sizes its restart similarly): enough history to keep converging,
        # while still discarding the directions that pull toward the periphery.
        keep = max(k + 1, min(restart_keep or m // 2, m - 1))
        basis, projected, residual_row = _harmonic_restart(
            basis,
            projected,
            residual_row,
            sigma=sigma,
            which=which,
            keep=keep,
            size=m,
        )
        active_basis_size = keep + 1

    gram = basis[:active_basis_size].conj() @ basis[:active_basis_size].T
    orthogonality = float(jnp.max(jnp.abs(gram - jnp.eye(active_basis_size, dtype=gram.dtype))))

    return EigenSolution(
        eigenvalues=jnp.asarray(values),
        eigenvectors=jnp.reshape(vectors, (k, *shape)),
        residuals=jnp.asarray(residuals),
        converged=jnp.asarray(residuals < tol),
        restarts=restarts,
        matvecs=matvecs,
        orthogonality=orthogonality,
    )


def _orthonormal_block(
    vectors: jax.Array,
    *,
    against: jax.Array | None = None,
    threshold: float | None = None,
) -> jax.Array:
    """Return independent rows of ``vectors`` after two-pass projection.

    Block/recycled iterations routinely receive nearly duplicate continuation
    vectors. Silently normalizing their roundoff components creates spurious
    search directions, so rank is decided against a dtype-scaled threshold.
    """

    vectors = jnp.asarray(vectors)
    real_dtype = jnp.real(jnp.empty((), dtype=vectors.dtype)).dtype
    cutoff = float(threshold or np.sqrt(np.finfo(np.dtype(real_dtype)).eps))
    accepted: list[jax.Array] = []
    fixed = [] if against is None else [vector for vector in against]
    for candidate in vectors:
        vector = candidate
        for _pass in range(2):
            for previous in (*fixed, *accepted):
                vector = vector - jnp.vdot(previous, vector) * previous
        norm = float(jnp.linalg.norm(vector))
        if np.isfinite(norm) and norm > cutoff:
            accepted.append(vector / norm)
    if not accepted:
        return jnp.zeros((0, vectors.shape[1]), dtype=vectors.dtype)
    return jnp.stack(accepted)


def _block_harmonic_ritz(
    basis: jax.Array,
    images: jax.Array,
    sigma: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic Ritz pairs for an arbitrary orthonormal search subspace.

    If ``W = (A - sigma I) V``, harmonic Petrov--Galerkin projection imposes
    ``W^H (A V y - theta V y) = 0``. This generalized small eigenproblem does
    not require the search basis to come from one scalar Arnoldi chain, which
    is what permits several recycled branch candidates to advance together.
    """

    vectors = np.asarray(basis).T
    operator_vectors = np.asarray(images).T
    shifted = operator_vectors - sigma * vectors
    left = shifted.conj().T @ operator_vectors
    right = shifted.conj().T @ vectors
    real_dtype = np.asarray(basis).real.dtype
    condition_limit = 1.0 / np.sqrt(np.finfo(real_dtype).eps)
    condition = np.linalg.cond(right)
    if not np.isfinite(condition) or condition > condition_limit:
        # If sigma is already represented by the subspace, W=(A-sigma I)V
        # loses a column and the generalized harmonic pencil sends that exact
        # mode to infinity. Rayleigh--Ritz is exact on an invariant direction,
        # so it is the stable extraction in precisely this successful case.
        quotient = vectors.conj().T @ operator_vectors
        return scipy.linalg.eig(quotient, check_finite=False)
    values, coefficients = scipy.linalg.eig(left, right, check_finite=False)
    finite = np.isfinite(values)
    if np.any(finite):
        return values[finite], coefficients[:, finite]

    # An invariant subspace containing sigma can make the harmonic pencil
    # singular. Standard Rayleigh--Ritz is exact on that subspace and is the
    # stable fallback.
    quotient = vectors.conj().T @ operator_vectors
    return scipy.linalg.eig(quotient, check_finite=False)


def _block_refined_vectors(
    basis: jax.Array,
    images: jax.Array,
    values: np.ndarray,
    wanted: np.ndarray,
) -> np.ndarray:
    """Minimum-residual coefficient vectors in a general search subspace."""

    vectors = np.asarray(basis).T
    operator_vectors = np.asarray(images).T
    coefficients = np.empty((basis.shape[0], len(wanted)), dtype=np.asarray(basis).dtype)
    for column, index in enumerate(wanted):
        residual_operator = operator_vectors - values[index] * vectors
        _left, _singular_values, vh = scipy.linalg.svd(
            residual_operator,
            full_matrices=False,
            check_finite=False,
        )
        coefficients[:, column] = vh.conj().T[:, -1]
    return coefficients


def _seed_block(
    v0: jax.Array,
    initial_subspace: jax.Array | None,
    *,
    block_size: int,
    seed: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Build a deterministic independent starting block."""

    shape = v0.shape
    seeds = [_flatten(jnp.asarray(v0, dtype=dtype))]
    if initial_subspace is not None:
        recycled = jnp.asarray(initial_subspace, dtype=dtype)
        if recycled.ndim != v0.ndim + 1 or recycled.shape[1:] != shape:
            raise ValueError(
                "initial_subspace must have shape (candidates, *v0.shape), got "
                f"{recycled.shape} for v0 shape {shape}"
            )
        seeds.extend(_flatten(vector) for vector in recycled)

    generator = np.random.default_rng(seed)
    while len(seeds) < block_size:
        random = generator.standard_normal(v0.size) + 1j * generator.standard_normal(v0.size)
        seeds.append(jnp.asarray(random, dtype=dtype))
    return _orthonormal_block(jnp.stack(seeds))


def block_harmonic_krylov(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    sigma: complex = 0.0,
    k: int = 1,
    m: int = 48,
    block_size: int = 4,
    tol: float = 1e-10,
    max_restarts: int = 60,
    which: str = "target",
    restart_keep: int | None = None,
    initial_subspace: jax.Array | None = None,
    subspace_apply: Callable[[jax.Array], jax.Array] | None = None,
    lock: bool = True,
    lock_overlap: float = 1.0 - 1e-8,
    seed: int = 0,
) -> EigenSolution:
    """Compute a cluster of interior eigenpairs from a block/recycled subspace.

    Unlike :func:`harmonic_krylov_schur`, which advances one Arnoldi chain, this
    method advances several candidate directions together. Harmonic extraction
    is posed directly over the resulting general subspace, refined vectors
    minimize the original residual, and a thick restart retains several nearby
    branches. Converged directions may be locked and deflated from subsequent
    cycles, preventing duplicate rediscovery when ``k > 1``.

    ``initial_subspace`` accepts right eigenvectors from a neighbouring
    parameter or resolution point. ``subspace_apply`` optionally replaces
    ``apply`` only for basis generation; passing an approximate shifted inverse
    creates a rational Krylov space while residuals and Rayleigh quotients are
    still evaluated with the original operator. A transformed
    ``which="largest_real"`` search uses Rayleigh--Ritz extraction because the
    transformation has already made the wanted modes extremal; target searches
    retain harmonic extraction about ``sigma``.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if m <= max(k, block_size):
        raise ValueError("m must exceed both k and block_size")
    if which not in {"largest_real", "target"}:
        raise ValueError(f"which must be 'largest_real' or 'target', got {which!r}")
    if not 0.0 <= lock_overlap <= 1.0:
        raise ValueError("lock_overlap must lie in [0, 1]")

    shape = v0.shape
    n = int(np.prod(shape))
    if m >= n:
        raise ValueError(
            f"m={m} must be smaller than the operator dimension {n}; "
            "use a dense eigendecomposition for problems this small"
        )
    dtype = jnp.result_type(v0, jnp.complex64)
    seeds = _seed_block(
        v0,
        initial_subspace,
        block_size=max(block_size, k),
        seed=seed,
        dtype=dtype,
    )
    if seeds.shape[0] == 0:
        raise ValueError("v0 and initial_subspace contain no finite independent vector")

    def apply_flat(vector: jax.Array) -> jax.Array:
        return _flatten(apply(jnp.reshape(vector, shape)))

    if subspace_apply is None:
        generate_flat = apply_flat
    else:

        def generate_flat(vector: jax.Array) -> jax.Array:
            return _flatten(subspace_apply(jnp.reshape(vector, shape)))

    locked_vectors = jnp.zeros((0, n), dtype=dtype)
    locked_values: list[complex] = []
    locked_residuals: list[float] = []
    matvecs = 0
    restarts = 0
    active_vectors = jnp.zeros((0, n), dtype=dtype)
    active_values = np.zeros((0,), dtype=complex)
    active_residuals = np.zeros((0,), dtype=float)
    final_basis = seeds

    for restart_index in range(1, max_restarts + 1):
        restarts = restart_index
        basis = _orthonormal_block(seeds, against=locked_vectors)
        if basis.shape[0] == 0:
            break

        vectors = [vector for vector in basis]
        images: list[jax.Array] = []
        source = 0
        while source < len(vectors) and len(vectors) < m:
            vector = vectors[source]
            image = apply_flat(vector)
            images.append(image)
            matvecs += 1
            generated = image if subspace_apply is None else generate_flat(vector)
            if subspace_apply is not None:
                matvecs += 1
            candidate = _orthonormal_block(
                generated[None, :],
                against=jnp.stack([*locked_vectors, *vectors])
                if locked_vectors.shape[0]
                else jnp.stack(vectors),
            )
            if candidate.shape[0]:
                vectors.append(candidate[0])
            source += 1

        final_basis = jnp.stack(vectors[:m])
        while len(images) < final_basis.shape[0]:
            images.append(apply_flat(final_basis[len(images)]))
            matvecs += 1
        operator_images = jnp.stack(images[: final_basis.shape[0]])

        if subspace_apply is not None and which == "largest_real":
            # A transformed subspace can make a rightmost eigenvalue extremal
            # without making it interior to the projected original operator.
            # Rayleigh--Ritz is the consistent extraction in that case;
            # harmonic extraction about an unrelated sigma can discard the
            # very directions the transformation amplified.
            vectors = np.asarray(final_basis).T
            operator_vectors = np.asarray(operator_images).T
            quotient = vectors.conj().T @ operator_vectors
            ritz_values, ritz_vectors = scipy.linalg.eig(
                quotient,
                check_finite=False,
            )
        else:
            ritz_values, ritz_vectors = _block_harmonic_ritz(
                final_basis,
                operator_images,
                sigma,
            )
        order = _order(ritz_values, which, sigma)
        wanted_count = min(max(k - len(locked_values), k), len(order))
        wanted = order[:wanted_count]
        refined = _block_refined_vectors(
            final_basis,
            operator_images,
            ritz_values,
            wanted,
        )
        lifted = jnp.asarray(refined.T, dtype=dtype) @ final_basis
        lifted_images = jnp.asarray(refined.T, dtype=dtype) @ operator_images
        norms = jnp.linalg.norm(lifted, axis=1, keepdims=True)
        lifted = lifted / norms
        lifted_images = lifted_images / norms

        active_values = np.empty(len(wanted), dtype=complex)
        active_residuals = np.empty(len(wanted), dtype=float)
        for index in range(len(wanted)):
            denominator = jnp.vdot(lifted[index], lifted[index])
            value = jnp.vdot(lifted[index], lifted_images[index]) / denominator
            active_values[index] = complex(value)
            active_residuals[index] = float(
                jnp.linalg.norm(lifted_images[index] - value * lifted[index])
                / max(abs(complex(value)), 1e-300)
            )
        active_vectors = lifted

        if lock:
            for value, vector, residual in zip(
                active_values,
                active_vectors,
                active_residuals,
                strict=True,
            ):
                if residual >= tol or len(locked_values) >= k:
                    continue
                overlaps = (
                    np.abs(np.asarray(locked_vectors) @ np.asarray(vector).conj())
                    if locked_vectors.shape[0]
                    else np.zeros((0,))
                )
                if overlaps.size and float(np.max(overlaps)) >= lock_overlap:
                    continue
                locked_values.append(value)
                locked_residuals.append(residual)
                locked_vectors = jnp.concatenate((locked_vectors, vector[None, :]), axis=0)

        if len(locked_values) >= k or (
            not lock and len(active_residuals) >= k and np.all(active_residuals[:k] < tol)
        ):
            break

        retain = restart_keep or max(block_size, 2 * k)
        retain = min(max(retain, block_size), final_basis.shape[0] - 1)
        retained_coefficients = ritz_vectors[:, order[:retain]]
        retained = jnp.asarray(retained_coefficients.T, dtype=dtype) @ final_basis
        seeds = _orthonormal_block(retained, against=locked_vectors)
        if seeds.shape[0] == 0:
            seeds = _seed_block(
                v0,
                None,
                block_size=max(block_size, k),
                seed=seed + restart_index,
                dtype=dtype,
            )

    combined_values = [*locked_values]
    combined_vectors = [vector for vector in locked_vectors]
    combined_residuals = [*locked_residuals]
    for value, vector, residual in zip(
        active_values,
        active_vectors,
        active_residuals,
        strict=True,
    ):
        if len(combined_values) >= k:
            break
        overlaps = (
            np.abs(np.asarray(jnp.stack(combined_vectors)) @ np.asarray(vector).conj())
            if combined_vectors
            else np.zeros((0,))
        )
        if overlaps.size and float(np.max(overlaps)) >= lock_overlap:
            continue
        combined_values.append(value)
        combined_vectors.append(vector)
        combined_residuals.append(residual)

    if len(combined_values) < k:
        missing = k - len(combined_values)
        combined_values.extend([complex(np.nan, np.nan)] * missing)
        combined_vectors.extend([jnp.zeros((n,), dtype=dtype)] * missing)
        combined_residuals.extend([np.inf] * missing)

    values_array = np.asarray(combined_values[:k])
    residual_array = np.asarray(combined_residuals[:k])
    output_order = _order(values_array, which, sigma)
    output_vectors = jnp.stack(combined_vectors[:k])[output_order]
    gram = final_basis.conj() @ final_basis.T
    orthogonality = float(
        jnp.max(jnp.abs(gram - jnp.eye(final_basis.shape[0], dtype=gram.dtype)))
    )
    return EigenSolution(
        eigenvalues=jnp.asarray(values_array[output_order]),
        eigenvectors=jnp.reshape(output_vectors, (k, *shape)),
        residuals=jnp.asarray(residual_array[output_order]),
        converged=jnp.asarray(residual_array[output_order] < tol),
        restarts=restarts,
        matvecs=matvecs,
        orthogonality=orthogonality,
    )


def _left_eigenvector(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    value: complex,
    **options,
) -> jax.Array:
    """Left eigenvector for ``value``, i.e. the right eigenvector of ``A^H``.

    Needed only for the derivative. ``A^H`` has the conjugated spectrum, so the
    same solver finds it with a conjugated target.
    """

    # A^H, obtained from the operator's transpose. Conjugation alone is NOT
    # enough: conj(A conj(x)) is the elementwise conjugate A-bar, not the
    # conjugate transpose, and using it silently returns a wrong derivative
    # (measured 6.7x off against finite differences). jax.linear_transpose gives
    # A^T for a linear callable, and A^H x = conj(A^T conj(x)).
    transpose = jax.linear_transpose(apply, v0)

    def adjoint(vector: jax.Array) -> jax.Array:
        return jnp.conj(transpose(jnp.conj(vector))[0])

    # The adjoint's spectrum is conjugated, so the target must be too; the
    # caller's sigma is replaced rather than passed through.
    adjoint_options = {**options, "sigma": np.conj(value)}
    return harmonic_krylov_schur(adjoint, v0, **adjoint_options).eigenvectors[0]


def eigenvalue(
    theta: jax.Array,
    build: Callable[[jax.Array], Callable[[jax.Array], jax.Array]],
    v0: jax.Array,
    **options,
) -> jax.Array:
    r"""Differentiable eigenvalue of the operator ``build(theta)``.

    ``build`` maps parameters to a matrix-free operator, so any model that can
    express its linearization as a matrix-vector product is supported without
    ever forming a matrix.

    The value comes from :func:`harmonic_krylov_schur`. The derivative does
    **not** differentiate the iteration; it uses the first-order perturbation
    identity for a simple eigenvalue,

    .. math::

        d\lambda = \frac{w^H (dA)\, v}{w^H v},

    with ``v`` and ``w`` the right and left eigenvectors. Differentiating
    through a restarted solver would be expensive and fragile — restart
    decisions are discontinuous in the parameters — whereas this identity is
    exact for a simple eigenvalue and costs one extra solve for ``w`` plus one
    JVP of the operator. The same posture as implicit differentiation of a
    nonlinear solve: differentiate the equation, not the algorithm.

    Parameters
    ----------
    theta:
        Differentiable parameters.
    build:
        ``theta -> (vector -> vector)``. Must be JAX-traceable in ``theta`` for
        the derivative; the eigen-iteration itself is not traced.
    v0:
        Starting vector, shaped like the operator's input.
    **options:
        Forwarded to :func:`harmonic_krylov_schur` (``sigma``, ``m``, ``tol``,
        ``which``, ...). ``k`` is fixed to 1: a derivative is defined for one
        simple eigenvalue at a time.

    Returns
    -------
    jax.Array
        The eigenvalue, differentiable with respect to ``theta``.

    Raises
    ------
    ValueError
        If the left/right eigenvector overlap vanishes, meaning the eigenvalue
        is numerically degenerate and its first-order derivative does not exist.
        Failing here is deliberate: the alternative is a silently meaningless
        number.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, numpy as np
    >>> from solvax import eigenvalue
    >>> a0 = jnp.asarray(np.diag([0.5 + 0.2j, -1.0 + 3j, -1.2 - 4j]))
    >>> b = jnp.asarray(np.diag([1.0 + 0j, 0.0, 0.0]))
    >>> v0 = jnp.ones(3, dtype=complex)
    >>> f = lambda t: jnp.real(eigenvalue(
    ...     t, lambda p: (lambda x: (a0 + p * b) @ x), v0,
    ...     sigma=0.5 + 0.2j, m=2, which="target"))
    >>> bool(abs(jax.grad(f)(0.0) - 1.0) < 1e-8)
    True
    """

    options = {**options, "k": 1}

    @jax.custom_jvp
    def _value(parameters):
        return harmonic_krylov_schur(build(parameters), v0, **options).eigenvalues[0]

    @_value.defjvp
    def _value_jvp(primals, tangents):
        (parameters,), (dparameters,) = primals, tangents
        apply = build(parameters)
        solution = harmonic_krylov_schur(apply, v0, **options)
        value = solution.eigenvalues[0]
        right = solution.eigenvectors[0]
        left = _left_eigenvector(apply, v0, complex(value), **options)

        denominator = jnp.vdot(left, right)
        if abs(complex(denominator)) < 1e-10:
            raise ValueError(
                "left/right eigenvector overlap is ~0: the eigenvalue is "
                "degenerate and its first-order derivative does not exist"
            )
        _, image = jax.jvp(lambda p: build(p)(right), (parameters,), (dparameters,))
        return value, jnp.vdot(left, image) / denominator

    return _value(theta)


def eigenpair(
    theta: jax.Array,
    build: Callable[[jax.Array], Callable[[jax.Array], jax.Array]],
    v0: jax.Array,
    *,
    tangent_solver: Callable[[Callable, jax.Array], jax.Array] | None = None,
    sensitivity_rtol: float = 1e-9,
    sensitivity_restart: int = 40,
    sensitivity_max_restarts: int = 20,
    condition_limit: float = 1e8,
    **options,
) -> tuple[jax.Array, jax.Array]:
    r"""Differentiable eigenvalue and right eigenvector of a matrix-free operator.

    The primal pair is found by :func:`harmonic_krylov_schur`. Its tangent is
    obtained from the bordered implicit eigenproblem, not by differentiating
    restarts. With left/right vectors normalized so ``w^H v = 1``,

    .. math::

        d\lambda &= w^H (dA) v,\\
        (A-\lambda I + v w^H)\,dv &= -(dA-d\lambda I)v.

    The rank-one border removes the eigenvector nullspace and imposes the gauge
    ``w^H dv = 0``. This is Nelson's method in matrix-free form. Eigenvector
    phase is intrinsically arbitrary, so derivatives are intended for
    phase-invariant observables such as normalized fluxes and quadratic mode
    weights.

    ``condition_limit`` guards exceptional points and unresolved clusters using
    the simple-eigenvalue condition number ``||w|| ||v|| / |w^H v|``. Returning
    an enormous finite gradient there would be less honest than refusing it.
    A caller with a physics-aware bordered preconditioner may provide
    ``tangent_solver(matvec, rhs)``; otherwise restarted GMRES is used.
    """

    if condition_limit <= 1.0:
        raise ValueError("condition_limit must exceed one")
    if sensitivity_rtol <= 0.0:
        raise ValueError("sensitivity_rtol must be positive")

    options = {**options, "k": 1}

    @jax.custom_jvp
    def _pair(parameters):
        solution = harmonic_krylov_schur(build(parameters), v0, **options)
        return solution.eigenvalues[0], solution.eigenvectors[0]

    @_pair.defjvp
    def _pair_jvp(primals, tangents):
        (parameters,), (dparameters,) = primals, tangents
        apply = build(parameters)
        solution = harmonic_krylov_schur(apply, v0, **options)
        value = solution.eigenvalues[0]
        right = solution.eigenvectors[0]
        left = _left_eigenvector(apply, v0, complex(value), **options)

        overlap = jnp.vdot(left, right)
        overlap_abs = abs(complex(overlap))
        condition = float(
            jnp.linalg.norm(left) * jnp.linalg.norm(right) / max(overlap_abs, 1e-300)
        )
        if not np.isfinite(condition) or condition > condition_limit:
            raise ValueError(
                "eigenpair sensitivity is ill-conditioned: "
                f"condition number {condition:.3e} exceeds {condition_limit:.3e}; "
                "differentiate an invariant subspace or smooth the branch selection"
            )

        # Normalize the left vector biorthogonally. Dividing by conj(overlap)
        # makes vdot(left, right) exactly one under JAX's conjugating vdot.
        left = left / jnp.conj(overlap)
        _, operator_tangent = jax.jvp(
            lambda p: build(p)(right),
            (parameters,),
            (dparameters,),
        )
        value_tangent = jnp.vdot(left, operator_tangent)
        rhs = value_tangent * right - operator_tangent

        def bordered(vector):
            return (
                apply(vector)
                - value * vector
                + right * jnp.vdot(left, vector)
            )

        if tangent_solver is None:
            from solvax.implicit import linear_solve
            from solvax.krylov import gmres

            restart = min(max(1, int(np.prod(v0.shape))), sensitivity_restart)

            def solve(matvec, right_hand_side):
                return gmres(
                    matvec,
                    right_hand_side,
                    restart=restart,
                    rtol=sensitivity_rtol,
                    max_restarts=sensitivity_max_restarts,
                ).x

            # The bordered iteration contains dynamic convergence loops. Treat
            # it as an implicit linear solve so reverse mode solves the
            # transposed bordered equation instead of differentiating those
            # loops, which JAX intentionally does not support.
            vector_tangent = linear_solve(bordered, rhs, solve)
        else:
            vector_tangent = tangent_solver(bordered, rhs)
        return (value, right), (value_tangent, vector_tangent)

    return _pair(theta)
