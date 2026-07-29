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

__all__ = ["EigenSolution", "eigenvalue", "harmonic_krylov_schur"]


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

        d\lambda = rac{w^H (dA)\, v}{w^H v},

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
