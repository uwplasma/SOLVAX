"""Randomized Nyström preconditioning for SPD systems.

For a symmetric positive semidefinite operator ``A`` and a regularized system
``(A + mu I) x = b``, a rank-``ell`` randomized Nystrom approximation
``A_nys = U diag(lam) U^T`` built from ``ell`` operator applications yields the
preconditioner

    P^{-1} v = U diag((lam_ell + mu) / (lam + mu)) U^T v + (v - U U^T v),

which is symmetric positive definite, costs one ``(n, ell)`` matmul pair per
application, and — when ``ell`` exceeds roughly twice the ``mu``-effective
dimension of ``A`` — bounds the preconditioned condition number by a small
constant *in expectation, independently of the spectrum's decay rate*
(Frangella, Tropp & Udell, SIAM J. Matrix Anal. Appl. 44, 718 (2023)). It is
the scalable coarse-correction alternative when no grid hierarchy or
structured coarse operator exists.

The sketch uses an explicit PRNG key, so construction is deterministic,
jit-able, and differentiable through both the sketch and the eigenfactors; the
stabilized-shift construction follows Frangella et al., Algorithm 2.1, whose
shift is calibrated to the unit roundoff of the arithmetic forming the core.
Since a GPU may answer that arithmetic in a lower precision than the dtype
names, the shift here is additionally floored at the roundoff measured on the
core itself; see :func:`nystrom_preconditioner`.

The operator passed in is applied through :func:`jax.vmap` over the sketch
columns, which makes the caller's matvec a matrix-shaped contraction even when
it is vector-shaped on a single vector. On Ampere and later NVIDIA GPUs XLA
satisfies such a contraction on the tensor cores in TF32 unless the caller pins
it, so the sketch may arrive with ~4.9e-04 relative error. That is the caller's
choice to make, and the shift floor below is what keeps it from becoming a NaN
Cholesky here.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

MatVec = Callable[[jax.Array], jax.Array]


def nystrom_preconditioner(
    matvec: MatVec,
    n: int,
    rank: int,
    key: jax.Array,
    *,
    mu: float = 0.0,
    dtype=None,
) -> MatVec:
    """Build a randomized Nystrom preconditioner for ``matvec + mu I``.

    Args:
        matvec: symmetric positive semidefinite operator ``v -> A v`` on flat
            ``(n,)`` arrays; must be pure JAX. Symmetry is assumed, not
            checked.
        n: static operand dimension.
        rank: static sketch size ``ell`` (1 <= rank <= n). Effective when it
            exceeds about twice the ``mu``-effective dimension of ``A``.
        key: PRNG key for the Gaussian test matrix; fixing it makes the
            preconditioner deterministic and differentiable.
        mu: regularization shift of the system being solved (``A + mu I``).
        dtype: sketch dtype (defaults to float32/float64 per x64 mode).

    Returns:
        A symmetric-positive-definite inverse action suitable as ``precond=``
        for :func:`solvax.pcg.pcg` on the system ``(A + mu I) x = b``.
    """
    if not 1 <= rank <= n:
        raise ValueError("need 1 <= rank <= n")
    dtype = jnp.zeros(0).dtype if dtype is None else dtype

    omega = jax.random.normal(key, (n, rank), dtype)
    omega, _ = jnp.linalg.qr(omega)
    sketch = jax.vmap(matvec, in_axes=1, out_axes=1)(omega)

    # Stabilized Nystrom factorization (Frangella-Tropp-Udell, Alg. 2.1):
    # shift by nu ~ eps ||Y|| so the core Cholesky exists for psd A.
    #
    # ``nu`` is proportional to ``||Y||``, so an operator whose sketch is zero
    # -- the zero operator, or one whose range misses the sketch entirely --
    # gets no shift at all. The core is then singular, its Cholesky is zero,
    # and the triangular solve divides by it: every downstream value is NaN
    # before the eigenvalue shift is even reached. A positive ``mu`` does not
    # rescue this, because the damage is done upstream of ``mu``. The floor
    # below keeps the shift strictly positive whatever the sketch contains.
    #
    # Alg. 2.1 line 4 reads ``nu = eps(norm(Y, 'fro'))``, so the shift is
    # calibrated to the unit roundoff of the arithmetic that forms the core --
    # and on a GPU that arithmetic can be coarser than ``eps`` names. XLA
    # satisfies an unpinned *matrix-shaped* contraction (a free axis left on
    # both operands) on the Ampere tensor cores in TF32, which keeps 10
    # mantissa bits: an effective unit roundoff of 2^-11 = 4.9e-04 rather than
    # float32's 2^-24. Two things keep the calibration honest.
    #
    # The Gram matrix is pinned, which is this module's own contraction to fix.
    # Unpinned it perturbed the core by ~440x ``nu``, measured on an RTX A4000.
    # The vector-shaped contractions in ``precond`` below measured exact
    # unpinned and are deliberately left alone.
    #
    # And ``nu`` is floored at the roundoff actually achieved, because the
    # sketch is *not* ours to fix: ``jax.vmap`` over the caller's matvec makes
    # that contraction matrix-shaped too, and its precision is the caller's to
    # choose. ``omega^T A omega`` is symmetric, so the antisymmetric part of
    # the computed Gram matrix is pure rounding and measures it for free. In
    # exact arithmetic the floor is inactive and this is Alg. 2.1 line 4
    # unchanged; with a TF32 sketch it is what keeps the core definite.
    eps = jnp.finfo(dtype).eps
    sketch_norm = jnp.linalg.norm(sketch)
    gram = jnp.matmul(omega.T, sketch, precision=jax.lax.Precision.HIGHEST)
    asymmetry = gram - gram.T
    # The absolute floor lives *inside* the square root. An exactly symmetric
    # Gram matrix is reachable -- whether the two triangles round identically
    # depends on the backend's reduction order, so it is not something to rely
    # on either way -- and ``sqrt`` at zero would return a 0/0 gradient through
    # a construction this module documents as differentiable. With a symmetric
    # Gram matrix this is exactly the ``eps**2`` floor described above.
    floor = jnp.asarray(eps, dtype) ** 2
    roundoff = jnp.sqrt(jnp.sum(asymmetry * asymmetry) + floor**2)
    nu = jnp.maximum(eps * sketch_norm, roundoff)
    shifted = sketch + nu * omega
    # ``omega`` has orthonormal columns, so ``omega^T (Y + nu omega)`` is
    # ``gram + nu I``: the shift enters exactly, and the shifted core costs no
    # second product. Symmetrizing is what ``jnp.linalg.cholesky`` would do to
    # its input anyway; it is written out because the symmetry is the same
    # property ``roundoff`` above measures, and neither should be silent.
    core = jnp.linalg.cholesky(
        0.5 * (gram + gram.T) + nu * jnp.eye(rank, dtype=dtype)
    )
    half = jax.scipy.linalg.solve_triangular(core, shifted.T, lower=True).T
    basis, singular, _ = jnp.linalg.svd(half, full_matrices=False)
    eigenvalues = jnp.maximum(singular**2 - nu, 0.0)

    smallest = eigenvalues[-1]

    # A posterior read on whether the sketch was wide enough. The Nystrom
    # approximation captures the leading part of the spectrum; the ratio of the
    # smallest retained eigenvalue to the largest says how much of the decay the
    # sketch actually spans. Near one, the spectrum inside the sketch is flat
    # and the rank is almost certainly too small to be preconditioning
    # anything; near zero, the sketch reaches into the tail. This is reported
    # rather than acted on, because growing the sketch means more operator
    # applications and only the caller knows what those cost.
    largest = eigenvalues[0]
    spectrum_span = jnp.where(largest > 0, smallest / largest, jnp.ones((), dtype))

    # A direction the operator does not see should be left alone: the scale
    # factor there is one, not an indeterminate 0/0.
    #
    # The test for "does not see" has to be *relative*, not `== 0`. On a
    # rank-deficient operator the null eigenvalues come out of an SVD as
    # rounding noise, positive or zero depending on the LAPACK path, so an
    # exact-zero test makes the result depend on the linear-algebra backend --
    # measured, the same zero operator gave the identity under one JAX release
    # and arbitrary O(1) values under another. Comparing against the largest
    # retained eigenvalue makes the degenerate case deterministic.
    # Two floors. Relative to the largest retained eigenvalue, for a spectrum
    # with genuine scale; and at ``nu`` itself, because the eigenvalues are
    # formed as ``singular**2 - nu`` and so cannot resolve anything below that
    # shift. Without the second floor a zero operator has no scale to be
    # relative *to*, and the comparison is against noise.
    null_threshold = jnp.maximum(jnp.finfo(dtype).eps * jnp.maximum(largest, 0.0), nu)
    seen = (eigenvalues > null_threshold) & (eigenvalues + mu > 0.0)
    denominator = jnp.where(seen, eigenvalues + mu, 1.0)
    numerator = jnp.where(seen, smallest + mu, 1.0)

    def precond(v: jax.Array) -> jax.Array:
        projected = basis.T @ v
        scaled = numerator / denominator * projected
        return basis @ (scaled - projected) + v

    # Attached rather than returned separately so existing callers are
    # unaffected and a caller who wants the diagnostic can read it.
    precond.retained_eigenvalues = eigenvalues       # type: ignore[attr-defined]
    precond.spectrum_span = spectrum_span            # type: ignore[attr-defined]
    return precond


def nystrom_preconditioner_adaptive(
    matvec: MatVec,
    n: int,
    key: jax.Array,
    *,
    mu: float = 0.0,
    span_target: float = 1.0e-2,
    initial_rank: int = 4,
    max_rank: int | None = None,
    dtype=None,
) -> tuple[MatVec, int]:
    """Grow the sketch until it spans enough of the spectrum.

    :func:`nystrom_preconditioner` takes a fixed rank, and a rank chosen once is
    either wasteful or inadequate whenever the spectrum changes -- which is
    exactly the continuation setting the recycling machinery targets. This
    doubles the rank until the posterior ``spectrum_span`` (the smallest
    retained eigenvalue over the largest) falls below ``span_target``, meaning
    the sketch has reached into the decaying tail rather than sitting on a flat
    plateau of the spectrum.

    The loop is ordinary Python: each sketch has a different static shape, so
    the growth cannot happen inside one traced computation. Every individual
    build is still pure JAX, and the returned preconditioner is traceable as
    usual. Call this once when the operator changes, not inside a hot loop.

    Args:
        matvec: symmetric positive semidefinite operator.
        n: problem dimension.
        key: PRNG key; each attempt draws its own sketch from a fresh split.
        mu: regularization shift of the system being solved.
        span_target: stop once ``spectrum_span`` is below this.
        initial_rank: first rank tried.
        max_rank: cap; defaults to ``n``.

    Returns:
        ``(preconditioner, rank)`` -- the rank actually used, so a caller can
        record what the operator required rather than what was requested.
    """
    if not 1 <= initial_rank <= n:
        raise ValueError("need 1 <= initial_rank <= n")
    if not 0.0 < span_target < 1.0:
        raise ValueError("span_target must lie strictly between zero and one")
    ceiling = n if max_rank is None else min(int(max_rank), n)
    if ceiling < initial_rank:
        raise ValueError("max_rank must be at least initial_rank")

    rank = int(initial_rank)
    while True:
        key, attempt = jax.random.split(key)
        precond = nystrom_preconditioner(matvec, n, rank, attempt, mu=mu, dtype=dtype)
        span = float(precond.spectrum_span)  # type: ignore[attr-defined]
        if span <= span_target or rank >= ceiling:
            return precond, rank
        rank = min(rank * 2, ceiling)
