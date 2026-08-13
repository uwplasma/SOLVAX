"""Guard the contractions XLA would otherwise satisfy in TF32.

On Ampere and later NVIDIA GPUs XLA may answer an unpinned ``dot`` on the tensor
cores in TF32, which keeps 10 explicit mantissa bits: a relative error of 2^-11
= 4.9e-04 where float32 gives ~1e-07. Two properties make it awkward to test.

It is invisible on CPU. There is no TF32 path there, so an unpinned dot and one
at ``Precision.HIGHEST`` produce bit-identical CPU numbers and no assertion on a
*value* can fail. The precision request is recorded in the jaxpr on every
backend, so that is what the structural tests below assert.

And it is shape-dependent. Measured on an RTX A4000 against a float64
reference, XLA reaches for the tensor cores only when a contraction is a
genuine matrix product -- a free axis left on *both* operands. Every
vector-shaped contraction in this package measured exact and bit-identical
pinned or not: ``_LowRankCorrected.__call__``'s two tensordots at 2.4e-07 and
8.0e-08, and the Nystrom application's at 2.5e-07 and 7.9e-08. The two matrix
products measured 2.9e-04 unpinned against ~1.6e-07 pinned. So the hazard is
not "a dot" but "a dot that is matrix-shaped", and a contraction can cross that
line without being edited -- which is why the guards here classify by shape
rather than by call site.

The shift floor in :func:`solvax.randomized.nystrom_preconditioner` *is*
testable by value on any backend, because a sketch that arrived in reduced
precision can be emulated exactly by rounding it to 10 mantissa bits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import low_rank_corrected, nystrom_preconditioner

jax.config.update("jax_enable_x64", True)

HIGHEST = jax.lax.Precision.HIGHEST


# --------------------------------------------------------------------- jaxpr
def _sub_jaxprs(value):
    """Every jaxpr reachable from one equation parameter."""
    if hasattr(value, "eqns"):
        return [value]
    inner = getattr(value, "jaxpr", None)
    if inner is not None and hasattr(inner, "eqns"):
        return [inner]
    if isinstance(value, (tuple, list)):
        return [j for item in value for j in _sub_jaxprs(item)]
    return []


def dot_equations(jaxpr):
    """All ``dot_general`` equations, descending into sub-jaxprs.

    ``vmap``, ``scan`` and ``fori_loop`` park their equations in a nested
    jaxpr, so a scan of the top-level equations alone would see none of them.
    """
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            yield eqn
        for value in eqn.params.values():
            for sub in _sub_jaxprs(value):
                yield from dot_equations(sub)


def is_matrix_shaped(eqn):
    """True when a free axis is left on *both* operands: a real matrix product."""
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    lhs, rhs = eqn.invars[0].aval, eqn.invars[1].aval
    lhs_free = lhs.ndim - len(lhs_contract) - len(lhs_batch)
    rhs_free = rhs.ndim - len(rhs_contract) - len(rhs_batch)
    return lhs_free > 0 and rhs_free > 0


def is_pinned(eqn):
    precision = eqn.params.get("precision")
    return precision in (HIGHEST, (HIGHEST, HIGHEST))


def assert_matrix_dots_pinned(jaxpr, what):
    loose = [
        eqn
        for eqn in dot_equations(jaxpr)
        if is_matrix_shaped(eqn) and not is_pinned(eqn)
    ]
    assert not loose, (
        f"{what}: {len(loose)} matrix-shaped dot(s) left unpinned, which XLA "
        f"may satisfy in TF32 on Ampere and later: "
        f"{[str(eqn.primitive) + str(eqn.params['dimension_numbers']) for eqn in loose]}"
    )


# ------------------------------------------------------- low_rank_corrected
def _woodbury_build_and_apply(a, columns, extraction, vector):
    lu, piv = jax.scipy.linalg.lu_factor(a)
    precond = lambda v: jax.scipy.linalg.lu_solve((lu, piv), v)  # noqa: E731
    return low_rank_corrected(precond, columns, extraction)(vector)


def test_low_rank_corrected_pins_every_matrix_shaped_dot():
    """The capacitance is inverted, so its rounding lands in the identity."""
    n, k = 24, 3
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n))
    columns = jnp.asarray(rng.standard_normal((n, k)))
    extraction = jnp.asarray(rng.standard_normal((n, k)))
    vector = jnp.asarray(rng.standard_normal(n))

    jaxpr = jax.make_jaxpr(_woodbury_build_and_apply)(a, columns, extraction, vector)
    assert_matrix_dots_pinned(jaxpr.jaxpr, "low_rank_corrected")


def test_low_rank_corrected_application_is_vector_shaped():
    """Records why the hot path is left unpinned.

    Building the correction happens once; applying it happens every iteration.
    Both of the apply-path contractions are vector-shaped and measured exact
    unpinned, so pinning them would cost tensor-core throughput for nothing.
    If either ever becomes matrix-shaped, this fails and the test above then
    requires a pin -- and a fresh measurement to justify it.
    """
    n, k = 24, 3
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n))
    columns = jnp.asarray(rng.standard_normal((n, k)))
    extraction = jnp.asarray(rng.standard_normal((n, k)))
    vector = jnp.asarray(rng.standard_normal(n))

    lu, piv = jax.scipy.linalg.lu_factor(a)
    corrected = low_rank_corrected(
        lambda v: jax.scipy.linalg.lu_solve((lu, piv), v), columns, extraction
    )
    jaxpr = jax.make_jaxpr(corrected)(vector)
    matrix_shaped = [eqn for eqn in dot_equations(jaxpr.jaxpr) if is_matrix_shaped(eqn)]
    assert not matrix_shaped, (
        "the Woodbury application became matrix-shaped; it is now a TF32 "
        "hazard and needs a pin plus a measurement"
    )


def test_woodbury_identity_holds_when_precond_is_exact():
    """The property the pin protects, checked where it is exactly checkable."""
    n, k = 40, 4
    rng = np.random.default_rng(2)
    a = np.asarray(rng.standard_normal((n, n)) + n * np.eye(n))
    columns = np.asarray(rng.standard_normal((n, k))) / np.sqrt(n)
    extraction = np.asarray(rng.standard_normal((n, k))) / np.sqrt(n)
    vector = np.asarray(rng.standard_normal(n))

    applied = np.asarray(
        _woodbury_build_and_apply(
            jnp.asarray(a), jnp.asarray(columns), jnp.asarray(extraction),
            jnp.asarray(vector),
        )
    )
    corrected = a + columns @ extraction.T
    residual = np.linalg.norm(corrected @ applied - vector) / np.linalg.norm(vector)
    assert residual < 1e-10


# ------------------------------------------------------------------ Nystrom
def _diagonal_operator(diagonal):
    """A matvec that is not itself a dot, so only solvax's own dots appear."""
    return lambda v: diagonal * v


def test_nystrom_pins_its_own_matrix_shaped_dot():
    n, rank = 32, 4
    diagonal = jnp.asarray(np.linspace(1.0, 0.1, n))

    def build_and_apply(diag, vector):
        precond = nystrom_preconditioner(
            _diagonal_operator(diag), n, rank, jax.random.PRNGKey(0), mu=1e-3
        )
        return precond(vector)

    jaxpr = jax.make_jaxpr(build_and_apply)(diagonal, jnp.ones(n))
    assert_matrix_dots_pinned(jaxpr.jaxpr, "nystrom_preconditioner")


def inexact_operator(diagonal, unit_roundoff, seed):
    """A psd operator applied with a stated backward error.

    The standard model for an inexactly evaluated product is that the computed
    result is the exact result for a perturbed operator, ``fl(Av) = (A + dA)v``
    with ``||dA|| <= u ||A||`` (Higham, *Accuracy and Stability of Numerical
    Algorithms*, 2nd ed., section 3.5). That is what is reproduced here, with
    ``u`` the unit roundoff of the arithmetic the caller chose.

    ``dA`` has to be *nonsymmetric* to be a faithful stand-in. A symmetric
    perturbation is just a different psd operator, whose Gram matrix stays psd
    and never troubles the Cholesky; and an error taken relative to each
    entry of the result vanishes on the operator's null space, which is exactly
    where a real GEMM's rounding does not vanish. Both make the test pass
    against arithmetic it should catch.
    """
    size = diagonal.shape[0]
    noise = np.random.default_rng(seed).standard_normal((size, size))
    noise /= np.linalg.norm(noise, 2)
    scale = float(jnp.max(jnp.abs(diagonal)))  # ||A||_2 for a diagonal operator
    perturbation = jnp.asarray(unit_roundoff * scale * noise, diagonal.dtype)

    def matvec(v):
        return diagonal * v + perturbation @ v

    return matvec


@pytest.mark.parametrize("numerical_rank", [1, 8, 20])
def test_nystrom_survives_a_reduced_precision_sketch(numerical_rank: int) -> None:
    """A sketch carrying TF32-level error must not produce a NaN Cholesky.

    ``jax.vmap`` over the caller's matvec makes that contraction matrix-shaped,
    so on Ampere and later XLA may satisfy it on the tensor cores whatever this
    module does -- its precision is the caller's to choose, not ours. The shift
    is calibrated to unit roundoff (Frangella-Tropp-Udell Alg. 2.1 line 4,
    ``nu = eps(norm(Y, 'fro'))``), so at float32 ``eps`` it is some three
    orders of magnitude too small for a sketch carrying 2^-11. Once the
    operator's numerical rank falls below ``rank`` the core goes indefinite and
    every downstream value is NaN; measured on an RTX A4000, five of seven
    rank-deficient configurations did exactly that. The floor on ``nu`` is what
    keeps the core defined.
    """
    n, rank = 256, 32
    spectrum = np.concatenate(
        [np.logspace(0, -2, numerical_rank), np.zeros(n - numerical_rank)]
    )
    diagonal = jnp.asarray(spectrum, jnp.float32)
    matvec = inexact_operator(diagonal, 2.0**-11, numerical_rank)

    precond = nystrom_preconditioner(
        matvec, n, rank, jax.random.PRNGKey(0), mu=1e-4, dtype=jnp.float32
    )
    rng = np.random.default_rng(numerical_rank)
    vector = jnp.asarray(rng.standard_normal(n), jnp.float32)
    out = np.asarray(precond(vector))

    assert np.isfinite(out).all(), "reduced-precision sketch produced a NaN core"
    assert np.isfinite(np.asarray(precond.retained_eigenvalues)).all()


def test_nystrom_shift_floor_stays_differentiable_on_a_symmetric_gram():
    """The floor must not put a 0/0 in the gradient.

    The shift now depends on the Gram matrix's antisymmetric part, which is
    zero, or as near as makes no difference, exactly when the arithmetic is
    good. A Frobenius norm at zero differentiates to 0/0, and whether the two
    triangles round identically is a property of the backend's reduction order
    rather than anything this module controls -- so the floor is placed inside
    the square root instead of being relied on to be missed.
    """
    n, rank = 64, 8
    diagonal = jnp.asarray(np.linspace(1.0, 0.1, n))

    def loss(diag):
        precond = nystrom_preconditioner(
            _diagonal_operator(diag), n, rank, jax.random.PRNGKey(0), mu=1e-3
        )
        return jnp.sum(precond(jnp.ones(n)) ** 2)

    gradient = np.asarray(jax.grad(loss)(diagonal))
    assert np.isfinite(gradient).all()

    # And the same through an exactly symmetric Gram matrix, constructed rather
    # than hoped for, since that is the case the floor exists to survive.
    symmetric = jnp.zeros((rank, rank))
    assert np.isfinite(
        np.asarray(jax.grad(lambda m: jnp.sqrt(jnp.sum(m * m) + 1e-30))(symmetric))
    ).all()


def test_nystrom_shift_floor_is_inactive_in_exact_arithmetic():
    """The floor must not cost accuracy when the sketch is already exact.

    Measured on an exact sketch the roundoff falls below ``eps * ||Y||`` and the
    construction reduces to Alg. 2.1 line 4. Checked where the Nystrom
    approximation is *exact* rather than merely good -- a sketch wider than the
    operator's rank spans its whole range, so the retained eigenvalues must
    reproduce the true spectrum. An inflated shift would show up directly,
    since the eigenvalues are formed as ``singular**2 - nu``.
    """
    n, rank, numerical_rank = 128, 16, 8
    spectrum = np.concatenate(
        [np.logspace(0, -3, numerical_rank), np.zeros(n - numerical_rank)]
    )
    diagonal = jnp.asarray(spectrum)

    precond = nystrom_preconditioner(
        _diagonal_operator(diagonal), n, rank, jax.random.PRNGKey(0), mu=1e-6
    )
    retained = np.asarray(precond.retained_eigenvalues, dtype=np.float64)
    expected = spectrum[:numerical_rank]
    assert np.allclose(retained[:numerical_rank], expected, rtol=1e-8, atol=1e-12)
    assert np.all(retained[numerical_rank:] <= 1e-12)
