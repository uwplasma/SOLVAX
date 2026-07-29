"""Tests for solvax.eigen: harmonic Krylov-Schur vs dense reference.

Every case is scored against ``numpy.linalg.eig`` on the same operator, so the
reference is exact rather than another approximation. Spectra are constructed
with prescribed eigenvalues (``Q diag(ev) Q^H`` with ``Q`` unitary) so the
answer is known in closed form and the difficulty can be dialled: what makes an
interior eigenproblem hard is the ratio of the spectral radius to the magnitude
of the wanted eigenvalue, and that ratio is a free parameter here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    EigenSolution,
    block_harmonic_krylov,
    eigenpair,
    eigenvalue,
    harmonic_krylov_schur,
)
from solvax.eigen import _extend, _harmonic_restart

jax.config.update("jax_enable_x64", True)


def operator_with_spectrum(eigenvalues, seed=0):
    """Dense normal operator whose spectrum is exactly ``eigenvalues``."""

    generator = np.random.default_rng(seed)
    n = len(eigenvalues)
    basis, _ = np.linalg.qr(
        generator.standard_normal((n, n)) + 1j * generator.standard_normal((n, n))
    )
    return jnp.asarray(basis @ np.diag(eigenvalues) @ basis.conj().T)


def start_vector(n, seed=1):
    generator = np.random.default_rng(seed)
    return jnp.asarray(generator.standard_normal(n) + 1j * generator.standard_normal(n))


def nonnormal_operator(eigenvalues, condition=30.0, seed=0):
    """Diagonalizable operator with a prescribed eigenvector condition."""

    generator = np.random.default_rng(seed)
    n = len(eigenvalues)
    left, _ = np.linalg.qr(
        generator.standard_normal((n, n)) + 1j * generator.standard_normal((n, n))
    )
    right, _ = np.linalg.qr(
        generator.standard_normal((n, n)) + 1j * generator.standard_normal((n, n))
    )
    singular_values = np.geomspace(1.0, condition, n)
    eigenvectors = left @ np.diag(singular_values) @ right.conj().T
    return jnp.asarray(eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors))


def interior_spectrum(n, target, imaginary_extent, seed=0):
    """One wanted interior eigenvalue against a cloud spread along the imaginary axis."""

    generator = np.random.default_rng(seed)
    bulk = (generator.standard_normal(n - 1) * 0.3 - 1.0) + 1j * generator.standard_normal(
        n - 1
    ) * imaginary_extent
    return np.concatenate([[target], bulk])


def test_peripheral_rightmost_is_exact():
    """The easy regime must be exact, not merely close.

    A rightmost eigenvalue well separated from a compact bulk is what plain
    Arnoldi already handles; if the restart machinery were wrong this is where
    it would show up first and unambiguously.
    """

    generator = np.random.default_rng(0)
    eigenvalues = np.concatenate([[2.0 + 0j], generator.standard_normal(199) * 0.3 - 1.0 + 0j])
    matrix = operator_with_spectrum(eigenvalues)
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(200),
        sigma=2.0 + 0j,
        k=1,
        m=32,
        tol=1e-9,
        which="largest_real",
    )

    assert bool(solution.converged[0])
    assert abs(complex(solution.eigenvalues[0]) - 2.0) < 1e-12
    assert float(solution.residuals[0]) < 1e-9


@pytest.mark.parametrize("imaginary_extent", [2.0, 10.0])
def test_interior_target_converges(imaginary_extent):
    """Interior eigenvalues are found where plain Arnoldi cannot reach them.

    With the bulk spread over ``+-imaginary_extent`` the wanted eigenvalue is not
    extremal in magnitude, so standard Rayleigh-Ritz on the same subspace returns
    the bulk. Harmonic extraction plus restarts is what makes it reachable.
    """

    target = 0.5 + 0.2j
    eigenvalues = interior_spectrum(200, target, imaginary_extent)
    matrix = operator_with_spectrum(eigenvalues)
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(200),
        sigma=target,
        k=1,
        m=32,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )

    assert bool(solution.converged[0])
    assert abs(complex(solution.eigenvalues[0]) - target) < 1e-10
    assert float(solution.residuals[0]) < 1e-9


def test_residual_is_measured_on_the_original_problem():
    """The reported residual must be ``||A x - lambda x|| / |lambda|`` on ``A``.

    Reporting the projected or transformed residual instead is the standard way
    an interior eigensolver claims success while returning the wrong eigenvalue,
    so the contract is checked directly rather than trusted.
    """

    target = 0.5 + 0.2j
    matrix = operator_with_spectrum(interior_spectrum(160, target, 5.0))
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(160),
        sigma=target,
        k=1,
        m=24,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )

    value = complex(solution.eigenvalues[0])
    vector = solution.eigenvectors[0]
    independent = float(jnp.linalg.norm(matrix @ vector - value * vector) / abs(value))
    assert independent == pytest.approx(float(solution.residuals[0]), rel=1e-6)


def test_eigenvector_matches_dense():
    """The eigenvector, not only the eigenvalue, must match the dense reference."""

    target = 0.5 + 0.2j
    eigenvalues = interior_spectrum(160, target, 5.0)
    matrix = operator_with_spectrum(eigenvalues)
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(160),
        sigma=target,
        k=1,
        m=24,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )

    reference_values, reference_vectors = np.linalg.eig(np.asarray(matrix))
    index = int(np.argmin(np.abs(reference_values - target)))
    reference = reference_vectors[:, index]
    overlap = abs(np.vdot(reference, np.asarray(solution.eigenvectors[0])))
    overlap /= np.linalg.norm(reference) * float(jnp.linalg.norm(solution.eigenvectors[0]))
    assert overlap > 1.0 - 1e-8


def test_structured_state_shape_is_preserved():
    """Callers with structured state should not have to flatten it.

    GKX's operator acts on ``(Nl, Nm, ky, kx, z)`` arrays; requiring a manual
    flatten at the call site is how shape bugs enter.
    """

    target = 0.5 + 0.2j
    shape = (4, 5, 2)
    matrix = operator_with_spectrum(interior_spectrum(40, target, 3.0))
    flat = start_vector(40)

    def apply(state):
        assert state.shape == shape
        return jnp.reshape(matrix @ jnp.reshape(state, (-1,)), shape)

    solution = harmonic_krylov_schur(
        apply,
        jnp.reshape(flat, shape),
        sigma=target,
        k=1,
        m=20,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )
    assert solution.eigenvectors.shape == (1, *shape)
    assert bool(solution.converged[0])


def test_basis_stays_orthonormal():
    """Loss of orthogonality invalidates the pairs regardless of their residuals."""

    target = 0.5 + 0.2j
    matrix = operator_with_spectrum(interior_spectrum(160, target, 5.0))
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(160),
        sigma=target,
        k=1,
        m=24,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )
    assert solution.orthogonality < 1e-8


def test_nonconvergence_is_reported_not_raised():
    """A hard case must return its best effort flagged, not raise.

    A caller running a resolution ladder wants the partial answer plus the
    diagnostics; raising would discard both.
    """

    target = 0.5 + 0.2j
    matrix = operator_with_spectrum(interior_spectrum(200, target, 60.0))
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(200),
        sigma=target,
        k=1,
        m=32,
        tol=1e-12,
        max_restarts=3,
        which="target",
    )

    assert isinstance(solution, EigenSolution)
    assert not bool(solution.converged[0])
    assert solution.restarts == 3
    assert np.isfinite(float(solution.residuals[0]))
    assert solution.orthogonality < 1e-8


def test_harmonic_restart_preserves_the_krylov_decomposition():
    """Translation, truncation, and recovery must preserve the original relation.

    An orthonormal retained basis is not sufficient: Schur-sorting the wrong
    projected matrix leaves an apparently healthy basis while destroying
    ``A U = U B + u b^H``. This assertion directly guards the STR-9 invariant.
    """

    generator = np.random.default_rng(11)
    n, m, keep = 72, 20, 9
    matrix = jnp.asarray(generator.standard_normal((n, n)) + 1j * generator.standard_normal((n, n)))
    initial = start_vector(n, seed=12)
    initial /= jnp.linalg.norm(initial)
    basis = jnp.zeros((m + 1, n), dtype=initial.dtype).at[0].set(initial)
    projected = jnp.zeros((m + 1, m), dtype=initial.dtype)
    residual_row = jnp.zeros(m, dtype=initial.dtype)
    basis, projected, residual_row, _matvecs = _extend(
        lambda vector: matrix @ vector,
        basis,
        projected,
        residual_row,
        0,
        m,
    )

    basis, projected, residual_row = _harmonic_restart(
        basis,
        projected,
        residual_row,
        sigma=0.3 - 0.2j,
        which="target",
        keep=keep,
        size=m,
    )
    retained = np.asarray(basis[:keep]).T
    residual_vector = np.asarray(basis[keep])
    quotient = np.asarray(projected[:keep, :keep])
    row = np.asarray(residual_row[:keep])
    defect = np.asarray(matrix) @ retained - retained @ quotient - np.outer(residual_vector, row)
    relative_defect = np.linalg.norm(defect) / np.linalg.norm(np.asarray(matrix) @ retained)

    assert relative_defect < 1e-12
    assert np.linalg.norm(retained.conj().T @ residual_vector) < 1e-12
    assert np.linalg.norm(residual_vector) == pytest.approx(1.0, abs=1e-12)


def test_refined_extraction_reduces_fixed_subspace_residual():
    """Refinement must improve the vector, not merely rename the extraction."""

    n = 120
    target = 0.5 + 0.2j
    eigenvalues = interior_spectrum(n, target, 20.0, seed=5)
    matrix = jnp.asarray(
        np.diag(eigenvalues) + np.diag(np.ones(n - 1), 1) + 0.2 * np.diag(np.ones(n - 2), 2)
    )
    options = dict(
        sigma=target,
        k=1,
        m=28,
        tol=1e-12,
        max_restarts=5,
        which="target",
    )
    unrefined = harmonic_krylov_schur(
        lambda vector: matrix @ vector,
        start_vector(n, seed=3),
        refined=False,
        **options,
    )
    refined = harmonic_krylov_schur(
        lambda vector: matrix @ vector,
        start_vector(n, seed=3),
        refined=True,
        **options,
    )

    assert float(refined.residuals[0]) < float(unrefined.residuals[0])


def test_nonnormal_interior_spectrum_converges():
    """A triangular nonnormal operator must not be mistaken for a normal case."""

    n = 120
    target = 0.5 + 0.2j
    eigenvalues = interior_spectrum(n, target, 20.0, seed=5)
    matrix = jnp.asarray(
        np.diag(eigenvalues) + np.diag(np.ones(n - 1), 1) + 0.2 * np.diag(np.ones(n - 2), 2)
    )
    solution = harmonic_krylov_schur(
        lambda vector: matrix @ vector,
        start_vector(n, seed=3),
        sigma=target,
        k=1,
        m=28,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )

    assert bool(solution.converged[0])
    assert abs(complex(solution.eigenvalues[0]) - target) < 1e-10


def test_block_harmonic_krylov_resolves_a_cluster_without_duplicates():
    """A candidate block must retain distinct nearby branches through restart."""

    targets = np.asarray([0.50 + 0.20j, 0.48 + 0.22j, 0.46 + 0.18j])
    bulk = interior_spectrum(77, -0.8 + 4.0j, 25.0, seed=9)
    matrix = nonnormal_operator(np.concatenate((targets, bulk)), condition=25.0, seed=4)
    dense_values, dense_vectors = np.linalg.eig(np.asarray(matrix))
    candidate_indices = [int(np.argmin(np.abs(dense_values - target))) for target in targets]
    generator = np.random.default_rng(6)
    candidates = dense_vectors[:, candidate_indices].T
    candidates += 1.0e-8 * (
        generator.standard_normal(candidates.shape)
        + 1j * generator.standard_normal(candidates.shape)
    )
    sigma = 0.48 + 0.2j
    shifted = matrix - sigma * jnp.eye(matrix.shape[0])
    solution = block_harmonic_krylov(
        lambda vector: matrix @ vector,
        start_vector(80, seed=5),
        initial_subspace=jnp.asarray(candidates),
        subspace_apply=lambda vector: jnp.linalg.solve(shifted, vector),
        sigma=sigma,
        k=3,
        m=36,
        block_size=5,
        restart_keep=10,
        tol=1e-9,
        max_restarts=50,
    )

    observed = np.asarray(solution.eigenvalues)
    matching = [
        np.min(np.abs(observed - target))
        for target in targets
    ]
    pair_separation = np.min(
        np.abs(observed[:, None] - observed[None, :] + np.eye(3))
    )
    assert np.max(matching) < 2e-7
    assert pair_separation > 1e-3
    assert np.all(np.asarray(solution.converged))
    assert solution.orthogonality < 1e-10


def test_block_harmonic_krylov_accepts_recycled_structured_candidates():
    """Continuation candidates keep their state shape and accelerate a nearby solve."""

    shape = (4, 5)
    n = int(np.prod(shape))
    targets = np.asarray([0.5 + 0.2j, 0.45 + 0.25j])
    eigenvalues = np.concatenate((targets, interior_spectrum(n - 2, -0.7 + 3j, 15.0)))
    matrix = nonnormal_operator(eigenvalues, condition=10.0, seed=7)
    dense_values, dense_vectors = np.linalg.eig(np.asarray(matrix))
    candidate_indices = [int(np.argmin(np.abs(dense_values - target))) for target in targets]
    candidates = jnp.reshape(
        jnp.asarray(dense_vectors[:, candidate_indices].T),
        (2, *shape),
    )

    def apply(state):
        return jnp.reshape(matrix @ jnp.ravel(state), shape)

    solution = block_harmonic_krylov(
        apply,
        jnp.reshape(start_vector(n, seed=8), shape),
        initial_subspace=candidates,
        sigma=0.48 + 0.22j,
        k=2,
        m=12,
        block_size=3,
        tol=1e-10,
        max_restarts=5,
    )

    assert solution.eigenvectors.shape == (2, *shape)
    assert np.all(np.asarray(solution.converged))
    assert np.max(
        [np.min(np.abs(np.asarray(solution.eigenvalues) - target)) for target in targets]
    ) < 1e-9


def test_block_harmonic_krylov_handles_an_exact_target_in_the_seed_subspace():
    """An exact target makes the harmonic pencil singular, not the eigenpair bad."""

    n = 24
    targets = np.asarray([0.5 + 0.2j, 0.46 + 0.18j])
    eigenvalues = np.concatenate((targets, interior_spectrum(n - 2, -1.0 + 3j, 20.0)))
    matrix = jnp.diag(jnp.asarray(eigenvalues))
    candidates = jnp.eye(n, dtype=matrix.dtype)[:2]
    solution = block_harmonic_krylov(
        lambda vector: matrix @ vector,
        candidates[0],
        initial_subspace=candidates,
        sigma=targets[0],
        k=2,
        m=10,
        block_size=3,
        tol=1e-12,
        max_restarts=2,
    )

    assert np.all(np.asarray(solution.converged))
    assert np.max(
        [np.min(np.abs(np.asarray(solution.eigenvalues) - target)) for target in targets]
    ) < 1e-12


def test_block_harmonic_krylov_supports_a_rational_subspace_operator():
    """A shifted inverse may generate the subspace without changing residual semantics."""

    n = 48
    targets = np.asarray([0.5 + 0.2j, 0.46 + 0.18j])
    eigenvalues = np.concatenate((targets, interior_spectrum(n - 2, -1.0 + 2j, 40.0)))
    matrix = nonnormal_operator(eigenvalues, condition=12.0, seed=10)
    sigma = 0.48 + 0.2j
    shifted = matrix - sigma * jnp.eye(n)
    solution = block_harmonic_krylov(
        lambda vector: matrix @ vector,
        start_vector(n, seed=11),
        subspace_apply=lambda vector: jnp.linalg.solve(shifted, vector),
        sigma=sigma,
        k=2,
        m=14,
        block_size=3,
        tol=1e-9,
        max_restarts=5,
    )

    assert np.all(np.asarray(solution.converged))
    assert np.max(
        [np.min(np.abs(np.asarray(solution.eigenvalues) - target)) for target in targets]
    ) < 1e-8


def test_transformed_rightmost_subspace_uses_rayleigh_ritz_extraction():
    """An amplification filter must not be undone by an unrelated target."""

    eigenvalues = np.concatenate(
        (
            np.asarray([0.5 + 0.2j, 0.46 - 0.3j]),
            interior_spectrum(38, -0.8 + 3.0j, 20.0, seed=15),
        )
    )
    matrix = jnp.diag(jnp.asarray(eigenvalues))
    amplification = jnp.diag(jnp.exp(20.0 * jnp.asarray(eigenvalues)))
    solution = block_harmonic_krylov(
        lambda vector: matrix @ vector,
        start_vector(40, seed=16),
        subspace_apply=lambda vector: amplification @ vector,
        sigma=100.0j,
        which="largest_real",
        k=2,
        m=12,
        block_size=3,
        tol=1e-9,
        max_restarts=3,
    )

    assert np.all(np.asarray(solution.converged))
    assert (
        np.max(
            [
                np.min(np.abs(np.asarray(solution.eigenvalues) - target))
                for target in eigenvalues[:2]
            ]
        )
        < 1e-8
    )


def test_invalid_arguments_are_rejected():
    matrix = operator_with_spectrum(interior_spectrum(40, 0.5 + 0.2j, 2.0))
    v0 = start_vector(40)

    with pytest.raises(ValueError, match="k must be positive"):
        harmonic_krylov_schur(lambda v: matrix @ v, v0, k=0, m=10)
    with pytest.raises(ValueError, match="m must exceed k"):
        harmonic_krylov_schur(lambda v: matrix @ v, v0, k=8, m=8)
    with pytest.raises(ValueError, match="smaller than the operator dimension"):
        harmonic_krylov_schur(lambda v: matrix @ v, v0, k=1, m=40)
    with pytest.raises(ValueError, match="finite and nonzero"):
        harmonic_krylov_schur(lambda v: matrix @ v, jnp.zeros_like(v0), k=1, m=10)
    with pytest.raises(ValueError, match="which must be"):
        harmonic_krylov_schur(lambda v: matrix @ v, v0, k=1, m=10, which="smallest")


def test_matvec_count_is_reported():
    """Matrix-vector products are the honest cost measure for a matrix-free solve."""

    target = 0.5 + 0.2j
    matrix = operator_with_spectrum(interior_spectrum(120, target, 3.0))
    calls = {"n": 0}

    def apply(v):
        calls["n"] += 1
        return matrix @ v

    solution = harmonic_krylov_schur(
        apply,
        start_vector(120),
        sigma=target,
        k=1,
        m=24,
        tol=1e-9,
        max_restarts=100,
        which="target",
    )
    assert solution.matvecs == calls["n"]


@pytest.mark.parametrize("imaginary_extent", [30.0, 60.0, 120.0])
def test_hard_interior_spectra_converge(imaginary_extent):
    """Spectra harder than the gyrokinetic case must still converge.

    The difficulty is the ratio of spectral radius to the wanted eigenvalue's
    magnitude. ``|Im| <= 60`` is roughly the linear gyrokinetic operator that
    motivated this solver; 120 is twice that, and is included so the method is
    not merely tuned to one application.
    """

    target = 0.5 + 0.2j
    matrix = operator_with_spectrum(interior_spectrum(200, target, imaginary_extent))
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v,
        start_vector(200),
        sigma=target,
        k=1,
        m=32,
        tol=1e-9,
        max_restarts=250,
        which="target",
    )

    assert bool(solution.converged[0]), f"residual {float(solution.residuals[0]):.2e}"
    assert abs(complex(solution.eigenvalues[0]) - target) < 1e-10


def test_eigenvalue_gradient_matches_finite_differences():
    """The analytic derivative must agree with a finite difference.

    This is the gate that distinguishes a correct adjoint from a plausible one:
    forming ``A^H`` as ``conj(A conj(x))`` gives the elementwise conjugate rather
    than the conjugate transpose, which leaves the value untouched and the
    derivative wrong by a factor of several.
    """

    target = 0.5 + 0.2j
    generator = np.random.default_rng(3)
    base = operator_with_spectrum(interior_spectrum(120, target, 5.0))
    perturbation = (
        jnp.asarray(
            generator.standard_normal((120, 120)) + 1j * generator.standard_normal((120, 120))
        )
        * 0.01
    )
    v0 = start_vector(120)
    options = dict(sigma=target, m=24, tol=1e-11, max_restarts=200, which="target")

    def build(parameter):
        return lambda x: (base + parameter * perturbation) @ x

    def value(parameter):
        return jnp.real(eigenvalue(parameter, build, v0, **options))

    analytic = float(jax.grad(value)(0.0))
    step = 1e-5
    difference = float((value(step) - value(-step)) / (2 * step))
    assert analytic == pytest.approx(difference, rel=1e-6)


def test_eigenvalue_is_differentiable_through_a_structured_operator():
    """A model that never forms a matrix must still be differentiable.

    Here the operator is assembled from a diagonal and a shift, as a PDE
    linearization would be, and is applied without materializing anything.
    """

    diagonal = jnp.asarray(np.array([0.5 + 0.2j, -1.0 + 6j, -1.1 - 7j, -0.9 + 9j]))
    v0 = jnp.ones(4, dtype=complex)
    options = dict(sigma=0.5 + 0.2j, m=3, tol=1e-11, max_restarts=100, which="target")

    def build(parameter):
        return lambda x: diagonal * x + parameter * jnp.roll(x, 1)

    def value(parameter):
        return jnp.real(eigenvalue(parameter, build, v0, **options))

    analytic = float(jax.grad(value)(0.0))
    step = 1e-6
    difference = float((value(step) - value(-step)) / (2 * step))
    assert analytic == pytest.approx(difference, abs=1e-6)


def test_eigenpair_gradient_matches_phase_invariant_finite_difference():
    """Nelson's bordered tangent must differentiate eigenvector observables."""

    n = 16
    target = 0.5 + 0.2j
    generator = np.random.default_rng(15)
    base = nonnormal_operator(interior_spectrum(n, target, 5.0), condition=8.0, seed=12)
    perturbation = jnp.asarray(
        0.02
        * (
            generator.standard_normal((n, n))
            + 1j * generator.standard_normal((n, n))
        )
    )
    weight = jnp.asarray(generator.standard_normal((n, n)))
    weight = 0.5 * (weight + weight.T)
    v0 = start_vector(n, seed=13)

    def build(parameter):
        return lambda vector: (base + parameter * perturbation) @ vector

    def objective(parameter):
        value, vector = eigenpair(
            parameter,
            build,
            v0,
            sigma=target,
            m=10,
            tol=1e-11,
            max_restarts=100,
            which="target",
            sensitivity_rtol=1e-11,
        )
        normalized = vector / jnp.linalg.norm(vector)
        quadratic = jnp.real(jnp.vdot(normalized, weight @ normalized))
        return jnp.real(value) + 0.1 * quadratic

    analytic = float(jax.grad(objective)(0.0))
    step = 2e-5
    difference = float((objective(step) - objective(-step)) / (2 * step))
    assert analytic == pytest.approx(difference, rel=2e-5, abs=2e-6)


def test_eigenpair_rejects_an_exceptional_point_condition():
    """Near-defective branches must fail rather than emit explosive gradients."""

    matrix = jnp.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0e-8, 0.0, 0.0, 0.0],
            [0.0, 0.0, -2.0 + 3.0j, 0.0],
            [0.0, 0.0, 0.0, -3.0 - 4.0j],
        ],
        dtype=jnp.complex128,
    )

    def objective(parameter):
        value, _vector = eigenpair(
            parameter,
            lambda p: (
                lambda vector: (
                    matrix + p * jnp.diag(jnp.asarray([1.0, 0.0, 0.0, 0.0]))
                )
                @ vector
            ),
            jnp.ones(4, dtype=jnp.complex128),
            sigma=1.0e-4,
            m=3,
            tol=1e-10,
            max_restarts=100,
            which="target",
            condition_limit=100.0,
        )
        return jnp.real(value)

    with pytest.raises(ValueError, match="ill-conditioned"):
        jax.grad(objective)(0.0)
