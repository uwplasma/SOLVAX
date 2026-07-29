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

from solvax import EigenSolution, eigenvalue, harmonic_krylov_schur

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
    return jnp.asarray(
        generator.standard_normal(n) + 1j * generator.standard_normal(n)
    )


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
    eigenvalues = np.concatenate(
        [[2.0 + 0j], generator.standard_normal(199) * 0.3 - 1.0 + 0j]
    )
    matrix = operator_with_spectrum(eigenvalues)
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v, start_vector(200), sigma=2.0 + 0j, k=1, m=32,
        tol=1e-9, which="largest_real",
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
        lambda v: matrix @ v, start_vector(200), sigma=target, k=1, m=32,
        tol=1e-9, max_restarts=100, which="target",
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
        lambda v: matrix @ v, start_vector(160), sigma=target, k=1, m=24,
        tol=1e-9, max_restarts=100, which="target",
    )

    value = complex(solution.eigenvalues[0])
    vector = solution.eigenvectors[0]
    independent = float(
        jnp.linalg.norm(matrix @ vector - value * vector) / abs(value)
    )
    assert independent == pytest.approx(float(solution.residuals[0]), rel=1e-6)


def test_eigenvector_matches_dense():
    """The eigenvector, not only the eigenvalue, must match the dense reference."""

    target = 0.5 + 0.2j
    eigenvalues = interior_spectrum(160, target, 5.0)
    matrix = operator_with_spectrum(eigenvalues)
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v, start_vector(160), sigma=target, k=1, m=24,
        tol=1e-9, max_restarts=100, which="target",
    )

    reference_values, reference_vectors = np.linalg.eig(np.asarray(matrix))
    index = int(np.argmin(np.abs(reference_values - target)))
    reference = reference_vectors[:, index]
    overlap = abs(np.vdot(reference, np.asarray(solution.eigenvectors[0])))
    overlap /= np.linalg.norm(reference) * float(
        jnp.linalg.norm(solution.eigenvectors[0])
    )
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
        apply, jnp.reshape(flat, shape), sigma=target, k=1, m=20,
        tol=1e-9, max_restarts=100, which="target",
    )
    assert solution.eigenvectors.shape == (1, *shape)
    assert bool(solution.converged[0])


def test_basis_stays_orthonormal():
    """Loss of orthogonality invalidates the pairs regardless of their residuals."""

    target = 0.5 + 0.2j
    matrix = operator_with_spectrum(interior_spectrum(160, target, 5.0))
    solution = harmonic_krylov_schur(
        lambda v: matrix @ v, start_vector(160), sigma=target, k=1, m=24,
        tol=1e-9, max_restarts=100, which="target",
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
        lambda v: matrix @ v, start_vector(200), sigma=target, k=1, m=32,
        tol=1e-12, max_restarts=3, which="target",
    )

    assert isinstance(solution, EigenSolution)
    assert not bool(solution.converged[0])
    assert solution.restarts == 3
    assert np.isfinite(float(solution.residuals[0]))


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
        apply, start_vector(120), sigma=target, k=1, m=24,
        tol=1e-9, max_restarts=100, which="target",
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
        lambda v: matrix @ v, start_vector(200), sigma=target, k=1, m=32,
        tol=1e-9, max_restarts=250, which="target",
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
    perturbation = jnp.asarray(
        generator.standard_normal((120, 120))
        + 1j * generator.standard_normal((120, 120))
    ) * 0.01
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
