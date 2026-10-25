"""Adaptive policies for extremal eigenmodes exposed by time propagators."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from solvax.autodiff import checkpointed_fori_loop


@dataclass(frozen=True)
class ExponentialActionPlan:
    """Static Taylor schedule for ``exp(t*A)``, valid for ``0 <= t <= horizon``.

    ``norm_bound`` must bound the induced Euclidean norm of ``A - shift*I``
    throughout the parameter neighborhood. The caller supplies and verifies it;
    neither Ritz values nor this plan validate that assumption. ``error_bound``
    bounds exact-arithmetic truncation per input norm, excluding roundoff and
    operator evaluation error. It is not a relative error bound on the output.
    """

    horizon: float
    norm_bound: float
    shift: complex
    degree: int
    substeps: int
    tolerance: float
    error_bound: float


class ExponentialActionSolution(NamedTuple):
    """Action, truncation bound, and finite/domain status (no roundoff certificate)."""

    value: jax.Array
    truncation_bound: jax.Array
    valid: jax.Array
    operator_applications: int


@dataclass(frozen=True)
class ChebyshevActionPlan:
    """Fixed-horizon action on an imaginary-major numerical-range rectangle.

    The caller verifies ``W(A-shift*I)`` lies in the centered rectangle with
    the supplied halfwidths, in the Euclidean norm of the supplied coordinates.
    For a physical norm, transform the operator and vectors consistently first.
    ``error_bound`` is an exact-arithmetic truncation estimate per input norm,
    excluding scalar coefficient, recurrence and operator roundoff. Coefficients
    and horizon are static: operator/input derivatives are supported, time
    derivatives require a Taylor plan or a separately converged construction.
    """

    horizon: float
    real_halfwidth: float
    imag_halfwidth: float
    shift: complex
    focus: float
    coefficients: tuple[complex, ...]
    tolerance: float
    error_bound: float

    @property
    def degree(self) -> int:
        """Number of operator applications per action."""
        return len(self.coefficients) - 1


def plan_chebyshev_action(
    *,
    horizon: float,
    real_halfwidth: float,
    imag_halfwidth: float,
    tolerance: float = 1.0e-10,
    shift: complex = 0.0,
    max_degree: int = 10000,
) -> ChebyshevActionPlan:
    r"""Plan a short recurrence using a verified numerical-range rectangle.

    Requires ``0 <= real_halfwidth < imag_halfwidth``. The rectangle is
    enclosed by the ellipse with semiaxes ``sqrt(2)`` times its halfwidths;
    using the halfwidths themselves would exclude the corners. For focal
    distance ``c``, ellipse parameter ``rho``, and ``r=t*c*rho/2``, the tail is
    bounded by ``2*(1+sqrt(2))*exp(t*Re(shift))*r**(m+1)/(m+1)!`` divided by
    ``1-r/(m+2)`` when ``m+2 > r``. This uses the established
    Crouzeix--Palencia constant and real Bessel coefficient bound.

    Planning runs on the host and requires SciPy (``pip install solvax[native]``)
    for Bessel coefficients; the device action needs only JAX. No numerical
    range or physical metric is inferred from Ritz values. Degree exhaustion
    and nonfinite coefficients raise; a finite status is not an accuracy proof.
    """
    horizon, alpha, beta = float(horizon), float(real_halfwidth), float(imag_halfwidth)
    tolerance, shift = float(tolerance), complex(shift)
    max_degree = operator.index(max_degree)
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError("horizon must be finite and nonnegative")
    if not math.isfinite(alpha) or not math.isfinite(beta) or not 0.0 <= alpha < beta:
        raise ValueError("require finite 0 <= real_halfwidth < imag_halfwidth")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if not math.isfinite(shift.real) or not math.isfinite(shift.imag):
        raise ValueError("shift must be finite")
    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    focus = math.sqrt(2.0) * beta * math.sqrt((1.0 - alpha / beta) * (1.0 + alpha / beta))
    if not math.isfinite(focus) or focus == 0.0:
        raise ValueError("ellipse focus is not representable; rescale the operator")
    if horizon == 0.0:
        return ChebyshevActionPlan(horizon, alpha, beta, shift, focus, (1.0 + 0j,),
                                   tolerance, 0.0)
    # c*rho = sqrt(2)*(alpha+beta); avoid overflow before dividing by two.
    r = horizon * (alpha / math.sqrt(2.0) + beta / math.sqrt(2.0))
    log_extent = math.log(beta) + math.log((1.0 + alpha / beta) / math.sqrt(2.0))
    argument = focus * horizon
    growth = shift.real * horizon
    if not math.isfinite(r) or not math.isfinite(argument) or not math.isfinite(growth):
        raise ValueError("Chebyshev truncation budget exhausted; rescale or use Krylov")
    for degree in range(max_degree + 1):
        if degree + 2 <= r:
            continue
        logtail = (math.log(2.0 * (1.0 + math.sqrt(2.0))) + growth
                   + (degree + 1) * (math.log(horizon) + log_extent)
                   - math.lgamma(degree + 2) - math.log1p(-r / (degree + 2)))
        if logtail < math.log(tolerance):
            bound = math.nextafter(math.exp(logtail), math.inf)
            if bound <= tolerance:
                break
    else:
        raise ValueError("Chebyshev truncation budget exhausted; increase max_degree or use Krylov")
    from scipy.special import jv

    coefficients = 2.0 * (1j ** np.arange(degree + 1)) * jv(np.arange(degree + 1), argument)
    coefficients[0] *= 0.5
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients *= np.exp(horizon * shift)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Chebyshev coefficients are nonfinite; rescale or use Krylov")
    return ChebyshevActionPlan(horizon, alpha, beta, shift, focus,
                               tuple(complex(x) for x in coefficients), tolerance, bound)


def plan_exponential_action(
    *,
    horizon: float,
    norm_bound: float,
    tolerance: float = 1.0e-10,
    shift: complex = 0.0,
    max_degree: int = 55,
    max_substeps: int = 4096,
) -> ExponentialActionPlan:
    r"""Minimize Taylor matvec count subject to a conservative norm tail bound.

    For degree ``m``, ``s`` substeps, and ``rho >= ||A-shift*I||``, telescoping
    the substep product gives ``s*exp(t*(rho+Re(shift)))*(t*rho/s)**(m+1)/(m+1)!``.
    Select ``m,s`` on the host, bounding its maximum over the horizon interval.
    This established Taylor-tail estimate can be much looser than adaptive
    scaling--Taylor selection. Budget exhaustion raises rather than silently
    returning an unqualified schedule. Plans are fixed during differentiation.
    """
    horizon, norm_bound, tolerance = float(horizon), float(norm_bound), float(tolerance)
    shift = complex(shift)
    max_degree, max_substeps = operator.index(max_degree), operator.index(max_substeps)
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError("horizon must be finite and nonnegative")
    if not math.isfinite(norm_bound) or norm_bound < 0.0:
        raise ValueError("norm_bound must be finite and nonnegative")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if not math.isfinite(shift.real) or not math.isfinite(shift.imag):
        raise ValueError("shift must be finite")
    if max_degree < 1 or max_substeps < 1:
        raise ValueError("max_degree and max_substeps must be positive")
    if horizon == 0.0 or norm_bound == 0.0:
        return ExponentialActionPlan(horizon, norm_bound, shift, 1, 1, tolerance, 0.0)
    candidates = []
    growth = norm_bound + shift.real
    for degree in range(1, max_degree + 1):
        peak = min(horizon, (degree + 1) / -growth) if growth < 0.0 else horizon
        log_tail = growth * peak + (degree + 1) * (math.log(peak) + math.log(norm_bound))
        log_tail -= math.lgamma(degree + 2)
        log_steps = max(0.0, (log_tail - math.log(tolerance)) / degree)
        if log_steps > math.log(max_substeps):
            continue
        substeps = max(1, math.ceil(math.exp(log_steps)))
        bound = math.nextafter(math.exp(log_tail - degree * math.log(substeps)), math.inf)
        # Protect the integer choice at a floating-point boundary.
        if bound > tolerance:
            substeps += 1
            bound = math.nextafter(math.exp(log_tail - degree * math.log(substeps)), math.inf)
        if substeps <= max_substeps:
            candidates.append((degree * substeps, substeps, degree, bound))
    if not candidates:
        raise ValueError("Taylor truncation budget exhausted; increase limits or use Krylov")
    _, substeps, degree, bound = min(candidates)
    return ExponentialActionPlan(horizon, norm_bound, shift, degree, substeps, tolerance, bound)


def exponential_action(
    apply: Callable[[jax.Array], jax.Array],
    vector: jax.Array,
    plan: ExponentialActionPlan | ChebyshevActionPlan,
    *,
    horizon: jax.Array | float | None = None,
    checkpoint: bool = True,
) -> ExponentialActionSolution:
    """Apply a fixed Taylor or Chebyshev plan to an arbitrary compatible array.

    ``apply`` is a time-independent linear map preserving shape and dtype.
    A block of RHSs is allowed when ``apply`` handles that block; norms then use
    its flattened Euclidean/Frobenius pairing. Promote inputs explicitly before
    calling if the operator or shift requires complex arithmetic. Numeric
    operator parameters, the input, and ``horizon`` remain JIT/JVP/VJP/VMAP
    operands; the host-selected plan is static. Derivatives are of this fixed
    polynomial, and need separate convergence checks against the exponential.

    Chebyshev plans require complex inputs and their fixed horizon: omit the
    ``horizon`` operand. Their three-term recurrence uses the same checkpoint
    loop; input and operator derivatives remain supported, including at zero.
    Taylor plans support real/complex inputs and a differentiable time operand.

    Checkpointing replays the recurrence using the existing bounded
    loop. ``valid`` flags nonfinite outputs/inputs and an out-of-plan horizon;
    it cannot detect a false norm bound or certify floating-point accuracy.
    """
    vector = jnp.asarray(vector)
    if not jnp.issubdtype(vector.dtype, jnp.inexact):
        raise TypeError("vector must have a floating or complex dtype")
    if isinstance(plan, ChebyshevActionPlan):
        if horizon is not None:
            raise ValueError("Chebyshev plans have a fixed horizon; omit horizon")
        if not jnp.iscomplexobj(vector):
            raise TypeError("a Chebyshev plan requires a complex vector")
    if plan.shift.imag and not jnp.iscomplexobj(vector):
        raise TypeError("a complex shift requires a complex vector")
    time = jnp.asarray(plan.horizon if horizon is None else horizon)
    if time.ndim != 0 or jnp.iscomplexobj(time):
        raise ValueError("horizon must be a real scalar")
    time = time.astype(jnp.real(vector).dtype)
    shift = jnp.asarray(plan.shift if jnp.iscomplexobj(vector) else plan.shift.real, vector.dtype)
    loop = checkpointed_fori_loop if checkpoint else jax.lax.fori_loop

    def image(state):
        result = apply(state)
        if result.shape != vector.shape or result.dtype != vector.dtype:
            raise ValueError("apply must preserve the vector shape and dtype")
        return result

    if isinstance(plan, ChebyshevActionPlan):
        coefficients = jnp.asarray(plan.coefficients, dtype=vector.dtype)
        if plan.degree == 0:
            value = coefficients[0] * vector
        else:
            first = (image(vector) - shift * vector) / (1j * plan.focus)

            def accumulate(index, carry):
                previous, current, total = carry
                following = 2.0 * (image(current) - shift * current) / (1j * plan.focus)
                following -= previous
                return current, following, total + coefficients[index] * following

            value = loop(2, plan.degree + 1, accumulate,
                         (vector, first, coefficients[0] * vector + coefficients[1] * first))[2]
        applications = plan.degree
    else:
        step = time / plan.substeps

        def substep(_index, state):
            def accumulate(index, carry):
                term, total = carry
                term = (step / index) * (image(term) - shift * term)
                return term, total + term

            _, total = loop(1, plan.degree + 1, accumulate, (state, state))
            return jnp.exp(step * shift) * total

        value = loop(0, plan.substeps, substep, vector)
        applications = plan.degree * plan.substeps
    valid = jnp.isfinite(time) & (time >= 0.0) & (time <= plan.horizon)
    valid &= jnp.all(jnp.isfinite(vector)) & jnp.all(jnp.isfinite(value))
    truncation = jnp.asarray(plan.error_bound, dtype=jnp.real(vector).dtype)
    scale = jnp.max(jnp.abs(vector), initial=0.0)
    norm = scale * jnp.linalg.norm(vector / jnp.where(scale > 0.0, scale, 1.0))
    bound = jnp.where(truncation == 0.0, 0.0, truncation * norm)
    bound = jnp.where(valid, bound, jnp.inf)
    return ExponentialActionSolution(value, bound, valid, applications)


class RK4Timestep(NamedTuple):
    """A stability-limited RK4 step inferred from an Arnoldi spectral sketch."""

    dt: float
    stability_boundary: float
    spectral_radius: float
    projected_dimension: int
    probe_count: int
    operator_applications: int


class AdaptiveEigenSolution(NamedTuple):
    """A residual-certified eigenpair and the work used to isolate it."""

    eigenvalue: jax.Array
    eigenvector: jax.Array
    residual: jax.Array
    converged: bool
    stable: bool
    restarts: int
    operator_applications: int
    filter_dt: float
    filter_steps: int
    filter_horizon: float
    filter_growth_defect: float


class PropagatorEigenSolution(NamedTuple):
    """Continuous eigenpairs extracted from one full-operator RK4 subspace."""

    eigenvalues: jax.Array
    eigenvectors: jax.Array
    residuals: jax.Array
    converged: jax.Array
    operator_applications: int


def _flatten(vector: jax.Array) -> jax.Array:
    return jnp.reshape(vector, (-1,))


def _arnoldi_spectrum(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    dimension: int,
) -> np.ndarray:
    """Estimate the peripheral spectrum with a two-pass Arnoldi sketch."""

    shape = v0.shape
    size = v0.size
    if not 1 < dimension < size:
        raise ValueError(f"dimension must lie between one and the operator size, got {dimension}")
    dtype = jnp.result_type(v0, jnp.complex64)
    vector = _flatten(jnp.asarray(v0, dtype=dtype))
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("v0 must be finite and nonzero")
    basis = jnp.zeros((dimension + 1, size), dtype=dtype).at[0].set(vector / norm)
    projected = jnp.zeros((dimension + 1, dimension), dtype=dtype)
    for column in range(dimension):
        work = _flatten(apply(jnp.reshape(basis[column], shape)))
        coefficients = basis[: column + 1].conj() @ work
        work = work - coefficients @ basis[: column + 1]
        correction = basis[: column + 1].conj() @ work
        work = work - correction @ basis[: column + 1]
        coefficients = coefficients + correction
        projected = projected.at[: column + 1, column].set(coefficients)
        norm = jnp.linalg.norm(work)
        if column + 1 < dimension:
            projected = projected.at[column + 1, column].set(norm)
        basis = basis.at[column + 1].set(
            jnp.where(norm > 0.0, work / jnp.where(norm > 0.0, norm, 1.0), 0.0)
        )
    return np.linalg.eigvals(np.asarray(projected[:dimension, :dimension]))


def _rk4_amplification(z: np.ndarray | complex) -> np.ndarray:
    z = np.asarray(z)
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def _arnoldi_basis(
    apply: Callable[[jax.Array], jax.Array],
    start: jax.Array,
    dimension: int,
) -> tuple[jax.Array, jax.Array]:
    """Build a twice-orthogonalized matrix-free Arnoldi factorization."""

    shape, dtype = start.shape, start.dtype
    norm = jnp.linalg.norm(start)
    basis = jnp.zeros((dimension + 1, *shape), dtype=dtype)
    basis = basis.at[0].set(start / jnp.where(norm > 0.0, norm, 1.0))
    projected = jnp.zeros((dimension + 1, dimension), dtype=dtype)

    def extend(column, carry):
        vectors, quotient = carry
        work = apply(vectors[column])
        operator_scale = jnp.linalg.norm(work)

        def orthogonalize(index, inner):
            candidate, matrix = inner
            coefficient = jnp.vdot(vectors[index], candidate)
            candidate = candidate - coefficient * vectors[index]
            return candidate, matrix.at[index, column].add(coefficient)

        work, quotient = jax.lax.fori_loop(0, column + 1, orthogonalize, (work, quotient))
        work, quotient = jax.lax.fori_loop(0, column + 1, orthogonalize, (work, quotient))
        next_norm = jnp.linalg.norm(work)
        threshold = (
            10.0
            * jnp.finfo(jnp.real(jnp.empty((), dtype=dtype)).dtype).eps
            * jnp.maximum(operator_scale, 1.0)
        )
        resolved = next_norm > threshold
        quotient = quotient.at[column + 1, column].set(jnp.where(resolved, next_norm, 0.0))
        vectors = vectors.at[column + 1].set(
            jnp.where(
                resolved,
                work / jnp.where(resolved, next_norm, 1.0),
                jnp.zeros_like(work),
            )
        )
        return vectors, quotient

    return jax.lax.fori_loop(0, dimension, extend, (basis, projected))


def _filtered_eigenpairs(
    apply: Callable[[jax.Array], jax.Array],
    filtered: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    krylov_dim: int,
    candidates: int,
    tol: float,
    operator_applications: int,
    restarts: int = 1,
) -> PropagatorEigenSolution:
    """Extract and certify continuous modes from a matrix-free filter."""

    @jax.jit
    def solve(initial):
        def extract(start, count):
            basis, projected = _arnoldi_basis(filtered, start, krylov_dim)
            filter_values, coefficients = jnp.linalg.eig(projected[:krylov_dim, :krylov_dim])
            indices = jnp.argsort(jnp.abs(filter_values))[-count:][::-1]
            lifted = jnp.tensordot(
                coefficients[:, indices].T,
                basis[:krylov_dim],
                axes=1,
            )
            flattened = lifted.reshape((count, -1))
            norms = jnp.linalg.norm(flattened, axis=1)
            return (flattened / jnp.where(norms > 0.0, norms, 1.0)[:, None]).reshape(
                (count, *initial.shape)
            )

        initial = jax.lax.fori_loop(
            1,
            restarts,
            lambda _index, start: extract(start, 1)[0],
            initial,
        )
        vectors = extract(initial, candidates)

        def certify(vector):
            image = apply(vector)
            denominator = jnp.vdot(vector, vector)
            value = jnp.vdot(vector, image) / jnp.where(
                denominator != 0.0,
                denominator,
                1.0 + 0.0j,
            )
            residual = jnp.linalg.norm(image - value * vector)
            residual /= jnp.maximum(
                jnp.abs(value) * jnp.linalg.norm(vector),
                jnp.finfo(jnp.real(vector).dtype).tiny,
            )
            nonzero = jnp.isfinite(denominator) & (jnp.real(denominator) > 0.0)
            residual = jnp.where(nonzero & jnp.isfinite(value), residual, jnp.inf)
            return value, residual

        values, residuals = jax.vmap(certify)(vectors)
        return values, vectors, residuals

    values, vectors, residuals = solve(v0)
    return PropagatorEigenSolution(
        eigenvalues=values,
        eigenvectors=vectors,
        residuals=residuals,
        converged=residuals < tol,
        operator_applications=operator_applications,
    )


def propagator_eigenpairs(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    dt: float,
    steps: int,
    krylov_dim: int = 24,
    candidates: int = 2,
    tol: float = 1.0e-9,
) -> PropagatorEigenSolution:
    """Return leading-growth continuous pairs from one compiled RK4 subspace.

    The full Arnoldi construction is compiled as one operation, including the
    RK4 loops, so this is suitable for an application-supplied adjoint action.
    Candidate ordering uses propagator magnitude, but values and convergence
    always come from the continuous operator.
    """

    size = int(v0.size)
    if not 1 <= candidates <= krylov_dim < size:
        raise ValueError("require 1 <= candidates <= krylov_dim < operator size")
    if dt <= 0.0 or steps < 1 or tol <= 0.0:
        raise ValueError("dt, steps, and tol must be positive")
    dtype = jnp.result_type(v0, jnp.complex64)
    initial = jnp.asarray(v0, dtype=dtype)
    dt_value = jnp.asarray(dt, dtype=jnp.real(initial).dtype)

    def rk4_step(state):
        first = apply(state)
        second = apply(state + 0.5 * dt_value * first)
        third = apply(state + 0.5 * dt_value * second)
        fourth = apply(state + dt_value * third)
        return state + (dt_value / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)

    def filtered(state):
        return jax.lax.fori_loop(
            0,
            steps,
            lambda _index, current: rk4_step(current),
            state,
        )

    return _filtered_eigenpairs(
        apply,
        filtered,
        initial,
        krylov_dim=krylov_dim,
        candidates=candidates,
        tol=tol,
        operator_applications=4 * steps * krylov_dim + candidates,
    )


def exponential_eigenpairs(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    horizon: float,
    inner_krylov_dim: int | None = None,
    outer_krylov_dim: int = 24,
    candidates: int = 2,
    tol: float = 1.0e-9,
    restarts: int = 1,
    action_plan: ExponentialActionPlan | ChebyshevActionPlan | None = None,
) -> PropagatorEigenSolution:
    """Return leading modes using Arnoldi actions of ``exp(horizon * A)``.

    Projecting each exponential action onto an inner Krylov space removes the
    explicit stability limit. A second, small Arnoldi space extracts the
    largest-magnitude propagator modes; every returned pair is certified
    against the original continuous operator.
    An optional ``action_plan`` replaces only the inner action with a fixed
    polynomial recurrence; provide either it or ``inner_krylov_dim``. Its caller-
    supplied norm bound must cover this operator and horizon. Outer extraction
    and continuous residual checks are unchanged.
    """

    size = int(v0.size)
    if not 1 <= candidates <= outer_krylov_dim < size:
        raise ValueError("require 1 <= candidates <= outer_krylov_dim < operator size")
    if action_plan is None:
        if inner_krylov_dim is None or not 1 < inner_krylov_dim <= size:
            raise ValueError("inner_krylov_dim must be in (1, operator size]")
        inner_applications = inner_krylov_dim
    else:
        if inner_krylov_dim is not None:
            raise ValueError("provide either action_plan or inner_krylov_dim")
        if not 0.0 < horizon <= action_plan.horizon:
            raise ValueError("horizon must lie in the action plan interval")
        if isinstance(action_plan, ChebyshevActionPlan) and horizon != action_plan.horizon:
            raise ValueError("Chebyshev plans have a fixed horizon")
        inner_applications = action_plan.degree
        if isinstance(action_plan, ExponentialActionPlan):
            inner_applications *= action_plan.substeps
    if horizon <= 0.0 or tol <= 0.0 or restarts < 1:
        raise ValueError("horizon, tol, and restarts must be positive")
    dtype = jnp.result_type(v0, jnp.complex64)
    initial = jnp.asarray(v0, dtype=dtype)
    horizon_value = jnp.asarray(horizon, dtype=jnp.real(initial).dtype)

    def filtered(vector):
        if action_plan is not None:
            if isinstance(action_plan, ChebyshevActionPlan):
                return exponential_action(apply, vector, action_plan).value
            return exponential_action(apply, vector, action_plan, horizon=horizon_value).value
        assert inner_krylov_dim is not None
        basis, projected = _arnoldi_basis(apply, vector, inner_krylov_dim)
        coefficients = jax.scipy.linalg.expm(
            horizon_value * projected[:inner_krylov_dim, :inner_krylov_dim]
        )[:, 0]
        return jnp.tensordot(
            coefficients,
            basis[:inner_krylov_dim],
            axes=1,
        )

    return _filtered_eigenpairs(
        apply,
        filtered,
        initial,
        krylov_dim=outer_krylov_dim,
        candidates=candidates,
        tol=tol,
        operator_applications=(restarts * inner_applications * outer_krylov_dim + candidates),
        restarts=restarts,
    )


def estimate_rk4_timestep(
    apply: Callable[[jax.Array], jax.Array],
    v0: jax.Array,
    *,
    dimension: int = 12,
    probe_count: int = 2,
    safety: float = 0.9,
    max_dt: float = np.inf,
    bisection_iterations: int = 48,
) -> RK4Timestep:
    """Choose an RK4 step that does not numerically amplify sketched modes.

    Arnoldi is used only to find inexpensive peripheral spectral sketches. The
    caller's seed is supplemented by deterministic broadband probes because a
    recycled eigenvector can be nearly invariant and blind to stability-limiting
    modes. The boundary is evaluated against the RK4 polynomial itself,
    including each Ritz value's complex angle, rather than assuming a purely
    imaginary spectrum. Positive physical growth is allowed; artificial growth
    beyond ``exp(dt * max(Re(lambda), 0))`` is not.
    """

    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie in (0, 1)")
    if probe_count < 1:
        raise ValueError("probe_count must be positive")
    if max_dt <= 0.0:
        raise ValueError("max_dt must be positive")
    seeds = [v0]
    real_dtype = jnp.real(v0).dtype
    for probe_index in range(1, probe_count):
        key_real = jax.random.PRNGKey(2 * probe_index - 1)
        probe = jax.random.normal(key_real, v0.shape, dtype=real_dtype)
        if jnp.iscomplexobj(v0):
            key_imag = jax.random.PRNGKey(2 * probe_index)
            probe = probe + 1j * jax.random.normal(
                key_imag,
                v0.shape,
                dtype=real_dtype,
            )
        seeds.append(jnp.asarray(probe, dtype=v0.dtype))
    values = np.concatenate([_arnoldi_spectrum(apply, seed, dimension) for seed in seeds])
    radius = float(np.max(np.abs(values)))
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("Arnoldi sketch did not produce a finite spectral radius")
    lower = 0.0
    upper = min(float(max_dt), 4.0 / radius)
    for _ in range(max(int(bisection_iterations), 1)):
        trial = 0.5 * (lower + upper)
        amplification = np.abs(_rk4_amplification(trial * values))
        physical = np.exp(trial * np.maximum(values.real, 0.0))
        if np.all(amplification <= physical * (1.0 + 1.0e-7)):
            lower = trial
        else:
            upper = trial
    return RK4Timestep(
        dt=safety * lower,
        stability_boundary=lower,
        spectral_radius=radius,
        projected_dimension=dimension,
        probe_count=probe_count,
        operator_applications=dimension * probe_count,
    )


def adaptive_eigenpair(
    apply: Callable[[jax.Array], jax.Array],
    restart_once: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    v0: jax.Array,
    *,
    tol: float,
    max_restarts: int,
    filter_dt: float,
    filter_steps: int,
    applications_per_restart: int,
    base_operator_applications: int = 0,
    stability_atol: float = 1.0e-7,
    stability_rtol: float = 1.0e-6,
) -> AdaptiveEigenSolution:
    """Restart until the original residual passes or RK4 stability is suspect."""

    if tol <= 0.0 or max_restarts < 1:
        raise ValueError("tol must be positive and max_restarts must be positive")
    if filter_dt <= 0.0 or filter_steps < 1:
        raise ValueError("filter_dt and filter_steps must be positive")
    vector = v0
    value = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=v0.dtype)
    residual = jnp.asarray(jnp.inf, dtype=jnp.real(v0).dtype)
    defect = np.inf
    stable = False
    restarts = 0
    for restart_index in range(1, max_restarts + 1):
        restarts = restart_index
        _projected_value, vector = restart_once(vector)
        image = apply(vector)
        denominator = jnp.vdot(vector, vector)
        value = jnp.vdot(vector, image) / denominator
        residual = jnp.linalg.norm(image - value * vector)
        residual = residual / jnp.maximum(
            jnp.abs(value) * jnp.linalg.norm(vector),
            jnp.finfo(jnp.real(vector).dtype).tiny,
        )
        scalar = complex(np.asarray(value))
        amplification = abs(complex(_rk4_amplification(filter_dt * scalar)))
        filter_growth = np.log(max(amplification, np.finfo(float).tiny)) / filter_dt
        defect = filter_growth - scalar.real
        stability_limit = stability_atol + stability_rtol * max(abs(scalar.real), 1.0)
        stable = bool(np.isfinite(defect) and defect <= stability_limit)
        if not stable or float(np.asarray(residual)) < tol:
            break
    converged = stable and float(np.asarray(residual)) < tol
    operator_applications = base_operator_applications + restarts * (
        4 * applications_per_restart * filter_steps + 2
    )
    return AdaptiveEigenSolution(
        eigenvalue=value,
        eigenvector=vector,
        residual=residual,
        converged=converged,
        stable=stable,
        restarts=restarts,
        operator_applications=operator_applications,
        filter_dt=filter_dt,
        filter_steps=filter_steps,
        filter_horizon=filter_dt * filter_steps,
        filter_growth_defect=defect,
    )


__all__ = [
    "AdaptiveEigenSolution",
    "PropagatorEigenSolution",
    "RK4Timestep",
    "ExponentialActionPlan",
    "ChebyshevActionPlan",
    "ExponentialActionSolution",
    "adaptive_eigenpair",
    "estimate_rk4_timestep",
    "exponential_eigenpairs",
    "exponential_action",
    "plan_exponential_action",
    "plan_chebyshev_action",
    "propagator_eigenpairs",
]
