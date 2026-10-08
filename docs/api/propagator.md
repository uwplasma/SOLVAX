# Adaptive propagator API

`estimate_rk4_timestep` uses small Arnoldi spectral sketches and the complex
RK4 stability polynomial to choose a conservative step for a matrix-free
operator. A deterministic broadband probe supplements the caller's seed so a
recycled, nearly invariant eigenvector cannot hide peripheral modes.
`adaptive_eigenpair` wraps a caller-supplied one-restart propagator solve with
original-operator residual stopping and a numerical growth-defect guard.
`exponential_eigenpairs` removes the explicit stability limit by projecting
each matrix-exponential action into an inner Arnoldi space, then extracting
leading modes in a smaller outer space. Returned values and residuals are
always recomputed with the original operator.

`plan_exponential_action` selects a fixed shifted Taylor schedule on the host.
`exponential_action` applies that schedule to arbitrary real or complex arrays,
including non-unit and zero vectors. Its two recurrence levels use the existing
checkpointed loop, so value, JVP and VJP differentiate the same finite polynomial.
Pass numeric operator parameters and inputs as operands of your jitted function;
keep the plan fixed within a derivative and check convergence of that derivative.

`plan_chebyshev_action` supplies an alternative for oscillatory operators with a
verified imaginary-major numerical-range rectangle. It uses a fixed-horizon,
three-term Chebyshev recurrence with checkpointed operator/input derivatives.
Its plan needs SciPy on the host (`pip install solvax[native]`); application needs
only JAX. Use a complex vector and omit the `horizon` operand for this plan.

```python
import jax
import jax.numpy as jnp
from solvax import exponential_action, plan_exponential_action

plan = plan_exponential_action(horizon=0.4, norm_bound=4.0, shift=-0.7)

@jax.jit
def propagate(matrix, vector, time):
    return exponential_action(lambda x: matrix @ x, vector, plan, horizon=time)

matrix = jnp.array([[-0.7, 3.2], [0.0, -0.7]])
result = propagate(matrix, jnp.array([2.0, -3.0]), 0.4)
```

The caller must verify `norm_bound >= ||A - shift*I||_2` over the plan's
parameter neighborhood. For degree `m`, `s` substeps and this bound `rho`,
the exact-arithmetic truncation estimate per input norm is
`s * exp(t * (rho + Re(shift))) * (t*rho/s)**(m+1) / (m+1)!`.
The planner bounds its maximum over `0 <= t <= horizon`, including an interior
maximum for strongly negative shifts. It minimizes operator applications within
the requested degree/substep limits and raises if the budget is exhausted.
This norm estimate can be much looser than adaptive scaling--Taylor or Krylov.

`result.truncation_bound` multiplies that estimate by the input Euclidean norm
(Frobenius norm for RHS blocks). It excludes roundoff and operator evaluation
error and is not an output-relative tolerance or an interval enclosure.
`result.valid` checks the horizon interval and finite input/output; it cannot
check the supplied norm assumption. A block is supported when `apply` handles
the entire block and preserves its shape/dtype. Promote real inputs explicitly
when a complex operator or shift requires complex arithmetic.

For Chebyshev, the caller verifies that `W(A - shift*I)` is contained in
`[-real_halfwidth, real_halfwidth] + 1j*[-imag_halfwidth, imag_halfwidth]`, with
`0 <= real_halfwidth < imag_halfwidth`. This is a numerical-range enclosure in
the Euclidean norm of the supplied coordinates, not an eigenvalue enclosure.
The planner encloses the rectangle by an ellipse with semiaxes `sqrt(2)` times
the halfwidths and bounds the Bessel/Chebyshev tail with the established
Crouzeix–Palencia constant `1 + sqrt(2)`.
See [the numerical-range spectral-set theorem](https://arxiv.org/abs/1702.00668).
The estimate excludes scalar coefficient error and operator/recurrence roundoff;
degree or coefficient overflow raises rather than providing silent success.

```python
from solvax import plan_chebyshev_action

# The Hermitian/skew-Hermitian parts must establish these extents beforehand.
plan = plan_chebyshev_action(horizon=0.4, real_halfwidth=0.2,
                             imag_halfwidth=5.0, tolerance=1e-10)
matrix = jnp.array([[0.0 + 2j, 0.3], [0.0, 0.0 - 3j]])
result = exponential_action(lambda x: matrix @ x,
                            jnp.array([2.0 + 0j, -3.0 + 0j]), plan)
```

| Caller requirements | Supported choice | Tradeoff |
| --- | --- | --- |
| Arbitrary autonomous linear operator, verified induced norm, numeric time derivatives | Taylor plan | Conservative bound may require many substeps. |
| Verified imaginary-major numerical range, repeated fixed-time actions/adjoints | Chebyshev plan | Short recurrence; fixed horizon, complex inputs, host SciPy. |
| No verified norm or numerical range; existing mode extraction | Inner Arnoldi dimension | Input-dependent approximation; validate actual action errors and continuous eigenpair residuals. |

Supplying a plan selects the polynomial action directly. There is no automatic
enclosure for a black-box operator, so SOLVAX does not silently switch an existing
unplanned Arnoldi caller. Neither option promises universal runtime superiority.
For a physical metric `M=L*L`, use `L A L^-1`, propagate `L v`, and transform the
result back. The metric and its parameter derivatives remain caller-owned.

For a complex linear action `f`, its Hermitian adjoint under `Re(vdot(w,f(v)))`
is `conj(jax.vjp(f, v)[1](conj(w))[0])`. A physical metric needs its own qualified
metric/constraint maps; this Euclidean action does not infer those from a model.
Fixed polynomial actions are linear in their input, including at zero. An
input-dependent Arnoldi approximation is generally not: its zero-start VJP is
not the exponential adjoint. For that method, construct the true generator
Hermitian adjoint and exponentiate it separately.

The optional `action_plan` argument of `exponential_eigenpairs` replaces only
the inner Arnoldi action. Supply either it or `inner_krylov_dim`; the existing
Arnoldi default, outer extraction and original continuous-operator residual
checks remain available. A zero lifted vector after breakdown has infinite
residual and cannot be reported as a converged eigenvector. A residual-qualified
nonzero pair alone does not prove that every leading mode has been found.
Chebyshev eigenpair filters must use exactly the plan horizon. Both polynomial
filters retain the outer extraction and original operator certification.

```{eval-rst}
.. automodule:: solvax.propagator
   :members:
   :member-order: bysource
```
