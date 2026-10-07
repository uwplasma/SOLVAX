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

For a complex linear action `f`, its Hermitian adjoint under `Re(vdot(w,f(v)))`
is `conj(jax.vjp(f, v)[1](conj(w))[0])`. A physical metric needs its own qualified
metric/constraint maps; this Euclidean action does not infer those from a model.

The optional `action_plan` argument of `exponential_eigenpairs` replaces only
the inner Arnoldi action. Supply either it or `inner_krylov_dim`; the existing
Arnoldi default, outer extraction and original continuous-operator residual
checks remain available. A zero lifted vector after breakdown has infinite
residual and cannot be reported as a converged eigenvector. A residual-qualified
nonzero pair alone does not prove that every leading mode has been found.

```{eval-rst}
.. automodule:: solvax.propagator
   :members:
   :member-order: bysource
```
