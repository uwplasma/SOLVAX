# Release 0.26.0

## Sparse-direct solves and eigenvalue derivatives inside traced JAX code

`solvax.sparse_direct` makes SOLVAX's SuperLU and MUMPS factorizations callable
from `jit`, `grad` and `vmap` (#121).

- `sparse_solve(pattern, values, b, options=...)` is a
  `jax.lax.custom_linear_solve` over a static `CsrPattern` and traced values.
  The forward, tangent and reverse (transposed) solves are host callbacks that
  reuse one cached factorization, so `value_and_grad` factors once. `vmap` over
  right-hand sides is a single multi-RHS solve. Gradients with respect to the
  values and `b` match dense `jnp.linalg.solve` autodiff.
- `sparse_eigenvalue(operator, params, pattern, values, shift, ...)` factors
  `A - shift I` once, runs shift-invert ARPACK for the eigenvalues nearest the
  shift, selects `max_real` or `nearest`, and computes the left eigenvector with
  conjugate-transposed solves on the same factor. Its derivative is the
  first-order eigenvalue perturbation, with no extra factorization.

Measured for GKX's linear gyrokinetic operator at 3,072 unknowns: the
growth-rate gradient matches a dense eigensolve to 3e-13 and takes 4-5.5 s
compiled, against 117-193 s for dense and matrix-free shift-invert routes.
Execution is host CPU; no GPU factorization is added in this release.
