"""Mixed-precision iterative refinement (defect correction).

Factor / solve in fast low precision (float32), then recover working-precision
(float64) accuracy with a few sweeps of `iterative_refinement`: each sweep
forms the residual in high precision and corrects it with a cheap low-precision
solve. `as_low_precision` wraps any solver to run internally in float32;
`mixed_precision` does the same for a preconditioner inside flexible GMRES.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 200
rng = np.random.default_rng(0)
A = jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n))  # moderate conditioning
b = jnp.asarray(rng.standard_normal(n))

# A dense solve wrapped to run in float32, used as the low-precision inner solve.
solve32 = sx.as_low_precision(jnp.linalg.solve, dtype=jnp.float32)


def matvec(x):
    return A @ x


def approx_solve(r):
    return solve32(A, r)


x, residual_norms = sx.iterative_refinement(matvec, b, approx_solve, iterations=3)
print("residual norm after each sweep:")
for i, r in enumerate(np.asarray(residual_norms)):
    print(f"  sweep {i}: {r:.3e}")

# `mixed_precision` runs any preconditioner in low precision inside FGMRES,
# which tolerates the inexactness while accumulating the residual in float64.
pc = sx.mixed_precision(sx.jacobi(jnp.diag(A)), dtype=jnp.float32)
sol = sx.gmres(matvec, b, precond=pc, rtol=1e-10)
print(f"\nmixed-precision-preconditioned GMRES: converged={bool(sol.converged)} "
      f"iters={int(sol.iterations)}")
