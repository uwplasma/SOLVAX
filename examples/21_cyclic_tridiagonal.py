"""Periodic (cyclic) tridiagonal solve via a Sherman-Morrison correction.

A periodic line couples the two endpoints, so the matrix is tridiagonal apart
from the two corners stored in `lower[0]` (top-right) and `upper[-1]`
(bottom-left). `cyclic_tridiagonal_solve` reduces it to a single ordinary
tridiagonal solve with two stacked right-hand sides, retaining the reproducible
Thomas / fused cuSPARSE backend and full differentiability.

Expected runtime: well under a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n, n_fields = 64, 3
rng = np.random.default_rng(0)

lower = jnp.asarray(rng.standard_normal(n))  # lower[0] is the top-right corner
diag = jnp.asarray(6.0 + rng.random(n))  # diagonally dominant
upper = jnp.asarray(rng.standard_normal(n))  # upper[-1] is the bottom-left corner
rhs = jnp.asarray(rng.standard_normal((n, n_fields)))

x = sx.cyclic_tridiagonal_solve(lower, diag, upper, rhs)
print("solution shape:", x.shape)

# Dense periodic reference including both corner couplings.
dense = (np.diag(np.asarray(diag))
         + np.diag(np.asarray(upper)[:-1], 1)
         + np.diag(np.asarray(lower)[1:], -1))
dense[0, -1] = float(lower[0])
dense[-1, 0] = float(upper[-1])
reference = np.linalg.solve(dense, np.asarray(rhs))
print("matches dense periodic solve:", bool(np.allclose(np.asarray(x), reference, atol=1e-10)))

# Same fused/Thomas backend selection as the non-periodic solve.
x_thomas = sx.cyclic_tridiagonal_solve(lower, diag, upper, rhs, method="thomas")
x_lax = sx.cyclic_tridiagonal_solve(lower, diag, upper, rhs, method="lax")
print("thomas vs lax (max abs diff):", float(jnp.max(jnp.abs(x_thomas - x_lax))))

# Differentiable through the coefficients.
g = jax.grad(lambda d: jnp.sum(sx.cyclic_tridiagonal_solve(lower, d, upper, rhs) ** 2))(diag)
print("gradient norm w.r.t. diagonal:", float(jnp.linalg.norm(g)))
