"""Non-pivoted banded LU: factor once, solve many right-hand sides, differentiate.

Advection-dominated 1-D operators give narrow banded matrices. `solvax.banded`
factors them without row pivoting (which XLA handles poorly) using row
equilibration + static pivoting, so the whole factor/solve is
jit/vmap/grad-transparent. Factor once, reuse across right-hand sides.

Expected runtime: about a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n, lower_bw, upper_bw = 128, 2, 1  # matrix size and sub/super bandwidths
rng = np.random.default_rng(0)

# scipy-style banded storage: shape (lower_bw + upper_bw + 1, n).
n_diags = lower_bw + upper_bw + 1
bands = rng.standard_normal((n_diags, n))
bands[upper_bw] = 6.0 + rng.random(n)  # dominant main diagonal
bands = jnp.asarray(bands)

factors = sx.lu_factor_banded(bands, lower_bw, upper_bw)
print("pivots clamped by static pivoting:", int(factors.n_clamped))

# One factorization, several right-hand sides (a single vector and a block).
b1 = jnp.asarray(rng.standard_normal(n))
b2 = jnp.asarray(rng.standard_normal((n, 4)))
x1 = sx.lu_solve_banded(factors, b1)
x2 = sx.lu_solve_banded(factors, b2)

# banded_matvec applies A without densifying — check the residuals.
r1 = sx.banded_matvec(bands, lower_bw, upper_bw, x1) - b1
r2 = sx.banded_matvec(bands, lower_bw, upper_bw, x2) - b2
print(f"||A x1 - b1|| = {float(jnp.linalg.norm(r1)):.2e}")
print(f"||A x2 - b2|| = {float(jnp.linalg.norm(r2)):.2e}")

# Differentiable end to end: gradient of a solve-based loss w.r.t. the bands.
def loss(ab):
    x = sx.lu_solve_banded(sx.lu_factor_banded(ab, lower_bw, upper_bw), b1)
    return jnp.sum(x**2)

print("gradient norm w.r.t. bands:", float(jnp.linalg.norm(jax.grad(loss)(bands))))
