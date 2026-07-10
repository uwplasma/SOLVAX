"""Block-tridiagonal solve with a low-precision factorization + refinement.

`mixed_precision_block_thomas` factors the block-Thomas Schur complements in
fast float32 (up to ~32x the float64 throughput of consumer GPU LU) and
recovers float64 accuracy with a few sweeps of iterative refinement — the
residual is formed with the working-precision operator, the correction with a
cheap low-precision solve. Accurate whenever kappa(A) * u_low < 1.

Expected runtime: about a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_blocks, m = 30, 6  # blocks and block size
rng = np.random.default_rng(0)

lower = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
upper = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
diag = jnp.asarray(rng.standard_normal((n_blocks, m, m)) + 4.0 * m * np.eye(m))
rhs = jnp.asarray(rng.standard_normal((n_blocks, m)))

x_ref = sx.block_thomas(lower, diag, upper, rhs)  # full float64 reference
for steps in (0, 1, 2):
    x = sx.mixed_precision_block_thomas(lower, diag, upper, rhs, refine_steps=steps)
    err = float(jnp.max(jnp.abs(x - x_ref)))
    print(f"refine_steps={steps}: max |mixed - float64| = {err:.2e}")
