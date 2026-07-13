"""Block-tridiagonal Schur elimination: factor reuse and the adjoint solve.

`block_thomas_factor` runs the downward Schur sweep once; `block_thomas_solve`
then applies it to any right-hand side, and with `transpose=True` solves the
*transposed* system A^T x = b from the *same* factors — exactly the pairing
implicit differentiation needs (one elimination covers the forward and the
adjoint solve).

Expected runtime: about a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_blocks, m = 24, 6  # number of blocks and block size
rng = np.random.default_rng(0)

lower = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
upper = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
diag = jnp.asarray(rng.standard_normal((n_blocks, m, m)) + 4.0 * m * np.eye(m))

# One elimination, reused across right-hand sides.
factors = sx.block_thomas_factor(lower, diag, upper)
rhs = jnp.asarray(rng.standard_normal((n_blocks, m)))
x = sx.block_thomas_solve(factors, rhs)
x_scaled = sx.block_thomas_solve(factors, 2.0 * rhs)
print("linearity check (x_scaled == 2 x):", bool(jnp.allclose(x_scaled, 2.0 * x)))

# The transposed solve reuses the same factors — no second elimination.
x_adj = sx.block_thomas_solve(factors, rhs, transpose=True)


def fwd(v):
    return sx.block_thomas_solve(factors, v)


(x_adj_ref,) = jax.linear_transpose(fwd, rhs)(rhs)
print("adjoint matches jax.linear_transpose:", bool(jnp.allclose(x_adj, x_adj_ref, atol=1e-10)))

# The convenience wrapper factors + solves in one call.
x_oneshot = sx.block_thomas(lower, diag, upper, rhs)
print("one-shot matches factor/solve:", bool(jnp.allclose(x_oneshot, x, atol=1e-12)))
