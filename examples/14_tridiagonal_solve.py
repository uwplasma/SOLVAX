"""Batched tridiagonal solve: Thomas on CPU, cuSPARSE on GPU, many columns at once.

`tridiagonal_solve` puts the system on the leading axis and solves every
trailing column (and stacked field) simultaneously — the layout that maps a
stack of independent tridiagonal systems onto the vendor batched kernel without
an outer vmap. `method="auto"` picks the bit-reproducible Thomas sweep when the
code lowers for CPU and the fused `jax.lax.linalg` (cuSPARSE) kernel on a GPU.
It is the fast path for 1-D radial preconditioners and line smoothers.

Expected runtime: about a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n, n_cols, n_fields = 256, 64, 3  # radial rows x spectral columns x force fields
rng = np.random.default_rng(0)

lower = jnp.asarray(rng.standard_normal((n, n_cols)))
upper = jnp.asarray(rng.standard_normal((n, n_cols)))
diag = jnp.asarray(6.0 + rng.random((n, n_cols)))  # diagonally dominant
rhs = jnp.asarray(rng.standard_normal((n, n_cols, n_fields)))

# One call solves n_cols * n_fields independent systems; backend per platform.
x = sx.tridiagonal_solve(lower, diag, upper, rhs)
print("solution shape:", x.shape)

# Verify one column / field against a dense solve.
c, f = 5, 1
A = (np.diag(np.asarray(diag)[:, c])
     + np.diag(np.asarray(upper)[:-1, c], 1)
     + np.diag(np.asarray(lower)[1:, c], -1))
x_ref = np.linalg.solve(A, np.asarray(rhs)[:, c, f])
print("matches dense reference:", np.allclose(np.asarray(x)[:, c, f], x_ref, atol=1e-10))

# The Thomas path is bitwise reproducible; the fused path agrees to roundoff.
x_thomas = sx.tridiagonal_solve(lower, diag, upper, rhs, method="thomas")
x_lax = sx.tridiagonal_solve(lower, diag, upper, rhs, method="lax")
print("thomas vs lax (max abs diff):", float(jnp.max(jnp.abs(x_thomas - x_lax))))

# Fully differentiable — e.g. through a preconditioner application.
g = jax.grad(lambda d: jnp.sum(sx.tridiagonal_solve(lower, d, upper, rhs) ** 2))(diag)
print("gradient norm w.r.t. diagonal:", float(jnp.linalg.norm(g)))
