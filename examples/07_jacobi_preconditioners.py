"""Point- and block-Jacobi preconditioning of GMRES.

The cheapest useful preconditioners: `jacobi` rescales by the diagonal,
`block_jacobi` inverts the dense diagonal blocks (batched LU). On a
block-diagonally-dominant operator, block-Jacobi collapses the iteration count
far more than point-Jacobi.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_blocks, m = 40, 4  # block count and block size
n = n_blocks * m
rng = np.random.default_rng(0)

# Block-diagonally-dominant matrix: strong dense diagonal blocks, weak coupling.
A = 0.1 * rng.standard_normal((n, n))
blocks = []
for k in range(n_blocks):
    s = slice(k * m, (k + 1) * m)
    blk = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
    A[s, s] = blk
    blocks.append(blk)
A = jnp.asarray(A)
blocks = jnp.asarray(np.stack(blocks))
b = jnp.ones(n)


def matvec(v):
    return A @ v


opts = dict(restart=40, rtol=1e-10, max_restarts=30)
plain = sx.gmres(matvec, b, **opts)
point = sx.gmres(matvec, b, precond=sx.jacobi(jnp.diag(A)), **opts)
block = sx.gmres(matvec, b, precond=sx.block_jacobi(blocks), **opts)

print(f"no preconditioner : iters={int(plain.iterations):3d} converged={bool(plain.converged)}")
print(f"point Jacobi      : iters={int(point.iterations):3d} converged={bool(point.converged)}")
print(f"block Jacobi      : iters={int(block.iterations):3d} converged={bool(block.converged)}")
