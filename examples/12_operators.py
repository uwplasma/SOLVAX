"""Matrix-free operator containers and the bordered (KKT) Schur preconditioner.

`solvax.operators` wraps the *action* of structured maps with closed-form
transposes: `BlockTridiagonalOperator` (whose `to_blocks()` feeds the direct
solver as its natural preconditioner), `SumOperator` (structured core + a
matrix-free perturbation), and `BorderedOperator` for saddle-point systems
`[[A, B], [C, 0]]`. `schur_projected_precond` turns an approximate inverse of
the physics block A into a preconditioner for the whole constrained system.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_blocks, m = 20, 4
n = n_blocks * m
rng = np.random.default_rng(0)

lower = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
upper = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
diag = jnp.asarray(rng.standard_normal((n_blocks, m, m)) + 4.0 * m * np.eye(m))
A = sx.BlockTridiagonalOperator(lower, diag, upper)

# The block-Thomas factors of the principal part are its natural preconditioner.
facs = sx.block_thomas_factor(*A.to_blocks())


def a_inv(r):
    return sx.block_thomas_solve(facs, r.reshape(n_blocks, m)).reshape(-1)


# SumOperator: the structured core plus a matrix-free low-rank perturbation.
u = jnp.asarray(rng.standard_normal(n))
pert = sx.MatrixFreeOperator(lambda v: 1e-3 * u * (u @ v), shape=A.shape)
full = sx.SumOperator((A, pert))
sol = sx.gmres(full, jnp.ones(n), precond=sx.coarse_operator(a_inv), rtol=1e-10)
print(f"Sum(block-tridiag + matrix-free): iters={int(sol.iterations)} "
      f"converged={bool(sol.converged)}")

# BorderedOperator [[A, B], [C, 0]] preconditioned by the projected Schur solve.
p = 2
B_cols = jnp.asarray(rng.standard_normal((n, p)))
C_rows = jnp.asarray(rng.standard_normal((p, n)))
K = sx.BorderedOperator(A, B_cols, C_rows)
kkt_precond = sx.schur_projected_precond(a_inv, B_cols, C_rows)
sol_k = sx.gmres(K, jnp.ones(n + p), precond=kkt_precond, rtol=1e-10)
print(f"bordered KKT (Schur precond)    : iters={int(sol_k.iterations)} "
      f"converged={bool(sol_k.converged)}")
