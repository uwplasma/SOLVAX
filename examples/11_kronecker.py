"""Kronecker-product matvec and nearest-Kronecker preconditioning.

`KroneckerOperator` applies A (x) B without forming the huge product. For a
separable (or nearly separable) operator, `nearest_kronecker` extracts the best
A (x) B factors from a dense matrix (Van Loan-Pitsianis rearrangement) and
`kronecker_nkp` inverts them with two small solves — an automatic structural
preconditioner that clusters the spectrum around 1.

Expected runtime: about a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_factor

import solvax as sx

jax.config.update("jax_enable_x64", True)

na, nb = 8, 6  # outer and inner factor sizes
rng = np.random.default_rng(0)
A = rng.standard_normal((na, na)) + na * np.eye(na)
B = rng.standard_normal((nb, nb)) + nb * np.eye(nb)
K = jnp.asarray(np.kron(A, B))  # the assembled (na nb) x (na nb) operator

# Matvec without forming K.
op = sx.KroneckerOperator(jnp.asarray(A), jnp.asarray(B))
v = jnp.asarray(rng.standard_normal(na * nb))
print("KroneckerOperator matvec matches jnp.kron:", bool(jnp.allclose(op(v), K @ v, atol=1e-10)))

# Recover the factors and build the inverse preconditioner.
A_fac, B_fac = sx.nearest_kronecker(K, na, nb)
precond = sx.kronecker_nkp(lu_factor(A_fac), lu_factor(B_fac))

# The preconditioned operator is ~ identity, so GMRES converges almost instantly.
sol = sx.gmres(lambda x: K @ x, v, precond=precond, rtol=1e-10)
print(f"Kronecker-preconditioned GMRES: iters={int(sol.iterations)} "
      f"converged={bool(sol.converged)}")
