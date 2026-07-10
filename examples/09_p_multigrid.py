"""A two-level multigrid V-cycle preconditioner.

`p_multigrid` runs a V-cycle over caller-supplied levels: pre-smooth, restrict
the residual, solve (or recurse) on the coarse level, prolong the correction,
post-smooth. Everything — matvecs, transfers, smoothers — is injected, so the
same routine covers geometric h-coarsening (shown here) and p-/spectral
coarsening alike. As a GMRES preconditioner it collapses the iteration count of
a Poisson-like operator to a handful.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_c = 31  # coarse interior points; fine grid is n_f = 2 n_c + 1
n_f = 2 * n_c + 1


def laplacian(n):
    h = 1.0 / (n + 1)
    tri = np.diag(2.0 * np.ones(n)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    return tri / h**2


A_f = jnp.asarray(laplacian(n_f))
A_c = np.asarray(laplacian(n_c))

# Linear interpolation (prolongation) and full-weighting restriction.
P = np.zeros((n_f, n_c))
for i in range(n_c):
    P[2 * i, i] = 0.5
    P[2 * i + 1, i] = 1.0
    P[2 * i + 2, i] = 0.5
P = jnp.asarray(P)
R = 0.5 * P.T

matvec_f = lambda v: A_f @ v
A_c_inv = np.linalg.inv(A_c)
coarse_solve = lambda r: jnp.asarray(A_c_inv) @ r
smoother = jnp.diag(A_f)  # one damped-Jacobi sweep per visit

precond = sx.p_multigrid(
    matvecs=[matvec_f],
    restricts=[lambda r: R @ r],
    prolongs=[lambda e: P @ e],
    coarse_solve=coarse_solve,
    smoothers=[smoother],
    cycles=1,
)

b = jnp.ones(n_f)
opts = dict(restart=40, rtol=1e-10, max_restarts=60)
plain = sx.gmres(matvec_f, b, **opts)
mg = sx.gmres(matvec_f, b, precond=precond, **opts)
print(f"no preconditioner : iters={int(plain.iterations):3d} converged={bool(plain.converged)}")
print(f"V-cycle           : iters={int(mg.iterations):3d} converged={bool(mg.converged)}")
