"""Precondition an advection-dominated solve with a coarse-operator inverse.

Unpreconditioned GMRES stagnates on strongly nonsymmetric (convection-
dominated) operators; a complete factorization of a *simplified* operator —
here the tridiagonal part — restores fast convergence. This is the same
strategy production kinetic codes use (LU of a coupling-dropped
"preconditioner matrix").

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 256
peclet = 1e3
sigma = 1e4  # reaction term keeps the operator (and the coarse core) nonsingular
h = 1.0 / n

# Periodic 1-D advection-diffusion-reaction: -u'' + peclet * u' + sigma * u
# (upwinded). Without sigma the periodic operator is singular (constant null
# space), and a singular coarse operator cannot precondition anything.
main = 2.0 / h**2 + peclet / h + sigma
lo = -1.0 / h**2 - peclet / h
up = -1.0 / h**2
A = np.zeros((n, n))
np.fill_diagonal(A, main)
A += np.diag(np.full(n - 1, lo), -1) + np.diag(np.full(n - 1, up), 1)
A[0, -1], A[-1, 0] = lo, up  # periodic wrap
A = A + 5.0 * np.random.default_rng(0).standard_normal((n, n)) / n  # long-range tail
A_j = jnp.asarray(A)

b = jnp.ones(n)
matvec = lambda v: A_j @ v

plain = sx.gmres(matvec, b, restart=30, rtol=1e-10, max_restarts=10)
print(f"unpreconditioned: converged={bool(plain.converged)} iters={int(plain.iterations)}")

# Coarse operator: keep only the periodic tridiagonal core, solve it exactly
# with the non-pivoted periodic banded LU (Woodbury corner correction).
bands = jnp.stack([
    jnp.full(n, up),
    jnp.full(n, main),
    jnp.full(n, lo),
])
corner_ul = jnp.array([[lo]])
corner_lr = jnp.array([[up]])
factors = sx.lu_factor_banded_periodic(bands, 1, 1, corner_ul, corner_lr)
precond = sx.coarse_operator(lambda v: sx.lu_solve_banded_periodic(factors, v))

fast = sx.gmres(matvec, b, precond=precond, restart=30, rtol=1e-10)
print(f"coarse-operator PC: converged={bool(fast.converged)} iters={int(fast.iterations)}")
