"""Randomized Nystrom preconditioning: spectral-decay SPD systems without a grid.

Builds a rank-ell Nystrom approximation from ell operator applications (fixed
PRNG key: deterministic and differentiable) and uses it to precondition PCG on
a regularized system (A + mu I) x = b whose spectrum decays fast ahead of a
flat tail -- the regime where the preconditioned condition number is bounded
by a small constant in expectation (Frangella, Tropp & Udell 2023).

Expected runtime: seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n, rank, mu = 400, 60, 1e-2
rng = np.random.default_rng(0)
q, _ = np.linalg.qr(rng.standard_normal((n, n)))
lam = np.concatenate([100.0 * 0.5 ** np.arange(40), 1e-2 * np.ones(n - 40)])
A = jnp.asarray((q * lam) @ q.T)
b = jnp.asarray(rng.standard_normal(n))

def system(v):
    return A @ v + mu * v

plain = sx.pcg(system, b, rtol=1e-10, max_steps=1000)
precond = sx.nystrom_preconditioner(lambda v: A @ v, n, rank, jax.random.PRNGKey(0), mu=mu)
fast = sx.pcg(system, b, precond=precond, rtol=1e-10, max_steps=1000)
print(f"plain PCG:    {int(plain.iterations)} iterations, converged={bool(plain.converged)}")
print(f"nystrom PCG:  {int(fast.iterations)} iterations, converged={bool(fast.converged)}")

# Differentiable end to end: gradient through sketch, eigenfactors, and solve.
def loss(scale):
    p = sx.nystrom_preconditioner(
        lambda v: scale * (A @ v), n, rank, jax.random.PRNGKey(0), mu=mu
    )
    x = sx.pcg_linear_solve(
        lambda v: scale * (A @ v) + mu * v, b, precond=p, rtol=1e-11, max_steps=600
    ).x
    return jnp.sum(x**2)

gradient = jax.grad(loss)(1.0)
step = 1e-6
finite = (loss(1.0 + step) - loss(1.0 - step)) / (2 * step)
matches = np.isclose(float(gradient), float(finite), rtol=1e-4)
print(f"grad through preconditioner+solve matches FD: {matches}")
