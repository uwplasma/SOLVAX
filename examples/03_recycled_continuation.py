"""Recycle Krylov subspaces across a parameter continuation.

Scans over a physical parameter (collisionality, electric field, geometry)
solve a sequence of slowly-varying linear systems. GCROT-style recycling
carries the deflation subspace from one solve to the next, cutting the
iteration count of every warm solve.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp

import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 200
key = jax.random.PRNGKey(1)
A0 = jax.random.normal(key, (n, n)) / jnp.sqrt(n) + 3.0 * jnp.eye(n)
dA = jax.random.normal(jax.random.PRNGKey(2), (n, n)) / jnp.sqrt(n)
b = jnp.ones(n)

cold_total, warm_total = 0, 0
recycle = None
for i in range(6):
    A_i = A0 + 0.02 * i * dA
    mv = lambda v, A=A_i: A @ v

    cold = sx.gcrot(mv, b, m=30, k=10, rtol=1e-10)
    warm = sx.gcrot(mv, b, m=30, k=10, rtol=1e-10, recycle=recycle)
    recycle = warm.recycle

    tag = "(cold start)" if i == 0 else ""
    print(f"step {i}: cold {int(cold.iterations):3d} iters | "
          f"recycled {int(warm.iterations):3d} iters {tag}")
    if i > 0:
        cold_total += int(cold.iterations)
        warm_total += int(warm.iterations)

print(f"\nsteps 1-5 totals: cold {cold_total} vs recycled {warm_total} iterations")
