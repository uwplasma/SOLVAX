"""Long-restart GCROT: stored ``Z`` basis vs ``fixed_precond=True``.

A 2-D convection-diffusion operator with a Jacobi preconditioner needs a
long restart; this times one fifo GCROT solve (after compilation) and reports
peak resident memory. Run each configuration in its own process:

    python benchmarks/benchmark_krylov_basis.py --n 400 --restart 1000 [--fixed]
"""

import argparse
import json
import resource
import sys
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from solvax import gcrot  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=400, help="grid points per side")
parser.add_argument("--restart", type=int, default=1000)
parser.add_argument("--fixed", action="store_true")
args = parser.parse_args()

n, h = args.n, 1.0 / (args.n + 1)
peclet = 50.0


def apply(u):
    u = u.reshape(n, n)
    p = jnp.pad(u, 1)
    lap = (4 * u - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]) / h**2
    adv = peclet * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * h)
    return (lap + adv).reshape(-1)


b = jnp.ones(n * n)
solve = jax.jit(lambda b: gcrot(
    apply, b, precond=lambda v: v * (h**2 / 4), m=args.restart, k=10,
    rtol=1e-10, max_restarts=20, fixed_precond=args.fixed,
) if "fixed_precond" in gcrot.__code__.co_varnames else gcrot(
    apply, b, precond=lambda v: v * (h**2 / 4), m=args.restart, k=10,
    rtol=1e-10, max_restarts=20,
))
sol = jax.block_until_ready(solve(b))
start = time.perf_counter()
sol = jax.block_until_ready(solve(b))
wall = time.perf_counter() - start
res = float(jnp.linalg.norm(b - apply(sol.x)) / jnp.linalg.norm(b))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
json.dump(dict(unknowns=n * n, restart=args.restart, fixed=args.fixed,
               iterations=int(sol.iterations), residual=res, seconds=wall,
               peak_rss_gib=rss, backend=jax.default_backend()), sys.stdout)
print()
