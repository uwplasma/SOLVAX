"""GMRES-IR: float32 block-Thomas factors preconditioning float64 GCROT.

Compares, on a block-tridiagonal system with condition number raised by ``--shift``:
float64 factor + solve; float32 factors under plain iterative refinement;
float32 factors as the fixed preconditioner of a float64 GCROT solve.

    python benchmarks/benchmark_gmres_ir.py --blocks 256 --size 128
"""

import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from solvax import gcrot  # noqa: E402
from solvax.direct import (  # noqa: E402
    block_thomas_factor,
    block_thomas_solve,
    block_tridiag_matvec,
)
from solvax.refine import iterative_refinement  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, default=256)
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--shift", type=float, default=0.2)
parser.add_argument("--repeat", type=int, default=3)
args = parser.parse_args()

rng = np.random.default_rng(0)
nb, m = args.blocks, args.size
# Block Laplacian in the block index, a nonnormal in-block part, and a
# negative shift that moves the spectrum toward the origin (larger kappa).
noise = rng.standard_normal((3, nb, m, m)) / np.sqrt(m)
diag = (2 - args.shift) * np.eye(m) + 0.5 * noise[0]
lower = -np.eye(m) + 0.05 * noise[1]
upper = -np.eye(m) + 0.05 * noise[2]
lower, diag, upper = (jnp.asarray(a) for a in (lower, diag, upper))
b = jnp.asarray(rng.standard_normal((nb, m)))
apply = lambda x: block_tridiag_matvec(lower, diag, upper, x)  # noqa: E731


def timed(fn, *a):
    out = jax.block_until_ready(fn(*a))
    best = float("inf")
    for _ in range(args.repeat):
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*a))
        best = min(best, time.perf_counter() - t)
    return out, best


def rel(x):
    return float(jnp.linalg.norm(b - apply(x)) / jnp.linalg.norm(b))


rows = []
factor64, t_f64 = timed(jax.jit(lambda: block_thomas_factor(lower, diag, upper)))
x, t_s64 = timed(jax.jit(lambda f: block_thomas_solve(f, b)), factor64)
dense = None
if nb * m <= 8192:
    dense = float(np.linalg.cond(np.asarray(jax.vmap(apply, 2, 2)(
        jnp.eye(nb * m).reshape(nb, m, nb * m)).reshape(nb * m, nb * m))))
rows.append(dict(route="float64 direct", cond=dense, factor_s=t_f64, solve_s=t_s64,
                 residual=rel(x)))

low = jax.jit(lambda: block_thomas_factor(
    lower.astype(jnp.float32), diag.astype(jnp.float32), upper.astype(jnp.float32)))
factor32, t_f32 = timed(low)
precond = lambda r: block_thomas_solve(  # noqa: E731
    factor32, r.reshape(nb, m).astype(jnp.float32)).astype(r.dtype).reshape(r.shape)

for sweeps in (2, 6):
    ir = jax.jit(lambda s=sweeps: iterative_refinement(apply, b, precond, iterations=s)[0])
    x, t = timed(ir)
    rows.append(dict(route=f"float32 factors, {sweeps}-sweep refinement",
                     factor_s=t_f32, solve_s=t, residual=rel(x)))

for tol in (1e-10, 1e-12):
    krylov = jax.jit(lambda t=tol: gcrot(apply, b, precond=precond, m=30, k=4, rtol=t,
                                        fixed_precond=True))
    sol, t = timed(krylov)
    rows.append(dict(route=f"float32 factors + float64 GCROT rtol {tol:g}",
                     factor_s=t_f32, solve_s=t, residual=rel(sol.x),
                     iterations=int(sol.iterations)))

for row in rows:
    row.update(backend=jax.default_backend(), blocks=nb, size=m)
    json.dump(row, sys.stdout)
    print()
