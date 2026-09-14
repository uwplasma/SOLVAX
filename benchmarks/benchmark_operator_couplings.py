"""Stored-band vs operator-coupled block Thomas on a kinetic-shaped system.

Couplings are ``L_k = x (cl_k S + ml_k diag(mu))`` and
``U_k = x (cu_k S + mu_k diag(mu))`` with one periodic central-difference
stencil ``S = alpha D_theta + beta D_zeta`` on a ``(T, Z)`` grid, and dense
diagonal blocks ``E + nu_k I``; ``x`` varies over a vmapped batch. Reports
per-block factor and apply wall time (best of ``--reps`` after warm-up),
stored factor bytes, compiled solve temporaries and agreement with the
stored-band route. Timings are indicative only on a shared machine.

    PYTHONPATH=src python benchmarks/benchmark_operator_couplings.py --T 21 --Z 37 --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import solvax  # noqa: E402
from solvax import (  # noqa: E402
    block_thomas_factor,
    block_thomas_factor_ops,
    block_thomas_solve,
    block_thomas_solve_ops,
    block_tridiag_relative_residual,
)


def _central(u, axis, transpose):
    d = 0.5 * u.shape[axis] / (2 * np.pi) * (jnp.roll(u, -1, axis) - jnp.roll(u, 1, axis))
    return -d if transpose else d  # periodic central differences are antisymmetric


def couple(params, k, z, *, which, transpose):
    alpha, beta, mu, coef, x = params
    T, Z = alpha.shape
    s_coef, mu_coef = (coef[k, 0], coef[k, 1]) if which == "lower" else (coef[k, 2], coef[k, 3])
    u = z.reshape(T, Z, -1)
    if transpose:
        s_u = _central(alpha[..., None] * u, 0, True) + _central(beta[..., None] * u, 1, True)
    else:
        s_u = alpha[..., None] * _central(u, 0, False) + beta[..., None] * _central(u, 1, False)
    return x * (s_coef * s_u.reshape(T * Z, -1) + mu_coef * mu[:, None] * z)


def build(T, Z, n, batch, seed=0):
    rng = np.random.default_rng(seed)
    m = T * Z
    k = np.arange(n, dtype=float)
    cl = np.where(k > 0, k / np.maximum(2 * k - 1, 1), 0.0)
    cu = (k + 1) / (2 * k + 3)
    coef = np.stack([cl, -cl * (k - 1), cu, cu * (k + 2)], axis=1)
    base = (
        jnp.asarray(1 + 0.3 * rng.random((T, Z))),
        jnp.asarray(0.4 + 0.3 * rng.random((T, Z))),
        jnp.asarray(0.5 * rng.standard_normal(m)),
        jnp.asarray(coef),
    )
    eye = jnp.eye(m)
    e_t = jax.vmap(lambda c: _central(c.reshape(T, Z), 0, False).reshape(m), 1, 1)(eye)
    e_z = jax.vmap(lambda c: _central(c.reshape(T, Z), 1, False).reshape(m), 1, 1)(eye)
    nu = 1.0 + 0.01 * k * (k + 1) / 2
    diag = jnp.stack([0.05 * (e_t + 0.3 * e_z) + nu_k * eye for nu_k in nu])
    xs = jnp.asarray(np.linspace(0.3, 2.5, batch))
    rhs = jnp.asarray(rng.standard_normal((batch, n, m)))
    return base, diag, xs, rhs


def _sync(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        np.asarray(leaf.ravel()[0])
    return tree


def _best(fn, *args, reps):
    _sync(fn(*args))
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        _sync(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def _temp_bytes(fn, *args):
    try:
        return int(jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes)
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


def _git_commit():
    # A copied tree has no repository; the runner then passes the commit it copied.
    if os.environ.get("SOLVAX_BENCH_COMMIT"):
        return os.environ["SOLVAX_BENCH_COMMIT"]
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_operator_couplings_benchmark(T=21, Z=37, n=16, batch=2, reps=3):
    """Time and size the stored-band, ``store="lu"`` and ``store="inverse"`` routes."""
    base, diag, xs, rhs = build(T, Z, n, batch)
    blocks = n * batch
    eye = jnp.eye(T * Z)

    def bands(x):
        params = (*base, x)
        lower, upper = (
            jax.vmap(lambda j, w=w: couple(params, j, eye, which=w, transpose=False))(
                jnp.arange(n, dtype=jnp.int32)
            )
            for w in ("lower", "upper")
        )
        return lower, diag, upper

    band_arrays = _sync(jax.jit(jax.vmap(bands))(xs))
    out = {
        "solvax": solvax.__version__,
        "git_commit": _git_commit(),
        "jax": jax.__version__,
        "device": jax.devices()[0].platform,
        "host": platform.node(),
        "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "T": T,
        "Z": Z,
        "m": T * Z,
        "blocks": n,
        "batch": batch,
        "routes": {},
    }
    dense_factor = jax.jit(jax.vmap(block_thomas_factor))
    dense_solve = jax.vmap(block_thomas_solve)
    routes = {"dense": (dense_factor, (*band_arrays,), dense_solve)}
    for store in ("lu", "inverse"):
        factor = jax.vmap(
            lambda d, x, store=store: block_thomas_factor_ops(d, couple, (*base, x), store=store),
            in_axes=(None, 0),
        )
        routes[store] = (jax.jit(factor), (diag, xs), jax.vmap(block_thomas_solve_ops))
    reference = None
    for name, (factor, factor_args, solve) in routes.items():
        factors = _sync(factor(*factor_args))
        jit_solve = jax.jit(solve)
        solution = _sync(jit_solve(factors, rhs))
        reference = solution if reference is None else reference
        residual = jax.vmap(block_tridiag_relative_residual)(*band_arrays, solution, rhs)
        out["routes"][name] = {
            "factor_s_per_block": _best(factor, *factor_args, reps=reps) / blocks,
            "apply_s_per_block": _best(jit_solve, factors, rhs, reps=reps) / blocks,
            # every retained leaf: LU/pivots (+ both bands for dense, + coupling params for ops)
            "stored_factor_bytes": int(sum(a.nbytes for a in jax.tree_util.tree_leaves(factors))),
            "solve_temp_bytes": _temp_bytes(solve, factors, rhs),
            "rel_diff_vs_dense": float(
                jnp.linalg.norm(solution - reference) / jnp.linalg.norm(reference)
            ),
            "max_rel_residual": float(jnp.max(residual)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--T", type=int, default=21)
    parser.add_argument("--Z", type=int, default=37)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = run_operator_couplings_benchmark(args.T, args.Z, args.blocks, args.batch, args.reps)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
