"""Time block-tridiagonal factor and apply routes on identical runtime arrays.

Bands come from an ``.npz`` with ``lower``, ``diag``, ``upper`` (for example
VMEX's scaled raw-force bands, captured by its ``benchmarks/block_solver.py``)
or from a synthetic nonsymmetric chain. Every route is compiled once with the
numeric arrays as arguments; warm samples are interleaved across routes so a
change of machine load affects all of them alike. Each route reports compile
time, executable memory, warm medians, the median of its paired ratio to the
first route with a bootstrap 95% interval, and the original-operator relative
residual of its primal and transposed solves.

    python benchmarks/benchmark_block_factors.py --n-blocks 31 --block-size 150
    python benchmarks/benchmark_block_factors.py --bands vmex_qa.npz --routes stored,normalized
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from solvax import (
    __version__,
    block_thomas_factor,
    block_thomas_solve,
    block_tridiag_relative_residual,
)

#: route -> factor function of ``(lower, diag, upper)``.
ROUTES = {
    "stored": block_thomas_factor,
    "normalized": partial(block_thomas_factor, normalized=True),
    "stored-f32lu": partial(block_thomas_factor, factor_dtype=jnp.float32),
    "normalized-f32lu": partial(block_thomas_factor, factor_dtype=jnp.float32, normalized=True),
    # All-FP32 factor arithmetic: the bands are rounded to FP32 before the sweep.
    "stored-f32": lambda *bands: block_thomas_factor(*(b.astype(jnp.float32) for b in bands)),
}


def synthetic_bands(n_blocks: int, block_size: int, seed: int = 0):
    """A nonsymmetric, weakly dominant chain whose block LUs pivot."""
    rng = np.random.default_rng(seed)
    lower, diag, upper = rng.standard_normal((3, n_blocks, block_size, block_size))
    diag += 2.0 * np.sqrt(block_size) * np.eye(block_size)
    return lower, diag, upper


def _paired_ratio(candidate, baseline, rng):
    ratio = np.asarray(candidate) / np.asarray(baseline)
    boot = np.median(rng.choice(ratio, (2000, ratio.size)), axis=1)
    interval = [float(np.quantile(boot, q)) for q in (0.025, 0.975)]
    return float(np.median(ratio)), interval


def run(bands, routes, n_rhs: int, repeats: int, warmup: int, seed: int = 0) -> dict:
    lower, diag, upper = (jax.device_put(jnp.asarray(b)) for b in bands)
    n_blocks, m = diag.shape[0], diag.shape[-1]
    rng = np.random.default_rng(seed)
    vector = jax.device_put(jnp.asarray(rng.standard_normal((n_blocks, m)), diag.dtype))
    matrix = jax.device_put(jnp.asarray(rng.standard_normal((n_blocks, m, n_rhs)), diag.dtype))

    stages = {}
    for name in routes:
        factor = ROUTES[name]
        t0 = time.perf_counter()
        f_exe = jax.jit(factor).lower(lower, diag, upper).compile()
        factor_compile = time.perf_counter() - t0
        factors = jax.block_until_ready(f_exe(lower, diag, upper))
        stage = {"factor": (f_exe, (lower, diag, upper))}
        compile_s = {"factor": factor_compile}
        for label, rhs, transpose in (
            ("apply_vector", vector, False),
            ("apply_matrix", matrix, False),
            ("apply_matrix_transpose", matrix, True),
        ):
            apply = lambda f, b, t=transpose: block_thomas_solve(f, b, transpose=t)  # noqa: E731
            t0 = time.perf_counter()
            stage[label] = (jax.jit(apply).lower(factors, rhs).compile(), (factors, rhs))
            compile_s[label] = time.perf_counter() - t0
        residual = {
            "primal": float(jnp.max(block_tridiag_relative_residual(
                lower, diag, upper, stage["apply_matrix"][0](factors, matrix), matrix))),
            "transpose": float(jnp.max(block_tridiag_relative_residual(
                jnp.swapaxes(jnp.roll(upper, 1, 0), -1, -2), jnp.swapaxes(diag, -1, -2),
                jnp.swapaxes(jnp.roll(lower, -1, 0), -1, -2),
                stage["apply_matrix_transpose"][0](factors, matrix), matrix))),
        }
        memory = f_exe.memory_analysis()
        stages[name] = {
            "executables": stage,
            "compile_seconds": compile_s,
            "residual": residual,
            "factor_bytes": int(sum(np.asarray(a).nbytes for a in jax.tree.leaves(factors))),
            "factor_temp_bytes": None if memory is None else int(memory.temp_size_in_bytes),
        }

    samples = {name: {s: [] for s in stages[name]["executables"]} for name in stages}
    for it in range(warmup + repeats):
        for stage_name in samples[routes[0]]:
            for name in (routes if it % 2 == 0 else routes[::-1]):
                exe, args = stages[name]["executables"][stage_name]
                t0 = time.perf_counter()
                jax.block_until_ready(exe(*args))
                if it >= warmup:
                    samples[name][stage_name].append(time.perf_counter() - t0)

    out = {}
    for name in routes:
        entry = {k: v for k, v in stages[name].items() if k != "executables"}
        entry["median_seconds"] = {s: float(np.median(v)) for s, v in samples[name].items()}
        entry["samples_seconds"] = samples[name]
        if name != routes[0]:
            entry["ratio_to_" + routes[0]] = {
                s: _paired_ratio(v, samples[routes[0]][s], rng) for s, v in samples[name].items()
            }
        out[name] = entry
    return {
        "benchmark": "block_factors",
        "n_blocks": int(n_blocks),
        "block_size": int(m),
        "n_rhs": n_rhs,
        "dtype": str(diag.dtype),
        "repeats": repeats,
        "warmup": warmup,
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "solvax_version": __version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "routes": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=Path, help=".npz with lower, diag, upper")
    parser.add_argument("--n-blocks", type=int, default=31)
    parser.add_argument("--block-size", type=int, default=150)
    parser.add_argument("--routes", default="stored,normalized")
    parser.add_argument("--n-rhs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.bands is None:
        bands = synthetic_bands(args.n_blocks, args.block_size)
    else:
        with np.load(args.bands) as data:
            bands = (data["lower"], data["diag"], data["upper"])
    result = run(bands, args.routes.split(","), args.n_rhs, args.repeats, args.warmup)
    result["bands"] = str(args.bands) if args.bands else "synthetic"
    rendered = json.dumps(result, indent=1) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
