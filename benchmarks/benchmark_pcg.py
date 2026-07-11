"""Cold/warm matrix-free PCG benchmark with correctness diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

from solvax import __version__, pcg, status_name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pcg_benchmark(*, size: int = 4096, repeats: int = 5) -> dict[str, object]:
    if size < 2:
        raise ValueError("size must be at least two")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    diagonal = jnp.logspace(0.0, 6.0, size, dtype=dtype)
    rhs = jnp.ones(size, dtype=dtype)

    def solve(value):
        return pcg(
            lambda x: diagonal * x,
            value,
            precond=lambda residual: residual / diagonal,
            rtol=100.0 * jnp.finfo(dtype).eps,
            max_steps=16,
        )

    compiled = jax.jit(solve)
    started = time.perf_counter()
    solution = compiled(rhs)
    solution.x.block_until_ready()
    cold_seconds = time.perf_counter() - started
    warm_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        solution = compiled(rhs)
        solution.x.block_until_ready()
        warm_samples.append(time.perf_counter() - started)
    exact = 1.0 / diagonal
    relative_error = float(jnp.linalg.norm(solution.x - exact) / jnp.linalg.norm(exact))
    pcg_module = importlib.import_module("solvax.pcg")
    return {
        "benchmark": "matrix_free_pcg",
        "cold_seconds": cold_seconds,
        "converged": bool(solution.converged),
        "device": str(jax.devices()[0]),
        "dtype": jnp.dtype(dtype).name,
        "implementation": {
            "benchmark_sha256": _sha256(Path(__file__)),
            "pcg_sha256": _sha256(Path(pcg_module.__file__)),
            "solvax_version": __version__,
        },
        "iterations": int(solution.iterations),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "relative_error": relative_error,
        "relative_residual_norm": float(solution.relative_residual_norm),
        "repeats": repeats,
        "size": size,
        "status": status_name(solution.status),
        "warm_median_seconds": statistics.median(warm_samples),
        "warm_samples_seconds": warm_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_pcg_benchmark(size=args.size, repeats=args.repeats)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
