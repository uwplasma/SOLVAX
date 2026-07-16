"""Benchmark the generated truncated block solver on kinetic-shaped systems."""

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

from solvax import __version__, block_thomas_truncated, block_thomas_truncated_fn


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _memory_dict(compiled) -> dict[str, int] | None:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {field: int(getattr(analysis, field)) for field in fields}


def run_generated_block_benchmark(
    *,
    n_theta: int = 13,
    n_zeta: int = 15,
    n_xi: int = 32,
    keep_lowest: int = 3,
    n_rhs: int = 2,
    repeats: int = 5,
) -> dict[str, object]:
    """Run a deterministic dense-block workload and return JSON-safe results."""
    if min(n_theta, n_zeta, n_xi, keep_lowest, n_rhs, repeats) < 1:
        raise ValueError("grid sizes, keep_lowest, n_rhs, and repeats must be positive")
    n_blocks = n_xi + 1
    if keep_lowest > n_blocks:
        raise ValueError("keep_lowest cannot exceed n_xi + 1")

    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    block_size = n_theta * n_zeta
    eye = jnp.eye(block_size, dtype=dtype)
    neighbor = jnp.roll(eye, 1, axis=0) - jnp.roll(eye, -1, axis=0)

    def block_fn(index):
        order = index.astype(dtype)
        lower = -(0.12 + 0.001 * order) * eye + 0.002 * neighbor
        diagonal = (3.0 + 0.03 * order) * eye + 0.01 * neighbor
        upper = -(0.10 + 0.0015 * order) * eye - 0.002 * neighbor
        return lower, diagonal, upper

    rhs_low = jnp.linspace(
        0.25, 1.25, keep_lowest * block_size * n_rhs, dtype=dtype
    ).reshape(keep_lowest, block_size, n_rhs)

    def solve(rhs):
        return block_thomas_truncated_fn(block_fn, n_blocks, rhs, keep_lowest)

    started = time.perf_counter()
    compiled = jax.jit(solve).lower(rhs_low).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    solution = compiled(rhs_low)
    solution.block_until_ready()
    first_execute_seconds = time.perf_counter() - started
    warm_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        solution = compiled(rhs_low)
        solution.block_until_ready()
        warm_samples.append(time.perf_counter() - started)

    indices = jnp.arange(n_blocks, dtype=jnp.int32)
    lower, diagonal, upper = jax.vmap(block_fn)(indices)
    reference = block_thomas_truncated(
        lower, diagonal, upper, rhs_low, keep_lowest
    )
    relative_error = float(
        jnp.linalg.norm(solution - reference) / jnp.linalg.norm(reference)
    )
    direct_module = importlib.import_module("solvax.direct")
    return {
        "benchmark": "generated_truncated_block_thomas",
        "block_size": block_size,
        "compile_seconds": compile_seconds,
        "device": str(jax.devices()[0]),
        "dtype": jnp.dtype(dtype).name,
        "executable_memory": _memory_dict(compiled),
        "first_execute_seconds": first_execute_seconds,
        "grid": {"n_theta": n_theta, "n_zeta": n_zeta, "n_xi": n_xi},
        "implementation": {
            "benchmark_sha256": _sha256(Path(__file__)),
            "direct_sha256": _sha256(Path(direct_module.__file__)),
            "solvax_version": __version__,
        },
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "keep_lowest": keep_lowest,
        "n_blocks": n_blocks,
        "n_rhs": n_rhs,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "relative_error_vs_materialized": relative_error,
        "repeats": repeats,
        "warm_median_seconds": statistics.median(warm_samples),
        "warm_samples_seconds": warm_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-theta", type=int, default=13)
    parser.add_argument("--n-zeta", type=int, default=15)
    parser.add_argument("--n-xi", type=int, default=32)
    parser.add_argument("--keep-lowest", type=int, default=3)
    parser.add_argument("--n-rhs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_generated_block_benchmark(
        n_theta=args.n_theta,
        n_zeta=args.n_zeta,
        n_xi=args.n_xi,
        keep_lowest=args.keep_lowest,
        n_rhs=args.n_rhs,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
