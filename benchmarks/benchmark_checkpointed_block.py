"""Measure full versus checkpointed generated block-Thomas solves."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from math import isqrt

import jax
import jax.numpy as jnp
import jaxlib

from solvax import (
    __version__,
    block_thomas_checkpointed_fn,
    block_thomas_factor_fn,
    block_thomas_solve,
)


def _measure(fn, rhs, repeats):
    started = time.perf_counter()
    compiled = jax.jit(fn).lower(rhs).compile()
    compile_seconds = time.perf_counter() - started
    samples = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        compiled(rhs).block_until_ready()
        samples.append(time.perf_counter() - started)
    memory = compiled.memory_analysis()
    return {
        "compile_seconds": compile_seconds,
        "temp_bytes": int(memory.temp_size_in_bytes),
        "warm_median_seconds": statistics.median(samples[1:]),
    }, compiled(rhs)


def run_checkpointed_block_benchmark(
    *, n_blocks: int = 128, block_size: int = 64, repeats: int = 5
) -> dict[str, object]:
    """Return reproducible runtime, accuracy, and executable-memory results."""
    if min(n_blocks, block_size, repeats) < 1:
        raise ValueError("n_blocks, block_size, and repeats must be positive")
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    eye = jnp.eye(block_size, dtype=dtype)
    neighbor = jnp.roll(eye, 1, axis=0) - jnp.roll(eye, -1, axis=0)
    rhs = jnp.linspace(0.25, 1.25, n_blocks * block_size, dtype=dtype).reshape(n_blocks, block_size)

    def block_fn(index):
        order = index.astype(dtype)
        lower = -(0.12 + 0.001 * order) * eye + 0.002 * neighbor
        diagonal = (3.0 + 0.03 * order) * eye + 0.01 * neighbor
        upper = -(0.10 + 0.0015 * order) * eye - 0.002 * neighbor
        return lower, diagonal, upper

    def full(b):
        return block_thomas_solve(block_thomas_factor_fn(block_fn, n_blocks), b)

    def checkpointed(b):
        return block_thomas_checkpointed_fn(block_fn, n_blocks, b)

    full_result, full_solution = _measure(full, rhs, repeats)
    checkpointed_result, checkpointed_solution = _measure(checkpointed, rhs, repeats)
    relative_error = jnp.linalg.norm(checkpointed_solution - full_solution)
    relative_error /= jnp.linalg.norm(full_solution)
    return {
        "benchmark": "checkpointed_generated_block_thomas",
        "block_size": block_size,
        "checkpoint_size": isqrt(n_blocks - 1) + 1,
        "device": str(jax.devices()[0]),
        "dtype": jnp.dtype(dtype).name,
        "full": full_result,
        "checkpointed": checkpointed_result,
        "n_blocks": n_blocks,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "relative_error": float(relative_error),
        "repeats": repeats,
        "solvax_version": __version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    result = run_checkpointed_block_benchmark(
        n_blocks=args.n_blocks,
        block_size=args.block_size,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
