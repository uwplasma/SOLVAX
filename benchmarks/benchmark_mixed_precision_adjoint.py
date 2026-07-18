"""Amortized implicit adjoint of the mixed-precision block-Thomas solve.

Measures the two claims behind ``mixed_precision_block_thomas(implicit_adjoint=True)``:

1. **Accuracy.** The gradient computed with float32 factors + working-precision
   refinement matches the exact float64 gradient to working precision, across a
   conditioning sweep — the gradient inherits the *refined forward error*, not
   the factorization precision. The bare (``refine_steps=0``) gradient shows
   what the factor precision alone would give.
2. **Cost.** The custom-VJP backward is refinement sweeps on the transposed
   factors — no differentiation through the factorization, no taped refinement
   loop — measured as compiled reverse-mode temp memory and warm wall time
   against the default unrolled path.

Deterministic and JSON-serializable for the reproducibility package.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from solvax import __version__, block_thomas, mixed_precision_block_thomas

jax.config.update("jax_enable_x64", True)


def _system(n_blocks: int, m: int, dominance: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    lower = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
    diag = jnp.asarray(rng.standard_normal((n_blocks, m, m)) + dominance * m * np.eye(m))
    upper = jnp.asarray(rng.standard_normal((n_blocks, m, m)))
    rhs = jnp.asarray(rng.standard_normal((n_blocks, m)))
    return lower, diag, upper, rhs


def _accuracy_row(dominance: float) -> dict[str, float]:
    lower, diag, upper, rhs = _system(10, 6, dominance, seed=11)
    exact = jax.grad(lambda d: jnp.sum(block_thomas(lower, d, upper, rhs) ** 2))(diag)

    def rel_error(steps, implicit):
        got = jax.grad(
            lambda d: jnp.sum(
                mixed_precision_block_thomas(
                    lower, d, upper, rhs,
                    refine_steps=steps, implicit_adjoint=implicit,
                ) ** 2
            )
        )(diag)
        return float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact))

    return {
        "dominance": dominance,
        "implicit_refined": rel_error(2, True),
        "implicit_bare_fp32": rel_error(0, True),
        "unrolled_refined": rel_error(2, False),
    }


def _accuracy(dominances: tuple[float, ...]) -> list[dict[str, float]]:
    return [_accuracy_row(dominance) for dominance in dominances]


def _cost(n_blocks: int, m: int) -> dict[str, dict[str, float]]:
    lower, diag, upper, rhs = _system(n_blocks, m, 4.0)
    out = {}
    for label, implicit in (("unrolled", False), ("implicit", True)):
        def loss(d, implicit=implicit):
            return jnp.sum(
                mixed_precision_block_thomas(
                    lower, d, upper, rhs, implicit_adjoint=implicit
                ) ** 2
            )

        grad = jax.jit(jax.grad(loss))
        t0 = time.perf_counter()
        compiled = grad.lower(diag).compile()
        compile_s = time.perf_counter() - t0
        grad(diag).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(10):
            result = grad(diag)
        result.block_until_ready()
        out[label] = {
            "backward_temp_bytes": int(compiled.memory_analysis().temp_size_in_bytes),
            "warm_ms": (time.perf_counter() - t0) / 10 * 1e3,
            "compile_s": compile_s,
        }
    return out


def run_mixed_precision_adjoint_benchmark(
    *,
    dominances: tuple[float, ...] = (6.0, 4.0, 2.0, 1.5, 1.2),
    cost_shape: tuple[int, int] = (128, 8),
) -> dict[str, object]:
    """Return JSON-safe accuracy-sweep and backward-cost measurements."""
    return {
        "solvax_version": __version__,
        "gradient_accuracy": _accuracy(dominances),
        "backward_cost": {"n_blocks": cost_shape[0], "m": cost_shape[1], **_cost(*cost_shape)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = run_mixed_precision_adjoint_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}")
    print(f"\n{'dominance':>10} {'implicit(2 steps)':>18} {'bare fp32':>12} {'unrolled(2)':>12}")
    for row in result["gradient_accuracy"]:
        print(
            f"{row['dominance']:>10.1f} {row['implicit_refined']:>18.2e}"
            f" {row['implicit_bare_fp32']:>12.2e} {row['unrolled_refined']:>12.2e}"
        )
    cost = result["backward_cost"]
    print(f"\nbackward cost at N={cost['n_blocks']}, m={cost['m']}:")
    for label in ("unrolled", "implicit"):
        row = cost[label]
        print(
            f"  {label:>9}: temp {row['backward_temp_bytes'] / 2**20:.2f} MiB,"
            f" warm {row['warm_ms']:.2f} ms, compile {row['compile_s']:.1f} s"
        )


if __name__ == "__main__":
    main()
