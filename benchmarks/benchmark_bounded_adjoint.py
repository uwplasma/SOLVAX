"""Bounded-memory adjoint of the truncated block-tridiagonal solve.

Demonstrates the two claims behind ``block_thomas_truncated(adjoint_window=w)``:

1. **Memory.** Compiled reverse-mode scratch (``temp_size_in_bytes`` from XLA's
   own memory analysis) is flat in the block count ``N`` for the windowed custom
   VJP, versus linear in ``N`` for the plain taped gradient.
2. **Accuracy.** The band-gradient error decays geometrically in the window
   ``w`` for a block diagonally dominant system (Demko-Moss-Smith decay); the
   right-hand-side gradient is exact at every window.

Deterministic and JSON-serializable for the reproducibility package.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from solvax import __version__, block_thomas_truncated

jax.config.update("jax_enable_x64", True)


def _system(n_blocks: int, m: int, keep: int, seed: int = 0, dominance: float = 3.0):
    rng = np.random.default_rng(seed)
    diag = jnp.asarray(rng.standard_normal((n_blocks, m, m)) + dominance * m * np.eye(m))
    lower = jnp.asarray(0.3 * rng.standard_normal((n_blocks, m, m)))
    upper = jnp.asarray(0.3 * rng.standard_normal((n_blocks, m, m)))
    rhs_low = jnp.asarray(rng.standard_normal((keep, m)))
    return lower, diag, upper, rhs_low


def _temp_bytes(n_blocks: int, m: int, keep: int, window: int | None) -> int:
    lower, diag, upper, rhs_low = _system(n_blocks, m, keep)

    def loss(d):
        return jnp.sum(
            block_thomas_truncated(lower, d, upper, rhs_low, keep, adjoint_window=window) ** 2
        )

    compiled = jax.jit(jax.grad(loss)).lower(diag).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes)


def run_bounded_adjoint_benchmark(
    *,
    m: int = 4,
    keep: int = 2,
    window: int = 4,
    block_counts: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024),
    decay_windows: tuple[int, ...] = (0, 2, 4, 6, 8),
) -> dict[str, object]:
    """Return JSON-safe memory-scaling and gradient-decay measurements."""
    memory = [
        {
            "n_blocks": n,
            "naive_temp_bytes": _temp_bytes(n, m, keep, None),
            "bounded_temp_bytes": _temp_bytes(n, m, keep, window),
        }
        for n in block_counts
    ]

    n_decay = max(decay_windows) + keep + 8
    lower, diag, upper, rhs_low = _system(n_decay, m, keep, seed=9)

    def band_grad(window):
        return jax.grad(
            lambda d: jnp.sum(
                block_thomas_truncated(lower, d, upper, rhs_low, keep, adjoint_window=window) ** 2
            )
        )(diag)

    exact = band_grad(n_decay)
    decay = []
    for w in decay_windows:
        err = float(jnp.linalg.norm(band_grad(w) - exact) / jnp.linalg.norm(exact))
        decay.append({"window": w, "band_grad_rel_error": err})

    return {
        "solvax_version": __version__,
        "params": {"m": m, "keep_lowest": keep, "window": window},
        "memory_scaling": memory,
        "band_gradient_decay": decay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = run_bounded_adjoint_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}  params={result['params']}")
    print(f"\n{'N':>6} {'naive_KiB':>12} {'bounded_KiB':>12}  {'ratio':>8}")
    for row in result["memory_scaling"]:
        naive, bounded = row["naive_temp_bytes"], row["bounded_temp_bytes"]
        n, ratio = row["n_blocks"], naive / bounded
        print(f"{n:>6} {naive / 1024:>12.1f} {bounded / 1024:>12.1f}  {ratio:>8.1f}x")
    print(f"\n{'window':>6} {'band_grad_rel_error':>22}")
    for row in result["band_gradient_decay"]:
        print(f"{row['window']:>6} {row['band_grad_rel_error']:>22.3e}")


if __name__ == "__main__":
    main()
