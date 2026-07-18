"""Communication accounting: collectives in compiled primal vs adjoint solves.

Counts collective operations (all-reduce, all-gather, reduce-scatter,
collective-permute, all-to-all) in the optimized HLO of jitted SOLVAX solves on
a sharded mesh, for 1/2/4/8 devices. Establishes, as measured data rather than
prose:

- sharded batched tridiagonal solves are collective-free, and stay so under
  ``jax.grad`` (the adjoint is columnwise too);
- ``single_reduction=True`` PCG compiles to fewer reductions than standard PCG;
- the implicit adjoint of a Krylov solve stays in the primal's communication
  class (about one extra solve's worth of collectives, never a different
  scaling).

Devices are CPU-emulated (``--xla_force_host_platform_device_count``), so the
counts are pure compiler facts: deterministic and hardware-independent.
"""

from __future__ import annotations

import argparse
import json
import os

_FLAG = "--xla_force_host_platform_device_count=8"
if _FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _FLAG).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from solvax import (  # noqa: E402
    __version__,
    gmres,
    linear_solve,
    pcg,
    pcg_linear_solve,
    tridiagonal_solve,
)

jax.config.update("jax_enable_x64", True)

_COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all")


def count_collectives(fn, *args) -> int:
    """Count collective ops in optimized HLO; async start/done pairs count once."""
    text = jax.jit(fn).lower(*args).compile().as_text()
    total = 0
    for line in text.splitlines():
        for op in _COLLECTIVES:
            if f"{op}-start(" in line or (f"{op}(" in line and f"{op}-done(" not in line):
                total += 1
                break
    return total


def _cases(n_devices: int) -> dict[str, dict[str, int]]:
    mesh = Mesh(np.array(jax.devices()[:n_devices]), axis_names=("i",))
    rng = np.random.default_rng(0)

    def shard(value, spec):
        return jax.device_put(value, NamedSharding(mesh, spec))

    out: dict[str, dict[str, int]] = {}

    # Batched tridiagonal, batch axis sharded: embarrassingly parallel.
    n, cols = 32, 64
    lower = shard(jnp.asarray(rng.standard_normal((n, cols))), P(None, "i"))
    upper = shard(jnp.asarray(rng.standard_normal((n, cols))), P(None, "i"))
    diag = shard(jnp.asarray(6.0 + rng.random((n, cols))), P(None, "i"))
    rhs = shard(jnp.asarray(rng.standard_normal((n, cols))), P(None, "i"))

    def tri(d):
        return tridiagonal_solve(lower, d, upper, rhs, method="thomas")

    out["tridiagonal_batch"] = {
        "primal": count_collectives(tri, diag),
        "adjoint": count_collectives(jax.grad(lambda d: jnp.sum(tri(d) ** 2)), diag),
    }

    # PCG on a sharded vector: standard vs single-reduction recurrence. The
    # adjoint is measured through the implicit path (pcg_linear_solve); the raw
    # while_loop primal is deliberately not reverse-differentiable.
    m = 256
    d_vec = jnp.asarray(2.0 + rng.random(m))
    b = shard(jnp.asarray(rng.standard_normal(m)), P("i"))
    # The loss must be nonlinear in the solution: a linear loss has a constant
    # cotangent, which XLA folds so the entire adjoint solve disappears from
    # the compiled module. sum(x**2) makes the cotangent 2x — sharded, input-
    # dependent, and therefore an honest adjoint measurement.
    for label, fused in (("pcg", False), ("pcg_single_reduction", True)):
        def solve(rhs_, fused=fused):
            return pcg(
                lambda v: d_vec * v, rhs_, rtol=1e-10, max_steps=64,
                single_reduction=fused,
            ).x

        def implicit(rhs_, fused=fused):
            return pcg_linear_solve(
                lambda v: d_vec * v, rhs_, rtol=1e-10, max_steps=64,
                single_reduction=fused,
            ).x

        out[label] = {
            "primal": count_collectives(solve, b),
            "adjoint": count_collectives(
                jax.grad(lambda rhs_, implicit=implicit: jnp.sum(implicit(rhs_) ** 2)), b
            ),
        }

    # Implicit-adjoint FGMRES via linear_solve.
    def gm(rhs_):
        return linear_solve(
            lambda v: d_vec * v, rhs_,
            solver=lambda mv, r: gmres(mv, r, restart=16, rtol=1e-10).x,
        )

    out["gmres_linear_solve"] = {
        "primal": count_collectives(gm, b),
        "adjoint": count_collectives(jax.grad(lambda rhs_: jnp.sum(gm(rhs_) ** 2)), b),
    }
    return out


def run_collectives_benchmark(
    device_counts: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, object]:
    """Return JSON-safe collective counts per device count and solver."""
    available = len(jax.devices())
    counts = [c for c in device_counts if c <= available]
    return {
        "solvax_version": __version__,
        "devices_available": available,
        "results": {str(c): _cases(c) for c in counts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = run_collectives_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}  devices={result['devices_available']}")
    for devices, cases in result["results"].items():
        print(f"\n-- {devices} device(s) --")
        print(f"{'case':>24} {'primal':>8} {'adjoint':>8}")
        for name, row in cases.items():
            print(f"{name:>24} {row['primal']:>8} {row['adjoint']:>8}")


if __name__ == "__main__":
    main()
