"""One-command reproduction of every committed benchmark record.

``python -m benchmarks.reproduce`` regenerates all measurement JSONs in
``benchmarks/results/`` from the current environment, after (a) writing a
hardware/software manifest and (b) validating the timer against a known
reference interval — so a third party can reproduce every documented number,
and trust the clock it was measured with, without contacting the authors.

``--quick`` runs reduced problem sizes (minutes, CI-friendly) into a scratch
directory and checks only that every driver executes and emits valid records;
the committed records are produced by the default full mode.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax

from benchmarks.benchmark_bounded_adjoint import run_bounded_adjoint_benchmark
from benchmarks.benchmark_collectives import run_collectives_benchmark
from benchmarks.benchmark_mixed_precision_adjoint import (
    run_mixed_precision_adjoint_benchmark,
)
from benchmarks.benchmark_sweeps import run_sweep_benchmark
from solvax import __version__

RESULTS = Path(__file__).parent / "results"


def hardware_manifest() -> dict[str, object]:
    """Machine/software state the measurements depend on."""
    import jaxlib

    return {
        "solvax": __version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "devices": [str(d) for d in jax.devices()],
        "default_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
    }


def validate_timer(interval_s: float = 0.1, tolerance: float = 0.25) -> dict[str, float]:
    """Check ``perf_counter`` against a known sleep; raise if the clock lies.

    A quarter-relative tolerance absorbs scheduler jitter while still catching
    a broken or misscaled timer — the failure mode that silently corrupts
    every wall-time table.
    """
    start = time.perf_counter()
    time.sleep(interval_s)
    measured = time.perf_counter() - start
    error = abs(measured - interval_s) / interval_s
    if error > tolerance:
        raise RuntimeError(
            f"timer validation failed: slept {interval_s}s, measured {measured:.4f}s"
        )
    return {"interval_s": interval_s, "measured_s": measured, "relative_error": error}


def run_all(quick: bool) -> int:
    out_dir = RESULTS if not quick else RESULTS / "quick"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = hardware_manifest()
    manifest["timer_validation"] = validate_timer()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest['platform']} | jax {manifest['jax']} | timer ok")

    if quick:
        jobs = {
            "bounded_adjoint": lambda: run_bounded_adjoint_benchmark(
                block_counts=(16, 64), decay_windows=(0, 2)
            ),
            "collectives": lambda: run_collectives_benchmark(device_counts=(1, 2)),
            "mixed_precision_adjoint": lambda: run_mixed_precision_adjoint_benchmark(
                dominances=(4.0, 1.5), cost_shape=(32, 4)
            ),
            "sweeps": run_sweep_benchmark,
        }
    else:
        jobs = {
            "bounded_adjoint": run_bounded_adjoint_benchmark,
            "collectives": run_collectives_benchmark,
            "mixed_precision_adjoint": run_mixed_precision_adjoint_benchmark,
            "sweeps": run_sweep_benchmark,
        }

    failures = 0
    for name, job in jobs.items():
        start = time.perf_counter()
        try:
            record = job()
        except Exception as error:  # noqa: BLE001 - report every driver failure
            print(f"{name}: FAILED ({error})")
            failures += 1
            continue
        (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
        print(f"{name}: ok ({time.perf_counter() - start:.1f}s)")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="reduced sizes into results/quick/ (CI smoke; does not overwrite records)",
    )
    args = parser.parse_args()
    raise SystemExit(run_all(quick=args.quick))


if __name__ == "__main__":
    main()
