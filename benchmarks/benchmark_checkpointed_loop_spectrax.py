"""Interleave released/candidate SOLVAX replay timings on the same plasma solve.

Use a Python environment with released SOLVAX and an editable SPECTRAX install.
The candidate is loaded from this checkout without changing the environment.
This reports steady execution times, not per-method allocator peaks: both
executables coexist. Use isolated processes for attributable memory measurements.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import spectrax
import spectrax._autodiff as plasma_ad

import solvax


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrax-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.spectrax_root.resolve()
    if Path(spectrax.__file__).resolve().parents[1] != root:
        parser.error("The imported SPECTRAX must match --spectrax-root")
    baseline_path = Path(inspect.getsourcefile(solvax.checkpointed_fori_loop))
    candidate_path = Path(__file__).resolve().parents[1] / "src/solvax/autodiff.py"
    if baseline_path.resolve() == candidate_path.resolve():
        parser.error("Use released SOLVAX as the baseline; remove candidate PYTHONPATH")
    example = load("phase_control", root / "Examples/2D_phase_control.py")
    candidate = load("candidate_autodiff", candidate_path)
    phase = np.random.default_rng(7).uniform(-np.pi, np.pi, 8)
    original = plasma_ad.checkpointed_fori_loop
    executables, compiled = {}, {}
    try:
        for name, loop in (
            ("released", solvax.checkpointed_fori_loop),
            ("candidate", candidate.checkpointed_fori_loop),
        ):
            plasma_ad.checkpointed_fori_loop = loop
            objective, _ = example.problem(32, 4, 2.0, args.steps)
            start = perf_counter()
            executables[name] = jax.jit(jax.value_and_grad(objective)).lower(phase).compile()
            compiled[name] = perf_counter() - start
    finally:
        plasma_ad.checkpointed_fori_loop = original
    reference = jax.block_until_ready(executables["released"](phase))
    comparison = jax.block_until_ready(executables["candidate"](phase))
    for x, y in zip(reference, comparison, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)
    for executable in executables.values():
        for _ in range(3):
            jax.block_until_ready(executable(phase))
    times = {name: [] for name in executables}
    for index in range(args.repeats):
        order = ("released", "candidate") if index % 2 == 0 else ("candidate", "released")
        for name in order:
            start = perf_counter()
            jax.block_until_ready(executables[name](phase))
            times[name].append(perf_counter() - start)
    paths = dict(
        released=baseline_path,
        candidate=candidate_path,
        phase_control=root / "Examples/2D_phase_control.py",
        plasma_solver=root / "spectrax/_autodiff.py",
        benchmark=Path(__file__),
    )
    report = dict(
        steps=args.steps,
        grid=32,
        hermite=4,
        controls=8,
        device=jax.devices()[0].device_kind,
        jax_version=jax.__version__,
        solvax_version=solvax.__version__,
        compile_seconds=compiled,
        samples_seconds=times,
        median_seconds={k: float(np.median(v)) for k, v in times.items()},
        source_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
