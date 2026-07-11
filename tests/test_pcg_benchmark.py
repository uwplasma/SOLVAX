from __future__ import annotations

import importlib.util
from pathlib import Path

_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_pcg.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_pcg", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_pcg_benchmark = _MODULE.run_pcg_benchmark


def test_pcg_benchmark_records_correctness_and_cold_warm_timings():
    result = run_pcg_benchmark(size=64, repeats=2)
    assert result["converged"] is True
    assert result["status"] == "converged"
    assert result["iterations"] <= 2
    assert result["relative_error"] < 1.0e-5
    assert result["implementation"]["solvax_version"] == "0.5.1"
    assert len(result["implementation"]["pcg_sha256"]) == 64
    assert result["jax_version"]
    assert result["python_version"]
    assert result["cold_seconds"] > 0.0
    assert result["warm_median_seconds"] > 0.0
    assert len(result["warm_samples_seconds"]) == 2
