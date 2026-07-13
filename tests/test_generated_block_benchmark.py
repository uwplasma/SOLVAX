"""Smoke tests for the reproducible generated-block benchmark."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_generated_block.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_generated_block", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_generated_block_benchmark_records_accuracy_timing_and_memory():
    result = _MODULE.run_generated_block_benchmark(
        n_theta=4, n_zeta=5, n_xi=7, keep_lowest=3, n_rhs=2, repeats=2
    )
    assert result["block_size"] == 20
    assert result["n_blocks"] == 8
    assert result["relative_error_vs_materialized"] < 1.0e-12
    assert result["compile_seconds"] > 0.0
    assert result["warm_median_seconds"] > 0.0
    assert len(result["warm_samples_seconds"]) == 2
    assert result["executable_memory"] is not None
