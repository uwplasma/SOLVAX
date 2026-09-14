"""Smoke test for the operator-coupled block Thomas benchmark."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_operator_couplings.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_operator_couplings", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_operator_couplings_benchmark_records_accuracy_storage_and_timing():
    result = _MODULE.run_operator_couplings_benchmark(T=4, Z=5, n=4, batch=2, reps=1)
    assert result["m"] == 20 and result["blocks"] == 4 and result["batch"] == 2
    assert set(result["routes"]) == {"dense", "lu", "inverse"}
    dense = result["routes"]["dense"]
    for name in ("lu", "inverse"):
        route = result["routes"][name]
        assert route["rel_diff_vs_dense"] < 1e-12
        assert route["max_rel_residual"] < 1e-12
        assert route["stored_factor_bytes"] < dense["stored_factor_bytes"]
        assert route["factor_s_per_block"] > 0.0 and route["apply_s_per_block"] > 0.0
