"""Smoke test for the checkpointed block benchmark."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_checkpointed_block.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_checkpointed_block", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_checkpointed_block_benchmark_records_memory_and_accuracy():
    result = _MODULE.run_checkpointed_block_benchmark(n_blocks=8, block_size=4, repeats=1)
    assert result["relative_error"] < 1e-12
    assert result["checkpointed"]["temp_bytes"] < result["full"]["temp_bytes"]
