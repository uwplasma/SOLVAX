import os
import re
import subprocess
import sys
from pathlib import Path

import jax

import solvax


def test_tests_import_the_worktree_package() -> None:
    """Avoid silently testing an older globally installed SOLVAX release."""
    source_root = Path(__file__).parents[1] / "src"
    assert Path(solvax.__file__).resolve().is_relative_to(source_root.resolve())


def test_pep561_marker_is_present() -> None:
    """Strict downstream type checking requires the PEP 561 source marker."""
    marker = Path(__file__).parents[1] / "src" / "solvax" / "py.typed"
    assert marker.is_file()


def test_minimum_ci_lane_pins_the_declared_floors() -> None:
    """A floor nothing installs is untested; the minimum lane pins each one."""
    root = Path(__file__).parents[1]
    project = (root / "pyproject.toml").read_text()
    floors = dict(re.findall(r'"([a-z]+)>=([^"]+)"', project.split("[project.urls]")[0]))
    workflow = (root / ".github/workflows/tests.yml").read_text()
    pins = dict(re.findall(r"'([a-z]+)==([^']+)'", workflow))
    for name in ("jax", "jaxlib", "equinox"):
        assert floors[name] == pins[name], name


def test_import_warns_exactly_on_jax_with_the_constant_folding_defect() -> None:
    """jax-ml/jax#33479: JAX 0.5.0 to 0.8.1 fold non-leading-axis gathers wrongly."""
    affected = (0, 5) <= jax.__version_info__[:3] < (0, 8, 2)
    env = {**os.environ, "PYTHONPATH": str(Path(solvax.__file__).parents[1])}
    command = [sys.executable, "-W", "error::RuntimeWarning", "-c", "import solvax"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert (result.returncode != 0) == affected, result.stderr[-300:]


def test_import_warning_names_the_defect_and_the_workaround(monkeypatch) -> None:
    """Re-run the import-time check in process under an affected and a fixed JAX."""
    import importlib
    import warnings

    try:
        monkeypatch.setattr(jax, "__version_info__", (0, 6, 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(solvax)
        messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("jax-ml/jax#33479" in m and "jit arguments" in m for m in messages)

        monkeypatch.setattr(jax, "__version_info__", (0, 8, 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(solvax)
        assert not [w for w in caught if "33479" in str(w.message)]
    finally:
        monkeypatch.undo()
        importlib.reload(solvax)
