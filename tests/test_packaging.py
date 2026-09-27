import re
import tomllib
from pathlib import Path

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
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    floors = dict(re.fullmatch(r"([a-z]+)>=(\S+)", dep).groups() for dep in deps)
    pins = dict(re.findall(r"'([a-z]+)==([^']+)'", (root / ".github/workflows/tests.yml").read_text()))
    for name in ("jax", "jaxlib", "equinox"):
        assert floors[name] == pins[name], name
