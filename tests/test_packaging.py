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
