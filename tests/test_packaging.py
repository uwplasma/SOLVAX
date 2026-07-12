from pathlib import Path


def test_pep561_marker_is_present() -> None:
    """Strict downstream type checking requires the PEP 561 source marker."""
    marker = Path(__file__).parents[1] / "src" / "solvax" / "py.typed"
    assert marker.is_file()
