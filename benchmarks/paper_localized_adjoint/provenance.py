"""Provenance stamped onto every localized-adjoint paper record.

A record that cannot be traced to an exact commit, environment, and problem
specification cannot support a published number, so every benchmark in this
package writes this block and no benchmark overwrites an existing record
silently.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

REPO_URL = "https://github.com/uwplasma/SOLVAX"
_ENV_KEYS = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "CUDA_VISIBLE_DEVICES",
)


def _git(*args: str) -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - git absent or not a checkout
        return "unknown"


def file_hashes(*paths: str | Path) -> dict[str, str]:
    """SHA-256 of the benchmark and solver files behind a record."""
    out: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if p.exists():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def _versions() -> dict[str, str]:
    mods = {"jax": jax, "numpy": np}
    try:
        import jaxlib

        mods["jaxlib"] = jaxlib
    except ImportError:  # pragma: no cover
        pass
    try:
        import scipy

        mods["scipy"] = scipy
    except ImportError:  # pragma: no cover
        pass
    try:
        import equinox

        mods["equinox"] = equinox
    except ImportError:  # pragma: no cover
        pass
    out = {name: getattr(m, "__version__", "unknown") for name, m in mods.items()}
    out["python"] = sys.version.split()[0]
    import solvax

    out["solvax"] = solvax.__version__
    return out


def provenance(*source_files: str | Path) -> dict[str, object]:
    """Repository, environment, and device block for a paper record."""
    devices = jax.devices()
    return {
        "repository": REPO_URL,
        "commit": _git("rev-parse", "HEAD"),
        "dirty_tree": bool(_git("status", "--porcelain")),
        "file_hashes": file_hashes(*source_files),
        "versions": _versions(),
        "backend": jax.default_backend(),
        "device_kind": devices[0].device_kind if devices else "none",
        "device_count": len(devices),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "platform": platform.platform(),
        "environment": {k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
    }
