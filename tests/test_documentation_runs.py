"""Execute the code the documentation shows, rather than only linting it.

Every public-facing snippet defect this file guards against was shipped: a
README gradient that differentiated a closed-over constant and returned zero, an
example that passed the advisor's record where an integer was expected, a
release note naming a dataclass field that does not exist. Linting caught none
of them, because all three are valid Python that does the wrong thing.

The README block is extracted and run under a small preamble that supplies the
symbols the prose introduces but the block does not define. If a snippet stops
running, or a documented attribute stops existing, these tests fail.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

import solvax as sx

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EXAMPLES = ROOT / "examples"
SRC = ROOT / "src"


def _env() -> dict[str, str]:
    """Subprocess environment that imports the tree under test.

    pytest puts ``src`` on the path through ``pythonpath`` in the project
    configuration, but a subprocess does not inherit that. Without this the
    snippets run against whatever ``solvax`` happens to be installed, which is
    exactly the mistake these tests exist to catch.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env

# The README's snippets are excerpts: they name `lower`, `block_fn`, `loss` and
# friends without constructing them, because the surrounding prose does. This
# preamble builds a small consistent problem so the excerpts run as written.
PREAMBLE = """
import jax
import jax.numpy as jnp
import solvax as sx

N, m, K = 10, 3, 3
_eye = jnp.eye(m)
lower = jnp.stack([_eye * 0.25 * (j > 0) for j in range(N)])
diag = jnp.stack([_eye * (4.0 + 0.5 * j) for j in range(N)])
upper = jnp.stack([_eye * 0.25 * (j < N - 1) for j in range(N)])
rhs = jnp.ones((N, m))
p = jnp.array([0.7])


def block_fn(params, j):
    return (
        _eye * 0.25 * (j > 0),
        _eye * (4.0 + params[0] * (1.0 + j)),
        _eye * 0.25 * (j < N - 1),
    )


# Unparameterized generator: the README's `row(j)`.
def row(j):
    return block_fn(p, j)


def loss(x):
    return jnp.sum(x ** 2)


b = rhs
b2 = rhs
x0 = jnp.zeros((N, m))
initial_state = x0


def matvec(v):
    return 4.0 * v - 0.25 * jnp.roll(v, 1, axis=0) - 0.25 * jnp.roll(v, -1, axis=0)


matvec2 = matvec


def preconditioner(v):
    return v / 4.0


coarse_inverse = preconditioner
approx_inverse = preconditioner


def coupling_sweep(v):
    return 0.25 * jnp.roll(v, 1, axis=0) + rhs


coupling_map = coupling_sweep


def residual_fn(v):
    return matvec(v) - rhs


# A scalar cyclic line for the tridiagonal excerpts.
n_line = 16
sub = jnp.full((n_line,), -1.0)
dia = jnp.full((n_line,), 4.0)
sup = jnp.full((n_line,), -1.0)
line_rhs = jnp.ones((n_line,))
rhs1 = rhs
rhs2 = 2.0 * rhs
"""


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.S)


@pytest.mark.slow_examples
@pytest.mark.parametrize("index", range(len(_python_blocks(README.read_text()))))
def test_readme_snippet_runs(index: int, tmp_path: Path) -> None:
    block = _python_blocks(README.read_text())[index]
    script = tmp_path / f"readme_{index}.py"
    script.write_text(PREAMBLE + "\n" + block)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_env(),
    )
    assert result.returncode == 0, (
        f"README python block {index} failed:\n{block}\n{result.stderr[-2000:]}"
    )


def test_readme_gradient_snippet_is_not_identically_zero() -> None:
    """The failure mode that shipped: a gradient of a closed-over constant.

    ``jax.grad(lambda p: loss(x_low))`` runs without error and returns zeros,
    because the lambda ignores its argument. Executing the snippet is therefore
    not enough on its own --- the value has to be looked at.
    """
    blocks = [b for b in _python_blocks(README.read_text()) if "jax.grad" in b]
    assert blocks, "the README no longer shows a gradient"
    for block in blocks:
        namespace: dict = {}
        exec(PREAMBLE + "\n" + block, namespace)  # noqa: S102
        grads = [v for k, v in namespace.items() if k.startswith("grad")]
        assert grads, "no gradient bound in the snippet"
        for g in grads:
            assert jnp.any(jnp.asarray(g) != 0.0), (
                "README gradient is identically zero; the objective probably "
                "closes over the solution instead of recomputing it"
            )


def test_localization_window_documented_fields_exist() -> None:
    """Release notes and docstrings name these; renaming one must fail here."""
    n_blocks, m, keep = 10, 2, 2

    def block_fn(j):
        eye = jnp.eye(m)
        return (
            eye * 0.2 * (j > 0),
            eye * (5.0 + 0.4 * j),
            eye * 0.2 * (j < n_blocks - 1),
        )

    advice = sx.localization_crossover_window(block_fn, n_blocks, keep_lowest=keep)
    for field in (
        "window",
        "crossover_row",
        "localized",
        "primal_profile",
        "certified",
        "status",
    ):
        assert hasattr(advice, field), f"documented field {field!r} is gone"
    assert advice.certified is False


@pytest.mark.slow_examples
@pytest.mark.parametrize(
    "script", sorted(p.name for p in EXAMPLES.glob("*.py")) if EXAMPLES.is_dir() else []
)
def test_example_runs(script: str) -> None:
    """Every shipped example must execute, not merely lint."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_env(),
        timeout=900,
    )
    assert result.returncode == 0, (
        f"{script} failed:\n{result.stdout[-1500:]}\n{result.stderr[-2000:]}"
    )


def test_example_run_lines_name_themselves() -> None:
    """A stale ``Run:`` line sends a reader to a file that does not exist."""
    wrong = []
    for path in sorted(EXAMPLES.glob("*.py")):
        head = path.read_text()[:2000]
        for match in re.findall(r"Run:\s+python\s+(\S+)", head):
            if not (ROOT / match).is_file():
                wrong.append(f"{path.name} points at missing {match}")
    assert not wrong, "\n".join(wrong)


def test_every_release_note_is_in_the_toctree() -> None:
    """A notes file outside the toctree fails the docs build, not the tests.

    ``sphinx -W`` treats "document isn't included in any toctree" as an error,
    so writing release notes and forgetting the index entry breaks CI at the
    build step -- after the whole test matrix has already run. Catching it here
    turns a fifteen-minute round trip into a one-line failure.
    """
    docs = ROOT / "docs"
    index = (docs / "index.md").read_text()
    missing = sorted(
        p.stem for p in docs.glob("release-notes-*.md") if p.stem not in index
    )
    assert not missing, (
        f"release notes not listed in docs/index.md: {missing}. Add them to the "
        f"Reference toctree or `sphinx -W` will fail the build."
    )
