"""Every test module and every marker used by a test is selected by some CI job.

Infrastructure test, no physics anchor: it reads the workflows, expands each job's
matrix rows and environment variables into its pytest commands, and compares the
modules and markers they select with `tests/test_*.py` and the markers the tests use.
A module that no job runs is a test that cannot fail.
"""

import itertools
import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULES = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_*.py")}
VALUE_OPTIONS = {"-n", "-m", "-k", "-p", "-c", "-o", "--rootdir", "--durations", "--maxfail"}


def _rows(job):
    matrix = dict((job.get("strategy") or {}).get("matrix") or {})
    include = matrix.pop("include", [])
    axes = {k: v for k, v in matrix.items() if isinstance(v, list)}
    base = [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]
    return base + include if axes else include or [{}]


def _expand(text, row, env):
    text = re.sub(r"\$\{\{\s*matrix\.(\w+)\s*\}\}", lambda m: str(row.get(m[1], "")), text)
    text = re.sub(r"\$\{\{[^}]*\}\}", "", text)
    return re.sub(r"\$\{?(\w+)\}?", lambda m: _expand(str(env.get(m[1], "")), row, {}), text)


def _runs(condition, row):
    """Evaluate a step's `if` when it reads only the matrix; otherwise assume it runs."""
    text = re.sub(r"\$\{\{|\}\}", "", str(condition)).replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    text = re.sub(r"matrix\.(\w+)", lambda m: repr(row.get(m[1])), text)
    text = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", text))
    other = r"[A-Za-z_]\w*\s*\(|\b(?!and\b|or\b|not\b|True\b|False\b|None\b)[A-Za-z_]\w*\."
    if re.search(other, text):
        return True
    return bool(eval(text, {"__builtins__": {}})) if text.strip() else True


def selection(command):
    """Modules and `-m` expression of one pytest command line."""
    words = shlex.split(command)
    words = words[next(i for i, w in enumerate(words) if w.endswith("pytest")) + 1 :]
    paths, ignored, expression, skip = [], set(), "", False
    for i, word in enumerate(words):
        if skip:
            skip = False
        elif word in VALUE_OPTIONS:
            skip, expression = True, words[i + 1] if word == "-m" else expression
        elif word.startswith("--ignore="):
            ignored.add(word.split("=", 1)[1])
        elif re.match(r"tests(/|$)", word):
            paths.append(word.split("::")[0].rstrip("/"))
    if not paths or "tests" in paths:
        return MODULES - ignored, expression
    return set(paths) - ignored, expression


def commands():
    """Every pytest command of every workflow, once per matrix row."""
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        for job in (yaml.safe_load(workflow.read_text()) or {}).get("jobs", {}).values():
            for row, step in itertools.product(_rows(job), job.get("steps", [])):
                if not _runs(step.get("if", ""), row):
                    continue
                env = {**(job.get("env") or {}), **(step.get("env") or {})}
                for line in str(step.get("run", "")).replace("\\\n", " ").splitlines():
                    if re.match(r"\s*(python -m |uv run )?pytest\b", line):
                        yield selection(_expand(line, row, env))


def _selects(expression, marker):
    names = r"\b(?!and\b|or\b|not\b)[\w.]+"
    python = re.sub(names, lambda m: str(m[0] == marker), expression)
    return expression == "" or eval(python, {"__builtins__": {}})


def test_selection_parser_reads_paths_and_markers():
    modules, expression = selection("pytest -n 3 tests/test_ode.py::test_a -m 'not gpu' --cov")
    assert (modules, expression) == ({"tests/test_ode.py"}, "not gpu")
    assert selection("python -m pytest -q")[0] == MODULES
    assert not _selects("not gpu", "gpu") and _selects("gpu or slow", "gpu")


def test_every_test_module_is_selected_by_a_job():
    selected = set().union(*(modules for modules, _ in commands()))
    missing = sorted(MODULES - selected)
    assert not missing, f"modules run by no CI job: {missing}"


def test_every_marker_in_use_is_selected_by_a_job(request):
    declared = [line.split(":")[0].strip() for line in request.config.getini("markers")]
    for marker in declared:
        users = {m for m in MODULES if f"mark.{marker}" in (ROOT / m).read_text()}
        runs = [users & mods for mods, expression in commands() if _selects(expression, marker)]
        assert not users or any(runs), f"marker {marker!r} is selected by no CI job"
