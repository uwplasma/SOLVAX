"""Compile once per operator code, not once per operator closure.

``gmres`` and ``gcrot`` take Python callables. Called eagerly, a fresh closure
retraces and recompiles the whole solve even when only the arrays it captured
changed. Here a plain function is split into its code object plus the values in
its closure cells and defaults: arrays and floats become jit arguments, other
values stay static. Same code and same shapes then reuse one executable.
Anything that cannot be split (bound methods, callable objects, tracers)
takes the direct path.
"""

from __future__ import annotations

import types
from typing import Any

import jax
import numpy as np


class _Unsplittable(Exception):
    pass


class _Id:
    """Static value compared by identity."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _Id) and other.value is self.value

    def __hash__(self):
        return id(self.value)


class _Fn:
    """Pytree view of a function: leaves are captured arrays, aux is static."""

    def __init__(self, spec, leaves):
        self.spec, self.leaves = spec, leaves


def _split(value, leaves, depth=0):
    if depth > 6:
        raise _Unsplittable
    if isinstance(value, (jax.Array, np.ndarray, float, complex, np.generic)):
        leaves.append(value)
        return ("leaf",)
    if value is None or isinstance(value, (bool, int, str, type, types.ModuleType)):
        return ("static", value)
    if isinstance(value, types.FunctionType):
        if value.__kwdefaults__:
            raise _Unsplittable
        cells = tuple(_split(c.cell_contents, leaves, depth + 1) for c in value.__closure__ or ())
        defaults = tuple(_split(d, leaves, depth + 1) for d in value.__defaults__ or ())
        return ("fn", value.__code__, _Id(value.__globals__), value.__name__, defaults, cells)
    flat, treedef = jax.tree.flatten(value)
    if treedef.num_leaves == 1 and treedef.num_nodes == 1:
        return ("static", _Id(value))
    if flat and all(isinstance(x, (jax.Array, np.ndarray, float, complex)) for x in flat):
        leaves.extend(flat)
        return ("tree", treedef)
    return ("static", _Id(value))


def _build(spec, it):
    kind = spec[0]
    if kind == "leaf":
        return next(it)
    if kind == "static":
        return spec[1].value if isinstance(spec[1], _Id) else spec[1]
    if kind == "tree":
        return jax.tree.unflatten(spec[1], [next(it) for _ in range(spec[1].num_leaves)])
    _, code, globs, name, defaults, cells = spec
    cell_values = [_build(c, it) for c in cells]
    default_values = tuple(_build(d, it) for d in defaults) or None
    closure = tuple(types.CellType(v) for v in cell_values) or None
    return types.FunctionType(code, globs.value, name, default_values, closure)


jax.tree_util.register_pytree_node(
    _Fn,
    lambda f: (f.leaves, f.spec),
    lambda spec, leaves: _build(spec, iter(leaves)),
)


def _pack(fn):
    if fn is None:
        return None
    if not isinstance(fn, types.FunctionType):
        raise _Unsplittable
    leaves: list[Any] = []
    spec = _split(fn, leaves)
    return _Fn(spec, leaves)


def call_reusing_compile(impl, fns: dict, args: dict, static: dict):
    """Run ``impl(**fns, **args, **static)`` under a module-level jit.

    ``impl`` must be one stable function object per solver so jit's cache is
    shared across calls.
    """
    try:
        packed = {k: _pack(f) for k, f in fns.items()}
    except (_Unsplittable, ValueError):
        packed = None
    leaves = jax.tree.leaves((packed, args))
    if packed is None or any(isinstance(x, jax.core.Tracer) for x in leaves):
        return impl(**fns, **args, **static)
    return impl_jit(impl, tuple(static))(**packed, **args, **static)


_JITS: dict = {}


def impl_jit(impl, static_names):
    key = (impl, static_names)
    if key not in _JITS:
        _JITS[key] = jax.jit(impl, static_argnames=static_names)
    return _JITS[key]
