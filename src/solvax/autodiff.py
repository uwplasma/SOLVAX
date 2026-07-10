r"""Memory-chunked Jacobian construction (the ``jac_chunk_size`` knob).

The dense Jacobian of ``f : R^n -> R^m`` is assembled column by column
(forward mode: one JVP against each input basis vector) or row by row
(reverse mode: one VJP against each output basis vector). Evaluating all
``n`` (or ``m``) directional derivatives in a single :func:`jax.vmap` — what
:func:`jax.jacfwd` / :func:`jax.jacrev` do — replicates the intermediate
program state that many times at once, so peak memory grows with the full
Jacobian width. **Chunking** trades that peak for a modest slowdown: the basis
is split into blocks of ``chunk_size``, each block is vmapped, and the blocks
are walked with :func:`jax.lax.map`, so peak memory scales with

    memory ~ m0 + m1 * chunk_size,     time ~ t0 + t1 * (n / chunk_size),

i.e. a knob between the fast/hungry ``chunk_size = n`` (plain
``jax.jacfwd``/``jacrev``) and the slow/lean ``chunk_size = 1``. This is the
memory lever that makes otherwise-OOM optimization Jacobians (residual of a
large parameter vector) and matrix-free operator materializations fit on a
single accelerator; it is the analogue of DESC's ``jac_chunk_size`` argument,
factored out here for reuse across kinetic and equilibrium codes.

The chunked Jacobians are numerically identical to their JAX counterparts (the
same JVP/VJP is evaluated for every basis vector; only the batching changes),
and remain jit/vmap/grad-transparent.

Contract: ``fun`` maps one array argument (selected by ``argnums``; arbitrary
shape, flattened internally) to one array output (arbitrary shape). The
returned Jacobian follows the JAX convention ``output_shape + input_shape``.
Pytree inputs/outputs are out of scope — ravel them (e.g. with
:func:`jax.flatten_util.ravel_pytree`) before calling.

References
----------
- D. Panici et al., *The DESC stellarator optimization code*, and the DESC
  ``jac_chunk_size`` memory option, https://github.com/PlasmaControl/DESC.
- JAX documentation for :func:`jax.jacfwd`, :func:`jax.jacrev`,
  :func:`jax.lax.map` (the ``batch_size`` chunking argument).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "chunk_map",
    "auto_chunk_size",
    "chunked_jacfwd",
    "chunked_jacrev",
    "chunked_jacobian",
]


def chunk_map(
    fun: Callable, xs: jax.Array, *, chunk_size: int | None = None
) -> jax.Array:
    """Map ``fun`` over the leading axis of ``xs`` in fixed-size chunks.

    A thin wrapper choosing between a single wide :func:`jax.vmap` and a
    chunked :func:`jax.lax.map`:

    - ``chunk_size is None``: one ``jax.vmap`` over all ``len(xs)`` slices
      (maximum parallelism, maximum memory).
    - ``chunk_size = k``: ``jax.lax.map(fun, xs, batch_size=k)`` — the leading
      axis is processed ``k`` slices at a time (vmapped within a chunk,
      scanned across chunks), so peak memory scales with ``k`` instead of
      ``len(xs)``. The final chunk is padded internally by ``lax.map`` and the
      padding discarded.

    Args:
        fun: callable applied to a single leading-axis slice of ``xs``.
        xs: array (or pytree of arrays) with a common leading axis.
        chunk_size: chunk width, or ``None`` for a single vmap.

    Returns:
        The stacked results, leading axis equal to ``len(xs)``.
    """
    if chunk_size is None:
        return jax.vmap(fun)(xs)
    return jax.lax.map(fun, xs, batch_size=int(chunk_size))


def _device_memory_limit() -> int | None:
    """Best-effort per-device memory limit in bytes, or ``None`` if unknown."""
    try:
        device = jax.local_devices()[0]
        stats = device.memory_stats()
    except Exception:  # pragma: no cover - platform dependent
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    return int(limit) if limit else None


def auto_chunk_size(
    dim: int,
    output_size: int = 1,
    *,
    max_memory_bytes: int | None = None,
    element_bytes: int = 8,
    memory_fraction: float = 0.5,
) -> int:
    r"""Pick a chunk width for a Jacobian with ``dim`` basis vectors.

    Two regimes:

    - **Memory budget** (``max_memory_bytes`` given, or a device limit is
      queryable): return the largest ``chunk`` whose block of directional
      derivatives fits the budget,

          chunk = floor(memory_fraction * budget / (output_size *
          element_bytes)),

      clamped to ``[1, dim]`` — DESC's "largest that fits" semantics.
    - **Heuristic** (no budget available): return the square-root-balanced
      width ``ceil(sqrt(dim))``, where the peak memory ``~ chunk`` and the
      chunk count ``~ dim / chunk`` are balanced, clamped to ``[1, dim]``.

    Args:
        dim: number of basis vectors (input size for forward mode, output
            size for reverse mode).
        output_size: element count of one directional-derivative result (the
            other Jacobian dimension), used only in the budget regime.
        max_memory_bytes: explicit byte budget; if ``None``, the first local
            device's ``bytes_limit`` is used when available, else the
            heuristic regime applies.
        element_bytes: bytes per array element (8 for float64, 4 for float32).
        memory_fraction: fraction of the budget the Jacobian block may use.

    Returns:
        A chunk width in ``[1, dim]`` (returns ``1`` for ``dim <= 1``).
    """
    dim = int(dim)
    if dim <= 1:
        return 1
    budget = max_memory_bytes if max_memory_bytes is not None else _device_memory_limit()
    if budget is None:
        return max(1, min(dim, math.ceil(math.sqrt(dim))))
    per_vector = max(1, int(output_size) * int(element_bytes))
    fit = int(memory_fraction * int(budget)) // per_vector
    return max(1, min(dim, fit))


def _sub(args: tuple, argnums: int, value) -> tuple:
    """Return ``args`` with position ``argnums`` replaced by ``value``."""
    return args[:argnums] + (value,) + args[argnums + 1 :]


def _resolve_chunk(chunk_size, dim: int, output_size: int) -> int | None:
    """Turn the public ``chunk_size`` argument into an int (or ``None``)."""
    if chunk_size is None:
        return None
    if isinstance(chunk_size, str):
        if chunk_size != "auto":
            raise ValueError(
                f"chunk_size string must be 'auto', got {chunk_size!r}"
            )
        return auto_chunk_size(dim, output_size)
    chunk = int(chunk_size)
    if chunk < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk}")
    return chunk


def chunked_jacfwd(
    fun: Callable, argnums: int = 0, *, chunk_size: int | str | None = "auto"
) -> Callable:
    """Forward-mode Jacobian assembled in column chunks.

    A drop-in for :func:`jax.jacfwd` (single array argument, single array
    output) whose columns — one JVP per input basis vector — are evaluated
    ``chunk_size`` at a time, bounding peak memory. Forward mode is the
    efficient choice when the input is smaller than the output (tall
    Jacobian).

    Args:
        fun: function of one array argument (position ``argnums``) returning
            an array; other positional arguments are held fixed.
        argnums: index of the argument to differentiate.
        chunk_size: column-chunk width. ``"auto"`` (default) uses
            :func:`auto_chunk_size` on the input dimension; ``None`` reproduces
            :func:`jax.jacfwd` exactly (a single vmap); an int fixes the width.

    Returns:
        A callable ``jac(*args) -> Jacobian`` of shape
        ``output_shape + input_shape``.
    """

    def jacfun(*args):
        x = jnp.asarray(args[argnums])
        in_shape = x.shape
        n = x.size

        def single(a):
            return fun(*_sub(args, argnums, a))

        basis = jnp.eye(n, dtype=x.dtype).reshape((n,) + in_shape)

        def column(tangent):
            _, jvp = jax.jvp(single, (x,), (tangent,))
            return jvp

        out_size = int(jax.eval_shape(single, x).size)
        chunk = _resolve_chunk(chunk_size, n, out_size)
        cols = chunk_map(column, basis, chunk_size=chunk)
        # cols: (n, *out_shape) -> (*out_shape, *in_shape) to match jacfwd.
        return jax.tree.map(
            lambda c: jnp.moveaxis(c, 0, -1).reshape(c.shape[1:] + in_shape), cols
        )

    return jacfun


def chunked_jacrev(
    fun: Callable, argnums: int = 0, *, chunk_size: int | str | None = "auto"
) -> Callable:
    """Reverse-mode Jacobian assembled in row chunks.

    A drop-in for :func:`jax.jacrev` (single array argument, single array
    output) whose rows — one VJP per output basis vector — are evaluated
    ``chunk_size`` at a time, bounding peak memory. Reverse mode is the
    efficient choice when the output is smaller than the input (wide
    Jacobian), e.g. a scalar-ish residual of a large parameter vector.

    Args:
        fun: function of one array argument (position ``argnums``) returning
            an array; other positional arguments are held fixed.
        argnums: index of the argument to differentiate.
        chunk_size: row-chunk width. ``"auto"`` (default) uses
            :func:`auto_chunk_size` on the output dimension; ``None``
            reproduces :func:`jax.jacrev` exactly; an int fixes the width.

    Returns:
        A callable ``jac(*args) -> Jacobian`` of shape
        ``output_shape + input_shape``.
    """

    def jacfun(*args):
        x = jnp.asarray(args[argnums])
        in_shape = x.shape

        def single(a):
            return fun(*_sub(args, argnums, a))

        y, vjp = jax.vjp(single, x)
        out_shape = y.shape
        m = y.size
        basis = jnp.eye(m, dtype=y.dtype).reshape((m,) + out_shape)

        def row(cotangent):
            return vjp(cotangent)[0]

        chunk = _resolve_chunk(chunk_size, m, x.size)
        rows = chunk_map(row, basis, chunk_size=chunk)
        # rows: (m, *in_shape) -> (*out_shape, *in_shape) to match jacrev.
        return rows.reshape(out_shape + in_shape)

    return jacfun


def chunked_jacobian(
    fun: Callable,
    argnums: int = 0,
    *,
    mode: str = "rev",
    chunk_size: int | str | None = "auto",
) -> Callable:
    """Memory-chunked Jacobian with forward/reverse/auto mode selection.

    Dispatches to :func:`chunked_jacfwd` or :func:`chunked_jacrev`.

    Args:
        fun: function of one array argument returning an array.
        argnums: index of the argument to differentiate.
        mode: ``"rev"`` (default, mirrors :func:`jax.jacobian`), ``"fwd"``, or
            ``"auto"`` — pick forward mode when the input has no more elements
            than the output (fewer basis vectors), reverse otherwise.
        chunk_size: forwarded to the chosen builder.

    Returns:
        A callable ``jac(*args) -> Jacobian`` of shape
        ``output_shape + input_shape``.
    """
    if mode == "fwd":
        return chunked_jacfwd(fun, argnums, chunk_size=chunk_size)
    if mode == "rev":
        return chunked_jacrev(fun, argnums, chunk_size=chunk_size)
    if mode != "auto":
        raise ValueError(f"mode must be 'fwd', 'rev' or 'auto', got {mode!r}")

    def jacfun(*args):
        x = jnp.asarray(args[argnums])
        out = jax.eval_shape(lambda a: fun(*_sub(args, argnums, a)), x)
        builder = chunked_jacfwd if x.size <= out.size else chunked_jacrev
        return builder(fun, argnums, chunk_size=chunk_size)(*args)

    return jacfun
