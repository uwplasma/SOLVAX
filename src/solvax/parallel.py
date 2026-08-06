"""Explicit sharding helpers for independent numerical batches."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map as _shard_map
except ImportError:  # JAX 0.4
    # A conditional import of the same name is a redefinition to a static
    # checker, which cannot know only one branch runs.
    from jax.experimental.shard_map import (  # type: ignore[no-redef]
        shard_map as _shard_map,
    )

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def shard_batch(
    local_function: Callable[[InputT], OutputT],
    *,
    mesh: Mesh,
    input_rank: int,
    output_rank: int,
    output_batch_axis: int = 0,
    axis_name: str = "device",
) -> Callable[[InputT], OutputT]:
    """Run independent batch shards without compiler-inserted replication.

    ``local_function`` receives only the batch items owned by one device. Its
    input batch axis is axis 0. The returned global function accepts an array
    whose batch axis is sharded over ``mesh`` and returns an array with the
    corresponding output batch axis sharded over the same mesh.

    This small wrapper is useful when ordinary ``jax.jit`` sharding propagates
    through a program but still replicates expensive batched work. It uses
    :func:`jax.shard_map` to make the per-device execution contract explicit.

    Args:
        local_function: Numerical program for one device's local batch.
        mesh: JAX device mesh containing ``axis_name``.
        input_rank: Rank of the global input array.
        output_rank: Rank of the global output array.
        output_batch_axis: Batch-axis position in the output.
        axis_name: Mesh axis that partitions the batch.

    Returns:
        A global function with the same call signature as ``local_function``.

    Raises:
        ValueError: If a rank, axis, or mesh setting is invalid.
    """
    if input_rank < 1:
        raise ValueError("input_rank must be at least 1")
    if output_rank < 1:
        raise ValueError("output_rank must be at least 1")
    if not -output_rank <= output_batch_axis < output_rank:
        raise ValueError(
            f"output_batch_axis must index an output with rank {output_rank}"
        )
    normalized_output_axis = output_batch_axis % output_rank
    if axis_name not in mesh.axis_names:
        raise ValueError(f"mesh has no axis named {axis_name!r}")

    input_spec = P(axis_name, *([None] * (input_rank - 1)))
    output_axes: list[str | None] = [None] * output_rank
    output_axes[normalized_output_axis] = axis_name
    output_spec = P(*output_axes)
    return _shard_map(
        local_function,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
    )


def axis_inner_product(axis_name: str):
    """Inner product for Krylov solves running inside ``shard_map``.

    Inside a ``shard_map`` region every operand leaf is a local shard, so
    the default inner product would return a partial sum. This helper
    computes the local Hermitian product and completes it with
    ``lax.psum`` over ``axis_name``, which makes :func:`solvax.gmres`,
    :func:`solvax.gcrot`, and :func:`solvax.newton_krylov` correct
    per-shard: pass it as ``inner_product=``.

    Args:
        axis_name: The mesh axis the caller's ``shard_map`` binds.

    Note:
        Call ``shard_map`` with ``check_vma=False`` around a solver using
        this inner product: the Krylov basis carry starts replicated and
        becomes shard-varying inside the iteration, which the
        varying-axis type checker rejects even though the computation is
        correct.

    Returns:
        A callable ``inner(left, right) -> scalar`` with the global value
        on every shard.
    """

    def inner(left, right):
        products = jax.tree.leaves(jax.tree.map(jnp.vdot, left, right))
        local = sum(products[1:], products[0])
        return jax.lax.psum(local, axis_name)

    return inner
