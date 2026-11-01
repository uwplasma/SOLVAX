"""Explicit sharding helpers for independent numerical batches."""

from __future__ import annotations

import math
import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map as _shard_map

    _CHECK_KEYWORD = "check_vma"
except ImportError:  # JAX 0.4
    # A conditional import of the same name is a redefinition to a static
    # checker, which cannot know only one branch runs.
    from jax.experimental.shard_map import (  # type: ignore[no-redef]
        shard_map as _shard_map,
    )

    _CHECK_KEYWORD = "check_rep"

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

_CGROUP_ROOT = "/sys/fs/cgroup"
_SCHEDULER_CPU_VARIABLES = ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS", "LSB_DJOB_NUMPROC")


def _cgroup_cpus() -> int | None:
    """CPU quota of this cgroup rounded up, or ``None`` when there is none."""
    root = Path(_CGROUP_ROOT)
    try:
        quota, period = (root / "cpu.max").read_text().split()
    except (OSError, ValueError):
        try:
            quota = (root / "cpu" / "cpu.cfs_quota_us").read_text()
            period = (root / "cpu" / "cpu.cfs_period_us").read_text()
        except OSError:
            return None
    try:  # An unlimited quota reads "max" on v2 and -1 on v1.
        quota_us, period_us = int(quota), int(period)
    except ValueError:
        return None
    return math.ceil(quota_us / period_us) if quota_us > 0 and period_us > 0 else None


def available_cpus() -> int:
    """CPUs this process may use: the smallest limit that applies to it.

    Takes the minimum of ``os.cpu_count()``, the scheduler affinity mask, the
    cgroup CPU quota, and the Slurm, PBS, SGE and LSF allocation variables.
    Unset or malformed limits are ignored. The result is at least 1.
    """
    limits = [os.cpu_count() or 1]
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        limits.append(len(affinity(0)))
    quota = _cgroup_cpus()
    if quota is not None:
        limits.append(quota)
    for name in _SCHEDULER_CPU_VARIABLES:
        value = os.environ.get(name, "")
        if value.isdecimal() and int(value) > 0:
            limits.append(int(value))
    return max(min(limits), 1)


def map_independent(
    fn: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    workers: int | None = None,
    executor: str = "thread",
    return_exceptions: bool = False,
) -> list[Any]:
    """Apply ``fn`` to independent Python tasks, keeping the input order.

    Use it for work that cannot share one compiled program, such as whole
    solves or file-backed cases. For array batches use :func:`shard_batch` or
    :func:`~solvax.autodiff.chunk_map`.

    Args:
        fn: Per-item computation sharing no mutable state with the others.
        items: Tasks; consumed once.
        workers: Pool size, clamped to ``[1, len(items)]``. ``None`` uses
            :func:`available_cpus`.
        executor: ``"thread"`` overlaps GIL-releasing work such as XLA
            execution; one worker runs inline. JAX context managers such as
            ``enable_x64`` are thread-local and do not reach pool threads.
            ``"process"`` always starts ``spawn`` children, even for one
            worker, because forking an initialized JAX is unsafe and a child
            may then change its own device environment. ``fn`` and the items
            must be picklable, and ``fn`` should return host data.
        return_exceptions: Put a raised ``Exception`` in that item's slot
            instead of raising the first one in input order and cancelling
            tasks that have not started.

    Returns:
        One result per item, in input order.

    Raises:
        ValueError: If ``executor`` is unknown.
    """
    if executor not in ("thread", "process"):
        raise ValueError(f"executor must be 'thread' or 'process', got {executor!r}")
    tasks = list(items)
    if not tasks:
        return []
    count = available_cpus() if workers is None else int(workers)
    count = max(1, min(count, len(tasks)))

    def collect(calls: Iterable[Callable[[], Any]]) -> list[Any]:
        results: list[Any] = []
        for call in calls:
            try:
                results.append(call())
            except Exception as error:
                if not return_exceptions:
                    raise
                results.append(error)
        return results

    if executor == "thread" and count == 1:
        return collect(partial(fn, task) for task in tasks)
    pool: ThreadPoolExecutor | ProcessPoolExecutor
    if executor == "thread":
        pool = ThreadPoolExecutor(count)
    else:
        pool = ProcessPoolExecutor(count, mp_context=multiprocessing.get_context("spawn"))
    with pool:
        try:
            return collect([pool.submit(fn, task).result for task in tasks])
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def shard_batch(
    local_function: Callable[[InputT], OutputT],
    *,
    mesh: Mesh,
    input_rank: int,
    output_rank: int,
    output_batch_axis: int = 0,
    axis_name: str = "device",
    check_vma: bool = True,
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
        check_vma: Keep JAX's varying-axis check (``check_rep`` on JAX 0.4).
            Pass ``False`` for a local program it wrongly rejects, such as one
            containing ``custom_linear_solve``.

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
    check: dict[str, Any] = {_CHECK_KEYWORD: check_vma}
    return _shard_map(
        local_function,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        **check,
    )


def _sharded_map(local_function: Callable, xs: Any, mesh: Mesh) -> Any:
    """Run ``local_function`` per device over ``xs``, whose axis 0 is the batch.

    The tail repeats the last item so that every shard has one shape, which
    keeps the padded items valid inputs; their results are dropped.
    """
    if len(mesh.axis_names) != 1:
        raise ValueError(f"mesh must have one axis, got {mesh.axis_names}")
    axis_name = mesh.axis_names[0]
    leaves = jax.tree.leaves(xs)
    size = jnp.shape(leaves[0])[0]
    padding = -size % mesh.shape[axis_name]

    def pad(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.concatenate([leaf, jnp.repeat(leaf[-1:], padding, axis=0)], axis=0)

    mapped = shard_batch(
        local_function,
        mesh=mesh,
        input_rank=1,
        output_rank=1,
        axis_name=axis_name,
        check_vma=False,
    )
    return jax.tree.map(lambda leaf: leaf[:size], mapped(jax.tree.map(pad, xs)))


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
        this inner product (the flag is spelled ``check_rep`` on JAX
        0.4): the Krylov basis carry starts replicated and becomes
        shard-varying inside the iteration, which the varying-axis type
        checker rejects even though the computation is correct.

    Returns:
        A callable ``inner(left, right) -> scalar`` with the global value
        on every shard.
    """

    def inner(left, right):
        products = jax.tree.leaves(jax.tree.map(jnp.vdot, left, right))
        local = sum(products[1:], products[0])
        return jax.lax.psum(local, axis_name)

    return inner
