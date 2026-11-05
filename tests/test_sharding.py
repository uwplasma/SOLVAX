"""Sharding preservation and communication accounting.

Runs on eight emulated CPU devices (see ``conftest.py``). Three claims are
pinned here rather than asserted in prose:

1. **Correctness under sharding** — sharded solves match their single-device
   references.
2. **Sharding preservation** — solver outputs keep the input leaves' named
   sharding; nothing is gathered to one device.
3. **Communication invariance of adjoints** — counting collective operations in
   the compiled HLO, the reverse-mode solve stays in the primal's communication
   class: embarrassingly parallel solves stay collective-free under ``grad``,
   and implicit-adjoint Krylov solves cost at most one extra solve's worth of
   collectives.
"""

import math
import multiprocessing
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import solvax.parallel
from solvax import (
    available_cpus,
    chunk_map,
    gmres,
    linear_solve,
    map_independent,
    pcg,
    pcg_linear_solve,
    shard_batch,
    tridiagonal_solve,
    tridiagonal_solve_checked,
)

jax.config.update("jax_enable_x64", True)

_COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all")

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 8, reason="needs 8 (emulated) devices; see conftest.py"
)


def count_collectives(fn, *args):
    """Count collective ops in the optimized HLO of ``jit(fn)(*args)``.

    Async pairs (``-start``/``-done``) are counted once via their start op.
    """
    text = jax.jit(fn).lower(*args).compile().as_text()
    total = 0
    for line in text.splitlines():
        for op in _COLLECTIVES:
            if f"{op}-start(" in line or (f"{op}(" in line and f"{op}-done(" not in line):
                total += 1
                break
    return total


def _mesh():
    return Mesh(np.array(jax.devices()[:8]), axis_names=("i",))


def _shard(mesh, value, spec):
    return jax.device_put(value, NamedSharding(mesh, spec))


def test_shard_batch_runs_only_each_devices_local_items():
    mesh = _mesh()
    values = _shard(mesh, jnp.arange(48.0).reshape(16, 3), P("i", None))

    def local_transpose(local_values):
        local_batch_size = local_values.shape[0]
        return local_values.T + 100.0 * local_batch_size

    mapped = shard_batch(
        local_transpose,
        mesh=mesh,
        input_rank=2,
        output_rank=2,
        output_batch_axis=1,
        axis_name="i",
    )
    result = jax.jit(mapped)(values)

    assert result.sharding.spec == P(None, "i")
    assert np.allclose(np.asarray(result), np.asarray(values).T + 200.0)
    assert count_collectives(mapped, values) == 0


def test_shard_batch_validates_layout():
    mesh = _mesh()

    def identity(value):
        return value

    with pytest.raises(ValueError, match="input_rank"):
        shard_batch(identity, mesh=mesh, input_rank=0, output_rank=1, axis_name="i")
    with pytest.raises(ValueError, match="output_rank"):
        shard_batch(identity, mesh=mesh, input_rank=1, output_rank=0, axis_name="i")
    with pytest.raises(ValueError, match="output_batch_axis"):
        shard_batch(
            identity,
            mesh=mesh,
            input_rank=1,
            output_rank=1,
            output_batch_axis=1,
            axis_name="i",
        )
    with pytest.raises(ValueError, match="no axis"):
        shard_batch(identity, mesh=mesh, input_rank=1, output_rank=1, axis_name="missing")


def test_shard_batch_check_vma_admits_custom_linear_solve():
    mesh = _mesh()
    values = jnp.arange(1.0, 17.0)

    def solve_one(value):
        return jax.lax.custom_linear_solve(lambda x: value * x, value**2, lambda _, b: b / value)

    def local(batch):
        return jax.lax.map(solve_one, batch)

    mapped = shard_batch(
        local, mesh=mesh, input_rank=1, output_rank=1, axis_name="i", check_vma=False
    )
    assert np.allclose(mapped(values), values)


@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize("size", [3, 8, 13])
def test_chunk_map_on_a_mesh_pads_shards_and_trims(chunk_size, size):
    mesh = _mesh()
    xs = {"a": jnp.arange(float(size)), "b": jnp.arange(2.0 * size).reshape(size, 2)}

    def fun(item):
        return {"sum": item["a"] + item["b"].sum(), "pair": item["b"] * item["a"]}

    got = jax.jit(lambda v: chunk_map(fun, v, chunk_size=chunk_size, mesh=mesh))(xs)
    expected = jax.vmap(fun)(xs)
    for key in expected:
        assert got[key].shape == expected[key].shape
        assert np.allclose(got[key], expected[key])


def test_chunk_map_on_a_mesh_differentiates_and_handles_edges():
    mesh = _mesh()
    xs = jnp.arange(1.0, 12.0)
    grad = jax.grad(lambda v: jnp.sum(chunk_map(lambda x: x**2, v, chunk_size=2, mesh=mesh)))(xs)
    assert np.allclose(grad, 2 * xs)
    assert chunk_map(lambda x: x, jnp.zeros((0, 2)), mesh=mesh).shape == (0, 2)
    with pytest.raises(ValueError, match="native backend"):
        chunk_map(lambda x: x, xs, backend="adv", mesh=mesh)
    with pytest.raises(TypeError, match="unsupported keyword"):
        chunk_map(lambda x: x, xs, mesh=mesh, in_axes=0)
    square = Mesh(np.array(jax.devices()[:8]).reshape(2, 4), axis_names=("i", "j"))
    with pytest.raises(ValueError, match="one axis"):
        chunk_map(lambda x: x, xs, mesh=square)


@pytest.mark.parametrize(
    ("variables", "cgroup", "expected"),
    [
        ({"SLURM_CPUS_PER_TASK": "3"}, {}, 3),
        ({"PBS_NP": "4", "LSB_DJOB_NUMPROC": "2"}, {}, 2),
        ({"NSLOTS": "bad", "PBS_NP": "0"}, {}, 16),
        ({}, {"cpu.max": "250000 100000\n"}, 3),
        ({}, {"cpu.max": "max 100000\n"}, 16),
        ({}, {"cpu.max": "broken\n"}, 16),
        ({}, {"cpu/cpu.cfs_quota_us": "500000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 5),
        ({}, {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, 16),
        ({"SLURM_CPUS_PER_TASK": "9"}, {"cpu.max": "400000 100000\n"}, 4),
    ],
)
def test_available_cpus_takes_smallest_limit(monkeypatch, tmp_path, variables, cgroup, expected):
    for name in solvax.parallel._SCHEDULER_CPU_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    for name, text in cgroup.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(solvax.parallel, "_CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(32)), raising=False)
    assert available_cpus() == expected
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    assert available_cpus() == min(expected, 2)
    monkeypatch.delattr(os, "sched_getaffinity")
    assert available_cpus() == expected


def _child_process(_):
    return multiprocessing.get_start_method(allow_none=True), os.getpid()


@pytest.mark.parametrize("workers", [None, 1, 3, 100])
def test_map_independent_threads_match_a_serial_loop(workers):
    assert map_independent(lambda x: x**2, range(7), workers=workers) == [x**2 for x in range(7)]


@pytest.mark.parametrize("workers", [1, 2])
def test_map_independent_reports_failures_in_input_order(workers):
    def fail_on_odd(x):
        if x % 2:
            raise ValueError(f"item {x}")
        return x

    results = map_independent(fail_on_odd, range(4), workers=workers, return_exceptions=True)
    assert [results[0], results[2]] == [0, 2]
    assert [str(results[1]), str(results[3])] == ["item 1", "item 3"]
    with pytest.raises(ValueError, match="item 1"):
        map_independent(fail_on_odd, range(4), workers=workers)


def test_map_independent_edges_and_spawned_processes():
    assert map_independent(abs, [], executor="process") == []
    with pytest.raises(ValueError, match="executor"):
        map_independent(abs, [1], executor="fork")
    assert map_independent(math.sqrt, [4.0, 9.0], workers=2, executor="process") == [2.0, 3.0]
    results = map_independent(
        math.sqrt, [4.0, -1.0], workers=2, executor="process", return_exceptions=True
    )
    assert results[0] == 2.0 and isinstance(results[1], ValueError)
    # One worker still gets a child: callers rely on it to set its own device environment.
    [(method, pid)] = map_independent(_child_process, [0], workers=1, executor="process")
    assert method == "spawn" and pid != os.getpid()


@pytest.mark.parametrize("checked", [False, True])
def test_sharded_batched_tridiagonal_is_collective_free_forward_and_reverse(checked):
    # Columns are independent systems: sharding the batch axis must produce a
    # solve with zero collectives -- and the same must hold for its gradient.
    n, cols = 32, 64
    rng = np.random.default_rng(0)
    mesh = _mesh()
    lower = _shard(mesh, jnp.asarray(rng.standard_normal((n, cols))), P(None, "i"))
    upper = _shard(mesh, jnp.asarray(rng.standard_normal((n, cols))), P(None, "i"))
    diag = _shard(mesh, jnp.asarray(6.0 + rng.random((n, cols))), P(None, "i"))
    rhs = _shard(mesh, jnp.asarray(rng.standard_normal((n, cols))), P(None, "i"))

    def solve(d):
        if checked:
            return tridiagonal_solve_checked(
                lower, d, upper, rhs, method="thomas"
            ).solution
        return tridiagonal_solve(lower, d, upper, rhs, method="thomas")

    solution = jax.jit(solve)(diag)
    reference = tridiagonal_solve(
        jax.device_get(lower), jax.device_get(diag), jax.device_get(upper),
        jax.device_get(rhs), method="thomas",
    )
    assert np.allclose(np.asarray(solution), np.asarray(reference), atol=1e-11)
    assert solution.sharding.spec == P(None, "i")  # batch stays sharded

    def loss(d):
        return jnp.sum(solve(d) ** 2)

    assert count_collectives(solve, diag) == 0
    grad_collectives = count_collectives(jax.grad(loss), diag)
    assert grad_collectives == 0  # the adjoint solve is columnwise too


def test_sharded_pytree_gmres_matches_reference_and_preserves_sharding():
    n1, n2 = 128, 64
    rng = np.random.default_rng(1)
    mesh = _mesh()
    d1 = jnp.asarray(2.0 + rng.random(n1))
    d2 = jnp.asarray(3.0 + rng.random(n2))
    b = {
        "distribution": _shard(mesh, jnp.asarray(rng.standard_normal(n1)), P("i")),
        "field": _shard(mesh, jnp.asarray(rng.standard_normal(n2)), P("i")),
    }

    def matvec(value):
        return {
            "distribution": d1 * value["distribution"],
            "field": d2 * value["field"],
        }

    solution = jax.jit(
        lambda rhs: gmres(matvec, rhs, restart=8, rtol=1e-12, max_restarts=3)
    )(b)
    assert bool(solution.converged)
    assert np.allclose(
        np.asarray(solution.x["distribution"]), np.asarray(b["distribution"] / d1), atol=1e-10
    )
    assert np.allclose(np.asarray(solution.x["field"]), np.asarray(b["field"] / d2), atol=1e-10)
    # Leaf-wise Arnoldi never concatenates leaves: each keeps its sharding.
    assert solution.x["distribution"].sharding.spec == P("i")
    assert solution.x["field"].sharding.spec == P("i")


def test_single_reduction_pcg_reduces_collectives():
    n = 256
    rng = np.random.default_rng(2)
    mesh = _mesh()
    d = jnp.asarray(2.0 + rng.random(n))
    b = _shard(mesh, jnp.asarray(rng.standard_normal(n)), P("i"))

    def solve(rhs, single_reduction):
        return pcg(
            lambda v: d * v, rhs, rtol=1e-10, max_steps=64,
            single_reduction=single_reduction,
        ).x

    x_std = jax.jit(lambda rhs: solve(rhs, False))(b)
    x_one = jax.jit(lambda rhs: solve(rhs, True))(b)
    assert np.allclose(np.asarray(x_std), np.asarray(b / d), atol=1e-9)
    assert np.allclose(np.asarray(x_one), np.asarray(b / d), atol=1e-9)

    standard = count_collectives(lambda rhs: solve(rhs, False), b)
    fused = count_collectives(lambda rhs: solve(rhs, True), b)
    # The single-reduction recurrence is algebraically one fused reduction per
    # iteration; whether the compiled module realizes that depends on the XLA
    # partitioner. Current JAX fuses (measured 2 vs 3); the 0.4-era GSPMD
    # lowering does not, so there the assertion is only that the rewrite does
    # not blow up the communication.
    jax_version = tuple(int(part) for part in jax.__version__.split(".")[:2])
    if jax_version >= (0, 5):
        assert fused < standard
    else:
        assert fused <= standard + 1


@pytest.mark.parametrize("solver", ["pcg", "gmres"])
def test_adjoint_stays_in_the_primal_communication_class(solver):
    # The implicit adjoint is one extra (transposed) solve: its compiled
    # collective count must stay within a small multiple of the primal's.
    n = 256
    rng = np.random.default_rng(3)
    mesh = _mesh()
    d = jnp.asarray(2.0 + rng.random(n))
    b = _shard(mesh, jnp.asarray(rng.standard_normal(n)), P("i"))

    if solver == "pcg":
        def primal(rhs):
            return pcg_linear_solve(lambda v: d * v, rhs, rtol=1e-10, max_steps=64).x
    else:
        def primal(rhs):
            return linear_solve(
                lambda v: d * v, rhs,
                solver=lambda mv, rhs_: gmres(mv, rhs_, restart=16, rtol=1e-10).x,
            )

    def loss(rhs):
        # Nonlinear on purpose: a linear loss has a constant cotangent, which
        # XLA folds so the adjoint solve vanishes from the compiled module.
        return jnp.sum(primal(rhs) ** 2)

    forward = count_collectives(primal, b)
    reverse = count_collectives(jax.grad(loss), b)
    assert forward > 0  # global inner products do communicate
    assert reverse > 0  # the adjoint really is compiled, not folded away
    assert reverse <= 3 * forward  # primal + one transposed solve, same class
