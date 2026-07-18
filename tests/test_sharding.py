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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from solvax import gmres, linear_solve, pcg, pcg_linear_solve, tridiagonal_solve

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


def test_sharded_batched_tridiagonal_is_collective_free_forward_and_reverse():
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
