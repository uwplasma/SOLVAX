"""Sharded-operand contract: Krylov solves on device-mesh pytrees.

Runs in a subprocess so the host device count is set before JAX
initializes. Asserts the spectral-MHD interop properties: complex
two-leaf pytree operands solve correctly under a NamedSharding, the
result matches the single-device solve to a stated tolerance (bitwise
equality across device counts is not achievable because reduction order
changes), and the compiled solve contains no operand-sized all-gather.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec

import solvax

assert jax.device_count() == 4, jax.devices()
mesh = Mesh(jax.devices(), axis_names=("m",))

modes = 512
key = jax.random.PRNGKey(1)
key_a, key_b, key_t, key_r1, key_r2 = jax.random.split(key, 5)
alpha_v = 1.0 + jax.random.uniform(key_a, (modes,))
alpha_b = 1.0 + jax.random.uniform(key_b, (modes,))
theta = 5.0 * jax.random.uniform(key_t, (modes,))


def matvec(x):
    v, b = x
    return (
        alpha_v * v - 1j * theta * b,
        -1j * theta * v + alpha_b * b,
    )


rhs = (
    jax.random.normal(key_r1, (modes,)) + 1j * jax.random.normal(key_r2, (modes,)),
    jax.random.normal(key_r2, (modes,)) + 1j * jax.random.normal(key_r1, (modes,)),
)
precond = solvax.alfven_block(alpha_v, alpha_b, theta)

# Single-device reference.
reference = solvax.gmres(matvec, rhs, precond=precond, rtol=1.0e-12)
assert bool(reference.converged)

# Sharded operands: leaves distributed along the mode axis.
spec = NamedSharding(mesh, PartitionSpec("m"))
rhs_sharded = jax.tree.map(lambda leaf: jax.device_put(leaf, spec), rhs)

solve = jax.jit(
    lambda b: solvax.gmres(matvec, b, precond=precond, rtol=1.0e-12)
)
sharded = solve(rhs_sharded)
assert bool(sharded.converged)

gap = max(
    float(jnp.max(jnp.abs(sharded.x[0] - reference.x[0]))),
    float(jnp.max(jnp.abs(sharded.x[1] - reference.x[1]))),
)
scale = float(jnp.max(jnp.abs(reference.x[0])))
assert gap < 1.0e-12 * max(scale, 1.0), gap

# The compiled solve must not gather operand-sized arrays: reductions
# produce scalars, and every leaf operation is elementwise.
hlo = solve.lower(rhs_sharded).compile().as_text()
for line in hlo.splitlines():
    if "all-gather" in line and f"{modes}" in line:
        raise AssertionError(f"operand-sized all-gather: {line}")

print("SHARDED-OK")
"""


def test_sharded_krylov_contract() -> None:
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORM_NAME"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "SHARDED-OK" in result.stdout
