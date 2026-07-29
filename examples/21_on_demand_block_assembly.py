"""Solve low blocks of a long hierarchy without materializing every block."""

import jax
import jax.numpy as jnp

import solvax as sx

n_blocks, block_size, keep = 32, 3, 2
eye = jnp.eye(block_size)


def block_at(k):
    """Assemble one row from compact mode-dependent physics coefficients."""
    collision = 3.0 + 0.02 * k * (k + 1)
    return -0.2 * eye, collision * eye, -0.2 * eye


rhs_low = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
x_low = sx.block_thomas_truncated_fn(block_at, n_blocks, rhs_low, keep)
print("requested solution shape:", x_low.shape)

# Verification materializes the same blocks. Production code can omit these
# arrays; the on-demand solve keeps memory independent of n_blocks.
lower, diagonal, upper = jax.vmap(block_at)(jnp.arange(n_blocks))
rhs = jnp.zeros((n_blocks, block_size)).at[:keep].set(rhs_low)
x_reference = sx.block_thomas(lower, diagonal, upper, rhs)[:keep]
print("matches materialized solve:", bool(jnp.allclose(x_low, x_reference)))
