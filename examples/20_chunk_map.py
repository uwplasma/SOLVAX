"""Bound peak memory when mapping an expensive calculation over many cases."""

import jax
import jax.numpy as jnp

import solvax as sx


def expensive_case(parameters):
    grid = jnp.linspace(0.0, 1.0, 128)
    field = jnp.sin(parameters[0] * grid) * jnp.exp(-parameters[1] * grid)
    return jnp.array([jnp.mean(field), jnp.linalg.norm(field)])


parameters = jnp.stack(
    [jnp.linspace(1.0, 4.0, 24), jnp.linspace(0.1, 1.0, 24)], axis=1
)

# Process four cases in parallel and scan over chunks. The result matches vmap
# without replicating intermediates across all 24 cases at once.
chunked = sx.chunk_map(expensive_case, parameters, chunk_size=4)
reference = jax.vmap(expensive_case)(parameters)

print("output shape:", chunked.shape)
print("matches a full vmap:", bool(jnp.allclose(chunked, reference)))
print("automatic width for 24 cases:", sx.auto_chunk_size(len(parameters)))
