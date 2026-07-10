"""Memory-chunked Jacobian: the jac_chunk_size knob for large residuals.

`jax.jacfwd` / `jax.jacrev` evaluate every directional derivative in one vmap,
replicating the intermediate program state across the full Jacobian width — an
easy way to run out of accelerator memory. The chunked builders split the basis
into blocks of `chunk_size` and walk them with `jax.lax.map`, so peak memory
scales with the chunk instead of the whole Jacobian, at a modest time cost. The
result is numerically identical to the JAX builders. This is the analogue of
DESC's `jac_chunk_size` optimization-memory option.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_params, n_res = 200, 40  # a wide Jacobian: many parameters, fewer residuals
rng = np.random.default_rng(0)
W = jnp.asarray(rng.standard_normal((n_res, n_params)))


def residual(theta):  # stand-in optimization residual: R^n_params -> R^n_res
    return jnp.tanh(W @ theta) - 0.1


theta = jnp.asarray(rng.standard_normal(n_params))
J_ref = jax.jacrev(residual)(theta)

# Wide Jacobian -> reverse mode; chunk the rows to bound memory.
for chunk in (None, 8, "auto"):
    J = sx.chunked_jacrev(residual, chunk_size=chunk)(theta)
    match = bool(jnp.allclose(J, J_ref, atol=1e-10))
    print(f"chunk={str(chunk):>4}: shape={J.shape}  matches jax.jacrev={match}")

print("auto chunk width for", n_res, "rows:", sx.auto_chunk_size(n_res))

# chunked_jacobian(mode="auto") picks forward/reverse by shape; all agree.
J_auto = sx.chunked_jacobian(residual, mode="auto", chunk_size="auto")(theta)
print("mode='auto' matches jax:", bool(jnp.allclose(J_auto, J_ref, atol=1e-10)))
