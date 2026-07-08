"""Solve a kinetic-style block-tridiagonal system with truncated storage.

Spectral discretizations of kinetic equations (e.g. a Legendre expansion in
pitch angle) couple only neighbouring modes l-1, l, l+1, giving a
block-tridiagonal system whose right-hand side lives in the lowest few modes
and whose observables (density, flow, pressure moments) touch only those same
modes. `block_thomas_truncated` exploits both facts: memory O(K m^2)
independent of the number of modes.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp

import solvax as sx

jax.config.update("jax_enable_x64", True)

n_modes, m = 64, 100  # Legendre-like modes x flux-surface grid points
key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)

# Streaming-like off-diagonal coupling and a collisional diagonal ~ l(l+1).
lower = 0.3 * jax.random.normal(k1, (n_modes, m, m))
upper = 0.3 * jax.random.normal(k2, (n_modes, m, m))
nu = 0.5 * jnp.arange(n_modes) * (jnp.arange(n_modes) + 1) + 5.0
diag = jax.random.normal(k3, (n_modes, m, m)) + nu[:, None, None] * jnp.eye(m)

# Two drives (radial + parallel), nonzero only in modes 0..2 — solved together.
rhs_low = jax.random.normal(k4, (3, m, 2))

x_low = sx.block_thomas_truncated(lower, diag, upper, rhs_low, keep_lowest=3)
print("lowest-mode solution block shape:", x_low.shape)

# Cross-check against the full solve.
rhs_full = jnp.zeros((n_modes, m, 2)).at[:3].set(rhs_low)
x_full = sx.block_thomas(lower, diag, upper, rhs_full)
err = jnp.max(jnp.abs(x_low - x_full[:3]))
print(f"max |truncated - full| = {err:.2e}")

# The whole solve is differentiable: gradient of a "flux" moment w.r.t. nu.
def flux(nu_vec):
    d = jax.random.normal(k3, (n_modes, m, m)) + nu_vec[:, None, None] * jnp.eye(m)
    x = sx.block_thomas_truncated(lower, d, upper, rhs_low, keep_lowest=3)
    return jnp.sum(x[1] ** 2)  # mode-1 moment ~ parallel flow

g = jax.grad(flux)(nu)
print("d(flux)/d(nu_0) =", float(g[0]))
