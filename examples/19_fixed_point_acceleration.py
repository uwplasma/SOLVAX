"""Accelerate a partitioned multiphysics fixed-point iteration.

A coupled simulation often exposes only ``state -> one coupling sweep``. This
example uses the complete Aitken solver and the lower-level Anderson primitive.
"""

import jax.numpy as jnp

import solvax as sx


def coupling_sweep(state):
    """Stand in for one fluid/field or transport/equilibrium exchange."""
    return jnp.array([0.75 * state[0] + 0.25, 0.60 * state[1] - 0.20])


x0 = jnp.zeros(2)
aitken = sx.aitken_fixed_point(coupling_sweep, x0, rtol=1e-6, max_steps=30)
print("Aitken solution:", aitken.x)
print("Aitken iterations:", int(aitken.iterations))
print("Aitken residual:", float(aitken.residual_norm))
assert bool(aitken.converged)

# Applications that own their stopping loop can call Anderson on a bounded
# history. First collect three ordinary fixed-point iterates.
history = []
residuals = []
x = x0
for _ in range(3):
    residual = coupling_sweep(x) - x
    history.append(x)
    residuals.append(residual)
    x = x + residual

x_anderson = sx.anderson_mixing(jnp.stack(history), jnp.stack(residuals))
before = jnp.linalg.norm(coupling_sweep(history[-1]) - history[-1])
after = jnp.linalg.norm(coupling_sweep(x_anderson) - x_anderson)
print("Anderson update:", x_anderson)
print("residual before/after Anderson:", float(before), float(after))

# The scalar update is also available when the application owns the loop.
omega = sx.aitken_relaxation(residuals[-2], residuals[-1])
print("next safeguarded Aitken relaxation:", float(omega))
