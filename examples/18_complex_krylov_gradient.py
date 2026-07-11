"""Complex matrix-free GMRES with an implicit parameter gradient.

The primal system is non-Hermitian and complex, while the design parameter and
objective are real. ``linear_solve`` differentiates the converged equation
instead of the GMRES iterations. The final central-difference check should agree
with the implicit gradient to roughly 1e-9 on an x64 CPU run.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp

import solvax as sx

jax.config.update("jax_enable_x64", True)

base = jnp.asarray(
    [[3.0 + 0.2j, 0.3 - 0.1j], [-0.2 + 0.4j, 2.0 - 0.3j]],
    dtype=jnp.complex128,
)
right_hand_side = jnp.asarray([1.0 + 0.5j, -0.3 + 0.2j])
parameter_direction = jnp.diag(jnp.asarray([1.0, 0.5]))


def response_energy(parameter):
    matrix = base + parameter * parameter_direction

    def matvec(vector):
        return matrix @ vector

    def solve(operator, rhs):
        return sx.gmres(
            operator, rhs, restart=2, max_restarts=4, rtol=1.0e-12
        ).x

    response = sx.linear_solve(matvec, right_hand_side, solve)
    return jnp.real(jnp.vdot(response, response))


parameter = 0.2
gradient = jax.grad(response_energy)(parameter)
step = 1.0e-5
finite_difference = (
    response_energy(parameter + step) - response_energy(parameter - step)
) / (2.0 * step)

print(f"response energy: {float(response_energy(parameter)):.12e}")
print(f"implicit gradient: {float(gradient):.12e}")
print(f"finite difference: {float(finite_difference):.12e}")
print(f"absolute mismatch: {float(jnp.abs(gradient - finite_difference)):.3e}")
