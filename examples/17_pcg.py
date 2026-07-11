"""Matrix-free pytree PCG and implicit differentiation."""

import jax
import jax.numpy as jnp

import solvax as sx

diagonal = {"velocity": jnp.array([2.0, 5.0]), "potential": jnp.array([10.0])}
rhs = {"velocity": jnp.array([1.0, -2.0]), "potential": jnp.array([3.0])}


def matvec(state):
    return jax.tree.map(lambda scale, value: scale * value, diagonal, state)


def precondition(residual):
    return jax.tree.map(lambda value, scale: value / scale, residual, diagonal)


solution = sx.pcg(matvec, rhs, precond=precondition, rtol=1.0e-10, max_steps=8)
print("status:", sx.status_name(solution.status))
print("iterations:", int(solution.iterations))
print("relative residual:", float(solution.relative_residual_norm))
print("solution:", solution.x)


def squared_solution_norm(scale):
    def operator(value):
        return jax.tree.map(lambda diagonal_value, x: scale * diagonal_value * x, diagonal, value)

    def scaled_precondition(residual):
        return jax.tree.map(
            lambda value, diagonal_value: value / (scale * diagonal_value),
            residual,
            diagonal,
        )

    solved = sx.pcg_linear_solve(
        operator,
        rhs,
        precond=scaled_precondition,
        max_steps=8,
        transpose_rtol=1.0e-10,
    )
    return sum(jnp.vdot(leaf, leaf).real for leaf in jax.tree.leaves(solved.x))


print("implicit gradient:", float(jax.grad(squared_solution_norm)(1.0)))
