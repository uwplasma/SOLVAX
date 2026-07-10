"""Implicit differentiation: gradients of a solve / root cost one extra solve.

`linear_solve` and `root_solve` wrap any black-box solver (Krylov, Newton, ...)
and register the implicit-function-theorem VJP, so `jax.grad` flows through the
converged answer without differentiating the solver's iterations — the adjoint
is one transposed / tangent solve, independent of the iteration count.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 50
rng = np.random.default_rng(0)
base = jnp.asarray(rng.standard_normal((n, n)) / np.sqrt(n) + 3.0 * np.eye(n))
b = jnp.ones(n)


# --- linear_solve: differentiate a Krylov solve w.r.t. a matrix parameter. ---
def energy(theta):
    a = base + theta * jnp.eye(n)
    matvec = lambda v: a @ v
    x = sx.linear_solve(matvec, b, solver=lambda mv, rhs: sx.gmres(mv, rhs, rtol=1e-12).x)
    return jnp.sum(x**2)


g = jax.grad(energy)(0.5)
eps = 1e-6
fd = (energy(0.5 + eps) - energy(0.5 - eps)) / (2 * eps)
print(f"linear_solve: grad = {float(g):+.6f}   finite-diff = {float(fd):+.6f}")


# --- root_solve: differentiate the root of f(x, theta) = 0 (here x = atanh). ---
def root(theta):
    f = lambda x: jnp.tanh(x) - theta

    def newton(f, x0):
        step = lambda _, x: x - f(x) / jax.grad(f)(x)
        return jax.lax.fori_loop(0, 30, step, x0)

    return sx.root_solve(f, 0.0, newton)


theta0 = 0.3
g2 = jax.grad(root)(theta0)
analytic = 1.0 / (1.0 - theta0**2)  # d/dtheta atanh(theta)
print(f"root_solve:   grad = {float(g2):+.6f}   analytic    = {analytic:+.6f}")
