"""Matrix-free Newton-Krylov (JFNK) for a nonlinear boundary-value problem.

Solves the 1-D nonlinear reaction-diffusion residual

    F(u)_i = -(u_{i-1} - 2 u_i + u_{i+1}) / h^2 + u_i^3 - f_i = 0

with homogeneous Dirichlet ends. `newton_krylov` never assembles the Jacobian:
each Newton correction obtains Jacobian-vector products from `jax.linearize`
and is solved by restarted FGMRES. The inner solve is preconditioned by the
frozen linearized diffusion operator, itself inverted with SOLVAX's batched
`tridiagonal_solve` -- a physics-based preconditioner built from another
structured solver, which collapses the Krylov work to a handful of steps. The
solver reports separate nonlinear and linear convergence flags.

Expected runtime: well under a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp

import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 128
h = 1.0 / (n + 1)
forcing = jnp.ones(n)
u0 = jnp.zeros(n)

# Tridiagonal linearization of -u'' + u^3 frozen at u0 (off-diagonals -1/h^2).
off = jnp.full(n, -1.0 / h**2)
precond_diag = 2.0 / h**2 + 3.0 * u0**2


def residual(u):
    laplacian = (-2.0 * u).at[:-1].add(u[1:]).at[1:].add(u[:-1]) / h**2
    return -laplacian + u**3 - forcing


def tridiagonal_preconditioner(vector):
    return sx.tridiagonal_solve(off, precond_diag, off, vector, method="thomas")


solution = sx.newton_krylov(
    residual,
    u0,
    precond=tridiagonal_preconditioner,
    rtol=1e-10,
    max_steps=20,
    linear_restart=40,
    linear_rtol=1e-3,
)

print("nonlinear converged:", bool(solution.converged))
print("linear converged:", bool(solution.linear_converged))
print("Newton iterations:", int(solution.newton_iterations))
print("total GMRES iterations:", int(solution.linear_iterations))
print("residual norm:", float(solution.residual_norm))
print("true residual check:", float(jnp.linalg.norm(residual(solution.x))))

# Fully jit-able; static Newton/GMRES limits keep the compiled shape fixed.
jitted = jax.jit(lambda guess: sx.newton_krylov(residual, guess, rtol=1e-10).x)
print("jit solution matches:", bool(jnp.allclose(jitted(u0), solution.x, atol=1e-8)))
