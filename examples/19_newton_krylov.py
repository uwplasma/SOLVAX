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

The second half calibrates a forcing amplitude with implicit derivatives.
This is a solver-contract example, not an MHD equilibrium benchmark.
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

if not (bool(solution.converged) and bool(solution.linear_converged)):
    raise RuntimeError("The forward root is not certified")

# Reuse the structured preconditioner in compiled solves too. Omitting it
# makes a stiff diffusion problem needlessly expensive.
def forward(f, guess):
    return sx.newton_krylov(
        f, guess, precond=tridiagonal_preconditioner, rtol=0.0,
        atol=1e-10, max_steps=12, linear_restart=8, linear_rtol=1e-5,
    )


def certified_root(f, guess):
    result = forward(f, guess)
    valid = result.converged & result.linear_converged
    return jnp.where(valid, result.x, jnp.nan)


def tangent_solve(g, rhs):
    # The derivative has the same tridiagonal sparsity. Recover its diagonal
    # with one action, avoiding a dense Jacobian or an iterative adjoint.
    neighbor_count = jnp.full(n, 2.0).at[0].set(1.0).at[-1].set(1.0)
    diagonal = g(jnp.ones(n)) - off * neighbor_count
    return sx.linear_solve(
        g, rhs,
        lambda _, vector: sx.tridiagonal_solve(
            off, diagonal, off, vector, method="thomas"
        ),
    )


def calibrated_state(amplitude):
    return sx.root_solve(
        lambda u: residual(u) + forcing - amplitude,
        u0, certified_root, tangent_solve=tangent_solve,
    )


mean_state = jax.jit(lambda amplitude: jnp.mean(calibrated_state(amplitude)))
mean_and_slope = jax.jit(jax.value_and_grad(mean_state))
value, slope = mean_and_slope(jnp.asarray(1.0))
errors = jnp.asarray([
    jnp.abs(mean_state(1.0 + step) - value - step * slope)
    for step in (0.1, 0.05, 0.025, 0.0125, 0.00625)
])
orders = jnp.log2(errors[:-1] / errors[1:])
print("re-solved Taylor orders:", orders)
assert bool(jnp.all((orders > 1.9) & (orders < 2.1)))

target = mean_state(jnp.asarray(2.0))
amplitude = jnp.asarray(1.0)
initial_error = float(jnp.abs(value - target))
for _ in range(4):
    value, slope = mean_and_slope(amplitude)
    if not bool(jnp.isfinite(value) & jnp.isfinite(slope) & (slope != 0)):
        raise RuntimeError("Rejected nonfinite or singular calibration step")
    amplitude = amplitude - (value - target) / slope
final_error = float(jnp.abs(mean_state(amplitude) - target))
print("calibrated amplitude:", float(amplitude))
print("mean-state error before/after:", initial_error, final_error)
assert final_error < 1e-8 * initial_error
assert abs(float(amplitude) - 2.0) < 1e-8
