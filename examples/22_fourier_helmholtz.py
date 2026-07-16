"""Spectral Fourier-Helmholtz elliptic solve: the drift-plane potential inversion.

Inverts d/dx(g11 dphi/dx) + g33 d^2phi/dz^2 = s * rho on a periodic z axis and
a bounded x axis. The periodic direction is Fourier-transformed (one rfft), and
every mode's remaining tridiagonal system in x is solved in a single batched
`tridiagonal_solve` call -- a direct O(nx nz log nz) inversion with no Krylov
iteration. Build the operator once per geometry and reuse it every timestep.

Expected runtime: well under a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

nx, nz = 64, 128
rng = np.random.default_rng(0)

# x-dependent metric weights (z-independent => separable), uniform spacings.
dx = jnp.full((nx,), 0.1)
dz = jnp.full((nx,), 0.05)
g11 = jnp.asarray(1.0 + 0.3 * rng.random(nx))
g33 = jnp.asarray(0.7 + 0.2 * rng.random(nx))
rhs_scale = jnp.ones((nx,))

operator = sx.build_fourier_helmholtz_operator(
    dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=nz
)
rho = jnp.asarray(rng.standard_normal((nx, nz)))

phi = sx.solve_fourier_helmholtz(rho, operator=operator)
print("solution shape:", phi.shape)

# Verify: apply the assembled per-mode operator to phi-hat and recover rho-hat.
phi_hat = jnp.fft.rfft(phi, axis=-1)
lower, diag, upper = (
    operator.lower_diagonals,
    operator.diagonals,
    operator.upper_diagonals,
)
applied = diag.T * phi_hat
applied = applied.at[1:, :].add(lower.T[1:, :] * phi_hat[:-1, :])
applied = applied.at[:-1, :].add(upper.T[:-1, :] * phi_hat[1:, :])
rho_hat = jnp.fft.rfft(rho * rhs_scale[:, None], axis=-1)
print("operator round trip matches rhs:", bool(jnp.allclose(applied, rho_hat, atol=1e-9)))

# jit + reuse across right-hand sides (the per-timestep pattern).
solve = jax.jit(lambda r: sx.solve_fourier_helmholtz(r, operator=operator))
print("jit solve matches:", bool(jnp.allclose(solve(rho), phi, atol=1e-12)))

# Differentiable through the geometry: gradient of a solution norm w.r.t. g11.
def loss(g):
    op = sx.build_fourier_helmholtz_operator(
        dx=dx, dz=dz, g11=g, g33=g33, rhs_scale=rhs_scale, nz=nz
    )
    return jnp.sum(sx.solve_fourier_helmholtz(rho, operator=op) ** 2)

gradient = jax.grad(loss)(g11)
step = 1e-6
probe = jnp.zeros_like(g11).at[nx // 2].set(step)
finite_difference = (loss(g11 + probe) - loss(g11 - probe)) / (2 * step)
print("grad wrt g11 ~ finite difference:",
      bool(np.isclose(float(gradient[nx // 2]), float(finite_difference), rtol=1e-5)))
