"""Alternating-direction line smoother for an anisotropic operator.

Point smoothers stall when the operator couples one grid direction far more
strongly than the other. A *line* smoother solves the strongly coupled
direction exactly (a tridiagonal solve along each line) and alternates
directions to cover mixed anisotropy — the classic multigrid remedy. Here the
line solves are built from `solvax.tridiagonal.tridiagonal_solve`.

Expected runtime: a few seconds on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

nx, ny = 32, 16  # grid; strong coupling along x
eps_x, eps_y, c = 1.0, 0.02, 0.5  # anisotropic diffusion + reaction
hx, hy = 1.0 / (nx + 1), 1.0 / (ny + 1)
n = nx * ny

wx, wy = eps_x / hx**2, eps_y / hy**2
d0 = 2.0 * wx + 2.0 * wy + c  # full diagonal of the 5-point stencil

# Dense operator (row-major index i*ny + j), Dirichlet boundaries.
A = np.zeros((n, n))
for i in range(nx):
    for j in range(ny):
        p = i * ny + j
        A[p, p] = d0
        if i > 0:
            A[p, p - ny] = -wx
        if i < nx - 1:
            A[p, p + ny] = -wx
        if j > 0:
            A[p, p - 1] = -wy
        if j < ny - 1:
            A[p, p + 1] = -wy
A = jnp.asarray(A)
b = jnp.ones(n)


def matvec(v):
    return A @ v


def x_line_solve(r):
    """Invert the x-direction tridiagonal block on every y-line at once."""
    grid = r.reshape(nx, ny)
    lower = jnp.full((nx, ny), -wx)
    upper = jnp.full((nx, ny), -wx)
    diag = jnp.full((nx, ny), d0)
    return sx.tridiagonal_solve(lower, diag, upper, grid).reshape(-1)


def y_line_solve(r):
    """Invert the y-direction tridiagonal block on every x-line at once."""
    grid = r.reshape(nx, ny).T  # system axis (y) first
    lower = jnp.full((ny, nx), -wy)
    upper = jnp.full((ny, nx), -wy)
    diag = jnp.full((ny, nx), d0)
    return sx.tridiagonal_solve(lower, diag, upper, grid).T.reshape(-1)


smoother = sx.line_smoother(matvec, [x_line_solve, y_line_solve], omega=0.8, sweeps=2)

opts = dict(restart=40, rtol=1e-10, max_restarts=40)
plain = sx.gmres(matvec, b, **opts)
smoothed = sx.gmres(matvec, b, precond=smoother, **opts)
print(f"no preconditioner    : iters={int(plain.iterations):3d} converged={bool(plain.converged)}")
print(f"line smoother (x, y) : iters={int(smoothed.iterations):3d} "
      f"converged={bool(smoothed.converged)}")
