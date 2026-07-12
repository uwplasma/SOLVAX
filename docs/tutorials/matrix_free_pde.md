# Tutorial: matrix-free anisotropic PDE with a line preconditioner

This tutorial solves a two-dimensional anisotropic diffusion-reaction equation
without passing an assembled matrix to FGMRES:

$$
-\epsilon_x u_{xx}-\epsilon_yu_{yy}+cu=f,
\qquad \epsilon_x\gg\epsilon_y.
$$

Point Jacobi treats the diagonal but not the strong $x$ coupling. We construct
an exact tridiagonal solve along every $x$ line and use it as a preconditioner.

## 1. Matrix-free stencil

```python
import jax
import jax.numpy as jnp
import solvax as sx

jax.config.update("jax_enable_x64", True)

nx, ny = 128, 32
eps_x, eps_y, reaction = 1.0, 1e-2, 0.5
hx, hy = 1.0 / (nx + 1), 1.0 / (ny + 1)
wx, wy = eps_x / hx**2, eps_y / hy**2
diagonal = 2.0 * wx + 2.0 * wy + reaction

def matvec(flat):
    u = flat.reshape(nx, ny)
    padded = jnp.pad(u, ((1, 1), (1, 1)))
    center = diagonal * u
    x_part = -wx * (padded[:-2, 1:-1] + padded[2:, 1:-1])
    y_part = -wy * (padded[1:-1, :-2] + padded[1:-1, 2:])
    return (center + x_part + y_part).reshape(-1)
```

The padding encodes homogeneous Dirichlet boundaries. The function accepts and
returns `(nx * ny,)`, matching FGMRES.

## 2. Exact solves along the stiff direction

```python
lower_x = jnp.full((nx, ny), -wx)
diag_x = jnp.full((nx, ny), diagonal)
upper_x = jnp.full((nx, ny), -wx)

def x_line_inverse(residual_flat):
    residual = residual_flat.reshape(nx, ny)
    correction = sx.tridiagonal_solve(
        lower_x, diag_x, upper_x, residual, method="auto"
    )
    return correction.reshape(-1)
```

The leading `nx` axis is the tridiagonal system; all `ny` columns are solved in
one call.

## 3. Build the smoother/preconditioner

```python
preconditioner = sx.line_smoother(
    matvec,
    [x_line_inverse],
    omega=0.9,
    sweeps=2,
)
```

Each sweep computes a true outer residual and applies the line inverse. Because
the preconditioner is an iterative action, flexible GMRES is appropriate.

## 4. Solve and inspect diagnostics

```python
rhs = jnp.ones(nx * ny)
solution = sx.gmres(
    matvec,
    rhs,
    precond=preconditioner,
    restart=40,
    rtol=1e-9,
    atol=1e-12,
    max_restarts=30,
)

if not solution.converged:
    raise RuntimeError(
        f"FGMRES stopped after {int(solution.iterations)} steps; "
        f"residual={float(solution.residual_norm):.3e}"
    )
```

## 5. Compare against point Jacobi

```python
point = sx.jacobi(jnp.full(nx * ny, diagonal))
point_solution = sx.gmres(
    matvec, rhs, precond=point, restart=40, rtol=1e-9, max_restarts=30
)

print("point iterations:", int(point_solution.iterations))
print("line iterations:", int(solution.iterations))
```

The meaningful comparison is total time including the preconditioner, not only
iteration count. On a GPU, the fused tridiagonal backend can make the line
solve especially attractive.

## 6. Extensions

- Add a $y$-line inverse and alternate directions when anisotropy changes
  spatially or both directions are stiff.
- Place line smoothing inside `p_multigrid` to eliminate high-frequency error
  while a coarse grid corrects smooth error.
- Use PCG only if the complete preconditioner action is symmetric positive
  definite. The sequential relaxed line smoother above is safest with FGMRES.
- Replace the stencil with spectral or finite-volume code; the outer API is
  unchanged as long as `matvec` preserves the flat shape.
