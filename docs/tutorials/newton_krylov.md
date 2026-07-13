# Tutorial: matrix-free Newton-Krylov for a nonlinear PDE

This tutorial solves a nonlinear reaction-diffusion system without assembling
or storing a Jacobian. It extends the {doc}`matrix_free_pde` line-preconditioner
tutorial from a linear operator to a nonlinear residual, using
{func}`solvax.implicit.newton_krylov` (JFNK).

$$
-\epsilon_x u_{xx}-\epsilon_yu_{yy}+u^3=f,
\qquad \epsilon_x\gg\epsilon_y,
$$

with homogeneous Dirichlet boundaries. The cubic reaction makes the operator
nonlinear, so each Newton step needs a fresh Jacobian action. `newton_krylov`
obtains that action from `jax.linearize` and never materializes a matrix.

## 1. The nonlinear residual

```python
import jax
import jax.numpy as jnp
import solvax as sx

jax.config.update("jax_enable_x64", True)

nx, ny = 128, 32
eps_x, eps_y = 1.0, 1e-2
hx, hy = 1.0 / (nx + 1), 1.0 / (ny + 1)
wx, wy = eps_x / hx**2, eps_y / hy**2
forcing = jnp.ones((nx, ny))

def residual(u):
    padded = jnp.pad(u, ((1, 1), (1, 1)))
    diffusion = (
        wx * (2.0 * u - padded[:-2, 1:-1] - padded[2:, 1:-1])
        + wy * (2.0 * u - padded[1:-1, :-2] - padded[1:-1, 2:])
    )
    return diffusion + u**3 - forcing
```

The state is a `(nx, ny)` array; `residual` maps a state to a matching residual.
It could equally be a nested pytree of species and field blocks — the solver
only requires that the input and output share tree structure.

## 2. A physics-based Jacobian preconditioner

The Jacobian's stiff part is the anisotropic diffusion along $x$. We reuse the
exact tridiagonal line solve from the linear tutorial as a right preconditioner
for the inner FGMRES, frozen at the initial iterate. Its diagonal includes the
linearized reaction $3u_0^2$.

```python
u0 = jnp.zeros((nx, ny))
lower_x = jnp.full((nx, ny), -wx)
upper_x = jnp.full((nx, ny), -wx)
diag_x = 2.0 * wx + 2.0 * wy + 3.0 * u0**2

def line_preconditioner(residual_block):
    return sx.tridiagonal_solve(lower_x, diag_x, upper_x, residual_block, method="auto")
```

Jacobian-free products remove matrix storage, but they do not by themselves make
Krylov convergence mesh-independent {cite}`knoll2004`; the structured
preconditioner is what keeps the inner iteration counts small.

## 3. Solve and inspect both convergence flags

```python
solution = sx.newton_krylov(
    residual,
    u0,
    precond=line_preconditioner,
    rtol=1e-10,
    max_steps=20,
    linear_restart=40,
    linear_rtol=1e-3,
)

if not (solution.converged and solution.linear_converged):
    raise RuntimeError(
        f"JFNK stopped: nonlinear={bool(solution.converged)}, "
        f"linear={bool(solution.linear_converged)}, "
        f"residual={float(solution.residual_norm):.3e}"
    )

print("Newton iterations:", int(solution.newton_iterations))
print("total GMRES iterations:", int(solution.linear_iterations))
```

The nonlinear test uses the true residual relative to the initial residual,
recomputed after the final accepted update, so a small stale estimate cannot
report false convergence. Always inspect *both* `converged` (nonlinear) and
`linear_converged`: an inner GMRES failure stops the Newton iteration, and only
the two flags together certify the result.

## 4. Inexact Newton and forcing terms

`linear_rtol` sets how tightly each Newton correction is solved. A loose inner
tolerance early (inexact Newton) avoids over-solving corrections far from the
root; tightening it near convergence preserves the asymptotic Newton rate
{cite}`knoll2004`. Because `max_steps`, `linear_restart`, and
`linear_max_restarts` are static, the whole solve compiles to a fixed shape and
is safe under `jax.jit` and `jax.vmap`.

## 5. When the map is affine

If the residual is affine in the state -- a linearized or frozen-physics
coupling $G(x)=Lx+c$ -- Newton converges in one step, and
{func}`solvax.fixed_point.affine_fixed_point_gmres` expresses that directly:
it solves $(I-L)\,\delta=G(x_0)-x_0$ with the same matrix-free FGMRES,
preconditioner, and inner-product contracts, and skips the outer Newton loop.

## 6. Extensions

- Supply `inner_product=` for weighted or mesh-wide (distributed) Krylov
  reductions, and `norm=` to define the nonlinear residual norm independently.
- Recompute the preconditioner diagonal at each Newton iterate for a stiffer
  reaction, or wrap `p_multigrid` as the preconditioner for a mesh hierarchy.
- Replace the stencil with spectral or finite-volume physics; the JFNK contract
  is unchanged as long as `residual` preserves the state structure.

Runnable counterpart: `examples/19_newton_krylov.py`.
