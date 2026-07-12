# Tutorial: a reusable block-tridiagonal transport solve

We solve a one-dimensional two-field diffusion-reaction model. At each radial
cell, the unknown is $x_i=(u_i,v_i)^T$. Centered diffusion couples adjacent
cells, while reaction couples the two fields locally. The Jacobian is therefore
block tridiagonal with block size two.

## 1. Model

For interior cells,

$$
-D\frac{x_{i-1}-2x_i+x_{i+1}}{h^2}+R x_i=f_i,
$$

with

$$
D=\begin{bmatrix}1&0\\0&0.2\end{bmatrix},
\qquad
R=\begin{bmatrix}3&-0.4\\-0.4&2\end{bmatrix}.
$$

Homogeneous Dirichlet boundary values are incorporated into the first and last
right-hand-side blocks.

## 2. Assemble block bands

```python
import jax
import jax.numpy as jnp
import solvax as sx

jax.config.update("jax_enable_x64", True)

n_cells = 128
block_size = 2
h = 1.0 / (n_cells + 1)

D = jnp.diag(jnp.array([1.0, 0.2]))
R = jnp.array([[3.0, -0.4], [-0.4, 2.0]])

off = -D / h**2
center = 2.0 * D / h**2 + R

lower = jnp.broadcast_to(off, (n_cells, block_size, block_size))
diag = jnp.broadcast_to(center, (n_cells, block_size, block_size))
upper = jnp.broadcast_to(off, (n_cells, block_size, block_size))
```

`lower[0]` and `upper[-1]` are unused. Keeping full-length arrays makes all
bands share a shape and feeds the factorizer directly.

## 3. Solve two forcing cases together

```python
grid = h * jnp.arange(1, n_cells + 1)
rhs_a = jnp.stack([jnp.sin(jnp.pi * grid), jnp.zeros_like(grid)], axis=1)
rhs_b = jnp.stack([jnp.zeros_like(grid), jnp.sin(2 * jnp.pi * grid)], axis=1)
rhs = jnp.stack([rhs_a, rhs_b], axis=-1)  # (cell, field, forcing)

factors = sx.block_thomas_factor(lower, diag, upper)
solution = sx.block_thomas_solve(factors, rhs)
assert solution.shape == (n_cells, block_size, 2)
```

The factorization is independent of the forcing. Solving both columns together
uses the same block triangular sweeps.

## 4. Verify the residual without dense assembly

```python
operator = sx.BlockTridiagonalOperator(lower, diag, upper)

def apply_each_forcing(x):
    return jax.vmap(operator, in_axes=1, out_axes=1)(x.reshape(-1, 2))

flat_solution = solution.reshape(n_cells * block_size, 2)
flat_rhs = rhs.reshape(n_cells * block_size, 2)
residual = apply_each_forcing(flat_solution) - flat_rhs
print(jnp.linalg.norm(residual, axis=0))
```

The direct routine returns no convergence flag, so an independently applied
operator is the right diagnostic.

## 5. Reuse the transpose solve

Suppose an objective supplies a cotangent $\bar x$. The adjoint equation is

$$
A^T\lambda=\bar x.
$$

```python
cotangent = jnp.ones((n_cells, block_size))
adjoint = sx.block_thomas_solve(factors, cotangent, transpose=True)
```

No second factorization is required.

## 6. Turn the direct solve into a preconditioner

If a weak nonlocal term $E$ is later added, solve $(A+E)x=b$ with FGMRES while
retaining $A^{-1}$:

```python
def a_inverse(r_flat):
    r = r_flat.reshape(n_cells, block_size)
    return sx.block_thomas_solve(factors, r).reshape(-1)

def full_matvec(x_flat):
    return operator(x_flat) + nonlocal_correction(x_flat)

krylov = sx.gmres(
    full_matvec,
    rhs_a.reshape(-1),
    precond=sx.coarse_operator(a_inverse),
    restart=30,
    rtol=1e-10,
)
```

This preserves exact local physics and lets the matrix-free outer iteration
correct the discarded coupling.

## 7. What changes for other cases?

- A scalar field uses `tridiagonal_solve` and can batch many modes on trailing
  axes.
- Coupling to the next two cells creates a block-pentadiagonal system; use a
  wider structured approximation as a preconditioner or reformulate blocks.
- Periodic scalar wrap couplings use periodic banded LU.
- If only low spectral modes are forced and observed, consider the truncated
  block solver.
