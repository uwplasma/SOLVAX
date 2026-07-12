# Batched tridiagonal solve

This specialized solver handles

$$
\ell_jx_{j-1}+d_jx_j+u_jx_{j+1}=b_j,
\qquad j=0,\ldots,n-1,
$$

with the system dimension on the leading axis. Every trailing axis is an
independent right-hand side or coefficient batch.

```python
x = sx.tridiagonal_solve(lower, diag, upper, rhs, method="auto")
```

If `rhs.shape == (n, n_columns, n_fields)`, one call solves
`n_columns * n_fields` systems. `lower`, `diag`, and `upper` may carry matching
trailing batch axes and are broadcast across extra right-hand-side axes.

## Thomas derivation

Forward elimination defines modified coefficients

$$
c'_0=\frac{u_0}{d_0},\qquad
d'_0=\frac{b_0}{d_0},
$$

$$
q_j=d_j-\ell_jc'_{j-1},\qquad
c'_j=\frac{u_j}{q_j},\qquad
d'_j=\frac{b_j-\ell_jd'_{j-1}}{q_j}.
$$

Back substitution gives

$$
x_{n-1}=d'_{n-1},\qquad x_j=d'_j-c'_jx_{j+1}.
$$

The work is $O(n)$ per system and storage is $O(n)$; no pivoting occurs
{cite}`thomas1949,golub2013`.

## Backend selection

| `method` | Implementation | Intended use |
|---|---|---|
| `"thomas"` | two `jax.lax.scan` sweeps | CPU, deterministic arithmetic, small systems |
| `"lax"` | `jax.lax.linalg.tridiagonal_solve` | fused accelerator/vendor path |
| `"auto"` | platform-dependent selection | default portable behavior |

`auto` selects Thomas when lowering for CPU and the fused operation otherwise.
Systems with fewer than three rows use Thomas because accelerator kernels may
require a larger minimum dimension.

The Thomas arithmetic order is fixed and therefore reproducible on a fixed
backend. The fused and Thomas paths should agree to floating-point accuracy,
not necessarily bit for bit.

## Boundary entries

`lower[0]` and `upper[-1]` do not correspond to matrix entries. Set them to
zero for clarity, although the algorithms ignore their out-of-domain role.
Periodic corner coupling is not represented; use periodic banded LU instead.

## Example: many field-line columns

```python
n, n_modes, n_fields = 256, 64, 3
lower = jnp.full((n, n_modes), -1.0)
diag = jnp.full((n, n_modes), 4.0)
upper = jnp.full((n, n_modes), -1.0)
rhs = jnp.ones((n, n_modes, n_fields))

x = sx.tridiagonal_solve(lower, diag, upper, rhs)
```

## Use as a line preconditioner

For a two-dimensional anisotropic PDE, reshape a residual so the strongly
coupled direction is leading, solve all transverse lines together, and reshape
back. Combine orthogonal line solves with {func}`solvax.precond.line_smoother`.

## Comparison with alternatives

- Compared with general banded LU, Thomas has less storage and lower constants
  but handles only one sub- and superdiagonal.
- Compared with block Thomas, it treats trailing axes as independent systems,
  not dense within-point coupling.
- Compared with parallel cyclic reduction, Thomas is sequential along the
  system axis; the fused vendor path is important on accelerators.
- Compared with dense solve, it preserves $O(n)$ rather than $O(n^3)$ work.

## Stability and diagnostics

Thomas elimination is safe for standard strictly diagonally dominant and many
SPD tridiagonal operators. A zero or tiny modified pivot $q_j$ causes numerical
failure. There is no pivot count, so validate difficult coefficient regimes by
checking the residual or using a pivoted banded/native solver.

The function returns only `x`; it does not return convergence diagnostics
because it is a direct algorithm.

## Differentiation

Both backends are JAX-traceable. Gradients can flow through coefficients and
right-hand sides:

```python
loss = lambda d: jnp.sum(
    sx.tridiagonal_solve(lower, d, upper, rhs, method="thomas") ** 2
)
gradient = jax.grad(loss)(diag)
```

## API summary

- {func}`solvax.tridiagonal.tridiagonal_solve`

Runnable counterpart: `examples/14_tridiagonal_solve.py`.
