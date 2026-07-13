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

For a periodic line, retain the two corner coefficients in `lower[0]` and
`upper[-1]` and use `cyclic_tridiagonal_solve`. It applies an exact
Sherman--Morrison correction through one tridiagonal call with two right-hand
sides, so the same fused accelerator backend remains available. Its derivation
is given in the cyclic-systems section below.

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

For `tridiagonal_solve`, `lower[0]` and `upper[-1]` do not correspond to matrix
entries. Set them to zero for clarity, although the algorithm ignores their
out-of-domain role. `cyclic_tridiagonal_solve` reuses those two slots to carry
the periodic corner coupling (see below).

## Cyclic (periodic) systems

A periodic line couples the two endpoints,

$$
d_0x_0+u_0x_1+\beta x_{n-1}=b_0,\qquad
\alpha x_0+\ell_{n-1}x_{n-2}+d_{n-1}x_{n-1}=b_{n-1},
$$

with the corner entries stored as $\beta=$ `lower[0]` and $\alpha=$
`upper[-1]`. The matrix is tridiagonal apart from those two corners, so it is a
rank-one update of an ordinary tridiagonal matrix
{cite}`press2007,golub2013`. Pick any $\gamma\neq0$ and write

$$
A=T+uv^{H},\qquad
u=\gamma e_0+\alpha e_{n-1},\qquad
v=e_0+\tfrac{\beta}{\gamma}e_{n-1},
$$

where $T$ equals $A$ with the corners removed and the endpoints shifted,
$T_{00}=d_0-\gamma$ and $T_{n-1,n-1}=d_{n-1}-\alpha\beta/\gamma$. SOLVAX takes
$\gamma=-d_0$ (falling back to $\gamma=-1$ when $d_0$ underflows) so the first
pivot stays well scaled. The Sherman--Morrison identity then gives the solution
from two ordinary tridiagonal solves $Ty=b$ and $Tz=u$,

$$
x=y-\frac{v^{H}y}{1+v^{H}z}\,z .
$$

Both right-hand sides are stacked into a single `tridiagonal_solve` call, so a
periodic line costs one tridiagonal solve with two columns and inherits the
same reproducible-Thomas / fused-cuSPARSE `method` selection. The construction
is exact (not iterative) and fully differentiable.

```python
x = sx.cyclic_tridiagonal_solve(lower, diag, upper, rhs, method="auto")
```

Trailing axes of `rhs` are extra right-hand sides solved together, exactly as
for `tridiagonal_solve`. Use this instead of periodic banded LU when the band
is a single sub- and superdiagonal; for wider periodic bands use
{func}`solvax.banded.lu_solve_banded_periodic`.

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
- {func}`solvax.tridiagonal.cyclic_tridiagonal_solve`

Runnable counterparts: `examples/14_tridiagonal_solve.py` and
`examples/21_cyclic_tridiagonal.py`.
