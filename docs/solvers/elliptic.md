# Spectral Fourier--Helmholtz elliptic solve

`solvax.elliptic` inverts a separable elliptic operator of Helmholtz type on a
tensor-product grid with one periodic axis ($z$) and one bounded axis ($x$):

$$
\frac{\partial}{\partial x}\!\left(g_{11}(x)\,
\frac{\partial\phi}{\partial x}\right)
+g_{33}(x)\,\frac{\partial^2\phi}{\partial z^2}
=s(x)\,\rho(x,z),
$$

with metric weights $g_{11}(x)$, $g_{33}(x)$ and a right-hand-side scale
$s(x)$ that depend only on the bounded coordinate. This is the
$\nabla_\perp^2\phi=\rho$ potential inversion used by reduced drift-plane and
vorticity models, where $\phi$ is the electrostatic potential and $\rho$ the
generalized vorticity.

## Method

Because the coefficients are independent of $z$, a real FFT diagonalizes the
periodic direction {cite}`swarztrauber1977`. For each Fourier mode $k_z$,

$$
\frac{\partial}{\partial x}\!\left(g_{11}\,
\frac{\partial\hat\phi_{k_z}}{\partial x}\right)
-k_z^2\,g_{33}\,\hat\phi_{k_z}
=\widehat{s\rho}_{k_z},
$$

a *tridiagonal* system along $x$ after second-order finite differencing. All
$n_z/2+1$ modes are solved simultaneously through one call to
{func}`solvax.tridiagonal.tridiagonal_solve` with the $x$ axis leading and the
mode index batched, so the whole inversion costs one FFT, one batched Thomas
sweep of $O(n_xn_z)$ work, and one inverse FFT — direct, not iterative.

The $x$ boundaries use a reflected closure consistent with the reduced
drift-plane potential solve; the discrete operator applied to the returned
solution reproduces the (scaled, transformed) right-hand side mode by mode,
which is how the implementation is pinned in the test suite.

## Usage

Build the operator once for a fixed geometry, then reuse it across right-hand
sides (e.g. every timestep of a turbulence simulation):

```python
operator = sx.build_fourier_helmholtz_operator(
    dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=nz,
)
phi = sx.solve_fourier_helmholtz(rho, operator=operator)
```

| Input | Meaning |
|---|---|
| `dx`, `g11`, `g33`, `rhs_scale` | length-`nx` arrays along the bounded axis |
| `dz`, `nz` | periodic spacing and length; `zlength = dz[0] * nz` |
| `rhs` | real `(nx, nz)` right-hand side |
| `method` | tridiagonal backend; `"thomas"` (default) is complex-safe everywhere |

The result is the real `(nx, nz)` solution. `method="auto"`/`"lax"` selects
the fused vendor kernel on JAX builds whose complex tridiagonal solve supports
it; the pure-`lax.scan` Thomas default is portable and bit-reproducible.

## Transforms and differentiation

Both routines are pure JAX: the solve composes with `jit`, batches under
`vmap` (e.g. over stacked right-hand sides), and differentiates through the
FFTs and the tridiagonal sweep with `jax.grad` — so the geometry weights
`g11`/`g33` and the right-hand side are all valid differentiation targets.

## Comparison with alternatives

- Compared with FGMRES/PCG on the assembled 2-D operator, the spectral solve
  is direct: no iteration count, no preconditioner tuning, exact separable
  inverse. Use the Krylov path instead when the coefficients depend on $z$
  (non-separable geometry) or the stencil is not Helmholtz-like.
- Compared with `cyclic_tridiagonal_solve` per $z$ line, the Fourier route
  treats the *periodic* direction spectrally and keeps the bounded direction
  tridiagonal, which is the natural split when $g_{11}, g_{33}$ vary in $x$.
- As a preconditioner, the operator built from a frozen or simplified geometry
  is an effective coarse inverse for FGMRES on the full non-separable problem.

## API summary

- {func}`solvax.elliptic.build_fourier_helmholtz_operator`
- {func}`solvax.elliptic.solve_fourier_helmholtz`
- {class}`solvax.elliptic.FourierHelmholtzOperator`

Runnable counterpart: `examples/22_fourier_helmholtz.py`.
