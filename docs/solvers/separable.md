# Separable inverses and the flux-form operator

On a rectilinear cell-centred grid with cell widths $w_a$ per axis, the finite-volume
Laplacian is a Kronecker sum of one-dimensional operators $M_a=W_a^{-1}K_a$, each
homogeneous Dirichlet (the face lies half a cell from the centre), Neumann or periodic.
`fast_diagonal` diagonalizes every $K_a v=\lambda W_a v$ once on the host
{cite}`lynch1964`, so $(s-c\nabla^2)x=r$ is three contractions and a divide:

```python
fd = sx.fast_diagonal(widths, ("dirichlet", "neumann", "periodic"), shift=0.5, coefficient=2.0)
x = fd.solve(r)                      # (shift - coefficient * laplacian) x = r, per unit volume
xt = fd.solve(r, transpose=True)     # the plain transpose, which differs: M = W^{-1} K
```

`shift` and `coefficient` are differentiable leaves; the widths are geometry. With
`shift == 0` and no Dirichlet axis the constants are the null space: `solve` is the
pseudo-inverse, drops the volume-mean part of `r` and returns zero volume mean.

## Variable coefficients

`flux_form_operator` is $A x=V\,(s\,x-c\,\nabla\!\cdot(G\nabla x))$ for cell arrays
$k,s>0$ and a scalar $c>0$. The flux per unit area through a face is
$(x_2-x_1)/(\delta_1/k_1+\delta_2/k_2)$, the series resistance of the two half cells
{cite}`patankar1980`, so what leaves one cell enters the next; a boundary face takes its
cell's $k$. Face area and cell volume enter once, in the divergence.

```python
op = sx.flux_form_operator(widths, kinds, conductivity, shift, coefficient)
x, info = op.solve(rhs)              # A x = rhs, rhs includes the cell volume
x, info = op.solve(rhs + op.lift([(0.0, 1.0), None]))   # Dirichlet / Neumann boundary data
op2 = op.with_coefficients(conductivity=k_traced, shift=s_traced)   # same structure, no recompile
```

$A$ is symmetric positive definite and is solved by `pcg_linear_solve`, preconditioned by
the separable inverse at the volume-mean $k$ and $s$ fixed when the operator is built
(one material: one iteration; large contrasts cost more iterations as the mesh refines).

- **Derivatives** are implicit in `rhs`, `conductivity`, `shift` and `coefficient`; the
  tangent and adjoint solves start from zero. $A$ is symmetric, so the transpose solve is
  the solve itself.
- **Status**: `info` is a `SpdSolveInfo` (`converged`, `iterations`, `residual`,
  `status`, `defect`), all traced. An unconverged solve, a nonpositive coefficient or a
  nonfinite value returns NaN, and NaN derivatives in every input.
- **Singular variant** (`shift=0`, no Dirichlet axis): the rank-one term
  $\alpha V (V\!\cdot\!x)/\sum V$ makes $A$ positive definite, the preconditioner gains the
  matching inverse on the constants, and the volume mean is removed after the solve. The
  compatibility defect $\sum r/\sum V$ is reported in `info.defect`; a right-hand side
  whose defect exceeds `defect_rtol` of its absolute sum is a net source the gauge would
  absorb, so the solve returns NaN rather than a gauge-fixed answer.

## Smallest eigenpair from a solve

`inverse_iteration(solve, weight, v0, tol=..., max_steps=...)` returns the smallest
eigenpair of $Kv=\lambda Wv$, $W=\mathrm{diag}(\text{weight})$, from $K^{-1}$ alone, with
an `InverseIterationInfo(converged, residual, iterations)`. The residual
$\|K^{-1}Wv-\mu v\|_W/(\mu\|v\|_W)$, $\mu=1/\lambda$, certifies an eigenpair (and bounds
the eigenvalue error relative to the nearest eigenvalue); that it is the smallest needs
`v0` to have a component along the lowest mode. Steps needed:
$\log\mathrm{tol}/\log(\lambda_1/\lambda_2)$.

Derivative contract: `lam` is the Rayleigh quotient of the converged vector held fixed,
stationary there, so its **first derivative** in everything `solve` and `weight` depend
on is exact and costs one differentiation through `solve`, never the loop. The
**eigenvector derivative is unsupported**: `v` is returned under `stop_gradient`
(zero), so differentiate functionals of `lam` only. Second derivatives are unsupported.
Unconverged, `lam` is NaN and its derivatives are NaN, provided `solve` carries a NaN
cotangent (`pcg_linear_solve` does).

## API summary

- {func}`solvax.separable.fast_diagonal`, {class}`solvax.separable.FastDiagonal`
- {func}`solvax.separable.flux_form_operator`, {class}`solvax.separable.FluxFormOperator`,
  {class}`solvax.separable.SpdSolveInfo`
- {func}`solvax.eigen.inverse_iteration`, {class}`solvax.eigen.InverseIterationInfo`
