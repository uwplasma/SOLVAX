# Targeted generalized eigenpairs

For a Hermitian block-tridiagonal pencil $F y=\lambda G y$ with $G$ positive
definite, `generalized_eigenpairs` returns the `count` eigenpairs nearest a real
`target`, with first-order derivatives with respect to the bands of $F$ and $G$.
Both may depend on the parameters. The motivating case is a finite-element
spectrum problem: $F$ is a stiffness matrix, $G$ a mass matrix that moves with
the geometry, and the unknown is ordered by radial element.

```python
F = (lower, diag, upper)   # (n_blocks, m, m) each; Hermitian pencil, G positive definite
G = (lower_g, diag_g, upper_g)
result = sx.generalized_eigenpairs(F, G, target=0.2, count=4, tol=1e-10, max_iter=100)
result.values, result.vectors   # (4,), (n_blocks, m, 4)
```

## Input convention

`F` and `G` are SOLVAX band triples. They are real symmetric or complex
Hermitian, and the code makes them so: it reads only `diag` and `upper[:-1]`,
rebuilds `lower[k]` as `upper[k-1]` conjugate-transposed, and replaces `diag` by
its Hermitian part. Transpose and conjugate transpose differ for complex input;
the Hermitian structure uses the conjugate transpose, and
{func}`~solvax.direct.block_thomas_solve` offers `transpose=True` ($A^T$) and
`adjoint=True` ($A^H$) separately. For a Hermitian band $A^H=A$, so the solver
needs only the forward solve.

## Algorithm

Shift-invert subspace iteration. With $S=F-\sigma G$ factored once by
{func}`~solvax.direct.block_thomas_factor`, a block of
$p=\min(2\,\text{count}+8,\ N m)$ vectors $Y$ is advanced by $Y\leftarrow
\mathrm{qr}(S^{-1} G Y_{\text{Ritz}})$, followed by Rayleigh--Ritz on
$(Y^H F Y,\ Y^H G Y)$ through a Cholesky reduction. The `count` Ritz values
nearest `target`, and the nearest excluded one, must reach the relative residual

$$
\frac{\lVert F y-\lambda G y\rVert}{(\lVert F\rVert+|\lambda|\lVert G\rVert)\,\lVert y\rVert}
\le \texttt{tol},
$$

with $\lVert\cdot\rVert$ the infinity norm (largest absolute row sum) of a band.
The shift is $\sigma=\texttt{target}$ moved by $16\,\epsilon\,\lVert F\rVert/\lVert G\rVert$ so that a
target equal to an eigenvalue does not produce a singular factor.

### Bad pivots

Block Thomas pivots inside each Schur complement but not across blocks. It
therefore fails on an invertible matrix whose Schur complement is singular, for
example $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ with one scalar block per row, or
on an indefinite pencil whose shift lands on a diagonal block. A probe solve
with a fixed vector is checked against the band operator after every
factorization. If its relative residual is not below $\sqrt{\epsilon}$ or is not
finite, a dense pivoted LU of the assembled band is used instead
(`result.fallback` is True). The dense path is compiled only when
`n_blocks * m <= dense_limit` (default 2048); for larger problems there is no
fallback and a breakdown shows as non-finite residuals and `converged=False`.
Under `vmap` a `cond` runs both branches, so pass `dense_limit=0` when batching
large problems.

## Result fields

| Field | Meaning |
|---|---|
| `values` | `(count,)` real, ascending |
| `vectors` | `(n_blocks, m, count)`, $y^H G y=1$, phase arbitrary |
| `residuals` | the relative residual above, per pair |
| `orthogonality` | $\max\lvert Y^H G Y-I\rvert$ |
| `iterations` | shift-invert applications |
| `converged` | `max(residuals) <= tol`, False if non-finite |
| `gap` | $\lvert\lambda-\text{target}\rvert$ of the nearest excluded Ritz value minus that of the farthest selected one |
| `fallback` | the dense pivoted path ran |

`gap` is zero when the selection splits a degenerate or tied set. It sees only
the subspace: an eigenvalue the iteration never captured is not counted.

## Derivatives

The iteration is not differentiated. With $y^H G y=1$ the rules are implicit,
valid only when `converged` is True, and exact to the accuracy of the solve.

**Eigenvalue.** $d\lambda=\operatorname{Re}\,y^H(dF-\lambda\,dG)\,y$.

**Cluster mean.** For a set of eigenvalues that is isolated from the rest of the
spectrum, $d\sum_i\lambda_i=\sum_i y_i^H(dF-\lambda_i\,dG)\,y_i$ is the trace of the
perturbation projected on the cluster. It is invariant under rotations inside
the cluster, so it stays smooth through crossings and exact degeneracies where
the individual values are not differentiable. Take `result.values.mean()`. It
needs `gap > 0`: if an excluded eigenvalue ties a selected one, the selected
set itself jumps.

**Eigenvector.** The tangent solves the bordered system

$$
(F-\lambda G)\,dy=-(dF-\lambda\,dG)\,y+d\lambda\,G y,\qquad
y^H G\,dy=-\tfrac12\,y^H dG\,y,
$$

whose second equation is the normalization $\operatorname{Re}(y^H G\,dy)=-\tfrac12 y^H dG\,y$
together with the phase condition $\operatorname{Im}(y^H G\,dy)=0$. The phase of the
primal vector is arbitrary, so only phase-invariant functions of `vectors`
($y y^H$, $|y|$) have meaningful derivatives. The projected right-hand side is solved
at the shift $\lambda+\text{sep}/10^3$, where `sep` is the distance to the nearest other Ritz value,
followed by three refinement sweeps on the exact singular operator, so the error
contracts by $10^{-3}$ per sweep. If $\text{sep}\le\sqrt{\epsilon}\,\lVert F\rVert/\lVert G\rVert$ the tangent
is NaN rather than wrong: differentiate the cluster mean (or the projector) of a
degenerate pair, not one of its vectors.

`target`, `tol` and every diagnostic have zero derivative.

## Support

| | real symmetric | complex Hermitian |
|---|---|---|
| eager, `jit` | yes | yes |
| `vmap` | yes (use `dense_limit=0` for large problems) | yes |
| `jvp` (forward), `values` | yes | yes |
| `jvp`, `vectors` | yes, phase-gauge tangent | yes, phase-gauge tangent |
| `vjp` / `grad` | yes | yes |
| first order | tested against finite differences, dense perturbation theory and the dot-product identity | same |
| second order, `values` | tested against finite differences of the gradient | same |
| second order, `vectors` | not supported (not tested; do not rely on it) | not supported |
| precision | float64 validated; float32 is not | float64 validated |
| CPU / GPU | CPU tested; GPU untested | CPU tested; GPU untested |
| JAX | `>=0.4.38`, Python `>=3.10` tested | same |

The tangent scan carries its linear inputs in the carry, because JAX before
0.10 cannot transpose a scan whose linear inputs are `xs`.

## Cost

One block-Thomas factorization of $F-\sigma G$ ($O(Nm^3)$), then per iteration one
solve with $p$ right-hand sides ($O(Nm^2p)$), two band products, and a
$p\times p$ Ritz problem. `jit` compiles the solve with a Python loop over blocks, so
compile time grows with `n_blocks` (about 7 s for 99 blocks of size 13, complex128, 6
pairs on one CPU core; the compiled solve then takes 0.1 s). The eigenvector tangent
factors one shifted band per returned eigenpair.

## API summary

- {func}`solvax.eigen.generalized_eigenpairs`
- {class}`solvax.eigen.EigenpairResult`
- {func}`solvax.direct.block_thomas_solve` (`adjoint=True`)
