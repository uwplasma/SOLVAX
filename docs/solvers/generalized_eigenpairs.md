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
Hermitian, and the solver uses the Hermitian band $A_h$: `lower[k]` is
`upper[k-1]` conjugate-transposed and `diag` is replaced by its Hermitian part.
It does not do so silently: `hermitian_defect` is $\lVert A-A_h\rVert/\lVert A_h\rVert$
(the larger over `F` and `G`), counting `diag` and `lower[1:]`, and
`converged` is False when it is above `tol`. Pass `lower=None` to say it is not
provided; then only `diag` is checked. A real non-symmetric operator such as
`lower=-0.2`, `diag=2`, `upper=-1` (40 points, spectrum in $[1.106, 2.894]$) gives
defect 0.2 and `converged=False`; its Hermitian part has eigenvalues 0.36, 0.46, 0.56.
To first order a defect $d$ moves an eigenvalue by at most
$d\,(\lVert F\rVert+\lvert\lambda\rvert\lVert G\rVert)$, the scale of the residual,
which is why the bound is `tol`; on the random 30 by 30 real and complex test
pencils the measured move is at most $0.03\,d$ of that scale for $d$ from
$10^{-12}$ to $10^{-3}$. Eigenvectors move by about $d$ divided by the relative gap
(about $7\,d$ on the complex test pencil), which `tol` does not bound.
`tol=None`, the default, is $\max(10^{-10}, 100\,\epsilon)$ of the pencil's dtype.
Transpose and conjugate transpose differ for complex input;
the Hermitian structure uses the conjugate transpose, and
{func}`~solvax.direct.block_thomas_solve` offers `transpose=True` ($A^T$) and
`adjoint=True` ($A^H$) separately. For a Hermitian band $A^H=A$, so the solver
needs only the forward solve.

## Algorithm

Shift-invert subspace iteration. With $S=F-\sigma G$ factored once by
{func}`~solvax.direct.block_thomas_factor`, a block of
$p=\min(2\,\text{count}+8,\ N m)$ vectors $Y$ spans, per sweep, the block Krylov space
$[Y,\ TY,\ \dots,\ T^5Y]$ of $T=S^{-1}G$ (orthogonalized twice; fewer blocks if it would exceed $N m$).
Rayleigh--Ritz for $T$ on that space, through an `eigh` of the Gram matrix $V^H G V$, gives real values $\mu$; the
$p$ largest $|\mu|$ are kept, with $\lambda=\sigma+1/\mu$. Ranking by $|\mu|$ keeps a Ritz ghost,
a mixture of eigenvectors on both sides of `target` with $\mu\approx0$, out of the selection;
Rayleigh--Ritz on $F$ itself cannot. The `count` Rayleigh quotients
nearest `target` must reach the relative residual

$$
\frac{\lVert F y-\lambda G y\rVert}{(\lVert F\rVert+|\lambda|\lVert G\rVert)\,\lVert y\rVert}
\le \texttt{tol},
$$

with $\lVert\cdot\rVert$ the infinity norm (largest absolute row sum) of a band.
The nearest excluded one, which sets `gap`, must reach `tol` too or resolve the gap.
For a Ritz pair $(\theta, x)$ with $x^H G x=1$ and $r=Fx-\theta Gx$ some eigenvalue
$\lambda$ of the pencil has $\lvert\lambda-\theta\rvert\le e=\lVert r\rVert_{G^{-1}}
=(r^H G^{-1} r)^{1/2}$ (Bai, Demmel, Dongarra, Ruhe and van der Vorst, *Templates for the
solution of algebraic eigenvalue problems*, SIAM 2000, chapter 5, generalized Hermitian
problem, error bounds for computed eigenvalues, eq. (5.31), which also gives the weaker
$\lVert G^{-1}\rVert_2\lVert r\rVert_2$).
$e$ is computed with one solve by the block-Thomas factor of $G$ per sweep. The gap is
resolved when $e$ of the excluded pair plus the largest $e$ of the selected ones is at most
`gap / 2`: the intervals around the selected and the excluded Ritz values are then
disjoint, and the true gap is at least half the reported one. On a diagonal pencil with a
cluster 0.1 behind the target this stops after one sweep instead of 13, with the excluded
residual at $2.9\times10^{-4}$. In a near tie (gap below the excluded pair's error) the
iteration goes on until the excluded pair reaches `tol`, as before.

`start`, of shape `(n_blocks, m, k)` with `k <= p`, replaces the first `k` columns of
the random starting block, for example by the vectors of a nearby pencil in a scan or an
optimization loop; `start=None` is the cold start, bit for bit. On a 400-point
Laplacian with target -0.2 and three pairs, the vectors at `t = 0` start the pencil
`t = 0.01` in 4 sweeps instead of 7. `max_iter` is traced: changing it does not compile
again (it took 1.1 to 2.6 s per value on CPU).
The shift is $\sigma=\texttt{target}$ moved by
$\sqrt{\epsilon}\,(\lVert F\rVert+\lvert\texttt{target}\rvert\lVert G\rVert)/\lVert G\rVert$.
A shift a distance $\delta$ from an eigenvalue leaves a solve error of about
$\epsilon/\delta$ in the other directions, so with the former $16\,\epsilon$ a target
equal to an eigenvalue stalled near $10^{-4}$: the error sits in the shift-invert
images, and the dense pivoted LU stalls the same way. On `tridiag(1, 0, 1)` with
401 unknowns and target 0, the Gram matrix of the first sweep lost 3 of 84
directions at the $16\,\epsilon$ offset (where its Cholesky factor was NaN) and
none at $\sqrt{\epsilon}$. As a safeguard the Ritz step cuts directions whose Gram
eigenvalue is below $\sqrt{\epsilon}$ of the largest; a cut direction returns as a
zero vector ranked last, and if one were ever selected its residual is NaN and
`converged` is False.

### Bad pivots

Block Thomas pivots inside each Schur complement but not across blocks. It
therefore fails on an invertible matrix whose Schur complement is singular, for
example $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ with one scalar block per row, or
on an indefinite pencil whose shift lands on a diagonal block. A probe solve
with a fixed vector is checked against the band operator after every
factorization; its relative residual is `result.probe_residual`. If it is above
$\epsilon^{3/4}$ or is not finite, `result.fallback` is True, at any size. Near an
eigenvalue (within about $10^{-4}$ relative) the probe exceeds this although block
Thomas is fine, so with `dense_limit > 0` the solve, and the derivative solves,
then use the dense LU; a shift on a singular Schur pivot, $\sqrt{\epsilon}$ away,
gives a probe of about $2.5\times10^{-9}$ and reaches it too.
A dense pivoted LU of the assembled band then takes over only if
`n_blocks * m <= dense_limit`, and the default is 0: there is no dense path
unless it is asked for, and a breakdown shows as large or non-finite residuals
and `converged=False`. Under `vmap` a `cond` runs both branches, so a nonzero
`dense_limit` pays for the dense factorization on every batched call.
The dense path ran exactly when `fallback` is True and
`n_blocks * m <= dense_limit`.

## Result fields

| Field | Meaning |
|---|---|
| `values` | `(count,)` real, ascending |
| `vectors` | `(n_blocks, m, count)`, $y^H G y=1$, phase arbitrary |
| `residuals` | the relative residual above, per pair |
| `orthogonality` | $\max\lvert Y^H G Y-I\rvert$ |
| `iterations` | sweeps after the first, each applying $T$ six times |
| `converged` | `max(residuals)` and `hermitian_defect` at most `tol`, and `excluded_residual` at most `tol` or the gap resolved (see the algorithm), False if non-finite |
| `gap` | $\lvert\lambda-\text{target}\rvert$ of the nearest excluded Ritz value minus that of the farthest selected one |
| `fallback` | the probe solve failed; the dense pivoted path ran if `n_blocks * m <= dense_limit` |
| `excluded_residual` | the relative residual of the nearest excluded Ritz pair, the one that sets `gap`; zero if every Ritz value was selected; may exceed `tol` when `converged` if that pair resolved the gap |
| `probe_residual` | relative residual of the block-Thomas probe solve at the shift |
| `hermitian_defect` | relative size of the non-Hermitian part of `diag` that was dropped (see the input convention) |

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
at the shift $\lambda+\text{sep}/10^3$, where `sep` is the distance to the nearest Ritz value outside the
cluster of $y$, followed by three refinement sweeps on the exact singular operator, so the error
contracts by $10^{-3}$ per sweep. Ritz values within $\sqrt{\epsilon}\,(\lVert F\rVert+|\lambda|\lVert G\rVert)/\lVert G\rVert$
of $\lambda$ are degenerate with it. For such a cluster $C$ the tangent is the gauge-fixed subspace tangent:
the right-hand side and the solve are projected $G$-orthogonally to the whole cluster, and the
cluster block of $Y^H G\,dY$ is $-\tfrac12 Y_C^H dG\,Y_C$, so there is no rotation inside the
cluster. Only subspace-invariant functions of `vectors` (the projector $Y_C Y_C^H$) are meaningful
there. The tangent is finite, never NaN, in forward and reverse mode: reverse mode transposes the
whole tangent map, and a NaN anywhere in it leaks into the value cotangent as NaN$\cdot 0$.
Value and cluster-mean tangents are unaffected. Certify the selection with `gap`.

`target`, `tol` and every diagnostic have zero derivative.

## Support

| | real symmetric | complex Hermitian |
|---|---|---|
| eager, `jit` | yes | yes |
| `vmap` | yes (a nonzero `dense_limit` runs the dense LU on every call) | yes |
| `jvp` (forward), `values` | yes | yes |
| `jvp`, `vectors` | yes, phase-gauge tangent | yes, phase-gauge tangent |
| `vjp` / `grad` | yes | yes |
| first order | tested against finite differences, dense perturbation theory and the dot-product identity | same |
| second order, `values` | tested against finite differences of the gradient | same |
| second order, `vectors` | not supported (not tested; do not rely on it) | not supported |
| precision | float64 validated; float32 is not | float64 validated |
| CPU / GPU | CPU tested; GPU untested | CPU tested; GPU untested |
| JAX | `>=0.4.38`; the test file passes on 0.4.38 (Python 3.10), 0.6.2, 0.7.2, 0.9.2 and 0.12 (Python 3.12), each a CI job | same |

Reverse mode transposes the tangent rule, and JAX's scan transpose is not
uniform across releases: a scan over the eigenpairs with the tangents in its
carry passed on 0.4.38 and 0.10 but raised `AssertionError` (0.6.2, 0.7.2) or
`TypeError` (0.9.2). The rule therefore loops over the `count` eigenpairs in
Python, and only the substitution sweeps of the solve are scans. Releases not
listed above are untested.

## Cost

One block-Thomas factorization of $F-\sigma G$ ($O(Nm^3)$), then per sweep six
solves with $p$ right-hand sides ($O(Nm^2p)$), eight band products, and a
$p\times p$ Ritz problem. The eigenvector tangent factors one shifted band per
returned eigenpair.

The substitution sweeps are two `lax.scan`s, not the unrolled loops of
{func}`~solvax.direct.block_thomas_solve`, so compile time does not depend on
`n_blocks`. The solver is never differentiated, so unrolling is not needed here.
Measured with complex128, $m=13$, `count=6`, `tol=1e-10`, one CPU core, JAX 0.12
(cold is compile plus first call; warm is the median of five calls; grad is
`jax.grad` of the values and a quadratic form in the vectors with respect to `F`
and `G`; seconds):

| `n_blocks` | forward cold, before / after | forward warm, before / after | grad cold, before / after | grad warm, before / after |
|---|---|---|---|---|
| 50 | 1.77 / 0.69 | 0.019 / 0.019 | 7.9 / 1.9 | 0.032 / 0.028 |
| 100 | 3.00 / 0.52 | 0.035 / 0.032 | 20.4 / 2.0 | 0.070 / 0.056 |
| 200 | 4.15 / 0.50 | 0.080 / 0.068 | 49.8 / 1.6 | 0.136 / 0.094 |
| 400 | 10.3 / 0.47 | 0.078 / 0.076 | 167 / 1.8 | 0.226 / 0.198 |

The gradient's cold time carries a constant for the Python loop over `count`
(1.8-2.0 s here, against 0.8-1.0 s when the loop was a scan, still independent
of `n_blocks`); it grows with `count`. Gradients agree to all printed digits. The dense fallback is compiled only if
`n_blocks * m <= dense_limit` (0 by default), so when asked for its LU adds a constant at small sizes.

## API summary

- {func}`solvax.eigen.generalized_eigenpairs`
- {class}`solvax.eigen.EigenpairResult`
- {func}`solvax.direct.block_thomas_solve` (`adjoint=True`)
