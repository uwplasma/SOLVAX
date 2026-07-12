# Preconditioners

A preconditioner is an inexpensive inverse action $M^{-1}$ chosen so the
preconditioned operator has a more favorable spectrum. It need not reproduce
$A^{-1}$ accurately in every direction; it should remove the error components
that make the outer iteration slow.

SOLVAX builders return callables suitable for `precond=`.

## Jacobi scaling

For $M=\operatorname{diag}(A)$,

$$
M^{-1}r=r\oslash\operatorname{diag}(A).
$$

```python
precond = sx.jacobi(diagonal)
```

Jacobi is cheap, parallel, and useful for scale disparity. It cannot represent
strong off-diagonal or within-cell coupling. Zero diagonal entries are a model
error and should be addressed explicitly.

## Block Jacobi

Partition the unknown into independent preconditioning blocks:

$$
M=\operatorname{blockdiag}(D_0,\ldots,D_{N-1}).
$$

```python
precond = sx.block_jacobi(blocks)  # (N, m, m)
```

Each dense block is LU-factored and applied in a batch. This is effective when
within-point physics is stiff but inter-point coupling is weaker. It costs more
than point Jacobi but often reduces Krylov iterations substantially.

## Coarse or simplified operator

Let $A_s$ retain dominant physics and discard expensive weak couplings. Then

$$
A A_s^{-1}=I+(A-A_s)A_s^{-1}.
$$

If the second term is small in the difficult subspace, eigenvalues cluster near
one. `coarse_operator` documents and returns an existing solve action:

```python
factors = sx.block_thomas_factor(*local_bands)
solve_local = lambda r: sx.block_thomas_solve(factors, r)
precond = sx.coarse_operator(solve_local)
```

This is the preferred production pattern for a structured local principal part
plus nonlocal, nonlinearized, or dense-tail corrections.

## Line smoother

For anisotropic operators, point smoothers leave error that varies slowly along
the strongly coupled direction. A line solve updates

$$
x\leftarrow x+\omega_iM_i^{-1}(b-Ax)
$$

for each selected direction $i$ and sweep.

```python
precond = sx.line_smoother(
    matvec,
    [x_line_inverse, y_line_inverse],
    omega=[0.8, 0.8],
    sweeps=2,
)
```

The line inverses often use `tridiagonal_solve` or banded LU. Alternating
directions treats mixed anisotropy better than a single line family
{cite}`trottenberg2001`.

## Multigrid V-cycle

Let level $\ell$ have operator $A_\ell$, smoother $S_\ell$, restriction
$R_\ell$, and prolongation $P_\ell$. A V-cycle performs:

1. pre-smoothing on $A_\ell x=b$;
2. residual restriction $r_{\ell+1}=R_\ell(b-A_\ell x)$;
3. recursive coarse correction;
4. prolongation $x\leftarrow x+P_\ell e_{\ell+1}$;
5. post-smoothing.

```python
precond = sx.p_multigrid(
    matvecs=[A_fine, A_medium],
    restricts=[R_fine, R_medium],
    prolongs=[P_fine, P_medium],
    coarse_solve=solve_coarse,
    smoothers=[fine_diagonal, medium_smoother],
    cycles=1,
)
```

Despite its historical name, the routine is agnostic to whether levels arise
from mesh spacing $h$, polynomial degree $p$, spectral truncation, or a physics
coarsening. The caller owns consistency of shapes and transfer operators.

Arrays supplied as smoothers are interpreted as diagonal smoothers; callables
may implement richer applications. Multigrid quality depends on complementary
smoothing and coarse correction, not the recursion alone
{cite}`trottenberg2001`.

## Kronecker preconditioning

For a separable approximation $A\approx A_1\otimes A_2$,

$$
(A_1\otimes A_2)^{-1}=A_1^{-1}\otimes A_2^{-1}.
$$

`kronecker_nkp` accepts LU factors for the two factors and applies the inverse
through reshaping and two small solves:

```python
precond = sx.kronecker_nkp(lu_factor(A1), lu_factor(A2))
```

For a small dense matrix, `nearest_kronecker(matrix, na, nb)` obtains a
rank-one Kronecker approximation from the leading singular triplet of the Van
Loan-Pitsianis rearrangement {cite}`vanloan1993`.

The extraction itself requires the dense matrix and is therefore a model or
setup tool, not a large matrix-free operation.

## Mixed-precision wrapper

```python
precond32 = sx.mixed_precision(precond64, dtype=jnp.float32)
```

Inputs are cast down for the preconditioner and results cast back. Flexible
GMRES can tolerate this varying/inexact action. PCG requires additional care:
the effective preconditioner must remain positive definite.

## Constraint-aware preconditioning

For bordered systems, use `schur_projected_precond`; see {doc}`operators`.
It incorporates the small constraint Schur complement rather than ignoring
Lagrange multipliers.

## How to evaluate a preconditioner

Measure:

- outer iterations and true residual;
- setup/factorization time;
- application time per iteration;
- memory and compilation cost;
- robustness across the full parameter regime.

A preconditioner that halves iterations but costs ten operator applications per
use is not an improvement. Benchmark complete solves, including setup reuse.

## Compatibility table

| Preconditioner | FGMRES | GCROT | PCG |
|---|---|---|---|
| Jacobi | yes | yes | yes if positive |
| Block Jacobi | yes | yes | yes if HPD |
| changing/inexact nested solve | yes | yes | generally no |
| line smoother | yes | yes | only if resulting action is SPD |
| V-cycle | yes | yes | only with a symmetric positive cycle |
| mixed precision | yes | yes | validate positivity carefully |

## API summary

- {func}`solvax.precond.jacobi`
- {func}`solvax.precond.block_jacobi`
- {func}`solvax.precond.coarse_operator`
- {func}`solvax.precond.line_smoother`
- {func}`solvax.precond.p_multigrid`
- {func}`solvax.precond.mixed_precision`
- {func}`solvax.precond.kronecker_nkp`
- {func}`solvax.precond.nearest_kronecker`

Runnable counterparts: examples 02, 07, 08, 09, 10, 11, and 12.
