# Banded and periodic-banded LU

A scalar matrix is banded when

$$
A_{ij}=0\quad\text{for}\quad i-j>w_l\;\text{or}\;j-i>w_u.
$$

SOLVAX stores only the $w_l+w_u+1$ diagonals and provides a pure-JAX,
factor/solve implementation for nonperiodic and periodic systems.

## Storage convention

`bands` has shape `(lower_bw + upper_bw + 1, n)`. Entry $A_{ij}$ is stored at

$$
\mathtt{bands}[w_u+i-j,\,j].
$$

Thus the main diagonal is `bands[upper_bw]`. Padding entries outside the matrix
are ignored.

```python
factors = sx.lu_factor_banded(bands, lower_bw=2, upper_bw=1)
x = sx.lu_solve_banded(factors, rhs)
r = sx.banded_matvec(bands, 2, 1, x) - rhs
```

`rhs` may be `(n,)` or `(n, n_rhs)`.

## Doolittle factorization in band storage

The factorization writes $A=LU$ without creating entries outside the band. For
column $j$, only rows $j-w_u,\ldots,j+w_l$ participate. This reduces storage to
$O(n(w_l+w_u))$ and work to approximately
$O(n(w_l+w_u)^2)$ rather than dense $O(n^3)$ {cite}`golub2013`.

The XLA-friendly implementation does not perform dynamic row swaps. It uses:

1. **Row equilibration:** scale each row by the largest magnitude stored in
   that row.
2. **Static pivot floor:** replace a pivot whose magnitude is below a threshold
   and increment `factors.n_clamped`.

```python
factors = sx.lu_factor_banded(
    bands,
    lower_bw,
    upper_bw,
    equilibrate=True,
    static_pivot_floor=None,
)
print(factors.n_clamped)
```

A nonzero clamp count is a diagnostic, not proof that the answer is accurate.
Check the residual and consider an iterative-refinement or pivoted alternative.

## Periodic bands via Woodbury

A periodic stencil has corner entries outside the ordinary band. Write it as

$$
A=B+UV^T,
$$

where $B$ is the nonperiodic banded core and the low-rank update represents the
upper-right and lower-left corners. Sherman-Morrison-Woodbury gives

$$
A^{-1}=B^{-1}-B^{-1}U(I+V^TB^{-1}U)^{-1}V^TB^{-1}.
$$

SOLVAX factors $B$ and the small capacitance matrix
$I+V^TB^{-1}U$ once:

```python
factors = sx.lu_factor_banded_periodic(
    bands,
    lower_bw=1,
    upper_bw=1,
    corner_ul=corner_upper_left,
    corner_lr=corner_lower_right,
)
x = sx.lu_solve_banded_periodic(factors, rhs)
```

For a periodic tridiagonal matrix, each corner array has shape `(1, 1)`. Wider
bands use square corner blocks with dimensions set by the corresponding wrap
couplings.

## Use cases

- upwind and compact finite-difference operators;
- one-dimensional implicit advection-diffusion-reaction;
- exact line solves used in a smoother;
- a coupling-dropped banded preconditioner for a dense-tail operator;
- periodic field-line or angular discretizations.

## Comparison with alternatives

| Alternative | Prefer it when |
|---|---|
| `tridiagonal_solve` | both bandwidths are one and many independent columns are solved |
| block Thomas | each grid point contains several densely coupled fields |
| pivoted SciPy/SuperLU | the operator is not diagonally dominant or robust pivoting is essential on CPU |
| FGMRES | the banded part is only an approximation to a larger matrix-free operator |
| FFT diagonalization | the operator is truly circulant/constant-coefficient and transform overhead is favorable |

## Stability and failure modes

Gaussian elimination without pivoting is reliable for important diagonally
dominant classes but is not universally backward stable {cite}`golub2013`.
Equilibration reduces scale disparity; it cannot repair structural singularity.
For periodic solves, the banded core and capacitance matrix must both be
nonsingular.

Use `banded_matvec` to report an independent residual. For difficult cases,
consider the banded solve as an FGMRES preconditioner: the outer true residual
then protects against an inexact factorization.

## Inputs and outputs

- `lu_factor_banded` returns `BandedLUFactors`, including LU bands, row scales,
  bandwidths, and clamp count.
- `lu_solve_banded` returns an array with the same right-hand-side shape.
- `lu_factor_banded_periodic` returns `PeriodicBandedLUFactors`, which includes
  the core and capacitance factors.
- `lu_solve_banded_periodic` returns the periodic-system solution.

## API summary

- {func}`solvax.banded.banded_matvec`
- {func}`solvax.banded.lu_factor_banded`
- {func}`solvax.banded.lu_solve_banded`
- {func}`solvax.banded.lu_factor_banded_periodic`
- {func}`solvax.banded.lu_solve_banded_periodic`

Runnable counterparts: `examples/02_advection_preconditioning.py` and
`examples/04_banded_lu.py`.
