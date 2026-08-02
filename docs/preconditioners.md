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

## Symmetric additive composition

For fixed self-adjoint positive-definite inverse actions $B_i$ and positive
weights $w_i$, the sum $B=\sum_i w_iB_i$ remains self-adjoint positive definite
and is therefore suitable for PCG. `additive_preconditioner` defaults to the
arithmetic mean and accepts arrays or arbitrary matching pytrees:

```python
precond = sx.additive_preconditioner([x_line_inverse, y_line_inverse])
solution = sx.pcg(matvec, rhs, precond=precond)
```

This composes existing axis, block, or Schwarz inverse actions; geometry and
component construction stay with the caller. Use positive custom weights only
after verifying that every component is symmetric positive definite.

For structured arrays, SOLVAX can build those components directly from
nonperiodic tridiagonal bands and an optional cyclic final axis:

```python
precond = sx.additive_tridiagonal_line_preconditioner(
    diagonal,
    [(0, lower_x, upper_x), (1, lower_y, upper_y)],
    periodic_last_axis=(lower_z, upper_z),
)
solution = sx.pcg(matvec, rhs, precond=precond)
```

All arrays retain their original layout outside each batched line solve. The
builder is JIT- and gradient-transparent; the caller remains responsible for
coefficient boundary entries and component positive definiteness.

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

`multigrid` is the same cycle written over a list of `MultigridLevel` entries
and adds the remaining cycle parameters:

```python
precond = sx.multigrid(
    hierarchy.levels,
    coarse_solve,
    cycle="f",          # "v" (gamma = 1), "w" (gamma = 2) or the F-cycle
    cycle_index=None,   # or an explicit gamma, scalar or per level
    pre_smooth=2,
    post_smooth=1,
)
```

Arrays supplied as smoothers are interpreted as damped-diagonal smoothers,
with the weight set by `damping`; callables may implement richer applications.
Multigrid quality depends on complementary smoothing and coarse correction,
not the recursion alone {cite}`trottenberg2001`. The recursion is unrolled at
trace time, so a W-cycle over $L$ levels emits $2^{L-1}$ coarse solves into
the program; prefer the F-cycle for deep hierarchies.

## Geometric transfers and semicoarsening

`solvax.transfer` builds the restriction and prolongation of a structured
tensor-product grid as one small matrix per coarsened axis, applied as a
sequence of per-axis contractions. No N-D transfer operator is ever formed,
and an axis excluded by the mask is not contracted at all.

```python
restrict, prolong = sx.grid_transfer(
    (64, 64, 16),
    coarsen=(True, True, False),      # semicoarsening: the last axis stays fine
    boundary=("periodic", "periodic", "dirichlet"),
)
```

Three closures are available per axis — `"periodic"` ($n_f$ even, coarse point
$i$ at fine point $2i$), `"dirichlet"` ($n_f$ odd, interior unknowns only,
coarse point $i$ at fine point $2i+1$), and `"reflective"` (cell-centered with
a mirror closure) — with `"full_weighting"` or `"injection"` restriction and
`"linear"` or `"injection"` prolongation. The default pair satisfies the
variational relation

$$
R = 2^{-d}P^{\mathsf T}
$$

exactly over the $d$ coarsened axes, and arrays may carry extra trailing field
axes, which ride along untouched.

Semicoarsening — coarsening only the axes along which the operator is strongly
coupled — is what makes multigrid work for anisotropic and convection-dominated
operators. Under full coarsening those problems leave error components that no
local relaxation can damp, because they are inexpensive for the operator yet
invisible to the coarse grid {cite}`trottenberg2001`. `coarsening_plan` handles
the bookkeeping, stopping each axis as its parity or minimum size runs out
while the others continue.

`semicoarsening_hierarchy` assembles a complete level list. Coarse operators
are obtained by **rediscretization** — rebuilding the same discrete physics on
each coarse grid — rather than by the Galerkin product $RAP$: it is cheaper and
it keeps every coarse operator in the structured (tridiagonal, banded) form the
smoothers need.

```python
def rediscretize(shape):
    matvec, bands = build_operator(shape)          # your own discretization
    return matvec, sx.tridiagonal_smoother(*bands, axis=0, periodic=True)

hierarchy = sx.semicoarsening_hierarchy(
    (128, 16), coarsen=(True, False), rediscretize=rediscretize, levels=4
)
coarsest, _ = build_operator(hierarchy.shapes[-1])
precond = sx.multigrid(
    hierarchy.levels,
    sx.dense_coarse_solve(coarsest, hierarchy.shapes[-1]),
)
```

The recursion has to bottom out on an *exact* solve, or the cycle inherits the
coarsest level's condition number {cite}`trottenberg2001`. `dense_coarse_solve`
supplies one without asking for an assembled matrix: it probes the same
matrix-free operator with unit vectors, factors the result once with a dense
LU, and closes over the factors. Coarsest grids are tiny by construction, so
the cubic cost is irrelevant — and when semicoarsening has left leading axes
the operator does not couple (per species, per energy, per wavenumber),
`batch_dims=k` probes and factors one small block per index instead of the
whole level.

## Smoothers

`solvax.smoothers` builds relaxation sweeps in the multigrid protocol
`smooth(matvec, x, b) -> x`, which improves a running iterate rather than
applying a fixed inverse to a right-hand side. `relaxation` converts any
approximate inverse into one:

| builder | inverts | use when |
|---|---|---|
| `jacobi_smoother` | $\operatorname{diag}(A)$ | isotropic coupling |
| `block_jacobi_smoother` | dense blocks along one axis | stiff within-cell physics |
| `tridiagonal_smoother` | lines along one axis | one strongly coupled direction |
| `plane_smoother` | whole 2-D planes | two strongly coupled directions |
| `upwind_smoother` | the upstream-triangular part | streaming/advection |
| `alternating_smoother` | composition of the above | mixed or unknown anisotropy |

For a streaming operator the sweep *order* is what matters: relaxation must
visit points downstream, otherwise the error is simply transported back in
{cite}`brandt1993`. `upwind_smoother` expresses that ordering pointwise —
keeping only the coupling to the upstream neighbour makes the swept operator
triangular in the flow direction, so a single batched tridiagonal solve
performs the whole ordered sweep with no sequential grid loop.

Measure, do not guess. `smoothing_factor` estimates Brandt's smoothing factor
{cite}`brandt1977` — the asymptotic error reduction restricted to the modes the
coarsening cannot represent — by power iteration on the Fourier-projected error
propagation operator:

```python
mu = sx.smoothing_factor(smooth, matvec, shape, key=key, coarsen=(True, False))
```

A smoother with $\mu$ near one will not be rescued by a better cycle, more
levels, or a stronger coarse solve.

## Randomized Nystrom preconditioning

When no grid hierarchy or structured coarse operator exists, a rank-$\ell$
randomized Nystrom approximation $A_{\mathrm{nys}}=U\Lambda U^T$ built from
$\ell$ operator applications gives the SPD preconditioner

$$
P^{-1}v=U\,\frac{\lambda_\ell+\mu}{\Lambda+\mu}\,U^Tv+(v-UU^Tv)
$$

for the regularized system $(A+\mu I)x=b$. When $\ell$ exceeds about twice
the $\mu$-effective dimension of $A$, the preconditioned condition number is
bounded by a small constant in expectation, independently of how the spectrum
decays {cite}`frangella2023`.

```python
precond = sx.nystrom_preconditioner(matvec, n, rank, jax.random.PRNGKey(0), mu=mu)
solution = sx.pcg(lambda v: matvec(v) + mu * v, b, precond=precond)
```

The sketch key is explicit, so construction is deterministic, jit-able, and
differentiable through both the sketch and the eigenfactors — gradients flow
through the preconditioner exactly like every other SOLVAX component. On a
decaying-spectrum SPD operator with a flat tail the test suite pins at least a
2x PCG iteration reduction; the target regime is fast spectral decay ahead of
a regularization shift.

### Choosing the rank

The bound is stated in terms of the $\mu$-effective dimension, which a caller
does not usually know. `spectrum_span` is the posterior read: the smallest
retained eigenvalue over the largest. Near zero the sketch has reached into the
decaying tail and the approximation is capturing structure; near one the
spectrum inside the sketch is flat, which means the rank is too small to be
preconditioning anything and the bound's precondition is not met.

```python
precond = sx.nystrom_preconditioner(matvec, n, rank, key, mu=mu)
print(float(precond.spectrum_span))  # << 1 means the sketch reached the tail
```

`nystrom_preconditioner_adaptive` acts on that reading, doubling the rank until
the span clears a target and returning the rank it used:

```python
precond, rank = sx.nystrom_preconditioner_adaptive(matvec, n, key, mu=mu, span_target=1e-2)
```

The loop is ordinary Python, because each rank is a different static shape and
the growth cannot happen inside one traced computation. Build the
preconditioner when the operator changes, not inside a hot loop; each
individual build is pure JAX and the result is traceable as usual. On a
spectrum decaying geometrically the loop stops at rank 16; on a flat one it
grows to the cap and returns that rank, which is the honest report — no rank
preconditions a flat spectrum, and the caller learns that rather than paying
for a sketch that cannot help.

## Symmetric Galerkin deflation

PCG needs a fixed symmetric positive-definite preconditioner. Given a symmetric
smoother $S$, prolongation $P$, and Galerkin coarse operator
$A_c=P^TAP$, `galerkin_deflation` applies the balanced two-level inverse

$$
S + (I-SA)P A_c^{-1}P^T(I-AS).
$$

```python
coarse_template = jnp.zeros(coarse_shape)
precond = sx.galerkin_deflation(
    A_fine,
    symmetric_smoother,
    prolong,
    solve_galerkin_coarse,
    coarse_template,
)
```

Restriction is generated as the exact linear transpose of `prolong`, avoiding
an inconsistent transfer pair. The caller still owns the symmetry and positive
definiteness of $A$, $S$, and the coarse solve. Use the general V-cycle with a
flexible Krylov method when these requirements do not hold.

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
| positive additive composition | yes | yes | yes if components are SPD |
| line smoother | yes | yes | only if resulting action is SPD |
| V-cycle | yes | yes | only with a symmetric positive cycle |
| balanced Galerkin deflation | yes | yes | yes if components are SPD |
| mixed precision | yes | yes | validate positivity carefully |

## API summary

- {func}`solvax.precond.jacobi`
- {func}`solvax.precond.block_jacobi`
- {func}`solvax.precond.coarse_operator`
- {func}`solvax.precond.line_smoother`
- {func}`solvax.precond.additive_preconditioner`
- {func}`solvax.precond.p_multigrid`
- {func}`solvax.precond.galerkin_deflation`
- {func}`solvax.randomized.nystrom_preconditioner`
- {func}`solvax.precond.mixed_precision`
- {func}`solvax.precond.kronecker_nkp`
- {func}`solvax.precond.nearest_kronecker`

Runnable counterparts: examples 02, 07, 08, 09, 10, 11, and 12.
