# FGMRES and GCROT recycling

SOLVAX provides restarted flexible GMRES for general matrix-free systems and a
GCROT-style extension for sequences of related systems. Both support real or
complex one-dimensional arrays and right preconditioning.

## Flexible GMRES

Given $A x=b$ and an initial guess $x_0$, form $r_0=b-Ax_0$ and
$v_1=r_0/\beta$, where $\beta=\lVert r_0\rVert_2$. At Arnoldi step $j$,

$$
z_j=M_j^{-1}v_j, \qquad w_j=A z_j.
$$

Orthogonalization produces

$$
A Z_m=V_{m+1}\bar H_m,
$$

where $Z_m=[z_1,\ldots,z_m]$, $V_{m+1}$ has orthonormal columns, and
$\bar H_m$ is upper Hessenberg. The correction is

$$
x_m=x_0+Z_my_m,
\qquad
y_m=\arg\min_y\lVert\beta e_1-\bar H_my\rVert_2.
$$

Storing $Z_m$ instead of assuming a fixed $M^{-1}$ makes the method flexible:
the preconditioner may change at every step {cite}`saad2003`.

SOLVAX uses two passes of classical Gram-Schmidt (CGS2) and incremental Givens
rotations. For complex inputs, projections are Hermitian and the rotations use
a scaled unitary LAPACK-style convention. The true residual is recomputed at
restart boundaries and after the final cycle.

## API usage

```python
solution = sx.gmres(
    matvec,
    b,
    x0=None,
    precond=None,
    restart=30,
    rtol=1e-8,
    atol=0.0,
    max_restarts=50,
)
```

### Inputs

| Input | Meaning |
|---|---|
| `matvec` | pure JAX callable `v -> A v` on `(n,)` arrays |
| `b` | right-hand side `(n,)`, real or complex |
| `x0` | optional initial guess; zero by default |
| `precond` | right inverse action `v -> M_j^{-1}v`; identity by default |
| `restart` | maximum Arnoldi steps per cycle |
| `rtol`, `atol` | true residual stopping tolerances |
| `max_restarts` | maximum number of Arnoldi cycles |

### Output

`KrylovSolution` contains `x`, true `residual_norm`, total inner
`iterations`, `converged`, and `recycle=None`.

## Restart trade-off

Unrestarted GMRES minimizes over an expanding Krylov space but stores
$O(nm)$ basis data and performs $O(nm^2)$ orthogonalization through $m$ steps.
Restarting caps memory and compiled shapes, but discards spectral information
and can stagnate. Increase `restart` when memory permits and convergence stalls;
improve the preconditioner before relying on very large restart spaces.

## Staged restart cycles

For expensive nested or multi-device operators, compiling every restart into
one executable can dominate the solve. `gmres_cycle` exposes one bounded
FGMRES cycle so the operator and preconditioner remain behind a reusable JIT
call boundary:

```python
@jax.jit
def cycle(rhs, initial):
    return sx.gmres_cycle(
        matvec,
        rhs,
        x0=initial,
        precond=precond,
        restart=24,
        rtol=1e-10,
    )

initial = jnp.zeros_like(b)
for _ in range(8):  # fixed, bounded staging loop
    solution = cycle(b, initial)
    initial = solution.x
```

Each result reports the true residual and iterations for that cycle. Passing
`solution.x` back as `x0` continues the same restarted method. Keep the outer
loop fixed when tracing; do not branch in Python on a JAX `converged` value.
For reverse-mode gradients, use the staged loop as the black-box solver passed
to {func}`solvax.implicit.linear_solve`. Krylov iterations contain dynamic JAX
loops and are differentiated implicitly, not by reversing their execution.

## GCROT-style recycling

For a sequence $A_i x_i=b_i$, retain source and image bases $(U,C)$ satisfying

$$
A U=C,\qquad C^HC=I.
$$

Before each Arnoldi cycle, project the residual over the recycled space:

$$
x\leftarrow x+UC^Hr,\qquad
r\leftarrow(I-CC^H)r.
$$

The new Arnoldi cycle is built for the deflated operator

$$
(I-CC^H)AZ_m=V_{m+1}\bar H_m.
$$

SOLVAX inserts the cycle's normalized correction into a fixed-size FIFO
recycle space. When a recycle pair is supplied for a changed operator, it
recomputes $AU$ and uses a thin QR factorization to restore $AU=C$ for the
current system.

```python
recycle = None
for parameter in scan:
    solution = sx.gcrot(
        make_matvec(parameter),
        b,
        m=30,
        k=10,
        recycle=recycle,
        rtol=1e-10,
    )
    recycle = solution.recycle
```

`m` is the inner FGMRES cycle length and `k` is the fixed recycle dimension.
The returned `recycle` arrays have shape `(n, k)` and may contain zero-padded
unused columns.

## Relationship to the literature

The implementation follows the recycling framework of Parks et al. and the
deflated-restart motivation of Morgan {cite}`parks2006,morgan2002`. It is not a
full harmonic-Ritz GCRO-DR implementation: it retains one optimal cycle
correction per restart rather than selecting harmonic Ritz vectors. This keeps
the update shape-static and $O(nk)$ but may identify difficult invariant
subspaces more slowly.

## Choosing between Krylov methods

| Method | Matrix assumptions | Memory | Best use |
|---|---|---|---|
| PCG | Hermitian positive definite | short recurrences | SPD elliptic and energy systems |
| FGMRES | general square operator | restart-sized basis | isolated nonsymmetric/indefinite solve |
| GCROT | general sequence | basis plus recycle pair | continuation, time stepping, optimization |
| BiCGSTAB | general; not in SOLVAX | short recurrences | memory-limited nonsymmetric solves, but irregular residuals |
| direct structured | exact known structure | factors | repeated solves with exact band/block form |

## Preconditioning patterns

- `jacobi` for scaling problems;
- `block_jacobi` for strong within-cell coupling;
- exact banded or block-Thomas inverse of a coupling-dropped operator;
- `line_smoother` for anisotropy;
- `p_multigrid` for scale-separated elliptic error;
- mixed-precision or truncated inner solves, which flexibility permits.

See {doc}`../preconditioners` for formulas and examples.

## Failure and performance diagnostics

- `converged=False` means the true final residual missed the requested
  tolerance; increase work or improve the preconditioner.
- A small internal least-squares estimate is not used as the final truth;
  inspect `residual_norm`.
- Near linear dependence increases orthogonalization sensitivity. CGS2 reduces
  but does not eliminate finite-precision loss of orthogonality
  {cite}`giraud2005`.
- A stale recycle space remains algebraically refreshed for the current
  operator but can be ineffective. Drop it after discontinuous parameter
  changes.
- `restart`, `m`, `k`, and `max_restarts` are static compiled sizes.

## Complex systems and gradients

Complex systems use conjugate inner products. `gmres` can be supplied as the
primal and transposed solver to {func}`solvax.implicit.linear_solve`. For a
real objective of complex state, validate the adjoint convention with finite
differences; see `examples/18_complex_krylov_gradient.py`.

## API summary

- {func}`solvax.krylov.gmres`
- {func}`solvax.krylov.gmres_cycle`
- {func}`solvax.krylov.gcrot`
- {class}`solvax.krylov.KrylovSolution`

Runnable counterparts: `examples/02_advection_preconditioning.py`,
`examples/03_recycled_continuation.py`, and
`examples/18_complex_krylov_gradient.py`.
