# solvax

**Differentiable structured linear solvers, preconditioners and matrix-free
methods in JAX.**

`solvax` is the solver layer that kinetic and PDE codes keep re-implementing,
factored out once: structured direct solves (batched dense LU, block-tridiagonal
Schur elimination with truncated storage, banded and periodic-banded LU,
hardware-aware batched tridiagonal), preconditioned and recycled Krylov methods,
physics-agnostic preconditioners (coarse-operator LU, p-multigrid, Kronecker
approximations, line smoothers), mixed-precision iterative refinement,
memory-chunked autodiff, and implicit differentiation of every solve — all
jit/vmap/grad-transparent on CPU and GPU.

It builds on [lineax](https://github.com/patrick-kidger/lineax)'s operator
interface and adds the block-structured direct elimination,
coarse-operator/multigrid preconditioning, Krylov recycling, and chunked
Jacobian layer that lineax does not cover.

## Install

```bash
pip install solvax            # core
pip install solvax[native]    # + SciPy SuperLU host bridge
```

## Quickstart

```python
import solvax as sx

# Block-tridiagonal system: L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k
x = sx.block_thomas(lower, diag, upper, rhs)

# Reuse one elimination across right-hand sides (and the transposed/adjoint solve)
factors = sx.block_thomas_factor(lower, diag, upper)
x1 = sx.block_thomas_solve(factors, rhs1)
xT = sx.block_thomas_solve(factors, rhs1, transpose=True)

# Preconditioned, recycled Krylov across a parameter scan
sol = sx.gcrot(matvec, b, precond=coarse_inverse, m=50, k=10)
sol2 = sx.gcrot(matvec2, b2, precond=coarse_inverse, recycle=sol.recycle)

# Differentiable solve wrapping any black-box solver
x = sx.linear_solve(matvec, b, solver=lambda mv, rhs: sx.gmres(mv, rhs).x)

# Batched tridiagonal solve (Thomas on CPU, cuSPARSE on GPU) over many columns
x = sx.tridiagonal_solve(lower, diag, upper, rhs)

# Memory-chunked Jacobian (the jac_chunk_size knob)
J = sx.chunked_jacrev(residual, chunk_size="auto")(theta)
```

Everything is differentiable (`jax.grad` through the solve) and batchable
(`jax.vmap` over stacked systems).

## What's in the box

| Module | Contents |
|---|---|
| {mod}`solvax.operators` | Matrix-free, sum, Kronecker, block-tridiagonal and bordered (constraint-row) operator containers with closed-form transposes |
| {mod}`solvax.direct` | Block-tridiagonal Schur elimination (block Thomas): full, factor/solve split, truncated-storage and mixed-precision variants |
| {mod}`solvax.banded` | Non-pivoted banded LU with row equilibration + static pivoting; periodic variant via the Woodbury capacitance trick |
| {mod}`solvax.tridiagonal` | Backend-aware batched tridiagonal solve: bit-reproducible Thomas on CPU, fused cuSPARSE kernel on GPU, many columns/fields at once |
| {mod}`solvax.krylov` | Flexible restarted GMRES (CGS2 + Givens) and GCROT-style Krylov subspace recycling for parameter continuation |
| {mod}`solvax.fixed_point` | Safeguarded Aitken and bounded-memory Anderson acceleration |
| {mod}`solvax.precond` | Jacobi/block-Jacobi, coarse-operator LU, line smoothers, p-multigrid V-cycles, nearest-Kronecker, mixed-precision wrappers |
| {mod}`solvax.implicit` | Implicit-function-theorem `linear_solve` and `root_solve` — gradients cost one extra (transposed) solve |
| {mod}`solvax.autodiff` | Memory-chunked forward/reverse Jacobians (`jac_chunk_size`) and the `auto` sizing policy |
| {mod}`solvax.refine` | Mixed-precision iterative refinement (float32 factor, float64 residuals) |
| {mod}`solvax.native` | Host-side SuperLU bridge (non-differentiable, import-guarded) |

The {doc}`methods` page documents every capability — the equations, the source,
the inputs/outputs and the use case; the {doc}`api` renders the full signatures.
Runnable, pedagogic scripts (one per capability) live in the `examples/`
directory of the repository.

```{toctree}
:maxdepth: 2

methods
api
```

## References

```{bibliography}
```
