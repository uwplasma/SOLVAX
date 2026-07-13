# SOLVAX

[![tests](https://github.com/uwplasma/SOLVAX/actions/workflows/tests.yml/badge.svg)](https://github.com/uwplasma/SOLVAX/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/uwplasma/SOLVAX/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/SOLVAX)
[![PyPI](https://img.shields.io/pypi/v/solvax)](https://pypi.org/project/solvax/)
[![docs](https://readthedocs.org/projects/solvax/badge/?version=latest)](https://solvax.readthedocs.io)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Differentiable structured linear solvers, preconditioners and matrix-free methods in JAX.**

`solvax` provides the solver infrastructure that kinetic and PDE codes keep
re-implementing: structured direct solves (batched dense LU, block-tridiagonal
Schur elimination with truncated storage), preconditioned and recycled Krylov
methods, physics-agnostic preconditioners (coarse-operator LU, p-multigrid,
Kronecker approximations, line smoothers), mixed-precision iterative
refinement, and implicit differentiation of every solve — all
jit/vmap/grad-transparent, on CPU and GPU.

It complements general JAX solver libraries with block-structured direct
elimination, coarse-operator and multigrid preconditioning, and Krylov
subspace recycling for parameter continuation. SOLVAX operators are native
JAX pytrees; no external operator abstraction is required.

## Install

```bash
pip install solvax
```

## Quickstart

```python
import jax.numpy as jnp
import solvax as sx

# Solve a block-tridiagonal system L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k
x = sx.block_thomas(lower, diag, upper, rhs)

# Matrix-free PCG on arrays or arbitrary JAX pytrees
solution = sx.pcg(matvec, rhs, precond=preconditioner, rtol=1e-10)
assert solution.converged

# Solve an expensive affine coupling map without assembling its Jacobian
coupled = sx.affine_fixed_point_gmres(coupling_sweep, initial_state)

# Same diagnostics, but gradients use an implicit primal/transpose solve
implicit_solution = sx.pcg_linear_solve(matvec, rhs, precond=preconditioner)

# Reuse one elimination across many right-hand sides
factors = sx.block_thomas_factor(lower, diag, upper)
x1 = sx.block_thomas_solve(factors, rhs1)
x2 = sx.block_thomas_solve(factors, rhs2)

# Generate each block once when reusable factors are needed without a stored
# diagonal band.
generated_factors = sx.block_thomas_factor_fn(block_fn, n_blocks=N)

# Memory-truncated mode: rhs nonzero only in the lowest K blocks and only the
# lowest K solution blocks needed -> O(K m^2) memory, independent of N.
x_low = sx.block_thomas_truncated(lower, diag, upper, rhs[:3], keep_lowest=3)
```

Everything is differentiable (`jax.grad` through the solve) and batchable
(`jax.vmap` over stacked systems).

## What's in the box

| Module | Contents |
|---|---|
| `solvax.operators` | Matrix-free, sum, Kronecker, block-tridiagonal and bordered (constraint-row) operator containers with closed-form transposes |
| `solvax.precond` | Jacobi/block-Jacobi, coarse-operator LU, alternating-direction line smoothers, p-multigrid V-cycles, nearest-Kronecker, mixed-precision wrappers |
| `solvax.direct` | Block-tridiagonal Schur elimination (block Thomas): full, factor/solve split, truncated-storage mode |
| `solvax.banded` | Non-pivoted banded LU with row equilibration + static pivoting; periodic variant via the Woodbury capacitance trick |
| `solvax.krylov` | Flexible restarted GMRES (CGS2 + Givens) and GCROT-style Krylov subspace recycling for parameter continuation |
| `solvax.pcg` | Matrix-free pytree PCG with preconditioning, fixed-shape residual history, and explicit convergence/breakdown status |
| `solvax.fixed_point` | Safeguarded Aitken, bounded-memory Anderson, and matrix-free affine fixed-point FGMRES |
| `solvax.implicit` | Implicit-function-theorem `linear_solve` and `root_solve` — gradients cost one extra (transposed) solve |
| `solvax.refine` | Mixed-precision iterative refinement (float32 factor, float64 residuals) |
| `solvax.native` | Host-side SuperLU bridge (non-differentiable, import-guarded) |

Complex-valued GMRES/GCROT, tridiagonal solves, and fixed-point acceleration
use Hermitian inner products and real-valued safeguards. Remaining roadmap:
harmonic-Ritz recycle selection, pytree GCROT operands, and expanded GPU
batched-LU benchmarks.

```python
# Preconditioned, recycled Krylov across a parameter scan:
sol = sx.gcrot(matvec, b, precond=coarse_inverse, m=50, k=10)
sol2 = sx.gcrot(matvec2, b2, precond=coarse_inverse, recycle=sol.recycle)

# Differentiable solve wrapping any solver:
x = sx.linear_solve(matvec, b, solver=lambda mv, rhs: sx.gmres(mv, rhs).x)
```

## License

MIT. Developed by the [UW Plasma group](https://github.com/uwplasma).
