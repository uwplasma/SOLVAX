# Native SuperLU bridge

The native bridge solves general SciPy sparse matrices with SuperLU on the host
CPU. It is an explicit escape hatch for systems that do not fit SOLVAX's JAX
structured or matrix-free methods.

Install the optional dependency:

```bash
pip install "solvax[native]"
```

## Factor once

```python
import scipy.sparse as sp

A = sp.csr_matrix(...)
factorization = sx.SpluFactorization(A)
x1 = factorization.solve(b1)
x2 = factorization.solve(b2)
```

## One-shot solve

```python
x = sx.splu_solve(A, b)
```

## Execution model

The sparse matrix and solve execute through SciPy/SuperLU, outside the JAX
trace. Returned values are converted to JAX arrays for convenience, but the
operation is not:

- JIT compilable;
- vectorizable with `jax.vmap`;
- differentiable with `jax.grad`;
- accelerator resident.

Runtime guards raise a clear error if traced values are passed. Do not hide the
bridge inside a jitted outer function.

## When to use it

- a general sparse CPU system needs robust pivoted LU;
- factorization reuse is important;
- the solve is outside optimization/adjoint traces;
- a structured JAX solver is unavailable or insufficiently robust.

## Comparison with JAX-native methods

| Property | SuperLU bridge | FGMRES | structured direct |
|---|---|---|---|
| matrix representation | SciPy sparse | callable | bands/blocks |
| pivoting | sparse pivoted LU | not applicable | method dependent |
| accelerator | no | yes | yes |
| `jit`/`vmap`/`grad` | no | yes | yes |
| repeated RHS | excellent after factorization | repeated iteration | excellent after factorization |

Sparse LU fill-in can dominate memory even when the input matrix is sparse.
For large PDEs, a matrix-free Krylov method with a structured preconditioner may
scale better. For small-to-moderate difficult CPU systems, SuperLU is often the
more robust engineering choice.

## API summary

- {class}`solvax.native.SpluFactorization`
- {func}`solvax.native.splu_solve`

Runnable counterpart: `examples/13_native_splu.py`.
