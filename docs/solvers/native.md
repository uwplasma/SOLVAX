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

## Sparse shift-invert eigenpairs

For a sparse nonsymmetric operator whose rightmost mode is difficult to
discover from an unshifted transient, sample JAX operator columns in bounded
batches and reuse one shifted LU for both right and adjoint modes:

```python
import scipy.sparse as sp
import solvax as sx

matrix = sx.sparse_operator_matrix(
    apply, prototype, batch_size=64, drop_tolerance=1e-14
)
shift = 0.2 - 0.4j
factor = sx.SpluFactorization(matrix - shift * sp.eye(matrix.shape[0]))
right = sx.sparse_eigenpairs(matrix, shift=shift, factorization=factor)
left = sx.sparse_eigenpairs(
    matrix, shift=shift, factorization=factor, adjoint=True
)
```

The assembly never holds the full dense matrix, and the adjoint uses the
conjugate-transpose solve of the same factors. The bridge itself is eager, but
the converged pair can be supplied to `eigenpair_reverse`; derivatives then
come from the implicit eigenpair equations rather than from SciPy or the LU
iteration tape. Always certify a dropped sparse approximation against the
original application operator.

## API summary

- {class}`solvax.native.SpluFactorization`
- {func}`solvax.native.splu_solve`
- {func}`solvax.native_eigen.sparse_operator_matrix`
- {func}`solvax.native_eigen.sparse_eigenpairs`

Runnable counterpart: `examples/13_native_splu.py`.
