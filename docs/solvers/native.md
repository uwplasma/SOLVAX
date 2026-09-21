# Native sparse-direct bridge

The native bridge solves general SciPy sparse matrices on the host CPU. SuperLU
remains the default. MUMPS is an explicit optional backend for callers that need
symbolic memory admission before numerical factorization.

Install the optional dependency:

```bash
pip install "solvax[native]"
```

For MUMPS, first provide compatible MUMPS and MPI libraries, then install the
Python binding and SciPy integration:

```bash
pip install "solvax[mumps]"
```

PyMUMPS, mpi4py, and the MUMPS libraries must use a compatible MPI ABI. SOLVAX
does not install or configure those system libraries.

The MUMPS adapter always uses one process through `MPI.COMM_SELF`; it does not
expose parallel MUMPS execution or distributed matrix input.

## Factor once

```python
import scipy.sparse as sp

A = sp.csr_matrix(...)
factorization = sx.SpluFactorization(A)
x1 = factorization.solve(b1)
x2 = factorization.solve(b2)
```

MUMPS selection and its memory budget are explicit:

```python
with sx.SpluFactorization(
    A,
    backend="mumps",
    memory_limit_bytes=8_000_000_000,
    memory_safety_factor=1.2,
) as factorization:
    x = factorization.solve(b)
    xt = factorization.solve(b_many, trans="T")
    xh = factorization.solve(b_many, trans="H")
```

The MUMPS adapter runs symbolic analysis first and reads `INFOG(16)`, the
maximum estimated in-core working memory. In this single-rank adapter, that is
the estimate for the sole participating process. It multiplies that estimate by
`memory_safety_factor` and refuses numerical factorization when the result
exceeds `memory_limit_bytes`. It also gives MUMPS the budget through
`ICNTL(23)`, which caps MUMPS internal integer and real or complex workspace.
The [official MUMPS users' guide](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
defines `INFOG(16)` and `INFOG(21)` in decimal megabytes (millions of bytes).
The negative-value convention used by some entry-count fields does not apply to
these memory fields. After factorization,
`effective_memory_bytes` exposes `INFOG(21)` and
`symbolic_memory_bytes` exposes the unpadded `INFOG(16)` estimate.

This budget is not a process-RSS limit. It excludes the input SciPy matrix, the
COO conversion retained by PyMUMPS, Python and JAX objects, MPI runtime storage,
and memory used by symbolic analysis before `ICNTL(23)` takes effect. Leave
headroom for those allocations outside `memory_limit_bytes`.

The adapter uses the public PyMUMPS context API with `MPI.COMM_SELF`:
`set_centralized_sparse`, `run(job=1/2/3)`, `set_rhs`, `set_icntl`,
`get_infog`, and `destroy`.

Both backends accept vector or matrix right-hand sides and `trans="N"`, `"T"`,
or `"H"`. PyMUMPS exposes a vector right-hand side through its public API, so
the adapter solves matrix columns in sequence while reusing one factorization.
Use the context manager or call `close()` to release native MUMPS storage
promptly. Input-validation errors leave the factorization usable; an exception
from a native MUMPS solve closes it because the binding does not guarantee that
the context remains reusable after such a failure.

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

| Property | SuperLU/MUMPS bridge | FGMRES | structured direct |
|---|---|---|---|
| matrix representation | SciPy sparse | callable | bands/blocks |
| pivoting | sparse pivoted LU | not applicable | method dependent |
| accelerator | no | yes | yes |
| `jit`/`vmap`/`grad` | no | yes | yes |
| repeated RHS | factorization reused | repeated iteration | factorization reused |

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
