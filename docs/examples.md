# Examples by application

Start with the problem in the left column. Every script is self-contained,
prints a residual or comparison, and runs from a source checkout:

```bash
python examples/17_pcg.py
```

## Find an example

| Application or task | SOLVAX method | Script |
|---|---|---|
| Low-order moments of a large kinetic hierarchy | truncated block Thomas | `01_block_tridiagonal_kinetic.py` |
| Periodic upwind/advection system | periodic banded LU preconditioner | `02_advection_preconditioning.py` |
| Parameter continuation | GCROT recycle space | `03_recycled_continuation.py` |
| Repeated narrow-band solves | banded factor/solve split | `04_banded_lu.py` |
| Multiple drives and an adjoint | block-Thomas reuse and transpose | `05_block_thomas_factor_solve.py` |
| Sensitivity of a steady solution | `linear_solve` and `root_solve` | `06_implicit_differentiation.py` |
| Diagonal or local-block physics | Jacobi and block Jacobi | `07_jacobi_preconditioners.py` |
| Strong grid-aligned anisotropy | alternating line smoother | `08_line_smoother.py` |
| Spectral/polynomial hierarchy | p-multigrid V-cycle | `09_p_multigrid.py` |
| Fast low-precision factorization | iterative refinement | `10_mixed_precision_refinement.py` |
| Approximately separable dimensions | nearest-Kronecker preconditioning | `11_kronecker.py` |
| Matrix-free sums and constraints | operators and Schur projection | `12_operators.py` |
| Sparse CPU fallback | native SuperLU bridge | `13_native_splu.py` |
| Many independent 1-D columns | batched tridiagonal solve | `14_tridiagonal_solve.py` |
| Large optimization Jacobian | chunked forward/reverse autodiff | `15_chunked_jacobian.py` |
| Low-precision block hardware | mixed-precision block Thomas | `16_mixed_precision_block_thomas.py` |
| SPD arrays or nested parameter trees | pytree PCG and implicit gradient | `17_pcg.py` |
| Frequency-domain/complex system | complex Krylov solve and gradient | `18_complex_krylov_gradient.py` |
| Partitioned multiphysics coupling | Aitken and Anderson | `19_fixed_point_acceleration.py` |
| Batched expensive evaluations | memory-bounded `chunk_map` | `20_chunk_map.py` |
| Very long structured hierarchy | on-demand truncated blocks | `21_on_demand_block_assembly.py` |
| Interior stability modes and their sensitivities | harmonic Krylov--Schur | `22_interior_eigenvalues.py` |

The scripts live in the repository's
[`examples/` directory](https://github.com/uwplasma/SOLVAX/tree/main/examples).

## Four short starting points

### Symmetric positive-definite PDE operator

```python
import jax.numpy as jnp
import solvax as sx

A = jnp.array([[2.0, -1.0], [-1.0, 2.0]])
b = jnp.array([1.0, 0.0])
result = sx.pcg(lambda x: A @ x, b, rtol=1e-6)
assert result.converged
```

**Input:** an operator action and a matching array or pytree right-hand side.
**Output:** `PCGSolution`, including solution, status, iteration count, and
residual history.

### Nonsymmetric transport operator

```python
A = jnp.array([[3.0, -1.0], [2.0, 4.0]])
b = jnp.array([1.0, 2.0])
result = sx.gmres(lambda x: A @ x, b, restart=10, rtol=1e-6)
assert result.converged
```

**Input:** a flat-vector operator, right-hand side, and optional inverse-action
preconditioner. **Output:** `KrylovSolution`, including an optional recycle
space for `gcrot`.

### Coupled block hierarchy

```python
eye = jnp.eye(2)
lower = jnp.stack([jnp.zeros((2, 2)), -0.2 * eye, -0.2 * eye])
diagonal = jnp.stack([2.0 * eye, 2.0 * eye, 2.0 * eye])
upper = jnp.stack([-0.2 * eye, -0.2 * eye, jnp.zeros((2, 2))])
rhs = jnp.ones((3, 2))
x = sx.block_thomas(lower, diagonal, upper, rhs)
```

**Input:** lower, diagonal, and upper block arrays plus one or several
right-hand sides. **Output:** an array with the same block and right-hand-side
layout. Use the factor/solve split when the operator is reused.

### Partitioned nonlinear coupling

```python
mapping = lambda x: 0.8 * x + 1.0
result = sx.aitken_fixed_point(mapping, jnp.array([0.0]), rtol=1e-6)
assert result.converged
```

**Input:** one complete coupling sweep and an initial state. **Output:** the
accelerated fixed point, true map residual, iteration count, and relaxation.

## Coverage by feature

Every public capability has at least one runnable example. Result container
classes are exercised by the examples for their corresponding solver.

| Feature family | Public names | Examples |
|---|---|---|
| Banded | `banded_matvec`, factor/solve, periodic variants | 02, 04 |
| Block direct | full, factor/solve, truncated, on-demand, mixed precision | 01, 05, 16, 21 |
| Tridiagonal | `tridiagonal_solve` | 14 |
| Fixed point | `aitken_relaxation`, `aitken_fixed_point`, `anderson_mixing` | 19 |
| Krylov | `gmres`, `gcrot`, recycle output | 02, 03, 18 |
| Eigenvalues | `harmonic_krylov_schur`, `block_harmonic_krylov`, `eigenvalue`, `eigenpair` | 22 |
| PCG | `pcg`, `pcg_linear_solve`, diagnostics, `status_name` | 17 |
| Implicit solves | `linear_solve`, `root_solve` | 06 |
| Operators | matrix-free, sum, Kronecker, block-tridiagonal, bordered, Schur | 11, 12 |
| Preconditioners | Jacobi, coarse, line, p-multigrid, Kronecker, precision | 07–12 |
| Refinement | `as_low_precision`, `iterative_refinement` | 10 |
| Autodiff memory | `chunk_map`, `auto_chunk_size`, all `chunked_jac*` variants | 15, 20 |
| Native | `SpluFactorization`, `splu_solve` | 13 |

For assumptions and failure modes, continue to {doc}`choosing`. For larger
composed workflows, continue to {doc}`tutorials/index`.
