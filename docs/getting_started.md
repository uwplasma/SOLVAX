# Getting started

## Installation

Install the JAX-native core from PyPI:

```bash
pip install solvax
```

The optional native bridge needs SciPy:

```bash
pip install "solvax[native]"
```

Install a JAX accelerator build separately, following the JAX instructions for
the target CUDA or TPU platform. SOLVAX does not select the JAX backend.

The v0.7.2 wheel built from this tree is 54 KiB. Core dependencies are JAX and
Equinox; SciPy remains optional for the host-side native bridge. SOLVAX does
not depend on a second linear-operator package, which keeps installation and
operator ownership explicit.

## The operator contract

Most iterative SOLVAX routines accept a callable `matvec` implementing

$$
v \longmapsto A v.
$$

The matrix does not need to exist as an array:

```python
def matvec(v):
    return diffusion(v) + advection(v) + reaction * v
```

This contract supports dense arrays, stencil applications, spectral
transforms, and nested solver calls without changing the outer method. GMRES
and GCROT currently operate on flat one-dimensional arrays. PCG accepts an
arbitrary JAX pytree, provided `matvec`, `precond`, `b`, and `x0` all preserve
the same tree structure.

## Preconditioner contract

The `precond` argument is an *inverse action*:

$$
r \longmapsto M^{-1}r,
$$

not the matrix $M$. SOLVAX FGMRES uses right preconditioning,

$$
A M^{-1} y = b, \qquad x=M^{-1}y,
$$

while PCG applies a positive-definite preconditioner to the residual. A direct
SOLVAX factorization can therefore become an iterative preconditioner by
closing over its factors:

```python
factors = sx.block_thomas_factor(lower, diagonal, upper)

def precondition(r):
    shaped = r.reshape(n_blocks, block_size)
    return sx.block_thomas_solve(factors, shaped).reshape(-1)
```

See {doc}`preconditioners` for construction patterns.

## Tolerances and stopping

The linear iterative solvers use the residual criterion

$$
\lVert b-Ax_k\rVert_2
\leq \max\!\left(\mathtt{atol},\mathtt{rtol}\lVert b\rVert_2\right).
$$

Consequences:

- Use `atol > 0` when a zero or tiny right-hand side is meaningful.
- A small relative tolerance cannot overcome an inaccurate or unstable
  preconditioner.
- Always inspect `converged` and the true final residual; reaching an iteration
  limit is a valid result state, not an exception.

Fixed-point acceleration uses the true map residual
$\lVert G(x)-x\rVert_2$ and scales its relative tolerance with
$\max(\lVert x_0\rVert_2,1)$.

## Shape conventions

| Method | Principal shape convention |
|---|---|
| `block_thomas*` | `(n_blocks, block_size[, n_rhs])` |
| `tridiagonal_solve` | system dimension first; all trailing axes are batched |
| `gmres`, `gcrot` | flat `(n,)` vector |
| `pcg` | array or arbitrary matching pytree |
| `anderson_mixing` | history on axis 0 |
| `chunked_jacfwd/rev` | same output layout as the corresponding JAX transform |

For block tridiagonal storage, `lower[0]` and `upper[-1]` are unused but must be
present. For banded storage, the main diagonal is row `upper_bw`, matching the
SciPy banded convention.

## Result objects

### Krylov results

`gmres` and `gcrot` return `KrylovSolution`:

```python
solution.x
solution.residual_norm
solution.iterations
solution.converged
solution.recycle       # None for GMRES; (C, U) for GCROT
```

### PCG results

`pcg` and `pcg_linear_solve` return `PCGSolution`, adding relative residual,
status, and a fixed-shape residual history. Convert a materialized status to a
name with `status_name` outside a JAX trace.

### Fixed-point results

`aitken_fixed_point` returns the iterate, true residual norm, iteration count,
convergence flag, and final relaxation parameter.

## JAX transforms

All core routines are intended for `jit`; static algorithm sizes such as
`restart`, `m`, `k`, `max_steps`, and refinement counts determine compiled
shapes and should not vary inside a trace.

```python
solve = jax.jit(lambda rhs: sx.gmres(matvec, rhs, restart=30).x)
batched = jax.vmap(solve)(stacked_rhs)
```

Differentiating directly through a converged iterative method also
differentiates its algorithm. For gradients of the mathematical solution, use
`linear_solve`, `root_solve`, or `pcg_linear_solve`; see
{doc}`solvers/implicit`.

## Checking a solve

During development, check the residual independently:

```python
x = solution.x
absolute = jnp.linalg.norm(b - matvec(x))
relative = absolute / jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).tiny)
```

For a structured direct solver, also compare a small instance against
`jnp.linalg.solve`. A small dense reference is a test oracle, not a production
strategy.
