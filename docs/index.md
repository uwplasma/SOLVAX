# SOLVAX

**Differentiable structured solvers, preconditioners, and matrix-free methods
for JAX.**

SOLVAX is the numerical-solver layer that kinetic, transport, equilibrium, and
PDE codes often reimplement independently. It provides structured direct
factorizations, matrix-free Krylov methods, fixed-point acceleration,
preconditioners, mixed-precision refinement, bounded-memory Jacobians, and
implicit differentiation. With the exception of the explicitly host-side
SuperLU bridge, the library is designed to compose with `jax.jit`, `jax.vmap`,
and `jax.grad`.

This documentation is organized around decisions rather than modules:

- {doc}`getting_started` establishes the operator, tolerance, shape, and result
  conventions used everywhere.
- {doc}`choosing` maps mathematical structure to a solver and preconditioner.
- The solver guides derive each algorithm, document every input and output,
  describe failure modes, and compare the method with common alternatives.
- The tutorials build complete structured, matrix-free, and differentiable
  workflows.
- {doc}`api` is the generated signature-level reference.

## A first solve

```python
import jax.numpy as jnp
import solvax as sx

A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
b = jnp.array([1.0, 2.0])

solution = sx.pcg(lambda x: A @ x, b, rtol=1e-10)
if not solution.converged:
    raise RuntimeError(sx.status_name(solution.status))

print(solution.x)
print(solution.residual_norm)
```

For a nonsymmetric operator, use FGMRES:

```python
solution = sx.gmres(lambda x: A @ x, b, restart=20, rtol=1e-10)
```

For a block-tridiagonal operator, do not discard the structure:

```python
factors = sx.block_thomas_factor(lower, diagonal, upper)
x = sx.block_thomas_solve(factors, rhs)
```

## Documentation map

```{toctree}
:maxdepth: 2
:caption: Start here

getting_started
choosing
methods
```

```{toctree}
:maxdepth: 2
:caption: Structured direct solvers

solvers/block_tridiagonal
solvers/banded
solvers/tridiagonal
```

```{toctree}
:maxdepth: 2
:caption: Iterative and nonlinear solvers

solvers/krylov
solvers/pcg
solvers/fixed_point
```

```{toctree}
:maxdepth: 2
:caption: Solver infrastructure

operators
preconditioners
solvers/implicit
solvers/mixed_precision
autodiff
solvers/native
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
release-notes-0.7.2
release-notes-0.7.1
```

## Literature

SOLVAX follows established numerical linear algebra rather than introducing
new convergence theory. Each method page states the algorithmic variant used
by the implementation and cites the relevant literature. The full bibliography
is collected below.

```{bibliography}
```
