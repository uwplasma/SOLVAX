# Tutorials

These tutorials combine several SOLVAX components into complete workflows.
They are intentionally small enough to run on a laptop and explicit enough to
adapt to transport, kinetic, equilibrium, and PDE applications.

```{toctree}
:maxdepth: 1

structured_system
matrix_free_pde
differentiable_solve
```

## What each tutorial teaches

| Tutorial | Main ideas |
|---|---|
| {doc}`structured_system` | storage, factor reuse, multiple RHS, residual checks, transpose solve |
| {doc}`matrix_free_pde` | operator action, structured principal-part preconditioner, FGMRES diagnostics |
| {doc}`differentiable_solve` | primal/adjoint separation, implicit VJP, finite-difference validation |

The repository also contains focused scripts in `examples/`, one per major
capability. The documentation tutorials emphasize composition and engineering
decisions; the example scripts emphasize minimal runnable demonstrations.

## Kinetic generated-block example

| Script | Device | Expected runtime | Output | Assumptions |
|---|---|---:|---|---|
| `examples/01_block_tridiagonal_kinetic.py` | CPU or GPU | seconds | shapes, dense-reference error, one gradient | nearest-mode coupling; forcing and observable restricted to modes 0--2 |

The production solve in this example generates each dense block from compact
streaming and collision coefficients. It materializes full bands only for the
small validation cross-check; callers should not copy that reference step into
production workflows.
