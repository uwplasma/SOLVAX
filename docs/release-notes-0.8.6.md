# Release 0.8.6

SOLVAX 0.8.6 makes the tridiagonal solvers complex-capable without giving up
the fast real paths:

- `tridiagonal_solve` and `cyclic_tridiagonal_solve` accept complex operands.
  Real bands with a complex right-hand side solve the real and imaginary parts
  independently, keeping real band storage and the fused accelerator kernel;
  genuinely complex bands use the portable Thomas path, which is complex-safe
  on every supported JAX release. See {doc}`solvers/tridiagonal`.
- The fused `"lax"` backend is wrapped in an implicit linear solve, so it is
  now forward- and reverse-differentiable even on JAX versions whose primitive
  defines no differentiation rule. Gradients through `method="lax"` and
  `method="auto"` match the Thomas path.
- The cyclic Sherman--Morrison derivation is stated with the algebraic
  transpose, matching the implementation for complex corner couplings.
