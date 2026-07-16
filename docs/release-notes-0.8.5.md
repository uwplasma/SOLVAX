# Release 0.8.5

SOLVAX 0.8.5 expands reusable structured preconditioning:

- `additive_tridiagonal_line_preconditioner` builds a differentiable additive
  inverse from nonperiodic tridiagonal grid lines and an optional cyclic final
  axis. It is JIT-, gradient-, and variable-coefficient compatible. See
  {doc}`preconditioners`.
- `schur_projected_precond` accepts an optional border-border `d_block`, so the
  projected Schur construction covers general bordered systems
  `[[A, B], [C, D]]` as well as the saddle-point case. See {doc}`preconditioners`.
