# Release 0.8.4

SOLVAX 0.8.4 extends the implicit-differentiation and fixed-point layers:

- `linear_solve` accepts an independent `transpose_solver` and optional
  `has_aux` diagnostics, so the primal and adjoint solves can use different
  methods (or the same method with different tolerances) while preserving the
  implicit JVP and VJP behavior. See {doc}`solvers/implicit`.
- `anderson_weights` exposes the safeguarded Anderson weight solve as a
  standalone primitive, so one history's weights can be reused across
  differently shaped coupled-state blocks. See {doc}`solvers/fixed_point`.
