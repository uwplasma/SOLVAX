# Release 0.8.8

## Checked tridiagonal preconditioning

- `tridiagonal_solve_checked` reports per-column modified-pivot and backward
  residual diagnostics.
- Unsafe coefficient columns can fall back to the identity preconditioner,
  emit non-finite values for an outer guard, or remain unchanged for
  diagnostics-only use.
- The existing `tridiagonal_solve` API and its default numerical behavior are
  unchanged.
