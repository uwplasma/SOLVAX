# Release 0.7.1

SOLVAX 0.7.1 adds the generated-block foundation needed by memory-bounded
kinetic solvers:

- generated truncated block elimination shares one LU solve between Schur and
  multiple right-hand-side updates;
- `block_tridiag_matvec` and `block_tridiag_relative_residual` provide an
  independent full block-row diagnostic;
- vector, multiple-RHS, `jit`, `vmap`, reverse-mode, float32, float64, low-order,
  factor-reuse, and transpose contracts are covered by tests;
- a generated kinetic example and versioned CPU/GPU benchmark artifacts are
  included;
- the unused Lineax dependency and the corresponding operator-interface
  overclaim are removed;
- repository tests explicitly import `src/solvax`, preventing an older global
  installation from satisfying CI.

On the checked-in production-shaped benchmark, CPU warm runtime changes by
`+0.36%`, within the no-regression gate. An RTX A4000 multi-RHS workload is
`5.53%` faster, while cold compilation increases from `0.53 s` to `0.65 s`.
These are hardware-specific measurements, not universal performance claims.
