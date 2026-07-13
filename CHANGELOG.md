# Changelog

## Unreleased

- Added a batched, differentiable cyclic-tridiagonal solve that retains the
  hardware-aware Thomas/cuSPARSE backend through an exact rank-one correction.

## 0.7.0 - 2026-07-12

- Added opt-in single-reduction PCG for sharded systems. Its algebraically
  equivalent recurrence lets XLA batch per-iteration scalar products into one
  tuple all-reduce while retaining residual diagnostics and implicit gradients.

## 0.6.1 - 2026-07-11

- Mark the distributed package as PEP 561 typed so strict downstream type
  checking analyzes SOLVAX's annotated public API.

## 0.6.0 — 2026-07-11

- Added complex-valued GMRES/GCROT with scaled unitary Givens rotations and
  Hermitian Arnoldi/recycle projections.
- Added complex fixed-point acceleration with real Aitken safeguards and the
  Hermitian Anderson residual Gram matrix.
- Restored block-Thomas linear-transpose compatibility on current JAX while
  preserving reusable factors and warm-solve performance.
- Made Jacobi preconditioners explicit PyTrees so mixed-precision wrappers cast
  stored factor state as well as runtime vectors.
- Added supported-minimum/current JAX CI rows, manual draft-PR validation, GPU
  compatibility evidence, and a complex implicit-gradient example.

## 0.5.1 — 2026-07-11

- Added `pcg_linear_solve`, which retains fixed-shape primal diagnostics while
  applying an implicit VJP with independently controlled transpose solves.

## 0.5.0 — 2026-07-11

- Added matrix-free preconditioned conjugate gradients on arbitrary JAX pytrees.
- Added fixed-shape residual histories and explicit convergence, iteration-limit,
  non-positive-curvature, nonfinite, and preconditioner-breakdown statuses.
- Added real/complex, x32/x64, JIT, scale-invariance, preconditioning, and
  implicit-gradient tests plus a cold/warm benchmark fixture.

## 0.4.0

- Added safeguarded Aitken and bounded-memory Anderson fixed-point acceleration.
