# Changelog

## 0.8.4 - 2026-07-14

- Extended `linear_solve` with an independent `transpose_solver` and optional
  `has_aux` diagnostics while preserving implicit JVP and VJP behavior.
- Exposed safeguarded Anderson weights for reuse across differently shaped
  coupled-state histories.

## 0.8.3 - 2026-07-14

- Added `additive_preconditioner`, a positive weighted combination of
  inverse actions for symmetry-preserving additive line, block, and Schwarz
  preconditioning on arrays or arbitrary PyTrees.

## 0.8.2 - 2026-07-14

- Added `galerkin_deflation`, a balanced symmetry-preserving Galerkin coarse
  correction for fixed SPD preconditioners used with PCG.

## 0.8.1 - 2026-07-13

- Added `solvax.elliptic`: a spectral Fourier--Helmholtz elliptic solve for
  separable Helmholtz-type problems on a periodic axis and a bounded axis
  (`build_fourier_helmholtz_operator`, `solve_fourier_helmholtz`,
  `FourierHelmholtzOperator`). Fourier-transforms the periodic axis and solves
  the remaining per-mode tridiagonal system in the bounded axis; `jit`/`grad`/
  `vmap` transparent. This is the `lap phi = rhs` inversion used by reduced
  drift-plane / vorticity models.

## 0.8.0 - 2026-07-13

- Extended FGMRES beyond flat arrays: `gmres` now solves scalar, array, and
  arbitrary matching-pytree operands through a leaf-wise Arnoldi basis (no
  `ravel_pytree`, preserving leaf-level sharding), and accepts an optional
  `inner_product` callback for weighted or mesh-wide (distributed) products.
  The optimized flat-array and GCROT paths are unchanged.
- Added `newton_krylov`, a matrix-free Jacobian-free Newton-Krylov (JFNK) root
  solver. Jacobian-vector products come from `jax.linearize`; each correction
  is solved by SOLVAX FGMRES. It supports array or pytree states, right
  preconditioning, custom inner products, an independent nonlinear norm, and
  reports separate nonlinear and linear convergence flags.
- Added `affine_fixed_point_gmres`, which solves an affine fixed-point map
  `G(x)=Lx+c` as the matrix-free system `(I-L)x=c`, and gave `anderson_mixing`
  optional spectral condition filtering of ill-conditioned histories.
- Added a batched, differentiable cyclic-tridiagonal solve that retains the
  hardware-aware Thomas/cuSPARSE backend through an exact rank-one
  (Sherman-Morrison) correction.
- `lu_solve_banded` now promotes a real right-hand side against complex factors
  instead of silently truncating the imaginary part.

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
