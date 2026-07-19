# Changelog

## Unreleased

## 0.8.7 - 2026-07-19

- Added the GPU measurement records (`benchmarks/results/gpu/`, 2x RTX A4000)
  and their docs columns: collective counts identical to the emulated-mesh
  schedule over NCCL; ideal weak scaling for single-reduction PCG (and a
  recorded non-scaling finding for the fused lax tridiagonal layout under
  sharding); the bounded adjoint flat on GPU; the physics-scale
  generated-block inversion whose N=4096, m=195 gradient runs in 33.7 MiB
  where the naive tape would exceed the 16 GB card; and the per-platform
  framing of the mixed-precision backward cost. New kinetic-inversion
  benchmark docs page.

- Added the baseline comparison benchmark (`benchmarks/benchmark_baselines.py`,
  `pip install solvax[bench]`): head-to-head against `jax.scipy`, `lineax`, and
  `scipy.sparse` on the problem suite at identical tolerance with no
  preconditioning, performance-profile ratios, and work-precision series
  including a solution-plus-gradient series through the implicit adjoint.
  SOLVAX matches the SciPy reference iteration counts exactly and runs within
  ~20% of the fastest JAX baseline at these sizes.

- `block_thomas_truncated_fn` gained a `params`/`adjoint_window` path with a
  structure-preserving custom VJP for generated blocks: the right-hand-side
  gradient is an exactly generated truncated solve of the transpose, and the
  `params` gradient pulls windowed band cotangents back through `block_fn`'s
  own derivative — forward and reverse both run at memory independent of the
  block count, with no band arrays materialized in either direction. Measured
  flat from N=32 to N=1024 (~30x less reverse scratch than the taped generated
  path at N=1024); full-window gradients are bitwise identical to the
  array-band bounded adjoint.

- Added the transport-inversion application benchmark
  (`benchmarks/benchmark_kinetic_inversion.py`): damped Newton recovers a
  collisionality profile exactly (quadratic convergence to 1.6e-14) from
  truncated low-moment observations of a spectral kinetic ladder, with the
  gradient and Hessian both flowing through
  `block_thomas_truncated(adjoint_window=w)` and validated against finite
  differences. The extended-profile misfit Hessian spectrum documents that
  the quadratic profile coefficient is unidentifiable from truncated moments,
  and the memory record separates the solve-tape savings of the bounded
  adjoint from the band-array floor of the array-band API.

- Added `nystrom_preconditioner`: a rank-`ell` randomized Nystrom
  preconditioner for SPD systems `(A + mu I) x = b`, built from `ell` operator
  applications with an explicit PRNG key — deterministic, jit-able, and
  differentiable through the sketch and eigenfactors. Bounds the
  preconditioned condition number by a small constant in expectation when the
  rank exceeds about twice the mu-effective dimension (Frangella, Tropp &
  Udell 2023); the scalable coarse correction when no grid hierarchy exists.

- Warm-started :func:`gcrot` reports `recycle_drift`, the mean principal-angle
  sine between the incoming recycle image space and its re-established span
  under the current operator — zero for an unchanged operator, growing
  linearly with the operator step, so optimization loops can monitor whether
  their step size keeps recycling effective. Joint primal+adjoint recycling
  along a continuation trajectory is demonstrated and measured in
  `benchmarks/benchmark_recycling.py`.

- Added the one-command reproduction driver (`python -m benchmarks.reproduce`):
  regenerates every committed measurement record after writing a
  hardware/software manifest and validating the timer against a known
  reference interval; `--quick` is CI-smoked. Tagged releases carry Zenodo
  metadata (`.zenodo.json`) so archives include the records.

- Benchmarks are now part of the documentation: a new Benchmarks section
  renders the committed measurement records (`benchmarks/results/*.json`) —
  bounded-memory adjoint scaling, mixed-precision adjoint accuracy and cost,
  communication accounting, and the problem-suite sweeps — each with its exact
  reproduce command and methodology notes, alongside a test-taxonomy page.

- Added the research-grade benchmark problem suite (`benchmarks/problems.py`):
  convection-diffusion (Peclet sweep), indefinite Helmholtz (wavenumber sweep),
  anisotropic diffusion (ratio sweep), Poisson, and the kinetic
  block-tridiagonal operator, each dense-verifiable; plus the sweep driver
  (`benchmarks/benchmark_sweeps.py`) recording iterations-to-tolerance,
  convergence, achieved residual, and warm wall time against the
  `jax.scipy.sparse.linalg` baselines at identical tolerance. CI smoke-runs the
  dense verification mode.

- Added a sharding and communication test suite on an eight-device emulated CPU
  mesh (`tests/test_sharding.py`), pinning sharding preservation through pytree
  Krylov solves and collective-operation counts of compiled primal and adjoint
  solves, plus `benchmarks/benchmark_collectives.py` and a sharding guide. The
  measured invariant: reverse-mode solves cost exactly one extra solve's worth
  of collectives, and sharded batched tridiagonal solves are collective-free in
  both directions.

- `mixed_precision_block_thomas` gained an opt-in `implicit_adjoint` custom VJP:
  the adjoint system is solved by the same working-precision refinement reusing
  the transposed low-precision factors — zero additional factorizations, no
  differentiation through the factorization, and the gradient inherits the
  refined forward error rather than the factorization precision.

- `block_thomas_truncated` gained an opt-in `adjoint_window` argument selecting a
  structure-preserving custom VJP: the right-hand-side gradient is the exact
  transposed truncated solve and the band gradients come from a leading
  `(keep_lowest + adjoint_window)`-block re-solve, so the *differentiated* solve
  runs at memory independent of the block count (versus the linear-in-`N` tape of
  plain reverse mode). Band-gradient error decays geometrically in the window.

## 0.8.6 - 2026-07-17

- `tridiagonal_solve` and `cyclic_tridiagonal_solve` accept complex operands:
  real bands with a complex right-hand side solve the real and imaginary parts
  independently (keeping real band storage and the fused accelerator kernel),
  while genuinely complex bands use the portable Thomas path. The fused
  primitive is wrapped in an implicit linear solve, so the `"lax"` backend is
  now forward- and reverse-differentiable.

## 0.8.5 - 2026-07-16

- Added `additive_tridiagonal_line_preconditioner` for differentiable additive
  line inverses over nonperiodic array axes and an optional cyclic final axis.
- `schur_projected_precond` accepts an optional border-border block
  `d_block`, generalizing the projected Schur preconditioner from the
  saddle-point case `[[A, B], [C, 0]]` to a general bordered matrix
  `[[A, B], [C, D]]` with `S = C A^{-1} B - D`.

## 0.8.4 - 2026-07-15

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
