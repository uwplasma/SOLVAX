# Release 0.21.0

## One principal inverse per bordered preconditioner application

`schur_projected_precond` keeps the transformed border `A^{-1} B` from its
construction and applies `x = A^{-1} r_x - (A^{-1} B) y`, so each application
calls the principal inverse once instead of twice. The preconditioner is the
same linear map. Downstream, a Krylov iteration preconditioned this way does
one coarse solve instead of two; a DKX full-Fokker-Planck field sweep measured
16.4 s to 11.8 s on CPU with identical results (#101).

## Structured direct solves

- Generated selected-head full recovery starts from the actual top block,
  removing a redundant identity-system solve and its reverse-mode work (#100).
- Checkpointed full segments drop per-step guards (#103).
- Thomas and checked-pivot sweeps unroll two rows per loop on batches of at
  least four, reducing accelerator launch overhead without changing CPU or
  narrower sweeps (#102).

## Nonlinear solvers

Newton-Krylov and pseudo-transient continuation reject nonfinite residual norms
and stopping thresholds instead of reporting false convergence, and the
Newton-Krylov PDE example adds implicit forcing calibration with re-solved
Taylor checks (#104).

## Verification

Every change landed through a pull request with the hosted test and
documentation matrix passing on `main` at the release commit. No public API
was removed or renamed.
