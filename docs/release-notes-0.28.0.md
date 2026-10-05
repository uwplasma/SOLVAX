# Release 0.28.0

## Single-basis Krylov for fixed preconditioners

`gmres` and `gcrot` take `fixed_precond=True` when the preconditioner is a fixed
linear map. The flexible cycle then stores only the Arnoldi basis `V`, not
`Z = M^{-1} V`, and forms the update as `M^{-1}(V y)` with one extra
preconditioner application per cycle; this halves the basis memory. Cycles
longer than 64 also orthogonalize only against the filled basis rows instead of
the whole zero-padded basis. Both changes keep the iteration counts: on a
90,000-unknown convection-diffusion solve at restart 1,000 (8 CPU cores) every
build took 663 iterations, and the wall time fell from 211 s to 70 s and peak
memory from 2.83 GiB to 1.32 GiB. At 250,000 unknowns the filled-rows build
took 773 s and 6.4 GiB, the single-basis build 459 s and 2.6 GiB (1,238 and
1,237 iterations); 0.27.0 did not finish within 30 minutes.

## Float32 factors as a float64 Krylov preconditioner

Float32 block-Thomas or sparse-LU factors used as the fixed preconditioner of
a float64 `gcrot` solve (GMRES-IR) reach a 1e-12 residual in 9-21 iterations
on block-tridiagonal systems where float32 iterative refinement stalls. See
the preconditioner guide for the measured CPU and GPU trade-off, including the
TF32 default matmul precision that degrades float32 factors on GPUs.
