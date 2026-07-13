# Release 0.7.3

SOLVAX 0.7.3 adds `block_thomas_truncated_fn_with_residual`. The API evaluates
the retained tail-eliminated Schur residual directly from pivoted LU factors as
`L(Ux)-P b`, without reconstructing another block or materializing diagonal
bands. Multi-RHS callers may select one residual channel to limit diagnostic
overhead.

This is the minimum SOLVAX version required by NTX's bounded-memory solver
migration. On the measured NTX prepared solve, it preserves coefficient parity
through `N_xi=140`, reduces CPU/GPU temporary memory, and improves warm CPU/GPU
runtime; cold compilation remains slower and is reported separately by NTX.
