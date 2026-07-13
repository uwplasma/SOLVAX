# Release 0.7.2

SOLVAX 0.7.2 adds `block_thomas_factor_fn`, the generated full-factor path for
repeated primal and transpose block-tridiagonal solves. Each callback block is
assembled exactly once. The factorization retains the required Schur LU and
off-diagonal state without also materializing a complete diagonal input band.

Tests cover one and multiple right-hand sides, one-block and multi-block
systems, exact transpose reuse, float32, float64, `jax.jit`, reverse-mode
differentiation, and runtime callback counts. This patch is the minimum SOLVAX
version required by NTX's generated-factor custom-VJP integration.
