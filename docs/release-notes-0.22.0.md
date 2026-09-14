# Release 0.22.0

## Block Thomas with operator couplings

`block_thomas_factor_ops` and `block_thomas_solve_ops` solve block-tridiagonal
systems whose off-diagonal blocks are linear operators rather than dense bands,
supplied as `couple(params, k, Z, *, which, transpose)` (#106). A shared sparse
stencil such as `a_k S + b_k diag(mu)` is the intended case.

- The factorization runs the Schur recurrence of `block_thomas_factor` but forms
  `U_k Delta_{k+1}^{-1} L_{k+1}` by applying `U_k` to a matrix. It materializes
  at most one `m x m` block per step and stores only the Schur factors, a third
  of the stored-band factors.
- The solve is two carry-threaded scans over block indices. It supports
  transposed solves, `jit`, `vmap`, `jax.linear_transpose` and reverse mode
  without copying the factors, and coupling coefficients travel in `params` as
  pytree data.
- `factor_dtype` puts the Schur factors in lower precision under
  working-precision substitution, and `store="inverse"` keeps `Delta_k^{-1}` so
  each solve step is a matrix product instead of two triangular solves (#107).
- `docs/solvers/block_tridiagonal.md` documents the contract and the storage
  options, and `benchmarks/benchmark_operator_couplings.py` compares the routes
  (#108).

On a kinetic-shaped stencil system with `m = 777`, 16 blocks and a batch of 2 in
float64, on four pinned cores of a shared Intel Xeon W-2295 with one BLAS
thread:

| Route | Apply per block | Stored factors | Solve temporaries |
|---|---:|---:|---:|
| `block_thomas_factor` | 5.47 ms | 464 MB | 445 MB |
| `block_thomas_factor_ops`, `store="lu"` | 5.60 ms | 155 MB | 10.0 MB |
| `block_thomas_factor_ops`, `store="inverse"` | 0.90 ms | 155 MB | 10.0 MB |

All three agree to $3 \times 10^{-15}$; the checked-in results under
`benchmarks/results/` record versions and the source commit. On DKX's NCSX
coarse-preconditioner blocks, explicit inverses matched the LU map to
$3 \times 10^{-13}$ and gave identical GCROT iteration counts down to a
thousandth of the deck's collisionality (uwplasma/DKX#230).

## Verification

Every change landed through a pull request with the hosted test and
documentation matrix passing on `main` at the release commit. No public API was
removed or renamed.
