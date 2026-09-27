# Release 0.27.0

## Dependency floors

SOLVAX now declares `equinox>=0.13.3`, `jax>=0.4.38`, `jaxlib>=0.4.38` and
`numpy>=1.24`. Previously it declared bare `equinox` and `jax`, so installing
or upgrading SOLVAX into an environment that already held an older equinox kept
it; with jax >= 0.10 that equinox fails on import
(`jax.interpreters.batching.NotMapped`), and equinox 0.13.1-0.13.2 fail at
trace time under jax 0.11 (`jax.core.mapped_aval`). equinox 0.13.3 is the first
release that imports and traces under jax 0.9.2 through 0.11.2. The minimum CI
lane installs exactly these floors, and a test keeps the two in step.

## Also in this release

- Host-side iterative refinement for sparse-direct solves
  (`HostFactorOptions(refine_steps=k)`) and a traced `sparse_backward_error`.
- Fewer operator applications in flat-array `gmres` and `gcrot`.
- Complex operators stay complex in `matrix_from_products`, `verify_products`,
  `iterative_refinement` and `as_low_precision`.
- `column_groups` builds the column-intersection graph with one sparse product.
- The host factorization cache is keyed on the pattern's structure.
