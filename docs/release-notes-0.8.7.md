# Release 0.8.7

SOLVAX 0.8.7 makes the structure-preserving adjoints complete in both storage
directions and finishes the measured evidence program:

- **Generated-block bounded adjoint**: `block_thomas_truncated_fn` accepts
  `params`/`adjoint_window`, with a custom VJP whose right-hand-side gradient
  is an exactly generated truncated solve of the transpose and whose `params`
  gradient pulls windowed band cotangents back through `block_fn`'s own
  derivative — no band arrays exist in either direction, and the reverse-mode
  footprint is measured flat from 32 to 4096 blocks (33.7 MiB at block size
  195 where the naive tape would exceed a 16 GB GPU).
- **Bounded adjoint for array bands** (`block_thomas_truncated`'s
  `adjoint_window`), the **amortized implicit adjoint** for
  `mixed_precision_block_thomas`, the **recycle-drift diagnostic** on
  warm-started `gcrot`, and the **randomized Nystrom preconditioner**.
- The benchmark program is part of the documentation (committed records, CPU
  and 2x RTX A4000 GPU columns, reproduce commands), with the sharding and
  communication test suite pinning collective-count invariants in CI, the
  research problem suite, baseline comparisons, and the one-command
  reproduction driver with timer validation.
