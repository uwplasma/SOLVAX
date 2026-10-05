# Release 0.28.1

## Warm-started GCROT keeps O(n k) memory

The recycle drift diagnostic computed on every warm start (`gcrot(recycle=...)`)
ran an SVD of the n-by-k residual of the old recycle columns, which a backend
served with an n-by-n workspace: a 113 GB allocation on a 13 x 19, `Nxi = 48`
DKX deck. It now takes a thin QR of that residual and the SVD of the k-by-k
factor, which has the same singular values. A test walks the traced solve and
checks that every SVD operand is at most `2 (m + k)` on a side.

## Documentation

`fixed_precond=True` (0.28.0) forms `M^{-1}(V y)` once per cycle, which
amplifies roundoff by the preconditioner gain; the docstrings now advise keeping
the stored basis for nearly singular preconditioners.
