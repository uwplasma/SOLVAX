# Release 0.23.0

## Recovering a sparse matrix from products with it

`solvax.compression` builds a sparse matrix from products with a matrix-free
operator, given its sparsity pattern (#111).

- `column_groups(pattern)` partitions the columns so that no two in a group
  share a row, by a greedy largest-first distance-2 colouring of the column
  intersection graph.
- `matrix_from_products(apply, pattern, groups=None)` then costs one product per
  group rather than one per column. A product with the sum of a group's unit
  vectors carries every entry of each of its columns, because each row receives
  a contribution from at most one of them (Curtis, Powell & Reid, 1974).
- `verify_products(matrix, apply)` compares the result against the operator on
  random vectors.

`sparse_operator_matrix` samples every column and so costs `n` products, which
confines it to small problems. The number of groups cannot fall below the count
of entries in the densest row, and on structured operators it lands near that
bound: a drift-kinetic operator with 633,600 unknowns and 198 entries per row is
recovered in 8,800 products, 1.4 per cent of one per column.

The pattern must be a superset of the operator's nonzeros. An entry outside it
is not merely missed: it lands in a row where another column of the same group
also contributes, and the two are summed, so entries that *are* in the pattern
come back wrong. The resulting matrix factors successfully and answers a
different question, which is why `verify_products` exists and why a caller
assembling an operator it did not write should use it.

A caller whose operator has a stencil knows its groups already and can pass
them, skipping the colouring; that cost is the pattern's nonzero count times its
row density.

## Continuous integration

- The optional-backend job installs JAX below 0.11.2 (#112). `adv-jax-math` 1.2
  imports `VJPHiPrimitive` from `jax.experimental.hijax`, which JAX 0.11.2 no
  longer provides, so the extra installed and then failed to import. The library
  keeps no upper bound on JAX, so the cap lives on that job alone.
