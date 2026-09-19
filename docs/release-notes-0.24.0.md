# Release 0.24.0

## Scaling a matrix before it is factored

`solvax.equilibration` applies Ruiz's algorithm: scale rows and columns
alternately by the square root of their largest entry, until every row and
column maximum is near one (#114).

- `equilibrate(matrix)` returns an `Equilibration` carrying the scaled matrix
  and both diagonals, with `scale_rhs` and `unscale_solution`, so `A x = b` is
  solved as `D_r A D_c y = D_r b` with `x = D_c y`.
- Convergence is linear, so the default runs up to thirty sweeps and stops early
  once the maxima are within a per cent of one. The sweeps are cheap next to the
  factorization they prepare.

A factorization chooses its pivots from the matrix it is handed, so rows
spanning many orders of magnitude make it choose badly. Solvers that pivot
completely scale internally; a static-pivoting one does not, and returns a
factorization of a different matrix, which refinement cannot repair.

Measured on a drift-kinetic operator of 66,004 unknowns whose rows carry
streaming, collision and constraint terms at once, factored with MKL PARDISO on
four cores: as assembled, 201 s and a relative residual of 9.4e-2 that an outer
defect correction drove to 117; equilibrated first, 103 s and 1.2e-10 after
refinement. For reference SuperLU takes 2,732 s at 19.1 GiB on the same matrix
and host, and MUMPS, which scales and matches by default, about 40 s.
