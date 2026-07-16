# Release 0.8.0

SOLVAX 0.8.0 extends the matrix-free and nonlinear solver surface so that
structured kinetic, transport, and PDE codes can drive implicit solves over
their native state without flattening it.

## Structured and distributed FGMRES

`gmres` now accepts scalar, array, and arbitrary matching-pytree operands. The
pytree path builds the flexible Arnoldi basis leaf by leaf, so it never
concatenates heterogeneous state with `ravel_pytree` and preserves leaf-level
sharding for distributed applications. An optional `inner_product(left, right)`
callback replaces the Euclidean product throughout the Arnoldi projections,
residual norms, and convergence test, which is what a distributed solve needs
to define one global reduction with JAX collectives. Callers that omit both a
pytree operand and a custom product keep the original optimized flat-array path,
and GCROT is unchanged.

## Matrix-free Newton-Krylov

`newton_krylov` solves `F(x) = 0` without materializing a Jacobian. Each Newton
correction obtains Jacobian-vector products from `jax.linearize` and is solved
by SOLVAX FGMRES, so array and pytree states, right preconditioning, custom
inner products, and an independent nonlinear norm are all available. The
nonlinear stopping test uses the true residual recomputed after the final
accepted update, and the result exposes separate nonlinear and linear
convergence flags {cite}`knoll2004`.

## Affine fixed points and filtered Anderson

`affine_fixed_point_gmres` treats an affine coupling map `G(x) = L x + c` as the
matrix-free linear system `(I - L) x = c`, for weakly contractive couplings
where fixed-point relaxation and Anderson mixing stall. `anderson_mixing` gained
optional spectral condition filtering to drop the smallest, ill-conditioned
directions from a history.

## Cyclic tridiagonal solve

`cyclic_tridiagonal_solve` solves periodic tridiagonal lines through an exact
Sherman-Morrison rank-one correction, a single ordinary tridiagonal solve with
two stacked right-hand sides {cite}`press2007`. It keeps the reproducible-Thomas
/ fused-cuSPARSE backend selection and is fully differentiable.

## Fixes

- `lu_solve_banded` promotes a real right-hand side against complex factors
  rather than truncating its imaginary part.
