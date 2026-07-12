# Mathematical and implementation conventions

This page records the conventions shared across SOLVAX. Detailed derivations
and examples live on the individual solver pages.

## Design principles

1. **Operators are actions.** Iterative solvers accept `v -> A v`; they do not
   require an assembled matrix.
2. **Structure is explicit.** Direct methods accept block or band storage rather
   than discovering sparsity from a dense array.
3. **Factorization and application are separate.** Expensive factors can be
   reused for new right-hand sides, preconditioning, and adjoints.
4. **JAX shapes remain static.** Histories and Krylov bases have fixed compiled
   shapes even when convergence occurs early.
5. **Differentiation is orthogonal to the primal algorithm.** Implicit wrappers
   attach solution derivatives to caller-selected forward solvers.
6. **Breakdown is data.** Iterative methods return convergence diagnostics and,
   where applicable, explicit status codes.

## Linear systems

SOLVAX uses

$$A x=b$$

and reports the true residual $r=b-Ax$. Complex methods use the Hermitian inner
product

$$\langle x,y\rangle=x^H y.$$

The transpose in a complex implicit solve is therefore the adjoint action
provided by JAX's linear transpose machinery. Application code should make its
operator convention explicit when complex parameters are present.

## Right preconditioning

FGMRES and GCROT use right preconditioning. At iteration $j$,

$$z_j=M_j^{-1}v_j, \qquad w_j=A z_j.$$

Because each $z_j$ is stored, $M_j^{-1}$ may change between iterations. This is
why nested, truncated, mixed-precision, and nonlinear approximate solves may be
used as FGMRES preconditioners.

## Factor/solve split

The general pattern is:

```python
factors = factor(operator_data)
x1 = solve(factors, rhs1)
x2 = solve(factors, rhs2)
```

Factor objects are JAX pytrees or named tuples of arrays. Construct factors
outside repeated application loops when the operator is unchanged. Refactor
when coefficients change; reusing stale factors changes the preconditioner or
solved system.

## Static iteration storage

JAX compilation benefits from static shapes. Consequently:

- PCG residual history always has `max_steps + 1` entries; unused entries repeat
  the final residual.
- Krylov bases are allocated to the fixed restart size.
- GCROT recycle arrays retain fixed `(n, k)` shape with zero padding.
- Chunked Jacobians map fixed-width basis chunks, padding a final short chunk
  internally.

These choices bound compilation shapes and permit `jit`/`vmap`, but users
should include the allocated history and basis storage in memory estimates.

## Numerical safeguards

The structured solvers avoid explicit inverses. Banded LU uses row
equilibration and a caller-visible static pivot floor because dynamic row
pivoting maps poorly to static accelerator execution. PCG detects nonpositive
curvature and preconditioner breakdown. Aitken clips relaxation. Anderson adds
a scaled Tikhonov regularization to its history Gram matrix.

Safeguards diagnose or limit numerical damage; they do not replace a
well-posed discretization or a physically appropriate preconditioner.

## Complexity notation

- $n$: total scalar unknowns.
- $N$: number of structured blocks.
- $m$: scalar unknowns per block.
- $w_l,w_u$: lower and upper scalar bandwidths.
- $m_r$: Krylov restart length.
- $k$: recycle-space dimension or retained low-mode count, depending on context.

The method pages distinguish factorization work, repeated solve work, and
stored arrays using these symbols.
