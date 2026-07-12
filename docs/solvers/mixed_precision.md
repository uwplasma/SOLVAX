# Mixed-precision iterative refinement

Mixed precision uses a fast approximate low-precision solve and recovers
working-precision accuracy with residual correction.

## Defect-correction derivation

Let $\widetilde A^{-1}$ denote the low-precision approximate solve. Starting
from $x_0=\widetilde A^{-1}b$, repeat

$$
r_i=b-Ax_i,
$$

$$
d_i=\widetilde A^{-1}r_i,
$$

$$
x_{i+1}=x_i+d_i.
$$

The residual is accumulated in the requested high `residual_dtype`; only the
correction solve is low precision. In a classical analysis, refinement
converges when the low-precision solve is sufficiently accurate, commonly
summarized by

$$
\kappa(A)u_{low}<1,
$$

with refinements depending on the precision combination and residual quality
{cite}`carson2018,higham2002`.

## Generic refinement

```python
x, residual_history = sx.iterative_refinement(
    matvec,
    b,
    approx_solve,
    iterations=3,
    residual_dtype=jnp.float64,
)
```

Inputs:

- `matvec`: working-precision operator action;
- `b`: working-precision right-hand side;
- `approx_solve`: callable applying the approximate inverse;
- `iterations`: fixed number of correction sweeps;
- `residual_dtype`: dtype used to form and norm residuals.

The returned history records the residual norm after the initial approximate
solve and subsequent corrections. Treat stagnation or growth as evidence that
the approximate inverse is too inaccurate.

## Converting a solver to low precision

```python
solve32 = sx.as_low_precision(jnp.linalg.solve, dtype=jnp.float32)
approx_solve = lambda r: solve32(A, r)
```

`as_low_precision` casts floating/complex array arguments to the requested
dtype, runs the callable, and casts floating results back to the original
working dtype. Integer and nonarray arguments are preserved. The factorization
itself must actually occur inside the wrapped function to gain low-precision
throughput.

## Block-tridiagonal convenience path

```python
x = sx.mixed_precision_block_thomas(
    lower,
    diag,
    upper,
    rhs,
    factor_dtype=jnp.float32,
    refine_steps=2,
)
```

This factors Schur complements in float32, applies block-Thomas corrections in
that precision, and computes block-tridiagonal residuals in working precision.

## Low-precision preconditioning

```python
precond32 = sx.mixed_precision(precond64, dtype=jnp.float32)
solution = sx.gmres(matvec, b, precond=precond32)
```

FGMRES is designed for changing or inexact preconditioners and is the safest
outer method for this pattern. A low-precision preconditioner inside PCG must
still be positive definite in the arithmetic actually used.

## When it helps

- hardware has much higher float32 than float64 factorization throughput;
- factor/application time dominates the solve;
- the operator is moderately conditioned;
- high-precision residuals are affordable;
- a few correction sweeps suffice.

## When it does not

- $\kappa(A)u_{low}$ is too large;
- low precision overflows or underflows scaled coefficients;
- transfers and casts dominate small problems;
- the backend does not provide a faster low-precision kernel;
- half precision is requested for LU operations without supported kernels.

## Comparison with alternatives

Mixed precision changes arithmetic cost, not the underlying mathematical
preconditioner. Increasing Krylov work can handle conditioning that defeats
refinement, while equilibration or nondimensionalization can make refinement
viable. Full float64 factorization remains the robust reference.

## API summary

- {func}`solvax.refine.iterative_refinement`
- {func}`solvax.refine.as_low_precision`
- {func}`solvax.precond.mixed_precision`
- {func}`solvax.direct.mixed_precision_block_thomas`

Runnable counterparts: `examples/10_mixed_precision_refinement.py` and
`examples/16_mixed_precision_block_thomas.py`.
