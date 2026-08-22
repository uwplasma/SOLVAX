# Release 0.14.0

## Stationary fixed-point iteration

`fixed_point_iteration` runs the reusable algebra of a relaxed iteration while
the application supplies its map and, when needed, a physically scaled
residual norm:

```python
solution = sx.fixed_point_iteration(
    coupling_sweep,
    initial_state,
    residual_norm=physical_residual_norm,
    relaxation=0.8,
    rtol=0.0,
    atol=1e-8,
    max_steps=100,
)
```

The result uses the same `FixedPointSolution` fields as the Aitken loop:
`x`, `residual_norm`, `iterations`, `converged`, and `relaxation`. With the
default residual, the implementation evaluates the map once before the loop
and once per accepted update. With an application residual it avoids evaluating
the map until an update is required.

Tolerance-based stopping uses `jax.lax.while_loop`. Set `fixed_steps=True` only
when the finite algorithm itself must be differentiated; that path uses a
static `jax.lax.fori_loop` and always performs `max_steps` updates. For the
derivative of a converged equation, use `root_solve` around the primal solver
instead of differentiating through its iteration count.

Negative tolerances or step counts and nonpositive relaxation are rejected at
the API boundary. A finite step budget that does not meet tolerance returns
`converged=False` with the terminal residual and executed iteration count.

## Verification

Tests cover custom and default residuals, tolerance and fixed-step execution,
non-convergence, input validation, `jit`, and reverse-mode differentiation.
The release matrix covers Python 3.10 and 3.12, minimum/current/optional JAX
stacks, Linux and macOS, and reports 98.16% combined coverage.
