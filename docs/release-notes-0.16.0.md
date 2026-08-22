# Release 0.16.0

## Exact, memory-bounded recurrence derivatives

`checkpointed_fori_loop` evaluates the same static finite recurrence as
`jax.lax.fori_loop`, while its reverse pass stores segment boundaries and
replays one segment at a time:

```python
import jax
import jax.numpy as jnp
import solvax as sx


def final_energy(decay):
    def advance(step, state):
        return jnp.sin(decay * state + 1.0e-4 * step)

    initial = jnp.linspace(0.1, 0.9, 1024)
    final = sx.checkpointed_fori_loop(0, 256, advance, initial)
    return jnp.mean(final**2)


value, derivative = jax.jit(jax.value_and_grad(final_energy))(0.83)
```

For $N$ transitions and segment width $C$, retained recurrence state is
$O(N/C+C)$ rather than $O(N)$. The default $C=\lceil\sqrt{N}\rceil$ balances
the two terms; an explicit `checkpoint_size=` trades memory for recomputation.
The derivative is the exact JVP/VJP of the finite recurrence, not a continuous
adjoint. Converged steady equations should continue to use `linear_solve` or
`root_solve`, whose implicit derivative avoids recording solver iterations.

The verification suite compares primal values, forward tangents, and reverse
cotangents with a plain loop, exercises pytree state and input validation, and
requires compiled reverse temporary memory below half of a full recurrence
tape. On a 256-step, 4096-value float32 recurrence, the measured compiled
temporary memory fell from 4,227,160 to 575,272 bytes (7.35x), while the warm
value-and-gradient runtime changed from 2.78 to 4.14 ms on the audit machine.

## Traceable periodic geometry

`periodic_poisson_eigenvalues` now accepts traced scalar spacing values and
spacing tuples. This enables `jit(value_and_grad(...))` through periodic
geometry without changing the eager validation contract: nonpositive concrete
spacing raises `ValueError`, and invalid traced spacing produces a nonfinite
symbol that downstream numerical gates can reject.

## Verification

The complete test matrix covers Python 3.10 and 3.12, minimum and current JAX
stacks, the optional chunking backend, Linux and macOS, lint, typing, package
builds, solver benchmarks, and executable documentation. Combined line/branch
coverage remains above 95%.
