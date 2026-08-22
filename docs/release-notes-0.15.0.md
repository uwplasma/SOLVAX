# Release 0.15.0

## Fully periodic Poisson inversion

SOLVAX now solves

$$
-\nabla^2 u=f
$$

on an $N$-dimensional fully periodic grid with an explicit zero-mode contract:

```python
import jax.numpy as jnp
import solvax as sx

x = 2.0 * jnp.pi * jnp.arange(32) / 32
rhs = jnp.sin(x[:, None]) * jnp.sin(2.0 * x[None, :])
solution = sx.solve_periodic_poisson(rhs, spacing=(2.0 * jnp.pi / 32,) * 2)
```

`solve_periodic_poisson` transforms a physical right-hand side and returns the
physical solution. Time integrators that already own the FFT can build
`periodic_poisson_eigenvalues` once and call
`solve_periodic_poisson_spectral` at every stage. Both paths project the
right-hand-side mean, support a selectable solution mean, preserve real or
complex layouts, and compose with `jax.jit` and reverse-mode differentiation.

## Verification

Manufactured one-, two-, and three-dimensional modes verify the discrete
symbol and inversion. Additional gates cover nullspace projection, prescribed
means, shape and dtype contracts, reusable transforms, JIT, and gradients. The
complete suite remains above 95% combined line/branch coverage on the supported
minimum, current, optional-backend, Linux, and macOS matrix.
