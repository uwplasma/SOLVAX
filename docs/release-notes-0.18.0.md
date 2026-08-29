# Release 0.18.0

## Globalized matrix-free nonlinear solves

This release adds reusable nonlinear continuation primitives for applications
whose Jacobians are too large to assemble. `pseudo_transient_continuation`
solves

$$
(M/\Delta\tau + J)\,\delta=-F
$$

with JVP-based Krylov products, optional shifted right preconditioning,
safeguarded Eisenstat--Walker forcing, true-residual Armijo backtracking, and
hard finite/admissibility gates. Arrays and PyTrees are supported, and the
fixed-size histories remain compatible with `jax.jit`.

```python
import jax.numpy as jnp
import solvax as sx


def residual(x):
    return x**3 - 2.0


solution = sx.pseudo_transient_continuation(
    residual,
    jnp.asarray([0.25]),
    config=sx.PseudoTransientConfig(rtol=1.0e-7),
)
```

`adaptive_continuation` adds host-orchestrated residual continuation with a
complete accepted/rejected stage record. `pseudo_arclength_residual` and
`pseudo_arclength_corrector` provide square bordered helpers for applications
that must follow a branch through a fold. These are deterministic root-solving
tools, not minimizers; applications should differentiate a converged equation
implicitly rather than differentiating the iteration history.

## Adaptive Newton forcing

`newton_krylov(..., forcing="eisenstat_walker")` now shares the public
`eisenstat_walker_forcing` policy with pseudo-transient continuation. Constant
forcing remains the unchanged default. On the documented 32-variable
reaction-diffusion problem, adaptive forcing used 8 Arnoldi iterations instead
of 15 while retaining four Newton updates and a terminal true residual below
`6e-11`.

## Verification and CI

The release was accepted with 750 current-stack tests (6 optional-backend
skips), 371 minimum-stack compatibility tests, 100 optional-backend tests,
Ruff, MyPy, warning-clean documentation, isolated wheel/sdist builds, and a
98% hosted line/branch coverage report. The CI suite now runs two exhaustive
current-stack shards plus focused compatibility and backend lanes; the complete
hosted evidence cycle measured 9 minutes 21 seconds, down from jobs that had
exceeded 13 minutes, without dropping test-module ownership or the 95% branch
coverage gate.
