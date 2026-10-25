# Embedded explicit Runge-Kutta API

`erk_step` advances one explicit Runge-Kutta step of any `ERKPair` and returns
the embedded local-error estimate with the new state. Three tableaux ship:
`RK4_2` (classical RK4 with the explicit-midpoint rule as the embedded
solution, so the estimate costs no extra right-hand-side evaluation),
`BS3_2` (Bogacki-Shampine 3(2), first-same-as-last) and `DOPRI5_4`
(Dormand-Prince 5(4), first-same-as-last).

`error_norm` is a weighted RMS norm for pytrees. `step_factor` maps the
current and previous error norms to a step multiplier with a `PIController`
(`k_p = 0` gives an I controller). `erk_integrate` combines them into an
accept/reject loop under `jax.lax.while_loop`. All functions are pure and
trace under `jit` and `scan`, so a caller with its own scan, projection or
diagnostics can use `erk_step` and `step_factor` directly.

```{eval-rst}
.. automodule:: solvax.ode
   :members:
   :member-order: bysource
```
