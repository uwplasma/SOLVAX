# Time stepping API

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

`imex_step` advances one additive Runge-Kutta step of `y' = f_exp + f_imp`.
The caller supplies `solve(t, r, gamma_h)`, which returns `z` with
`z - gamma_h f_imp(t, z) = r`, so the stiff part can be a diagonal in Fourier
space, a fast-diagonal or multigrid solve, or a preconditioned `gmres`. Five
tableaux ship: `IMEX_EULER`, `IMEX_SSP2` (Pareschi-Russo), `ARS_343`, `ARK3_KC`
(Kennedy-Carpenter, with an embedded pair) and `SMR_RK3_CN` (the
Spalart-Moser-Rogers RK3 with Crank-Nicolson substeps).

`lawson_step` runs any `ERKPair` as a Lawson (integrating-factor) method for
`y' = L y + f(t, y)`, given `propagate(x, tau) = exp(tau L) x`. It is exact when
`f` vanishes and keeps the pair's order and embedded estimate. `ssp_step` runs a
two-register strong-stability-preserving scheme: `SSPRK3` (Shu-Osher),
`SSPRK43` and `SSPRK104` (Ketcheson), with SSP coefficients 1, 2 and 6.

`cfl_dt` turns a signal-speed bound into a step. `controlled` wraps any
one-step map into the contract of `integrate`: a fixed step, a CFL step, or PI
control from an embedded estimate. `integrate` runs `n_steps` attempts as a
`lax.scan` over checkpointed chunks. It rejects a non-finite state and retries
at half the step, stops after `max_retries` consecutive rejections with
`ok = False` and the last good state, stops at `t1`, and can print one flushed
progress line per chunk from inside the compiled program. It differentiates in
reverse mode with memory of one carry per chunk plus one chunk.

These steppers are not built on diffrax. diffrax's IMEX solvers solve each
implicit stage with its own Newton and lineax solve and take no user solve for
`I - gamma h L`, which is how the plasma codes use fast-diagonal, multigrid and
preconditioned Krylov solves; it has no Lawson or SSP low-storage schemes; and
its chunked progress and non-finite stop are not exposed. exponax has ETDRK
methods, tied to its own Fourier domain and a fixed step.

```{eval-rst}
.. automodule:: solvax.ode
   :members:
   :member-order: bysource
```
