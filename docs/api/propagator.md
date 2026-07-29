# Adaptive propagator API

`estimate_rk4_timestep` uses small Arnoldi spectral sketches and the complex
RK4 stability polynomial to choose a conservative step for a matrix-free
operator. A deterministic broadband probe supplements the caller's seed so a
recycled, nearly invariant eigenvector cannot hide peripheral modes.
`adaptive_eigenpair` wraps a caller-supplied one-restart propagator solve with
original-operator residual stopping and a numerical growth-defect guard.

The propagator action remains application-specific: SOLVAX owns the stability
and certification policy, while the caller owns the time-stepper and projected
eigenmode extraction. This keeps the policy reusable without assuming a
particular state layout or materializing the operator.

```{eval-rst}
.. automodule:: solvax.propagator
   :members:
   :member-order: bysource
```
