# Baseline comparisons

What it measures: SOLVAX against `jax.scipy.sparse.linalg`, `lineax`, and
`scipy.sparse.linalg` on the research problem families at identical tolerance
with **no preconditioning anywhere** — identical knobs isolate the solver
implementations (preconditioned SOLVAX numbers live in {doc}`sweeps`).

Reproduce (`lineax` via `pip install solvax[bench]`; its rows skip if absent):

```bash
python -m benchmarks.benchmark_baselines --json
```

Record: `benchmarks/results/baselines.json` (rtol 1e-8, float64, CPU; SciPy
runs on NumPy arrays on the same host, so iterations are the primary
cross-library metric and its wall times carry that caveat).

## Headline (grid-32 families, rtol 1e-8)

- **Iteration parity with the reference**: SOLVAX's PCG and FGMRES take
  exactly the SciPy iteration counts on every SPD and nonsymmetric point
  (e.g. 18/18 and 70/70 on anisotropic diffusion, 19/19 and 20/20 on
  Helmholtz) — same mathematics, verified head-to-head. `lineax` GMRES
  reports outer restart cycles rather than inner iterations, and its CG stops
  on a different criterion, so its counts are not directly comparable.
- **Time**: median time-to-best ratios over all problems — `jax.scipy` 1.00,
  **SOLVAX 1.19**, `lineax` 1.96, `scipy` 6.49. At these small unpreconditioned
  sizes the minimal-machinery baseline is fastest; SOLVAX's ~20% overhead buys
  iteration diagnostics, convergence flags, preconditioning hooks, recycling,
  and the structured/bounded adjoints no baseline offers. Honest losses
  included: `jax.scipy` wins most raw-time comparisons here.

## Work-precision

The record includes achieved-residual-versus-time series across rtol
{1e-4 … 1e-10} for all four solvers on three representative problems, and a
SOLVAX-only **solution-plus-gradient** series (one fused jit through the
implicit adjoint at each tolerance) — the integrated differentiable cost that
baselines would hand-roll as two separate solves.
