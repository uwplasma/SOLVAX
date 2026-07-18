# Problem-suite robustness sweeps

What it measures: iterations-to-tolerance, convergence, achieved residual, and
warm wall time across the research problem families in
`benchmarks/problems.py`, each swept over the parameter that makes it hard,
next to the `jax.scipy.sparse.linalg` baseline at identical tolerance and
preconditioner. Every family is verified against a dense reference in CI
(`--verify`).

Reproduce:

```bash
python -m benchmarks.benchmark_sweeps --json     # full sweep
python -m benchmarks.benchmark_sweeps --verify   # dense verification (CI)
```

Record: `benchmarks/results/sweeps.json` (grid 32 unless swept, rtol 1e-8,
Jacobi preconditioning, float64, CPU).

## Iterations to tolerance (all 16 points converge)

| family | sweep | iterations | behavior |
|---|---|---|---|
| Poisson | grid 16 → 64 | 8 → 44 | expected mesh dependence with point Jacobi |
| convection–diffusion | Pe 0.1 → 100 | 145 → 35 | upwinding adds dominance at high Péclet — the literature's non-monotone curve |
| Helmholtz | k 1 → 20 | ≈19 flat | into the indefinite regime at this size |
| anisotropic diffusion | ε 1 → 10⁻³ | 18 → 70 → 20 | hardening at ε=10⁻², then grid alignment |

The baselines (`jax.scipy` `cg`/`gmres`) do not expose iteration counts; their
achieved residuals and warm times are recorded in the JSON. Adapters with full
iteration parity (lineax, PETSc) and the Dolan–Moré profile aggregation are
the next stage of the benchmark program, together with line/additive
preconditioning of the anisotropic family — the sweep above uses plain Jacobi
everywhere precisely so the problem hardness, not the preconditioner, is what
varies.
