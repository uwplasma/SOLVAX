# Mixed-precision amortized adjoint

What it measures: the gradient accuracy and backward cost of
`mixed_precision_block_thomas(implicit_adjoint=True)` — float32 factors reused
(transposed) for the adjoint refinement — against the default unrolled
differentiation of the refinement loop and against the bare low-precision
gradient. See the {doc}`../solvers/mixed_precision` guide.

Reproduce:

```bash
python benchmarks/benchmark_mixed_precision_adjoint.py --json
```

Record: `benchmarks/results/mixed_precision_adjoint.json` (float32 factors,
float64 working precision, two refinement sweeps, CPU).

## Gradient accuracy across conditioning

Relative error against the gradient of the exact float64 solve. The implicit
adjoint sits at working precision across the entire dominance sweep — **the
gradient inherits the refined forward error, not the factorization
precision** {cite}`carson2018` — while the unrefined gradient carries
float32-level error:

| dominance | implicit (2 sweeps) | bare fp32 (0 sweeps) | unrolled (2 sweeps) |
|---|---|---|---|
| 6.0 | 3.56e-16 | 1.73e-7 | 3.01e-16 |
| 4.0 | 3.25e-16 | 1.19e-7 | 2.70e-16 |
| 2.0 | 2.93e-16 | 1.06e-7 | 3.21e-16 |
| 1.5 | 4.59e-16 | 1.13e-7 | 3.57e-16 |
| 1.2 | 4.52e-16 | 9.13e-8 | 4.80e-16 |

## Backward cost

Unrolled differentiation is equally accurate when the refinement converges —
the difference is cost. The custom VJP's backward is refinement sweeps on the
transposed factors, not a taped loop through a differentiated factorization:

| N, m | unrolled temp | implicit temp | unrolled warm | implicit warm |
|---|---|---|---|---|
| 64, 8 | 6.58 MiB | 0.06 MiB | 15.3 ms | 4.2 ms |
| 256, 8 | 104.3 MiB | 0.22 MiB | 257.8 ms | 37.8 ms |
| 256, 16 | 401.1 MiB | 0.59 MiB | 1274.5 ms | 31.9 ms |

At N=256, m=16 the implicit adjoint uses **680× less reverse-mode scratch and
runs 40× faster**, with roughly half the compile time.

## Platform contrast

The accuracy dichotomy is platform-independent: on the A4000 the implicit
gradient sits at 3.1e-16 with bare fp32 at 2.0e-7, identical to CPU
(`benchmarks/results/gpu/mixed_precision_adjoint.json`). The backward *cost*
advantage is platform-dependent: 238×/5.9× (memory/time) on CPU at N=128,
m=8, but 2×/1.3× on the GPU, whose executor absorbs the tape and factor-VJP
far better at this size (and compiles 32 s vs 302 s). Cost claims are
therefore stated per platform; the accuracy theorem is not.
