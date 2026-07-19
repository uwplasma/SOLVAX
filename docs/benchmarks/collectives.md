# Communication accounting

What it measures: collective operations in the compiled optimized HLO of
sharded primal and reverse-mode solves on an emulated multi-device CPU mesh.
Counts are pure compiler facts — deterministic and hardware-independent. See
the {doc}`../sharding` guide for the contracts these numbers pin.

Reproduce:

```bash
python benchmarks/benchmark_collectives.py --json
```

Record: `benchmarks/results/collectives.json` (eight emulated devices,
float64; counts identical on 2/4/8 devices).

## Collectives per compiled solve, primal vs adjoint

| case | primal | adjoint | ratio |
|---|---|---|---|
| batched tridiagonal (batch axis sharded) | 0 | 0 | — |
| PCG | 3 | 6 | 2.0 |
| PCG `single_reduction=True` | 2 | 4 | 2.0 |
| FGMRES via `linear_solve` | 6 | 12 | 2.0 |

Two facts carry the table. Embarrassingly parallel structured solves are
collective-free **and stay collective-free under `jax.grad`** — the adjoint of
a columnwise solve is columnwise. And every Krylov adjoint costs **exactly one
extra solve's worth of collectives** (ratio 2.0 = primal recompute plus one
transposed solve): reverse-mode differentiation never changes the
communication class of a solve.

The single-reduction rewrite lowers the per-solve reduction count 3→2 and its
adjoint follows 6→4. Its fused realization is compiler-dependent (current JAX
fuses; the oldest supported partitioner does not) — which is exactly why these
counts are measured per toolchain rather than asserted from the algebra.

## On real GPUs

The identical counts compile on 2x RTX A4000 over NCCL — PCG 3→6,
`single_reduction` 2→4, FGMRES 6→12, batched (Thomas) tridiagonal 0→0 — the
schedule is backend-invariant (`benchmarks/results/gpu/collectives.json`).

Weak scaling at constant per-device size
(`benchmarks/results/gpu/gpu_weak_scaling.json`): single-reduction PCG runs
1e6 unknowns/device in 31.6 ms on one GPU and 2e6 across two in 31.4 ms —
ideal. The fused `lax` tridiagonal path does **not** scale (3.9 → 7.8 ms for
2x work on 2 devices): its reshape/moveaxis layout defeats the sharding.
Under sharding, use the leaf-local Thomas path; a sharding-aware layout for
the fused path is the recorded follow-up.
