# Kinetic transport inversion

What it measures: the end-to-end application the truncated adjoints exist for —
recovering a collisionality profile of a spectral kinetic ladder from
truncated low-moment observations by damped Newton, with gradient and Hessian
flowing through the bounded adjoint. See {doc}`../solvers/block_tridiagonal`.

Reproduce:

```bash
python -m benchmarks.benchmark_kinetic_inversion --json
```

Records: `benchmarks/results/kinetic_inversion.json` (CPU) and
`benchmarks/results/gpu/` (A4000; `physics_scale_a2.json` for the large-scale
generated-block run).

## Inversion (m=36, N=96)

Damped Newton converges quadratically to the exact profile — loss
`4.7e-2 → 1.6e-14` in eight steps, recovered `(nu0, a) = (1.0, 0.6)` exactly,
adjoint gradient validated against finite differences (identically on CPU and
GPU). The extended-profile misfit Hessian spectrum `{3e-10, 1e-5, 1.5e-1}`
shows the quadratic coefficient is unidentifiable from truncated moments —
the adjoint machinery diagnoses observability for free.

## Physics scale on GPU (m=195, generated blocks)

With the generated-block bounded adjoint
(`block_thomas_truncated_fn(params=..., adjoint_window=8)`) at the
drift-kinetic block size m=195:

| N_modes | gradient scratch | naive tape (estimate) |
|---|---|---|
| 256 | 33.7 MiB | 0.9 GB |
| 1024 | 33.7 MiB | 3.7 GB |
| 4096 | **33.7 MiB (flat)** | **14.9 GB — exceeds the 16 GB card** |

The largest gradient is computable *only* through the bounded adjoint on this
hardware. A further observability finding comes with it: the profile slope's
effect on the observed low moments shrinks like `1/N_modes`, so at physics
scale only the local low-mode collisionality is recoverable — Newton pins
`nu0` to 0.9972 across all sizes while the slope direction is a near-flat
valley. Truncated observations bound both the memory *and* the information.

## Three-path memory record (CPU, m=36)

| N | naive tape | array-band bounded | generated bounded |
|---|---|---|---|
| 32 | 3.0 MiB | 2.4 MiB | 1.71 MiB |
| 256 | 23.5 MiB | 9.2 MiB | 1.71 MiB |
| 512 | 46.8 MiB | 17.0 MiB | **1.71 MiB (flat)** |
