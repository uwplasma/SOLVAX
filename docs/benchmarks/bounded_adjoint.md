# Bounded-memory truncated adjoint

What it measures: the reverse-mode memory of `block_thomas_truncated` with and
without the structure-preserving custom VJP (`adjoint_window`), and the
geometric convergence of the windowed band gradient. See the
{doc}`../solvers/block_tridiagonal` guide for the construction.

Reproduce:

```bash
python benchmarks/benchmark_bounded_adjoint.py --json
```

Record: `benchmarks/results/bounded_adjoint.json` (m=4, keep_lowest=2, window
4, block-diagonally-dominant, float64, CPU).

## Reverse-mode scratch memory vs block count

XLA compiled `temp_size_in_bytes` for `jax.grad` of a loss through the
truncated solve. The plain tape grows linearly with the block count `N`; the
windowed custom VJP is flat — the differentiated solve becomes block-count
independent, like the forward solve.

| N | naive tape | `adjoint_window=4` | ratio |
|---|---|---|---|
| 16 | 16.3 KiB | 8.7 KiB | 1.9× |
| 32 | 28.5 KiB | 8.7 KiB | 3.3× |
| 64 | 53.0 KiB | 8.7 KiB | 6.1× |
| 128 | 102.0 KiB | 8.7 KiB | 11.7× |
| 256 | 200.0 KiB | 8.7 KiB | 23.0× |
| 512 | 396.0 KiB | 8.7 KiB | 45.6× |
| 1024 | 788.0 KiB | 8.7 KiB | 90.7× |

## Band-gradient error vs window

The right-hand-side gradient is exact at every window; the band gradient
converges geometrically at the proved $O(\rho^{2w})$ rate
{cite}`demko1984,benzi2013`:

| window `w` | relative error |
|---|---|
| 0 | 2.08e-3 |
| 2 | 3.14e-8 |
| 4 | 7.91e-14 |
| 6 | 2.46e-16 |
| 8 | 2.46e-16 |

## On GPU

The same flat-versus-linear scaling compiles on the A4000: naive 33→802 KiB
across N=16→1024, bounded flat at ~32 KiB
(`benchmarks/results/gpu/bounded_adjoint.json`), with the windowed decay
reaching 7.1e-24 by w=8.
