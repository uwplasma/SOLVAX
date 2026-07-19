# Benchmarks

Every performance and accuracy claim in these docs is backed by a benchmark
script in `benchmarks/` whose committed JSON record lives in
`benchmarks/results/`. Each page below shows the measured tables, the exact
reproduce command, and the record it renders. CI runs the problem-suite dense
verification on every push, so the drivers cannot rot.

## Methodology

The measurements follow a fixed protocol:

- **Warm timings** are the median of repeated runs after a warm-up call, each
  closed with `jax.block_until_ready`; **cold compile time** is reported
  separately and never mixed into run time.
- **Memory** is XLA's own compiled-module accounting
  (`compiled.memory_analysis().temp_size_in_bytes`) — deterministic scratch
  requirements, not noisy runtime sampling.
- **Communication** is counted from the compiled optimized HLO: occurrences of
  `all-reduce`, `all-gather`, `reduce-scatter`, `collective-permute`, and
  `all-to-all`, with async start/done pairs counted once. Three subtleties
  make this measurement honest, each pinned by a test: the differentiated
  functional must be nonlinear (a linear loss has a constant cotangent and the
  compiler folds the adjoint solve away); sharded operands must be runtime
  arguments (closure constants may be replicated); and algebraic
  reduction-count guarantees are realized per compiler generation (the
  single-reduction fusion appears on current JAX, not the 0.4-era
  partitioner).
- **Solver comparisons** use identical tolerances, preconditioners, and
  precision across libraries, and report iterations and achieved residuals
  next to wall time wherever the API exposes them.

## Reproducing everything

One command regenerates every record from the current environment, after
writing a hardware/software manifest and validating the timer against a known
reference interval:

```bash
python -m benchmarks.reproduce            # full records into benchmarks/results/
python -m benchmarks.reproduce --quick    # reduced sizes, CI smoke, scratch dir
```

The manifest (`results/manifest.json`) pins the platform, processor, device
list, and JAX/jaxlib/solvax versions each table was measured with; the timer
validation guards the one failure mode that silently corrupts every wall-time
number. Tagged releases are archived on Zenodo with these records included.

```{toctree}
:maxdepth: 1

bounded_adjoint
mixed_precision_adjoint
collectives
sweeps
baselines
```
