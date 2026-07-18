# Test taxonomy

The suite (298 parameterized cases across 18 files, ~99% line coverage,
enforced at 95% in CI) is organized so that every solver carries the same five
kinds of evidence. The categories below say what is pinned and where.

## Correctness against dense references

Every solver family is checked against an assembled dense system solved by
NumPy/SciPy: block-Thomas (full, factored, generated, truncated), banded and
periodic-banded LU, tridiagonal and cyclic-tridiagonal (real and complex,
every backend), FGMRES/GCROT, PCG, the Fourier–Helmholtz elliptic solve, and
each preconditioner's action. `tests/test_direct.py`, `test_banded.py`,
`test_tridiagonal.py`, `test_krylov.py`, `test_pcg.py`, `test_elliptic.py`,
`test_operators.py`, `test_precond.py`.

## Differentiation exactness

Reverse- and forward-mode derivatives are validated against dense analytic
gradients, finite differences, and `jax.linear_transpose` self-consistency —
including the structure-preserving custom VJPs (bounded truncated adjoint at
full window equals the taped gradient to rounding; mixed-precision implicit
adjoint matches the exact float64 gradient at working precision) and the
implicit paths (`linear_solve`, `pcg_linear_solve`, `root_solve`,
`newton_krylov`). `test_autodiff.py`, `test_implicit.py`, `test_direct.py`,
`test_mixed_precision.py`, `test_tridiagonal.py`.

## Transform transparency

Solvers are exercised under `jit`, `vmap`, and combinations, with static
algorithm sizes ensuring fixed compiled shapes; scalar, array, and pytree
operands; float32/float64 and complex64/complex128 where supported.

## Sharding and communication

On an eight-device emulated CPU mesh (every CI run): sharded solves match
single-device references; pytree Krylov preserves each leaf's named sharding;
collective counts of compiled primal and adjoint solves obey the measured
invariants ({doc}`benchmarks/collectives`). `test_sharding.py` and
`tests/conftest.py`.

## Robustness and failure reporting

Breakdown statuses (PCG non-positive curvature, preconditioner breakdown,
iteration limits), tiny-pivot clamping, ill-conditioned Anderson histories,
input validation errors, and refinement behavior as conditioning degrades.
`test_pcg.py`, `test_banded.py`, `test_fixed_point.py`,
`test_mixed_precision.py`, `test_refine.py`.

Benchmark drivers are exercised by CI too: the problem-suite dense
verification runs on every push ({doc}`benchmarks/sweeps`).
