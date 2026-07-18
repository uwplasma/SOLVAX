# Sharding and communication

SOLVAX solvers run unchanged on sharded inputs: pass leaves placed with a
`NamedSharding` and the solve compiles to SPMD code on the mesh. Two design
rules make this work and are enforced by tests:

- **No flattening.** Pytree Krylov (GMRES, PCG) builds its basis leaf by leaf
  and never concatenates operands, so each leaf keeps its own sharding through
  the whole solve — solutions come back with the input's sharding.
- **Static shapes, pure collectives.** All communication a solve performs is
  whatever XLA emits for its inner products; there are no host round-trips.

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), axis_names=("i",))
b = jax.device_put(b, NamedSharding(mesh, P("i")))
solution = jax.jit(lambda rhs: sx.pcg(matvec, rhs, single_reduction=True))(b)
```

## Counting communication

Because solves are jit-compiled, their communication is inspectable: count the
collective operations (`all-reduce`, `all-gather`, ...) in the optimized HLO of
the compiled function. `benchmarks/benchmark_collectives.py` does exactly this,
and `tests/test_sharding.py` pins the results on an eight-device emulated CPU
mesh:

| case | collectives (primal) | collectives (adjoint) |
|---|---|---|
| batched tridiagonal, batch axis sharded | 0 | 0 |
| PCG | 3 | 6 |
| PCG, `single_reduction=True` | 2 | 4 |
| FGMRES via `linear_solve` | 6 | 12 |

Counts are independent of the device count — communication *volume* scales,
the number of synchronization points does not.

## The adjoint stays in the primal's communication class

The table's right column is the point: reverse-mode differentiation of a solve
costs **exactly one extra solve's worth of collectives** (the implicit
transposed solve), never a different communication pattern. Embarrassingly
parallel structured solves (batched tridiagonal lines sharded across columns)
are collective-free and *stay collective-free under* `jax.grad`, because the
adjoint of a columnwise solve is columnwise too. Deviations would indicate a
flattening or an unintended resharding — the tests fail on any such
regression.

Two measurement notes, learned the honest way: differentiate a *nonlinear*
functional of the solution (a linear loss has a constant cotangent and XLA
folds the entire adjoint solve out of the module), and pass sharded operands as
runtime arguments rather than closure constants (embedded constants may be
replicated).

## Reductions per iteration

`pcg(single_reduction=True)` rewrites the CG recurrence so the three inner
products of an iteration batch into one fused all-reduce — the measured 3→2
collective reduction above, preserved at 6→4 through the adjoint. On
high-latency meshes this halves the number of synchronization points per
iteration at identical arithmetic.

## Scope

Multi-host meshes and explicit `shard_map` layouts are exercised downstream
(SPECTRAX, DRBX) on multi-GPU hardware; the in-repo tests emulate an
eight-device mesh on CPU so every CI run checks sharding preservation and
communication counts without accelerators.
