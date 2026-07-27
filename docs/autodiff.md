# Memory-bounded Jacobians

`jax.jacfwd` and `jax.jacrev` batch all basis directions by default. If a
single derivative evaluation has large intermediates, full batching can exceed
device memory even when the final Jacobian fits.

SOLVAX chunks the derivative basis while preserving JAX's Jacobian layout.

## Forward and reverse mode

For $f:\mathbb{R}^n\to\mathbb{R}^m$:

- forward mode evaluates Jacobian columns and is usually favored when $n$ is
  small relative to $m$;
- reverse mode evaluates Jacobian rows and is usually favored when $m$ is
  small relative to $n$.

```python
J_fwd = sx.chunked_jacfwd(f, chunk_size=8)(x)
J_rev = sx.chunked_jacrev(f, chunk_size=8)(x)
J_auto = sx.chunked_jacobian(f, mode="auto", chunk_size="auto")(x)
```

The output shape follows JAX:

$$
\operatorname{shape}(J)=
\operatorname{shape}(f(x))+\operatorname{shape}(x).
$$

`argnums` selects the differentiated positional argument.

## Chunking model

Let $q$ be the number of basis directions and $c$ the chunk size. SOLVAX uses
`vmap` inside each width-$c$ block and `jax.lax.map` across blocks. A useful
cost model is

$$
M(c)\approx M_0+cM_1,
\qquad
T(c)\approx T_0+\left\lceil\frac{q}{c}\right\rceil T_1.
$$

Larger chunks expose more parallelism and use more memory. `chunk_size=None`
is the unchunked extreme; `chunk_size=1` minimizes batched intermediate state.

## Automatic chunk size

```python
chunk = sx.auto_chunk_size(
    dim=q,
    output_size=other_dimension,
    max_memory_bytes=None,
    element_bytes=8,
    memory_fraction=0.5,
)
```

With an explicit memory budget, the policy chooses the largest estimated chunk
fitting the chosen fraction. Without one, it uses $\lceil\sqrt q\rceil$,
balancing chunk width and chunk count. The automatic default is intentionally
device-independent: reported accelerator capacity cannot account for the
larger intermediates of an arbitrary traced program. Pass a byte budget
explicitly to opt into the wider policy.

## General chunked mapping

```python
ys = sx.chunk_map(expensive_function, xs, chunk_size=16)
```

`xs` may be an array or a pytree with a common leading axis. `None` performs a
single `vmap`; an integer uses batched `lax.map`, including internal handling
of a short final chunk. This helper is useful for parameter scans and batched
local physics even when no Jacobian is formed.

## Backends

Chunking has two interchangeable implementations, selected with `backend=`:

| | `"native"` (default) | `"adv"` (optional) |
|---|---|---|
| dependency | JAX only | `adv-jax-math` |
| JAX versions | whatever SOLVAX supports | narrower; see below |
| in-chunk `reduction` | no | yes |
| explicit `shard`/`mesh` | no | yes |

The native backend is the default and always available. It is built only on
`jax.vmap` and `jax.lax.map`, so it places **no constraint on which JAX version
a downstream code may use** — a deliberate policy for a library that other
solvers depend on.

The optional backend wraps
[`adv-jax-math`](https://pypi.org/project/adv-jax-math/) and adds capability the
native path does not have. Install it with:

```bash
pip install solvax[adv]
```

Be aware that `adv-jax-math` currently requires `jax>=0.6.2` and excludes
several later releases, which is narrower than the range SOLVAX itself
supports; installing the extra therefore constrains your environment, which is
why it is opt-in.

Select per call, or set `SOLVAX_CHUNK_BACKEND` to change the default:

```python
import solvax as sx

sx.chunked_jacfwd(f, chunk_size=16)                    # native
sx.chunked_jacfwd(f, chunk_size=16, backend="adv")     # optional backend
sx.chunked_jacfwd(f, chunk_size=16, backend="auto")    # adv if importable
```

Installing the extra never changes a default: `backend="auto"` is the only
setting that prefers it, and options the native path cannot honour raise
`TypeError` rather than being silently ignored.

### In-chunk reduction

The one capability worth reaching for the extra to get. `jax.lax.map` always
materializes the stacked `(n, *out_shape)` result; when the caller only wants a
reduction of it, that array is pure waste. The optional backend folds each
chunk into an accumulator instead:

```python
import jax.numpy as jnp

# stack all 256 outer products, then sum  -> 262 232 bytes of temporaries
jnp.sum(sx.chunk_map(f, xs, chunk_size=8), axis=0)

# fold each chunk into a running sum      ->  16 024 bytes
sx.chunk_map(f, xs, chunk_size=8, backend="adv",
             reduction=jnp.add, chunk_reduction=lambda y: jnp.sum(y, 0))
```

On the benchmark in `benchmarks/benchmark_chunk_backends.py` that is a
**16.4x** reduction in compiled temporary memory, with the two results agreeing
to floating-point tolerance. See `examples/26_chunked_jacobian_backends.py` for
a runnable version.

## Numerical equivalence

Chunking changes batching, not the JVP or VJP being evaluated. Results should
match `jax.jacfwd`/`jax.jacrev` to normal floating-point tolerance. Different
batching may change reduction order inside user code, so bitwise identity is
not a portable guarantee.

## Comparison with other memory strategies

| Strategy | Saves memory by | Trade-off |
|---|---|---|
| chunked Jacobian | reducing simultaneous derivative directions | more sequential chunks |
| matrix-free JVP/VJP | never materializing the Jacobian | only operator actions are available |
| rematerialization/checkpointing | recomputing primal intermediates | extra primal work |
| finite differences | avoiding AD trace | truncation error and one solve per direction |

If the consumer is a Krylov method, prefer a matrix-free JVP over materializing
the full Jacobian. Use chunked Jacobians when the matrix itself is required by
a direct factorization, export, or dense diagnostic.

## Failure and tuning guidance

- `mode="auto"` selects by input/output sizes, not by the cost of the traced
  function. Benchmark both modes for unusual programs.
- A device memory statistic may be unavailable or may not reflect concurrent
  allocations. Pass an explicit budget for predictable jobs.
- Chunk size is static under `jit`; changing it produces a different
  compilation.
- Choose `element_bytes` consistent with the differentiated dtype.

## API summary

- {func}`solvax.autodiff.chunk_map`
- {func}`solvax.autodiff.auto_chunk_size`
- {func}`solvax.autodiff.chunked_jacfwd`
- {func}`solvax.autodiff.chunked_jacrev`
- {func}`solvax.autodiff.chunked_jacobian`

Runnable counterpart: `examples/15_chunked_jacobian.py`.
