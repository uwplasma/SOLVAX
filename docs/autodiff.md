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

With an explicit or device-reported memory budget, the policy chooses the
largest estimated chunk fitting the chosen fraction. Without a usable device
budget, it uses $\lceil\sqrt q\rceil$, balancing chunk width and chunk count.
This is a heuristic because JAX intermediate storage depends on the traced
program, not only the final Jacobian.

## General chunked mapping

```python
ys = sx.chunk_map(expensive_function, xs, chunk_size=16)
```

`xs` may be an array or a pytree with a common leading axis. `None` performs a
single `vmap`; an integer uses batched `lax.map`, including internal handling
of a short final chunk. This helper is useful for parameter scans and batched
local physics even when no Jacobian is formed.

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
