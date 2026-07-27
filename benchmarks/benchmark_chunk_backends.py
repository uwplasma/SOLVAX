"""Native versus adv-jax-math chunking: what the optional backend buys.

Three measurements on one problem, a batched outer product whose per-slice
output is much larger than its input --- the shape where chunking matters:

1. **Chunk-size scaling.** Compiled temporary memory and wall time against
   chunk width, for both backends. This is the knob's whole purpose, and the
   two implementations should trace the same curve.

2. **In-chunk reduction.** The optional backend can fold each chunk into a
   running accumulator (``reduction=``) instead of stacking every slice and
   reducing afterwards. The native path cannot: ``jax.lax.map`` always
   materializes the stack. Where the caller only wants the reduced result,
   this removes the ``(n, *out_shape)`` array entirely.

3. **Jacobian parity.** Both backends against ``jax.jacfwd`` on the same
   function, to show that ``backend=`` selects an implementation and not an
   answer.

Run: ``python -m benchmarks.benchmark_chunk_backends``
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from solvax.autodiff import available_backends, chunk_map, chunked_jacfwd

HAS_ADV = "adv" in available_backends()


def _temp_bytes(fn, *args) -> int:
    return int(jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes)


def _time_ms(fn, *args, reps: int = 5) -> float:
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def chunk_scaling(n: int = 256, d: int = 16) -> list[dict]:
    """Memory and time against chunk width, per backend."""
    xs = jnp.linspace(0.0, 1.0, n * d).reshape(n, d)
    fun = lambda a: jnp.outer(a, a)  # noqa: E731
    rows = []
    for chunk in (1, 2, 4, 8, 16, 32, 64, 128, 256, None):
        row = {"chunk_size": chunk}
        for backend in available_backends():
            def f(v, c=chunk, b=backend):
                return chunk_map(fun, v, chunk_size=c, backend=b)

            row[backend] = {
                "temp_bytes": _temp_bytes(f, xs),
                "ms_median": _time_ms(f, xs),
            }
        rows.append(row)
    return rows


def in_chunk_reduction(n: int = 256, d: int = 16) -> dict:
    """Stack-then-reduce (native) against fold-in-chunk (adv)."""
    xs = jnp.linspace(0.0, 1.0, n * d).reshape(n, d)
    fun = lambda a: jnp.outer(a, a)  # noqa: E731
    chunk = 8

    def native_stack_then_sum(v):
        return jnp.sum(chunk_map(fun, v, chunk_size=chunk, backend="native"), axis=0)

    out = {
        "n_slices": n, "slice_out_elements": d * d, "chunk_size": chunk,
        "native_stack_then_reduce": {
            "temp_bytes": _temp_bytes(native_stack_then_sum, xs),
            "ms_median": _time_ms(native_stack_then_sum, xs),
        },
    }
    if not HAS_ADV:
        out["adv_fold_in_chunk"] = None
        return out

    def adv_fold(v):
        return chunk_map(fun, v, chunk_size=chunk, backend="adv",
                         reduction=jnp.add, chunk_reduction=lambda y: jnp.sum(y, 0))

    out["adv_fold_in_chunk"] = {
        "temp_bytes": _temp_bytes(adv_fold, xs),
        "ms_median": _time_ms(adv_fold, xs),
    }
    a = np.asarray(jax.jit(native_stack_then_sum)(xs))
    b = np.asarray(jax.jit(adv_fold)(xs))
    out["max_abs_difference"] = float(np.abs(a - b).max())
    out["memory_ratio"] = (
        out["native_stack_then_reduce"]["temp_bytes"]
        / max(out["adv_fold_in_chunk"]["temp_bytes"], 1)
    )
    return out


def jacobian_parity(n: int = 64) -> dict:
    f = lambda x: jnp.sin(x) * jnp.sum(x**2)  # noqa: E731
    x = jnp.linspace(0.1, 1.0, n)
    ref = np.asarray(jax.jacfwd(f)(x))
    out = {}
    for backend in available_backends():
        got = np.asarray(chunked_jacfwd(f, chunk_size=8, backend=backend)(x))
        out[backend] = {
            "max_abs_error_vs_jacfwd": float(np.abs(got - ref).max()),
            "ms_median": _time_ms(chunked_jacfwd(f, chunk_size=8, backend=backend), x),
        }
    return out


def main() -> None:
    result = {
        "backends_available": list(available_backends()),
        "jax": jax.__version__,
        "backend_device": jax.default_backend(),
        "chunk_scaling": chunk_scaling(),
        "in_chunk_reduction": in_chunk_reduction(),
        "jacobian_parity": jacobian_parity(),
    }
    print(json.dumps(result, indent=2))

    r = result["in_chunk_reduction"]
    print("\n--- summary ---")
    print(f"backends: {result['backends_available']}")
    if r["adv_fold_in_chunk"]:
        print(
            f"in-chunk reduction: {r['native_stack_then_reduce']['temp_bytes']} B "
            f"-> {r['adv_fold_in_chunk']['temp_bytes']} B "
            f"({r['memory_ratio']:.1f}x less), "
            f"agreeing to {result['in_chunk_reduction']['max_abs_difference']:.2e}"
        )
    else:
        print("in-chunk reduction: adv-jax-math not installed (pip install solvax[adv])")


if __name__ == "__main__":
    main()
