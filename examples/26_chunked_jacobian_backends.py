"""Chunked Jacobians: choosing a backend, and what the optional one adds.

The chunk-size knob trades peak memory for a modest slowdown when assembling a
dense Jacobian. SOLVAX ships two implementations of it. Run this file to see
both, and to see the one thing the optional backend can do that the built-in
one cannot.

    python examples/26_chunked_jacobian_backends.py
"""

import jax
import jax.numpy as jnp

from solvax.autodiff import (
    available_backends,
    chunk_map,
    chunked_jacfwd,
    default_backend,
)


def main() -> None:
    print(f"available backends : {available_backends()}")
    print(f"default backend    : {default_backend()}")
    print()

    # ------------------------------------------------------------------
    # 1. The knob. Narrower chunks hold less at once and take a little longer.
    # ------------------------------------------------------------------
    def residual(p):
        return jnp.sin(p) * jnp.sum(p**2) + jnp.cos(p[::-1])

    p = jnp.linspace(0.1, 1.0, 128)
    reference = jax.jacfwd(residual)(p)

    print("chunk_size   temp bytes   max error vs jax.jacfwd")
    for chunk in (4, 16, 64, None):
        jac_fn = chunked_jacfwd(residual, chunk_size=chunk)
        temp = jax.jit(jac_fn).lower(p).compile().memory_analysis().temp_size_in_bytes
        err = float(jnp.abs(jac_fn(p) - reference).max())
        label = "None (=jacfwd)" if chunk is None else str(chunk)
        print(f"{label:>14}  {temp:>10}   {err:.2e}")

    # ------------------------------------------------------------------
    # 2. Backends are interchangeable: same answer, your choice of engine.
    # ------------------------------------------------------------------
    print()
    for backend in available_backends():
        jac = chunked_jacfwd(residual, chunk_size=16, backend=backend)(p)
        print(f"backend={backend:<7} max error vs jax.jacfwd: "
              f"{float(jnp.abs(jac - reference).max()):.2e}")

    # ------------------------------------------------------------------
    # 3. What the optional backend adds: fold each chunk into an accumulator
    #    instead of stacking every slice and reducing afterwards. When only the
    #    reduced result is wanted, the (n, *out_shape) stack never exists.
    # ------------------------------------------------------------------
    print()
    if "adv" not in available_backends():
        print("adv-jax-math is not installed; install with:")
        print("    pip install solvax[adv]")
        print("to enable in-chunk reduction and explicit sharding.")
        return

    slices = jnp.linspace(0.0, 1.0, 256 * 16).reshape(256, 16)
    per_slice = lambda a: jnp.outer(a, a)  # noqa: E731

    def stack_then_reduce(v):
        return jnp.sum(chunk_map(per_slice, v, chunk_size=8, backend="native"), axis=0)

    def fold_in_chunk(v):
        return chunk_map(per_slice, v, chunk_size=8, backend="adv",
                         reduction=jnp.add, chunk_reduction=lambda y: jnp.sum(y, 0))

    def temp(fn):
        return jax.jit(fn).lower(slices).compile().memory_analysis().temp_size_in_bytes

    a, b = temp(stack_then_reduce), temp(fold_in_chunk)
    agree = float(jnp.abs(jax.jit(stack_then_reduce)(slices)
                          - jax.jit(fold_in_chunk)(slices)).max())
    print(f"stack then reduce (native) : {a:>8} bytes")
    print(f"fold in chunk     (adv)    : {b:>8} bytes   -> {a / b:.1f}x less")
    print(f"the two agree to           : {agree:.2e}")

    # An option the native path cannot honour is refused, not ignored.
    try:
        chunk_map(per_slice, slices, chunk_size=8, backend="native", reduction=jnp.add)
    except TypeError as exc:
        print(f"\nnative backend refuses adv-only options:\n  {exc}")


if __name__ == "__main__":
    main()
