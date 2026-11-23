"""Fixed-order FP64 products for sensitive Schur elimination on CUDA.

The CUDA lowering owns the reduction order instead of letting independent
cuBLAS autotuning runs choose different orders. CPU and non-FP64 arithmetic
retain the normal JAX matrix product. No compiler flags or cache are needed.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.extend import core as extend_core
from jax.interpreters import ad, batching, mlir, xla


def _cuda_matmul(a, b):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    # Pallas moved masked loads/stores into its Triton namespace after 0.7.
    load = getattr(plt, "load", None) or pl.load
    store = getattr(plt, "store", None) or pl.store
    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk = 16, 16, 8

    def kernel(a_ref, b_ref, out_ref):
        rows = pl.program_id(0) * bm + jnp.arange(bm)
        cols = pl.program_id(1) * bn + jnp.arange(bn)
        inner = jnp.arange(bk)

        def step(i, total):
            kk = i * bk + inner
            left = load(a_ref, (rows[:, None], kk[None, :]),
                        mask=(rows[:, None] < m) & (kk[None, :] < k), other=0.)
            right = load(b_ref, (kk[:, None], cols[None, :]),
                         mask=(kk[:, None] < k) & (cols[None, :] < n), other=0.)
            products = left[:, :, None] * right[None, :, :]
            # Explicit balanced additions: no dot, split-K or atomic reduction.
            for width in (4, 2, 1):
                products = products[:, :width, :] + products[:, width:2*width, :]
            return total + products[:, 0, :]

        total = jax.lax.fori_loop(0, (k + bk - 1) // bk, step,
                                  jnp.zeros((bm, bn), dtype=a.dtype))
        store(out_ref, (rows[:, None], cols[None, :]), total,
              mask=(rows[:, None] < m) & (cols[None, :] < n))

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=((m + bm - 1) // bm, (n + bn - 1) // bn),
        backend="triton",
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
        name="solvax_fixed_order_f64_matmul",
    )(a, b)


def _shape(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("fixed-order product requires compatible rank-two matrices")
    if a.dtype != np.dtype("float64") or b.dtype != a.dtype:
        raise TypeError("fixed-order product requires two float64 operands")
    return core.ShapedArray((a.shape[0], b.shape[1]), a.dtype)


_product = extend_core.Primitive("solvax_fixed_order_product")
_product.def_impl(partial(xla.apply_primitive, _product))
_product.def_abstract_eval(_shape)
mlir.register_lowering(_product, mlir.lower_fun(lambda a, b: a @ b, multiple_results=False))
mlir.register_lowering(_product, mlir.lower_fun(_cuda_matmul, multiple_results=False), platform="cuda")

# The ordinary bilinear derivative; both the primal and its transposes use
# the same primitive, so an outer jit/scan/AD transform cannot retune them.
ad.defbilinear(_product,
              lambda bar, a, b: _product.bind(bar, b.T),
              lambda bar, a, b: _product.bind(a.T, bar))


def _batch(args, axes):
    mapped = [(x, axis) for x, axis in zip(args, axes) if axis is not None]
    size = mapped[0][0].shape[mapped[0][1]]
    arrays = [jnp.broadcast_to(x, (size, *x.shape)) if axis is None
              else jnp.moveaxis(x, axis, 0) for x, axis in zip(args, axes)]
    return jax.lax.map(lambda pair: _product.bind(*pair), tuple(arrays)), 0


batching.primitive_batchers[_product] = _batch


def schur_product(a, b):
    """Matrix/source product with a fixed CUDA FP64 accumulation order."""
    if a.dtype != np.dtype("float64") or b.dtype != a.dtype or a.ndim != 2 or b.ndim not in (1, 2):
        return a @ b
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    if a.shape[1] != b.shape[0]:
        raise ValueError("matrix product contraction dimensions differ")
    if not all((*a.shape, *b.shape)):
        result = jnp.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
    else:
        result = _product.bind(a, b)
    return result[:, 0] if vector else result
