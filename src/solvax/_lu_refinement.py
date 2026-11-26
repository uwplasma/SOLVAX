"""Experimental one-step LU application correction; not enabled by direct.py.

CUDA FP64 residuals use two-word triangular products. The factors and pivots
are reused. Implicit linear differentiation avoids taping the correction.
CPU residual products retain ordinary arithmetic; other dtypes use lu_solve.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.extend import core as extend_core
from jax.interpreters import batching, mlir, xla
from jax.scipy.linalg import lu_solve

from solvax._reproducible import schur_product


def _cuda_triangular_product(a, hi, lo, *, lower, transpose):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    m, n = hi.shape
    bm, bn, bk = 8, 16, 8

    def two_sum(x, y):
        # Explicit rounded instructions preserve the error-free transform.
        return plt.elementwise_inline_asm(
            '{ .reg .f64 z, t, v; add.rn.f64 $0, $2, $3; '
            'sub.rn.f64 z, $0, $2; sub.rn.f64 t, $0, z; '
            'sub.rn.f64 t, $2, t; sub.rn.f64 v, $3, z; add.rn.f64 $1, t, v; }',
            args=[x, y], constraints='=&d,=&d,d,d', pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)]*2)

    def product(x, y, low):
        return plt.elementwise_inline_asm(
            'mul.rn.f64 $0, $2, $3; fma.rn.f64 $1, $2, $3, -$0; '
            'fma.rn.f64 $1, $2, $4, $1;',
            args=[x, y, low], constraints='=&d,=&d,d,d,d', pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)]*2)

    def kernel(a_ref, hi_ref, lo_ref, out_hi, out_lo):
        rows = pl.program_id(0)*bm+jnp.arange(bm)
        cols = pl.program_id(1)*bn+jnp.arange(bn)
        inner = jnp.arange(bk)

        def step(i, carry):
            total, error = carry
            kk = i*bk+inner
            indices = (kk[None, :], rows[:, None]) if transpose else (rows[:, None], kk[None, :])
            left = plt.load(a_ref.at[indices],
                            mask=(rows[:, None] < m) & (kk[None, :] < m), other=0.)
            strict = kk[None, :] < rows[:, None] if lower != transpose else kk[None, :] > rows[:, None]
            diagonal = 1. if lower else left
            left = jnp.where(kk[None, :] == rows[:, None], diagonal, jnp.where(strict, left, 0.))
            high = plt.load(hi_ref.at[kk[:, None], cols[None, :]],
                            mask=(kk[:, None] < m) & (cols[None, :] < n), other=0.)
            low = plt.load(lo_ref.at[kk[:, None], cols[None, :]],
                           mask=(kk[:, None] < m) & (cols[None, :] < n), other=0.)
            left, high, low = jnp.broadcast_arrays(left[:, :, None], high[None, :, :], low[None, :, :])
            ph, pe = product(left, high, low)
            for width in (4, 2, 1):
                ah, bh = jax.lax.split(ph, (width, width), axis=1)
                ae, be = jax.lax.split(pe, (width, width), axis=1)
                ph, roundoff = two_sum(ah, bh)
                pe = roundoff+ae+be
            ph, pe = jnp.squeeze(ph, axis=1), jnp.squeeze(pe, axis=1)
            summed, roundoff = two_sum(total, ph)
            return tuple(two_sum(summed, error+pe+roundoff))

        zero = jnp.zeros((bm, bn), dtype=a.dtype)
        high, low = jax.lax.fori_loop(0, (m+bk-1)//bk, step, (zero, zero))
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        plt.store(out_hi.at[rows[:, None], cols[None, :]], high, mask=mask)
        plt.store(out_lo.at[rows[:, None], cols[None, :]], low, mask=mask)

    shape = jax.ShapeDtypeStruct(hi.shape, hi.dtype)
    return pl.pallas_call(kernel, out_shape=(shape, shape),
                          grid=((m+bm-1)//bm, (n+bn-1)//bn),
                          compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
                          name='solvax_two_word_triangular_product')(a, hi, lo)


def _ordinary_product(a, hi, lo, *, lower, transpose):
    triangle = jnp.tril(a, -1)+jnp.eye(a.shape[0], dtype=a.dtype) if lower else jnp.triu(a)
    if transpose:
        triangle = triangle.T
    return triangle@hi, triangle@lo


def _shape(a, hi, lo, *, lower, transpose):
    del lower, transpose
    if a.ndim != 2 or a.shape[0] != a.shape[1] or hi.ndim != 2 or hi.shape != lo.shape or hi.shape[0] != a.shape[0]:
        raise ValueError('Require square factors and compatible two-word matrix RHS')
    if any(value.dtype != np.dtype('float64') for value in (a, hi, lo)):
        raise TypeError('Two-word triangular products require float64')
    return (core.ShapedArray(hi.shape, hi.dtype),)*2


_product = extend_core.Primitive('solvax_two_word_triangular_product')
_product.multiple_results = True
_product.def_impl(partial(xla.apply_primitive, _product))
_product.def_abstract_eval(_shape)
mlir.register_lowering(_product, mlir.lower_fun(_ordinary_product, multiple_results=True))
mlir.register_lowering(_product, mlir.lower_fun(_cuda_triangular_product, multiple_results=True), platform='cuda')


def _batch(args, axes, **params):
    size = next(x.shape[axis] for x, axis in zip(args, axes) if axis is not None)
    arrays = [jnp.broadcast_to(x, (size, *x.shape)) if axis is None
              else jnp.moveaxis(x, axis, 0) for x, axis in zip(args, axes)]
    return jax.lax.map(lambda values: _product.bind(*values, **params), tuple(arrays)), (0, 0)


batching.primitive_batchers[_product] = _batch


def _correct_once(lu, piv, perm, inverse, rhs, trans):
    x = lu_solve((lu, piv), rhs, trans=trans)
    zero = jnp.zeros_like(x)
    if trans == 0:
        high, low = _product.bind(lu, x, zero, lower=False, transpose=False)
        high, low = _product.bind(lu, high, low, lower=True, transpose=False)
        residual = ((rhs[perm]-high)-low)[inverse]
    else:
        high, low = _product.bind(lu, x[perm], zero, lower=True, transpose=True)
        high, low = _product.bind(lu, high, low, lower=False, transpose=True)
        residual = (rhs-high)-low
    return x+lu_solve((lu, piv), residual, trans=trans)


def refined_lu_solve(factors, rhs, trans=0):
    """Apply existing real FP64 LU factors with exactly one residual correction.

    Differentiates the factored equation, using the same correction in forward
    and transpose solves. This function performs no LU factorization.
    """
    lu, piv = factors
    if lu.dtype != np.dtype('float64') or rhs.dtype != lu.dtype:
        return lu_solve(factors, rhs, trans=trans)
    if trans not in (0, 1, 2):
        raise ValueError('trans must be 0, 1 or 2')
    trans = int(trans != 0)  # Real conjugate transpose equals transpose.
    vector = rhs.ndim == 1
    rhs = rhs[:, None] if vector else rhs
    if lu.ndim != 2 or rhs.ndim != 2 or lu.shape[0] != lu.shape[1] or rhs.shape[0] != lu.shape[0]:
        raise ValueError('Require one square LU factor and a vector/matrix RHS')
    if not all(rhs.shape):
        return jnp.zeros_like(rhs[:, 0] if vector else rhs)
    perm = jax.lax.linalg.lu_pivots_to_permutation(piv, lu.shape[0])
    inverse = jnp.argsort(perm)
    lower, upper = jnp.tril(lu, -1), jnp.triu(lu)

    def action(value):
        if trans == 0:
            z = schur_product(upper, value)
            return (z+schur_product(lower, z))[inverse]
        z = value[perm]
        return schur_product(upper.T, z+schur_product(lower.T, z))

    result = jax.lax.custom_linear_solve(action, rhs,
        solve=lambda _, b: _correct_once(lu, piv, perm, inverse, b, trans),
        transpose_solve=lambda _, b: _correct_once(lu, piv, perm, inverse, b, 1-trans))
    return result[:, 0] if vector else result
