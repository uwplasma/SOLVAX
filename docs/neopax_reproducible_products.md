# Reproducible generated Schur products

The generated block-Thomas solve uses fixed-order products at its Schur,
source and substitution contractions. CUDA FP64 uses a fixed Pallas/Triton
reduction; CPU and other dtypes retain the ordinary JAX product. No cache,
compiler flags, solver replacement or LU correction is required.

JVP, VJP and batched products use the usual bilinear derivative. Fixed ordering
addresses compilation-dependent rounding, not ill-conditioning or coefficient
accuracy. The existing block equations and factorization are unchanged.

The two NumPy/API checks run without importing JAX. Enable the four CUDA
regressions for nested compilation, derivatives, edge cases and generated solves:

```bash
SOLVAX_TEST_REPRODUCIBLE_GPU=1 python -m unittest discover \
  -s tests -p test_reproducible_products.py -v
```
