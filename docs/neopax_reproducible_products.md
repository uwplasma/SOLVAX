# Fixed-order products for the NTX Schur solve

This branch is a candidate correction for the captured NTX forward split.
Independent compilations of the same arithmetic graph selected different
FP64 cuBLAS implementations. With all 48 prepared arrays and both scan input
rows byte-identical, stored D13 differed by `2.531143490e-03`. Disabling
autotuning, or sharing its decisions, removed the split for all 880 captured
coefficients. A shared cache is a diagnostic control, not this correction.

The candidate replaces four product sites in
`_block_thomas_selected_fn_state`: tail Schur, head Schur, head source and
upward substitution. On CUDA, FP64 products use one Pallas/Triton kernel with
a fixed tile shape and reduction order. It uses ordinary FP64 multiplication
and addition, without a GEMM autotuning choice, atomic accumulation or split-K
reduction. It changes neither the block equations, LU factorization, source
support, retained modes nor the selected-head solver algorithm. CPU products
and other dtypes retain the ordinary JAX product.

The product primitive uses the ordinary bilinear derivative:
`d(AB) = dA B + A dB`, with transpose products for reverse mode. Those products
use the same primitive inside nested JITs, scans and mapped cotangents. No
compiler options are added to nested JITs, and no XLA flags or cache are required.

Local validation executes the actual tiled kernel body with NumPy-backed
masked loads/stores. Five shapes, including partial tiles and partial inner
blocks, agree with an extended-precision reference to `3e-13` absolute/relative
tolerance. This is an arithmetic/indexing check, not a CUDA compiler test.

Run the bounded CUDA checks with caching disabled:

```bash
env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
  JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  SOLVAX_TEST_REPRODUCIBLE_GPU=1 \
  "$HOME/jax_env/bin/python" -m unittest discover -s tests \
  -p test_reproducible_products.py -v
```

The five CUDA tests cover nested compilation, products, JVP/VJP, mapped
operands, vector and empty inputs, a generated solve against a dense reference,
its parameter derivative, and timing/buffer reports for `775x775` and `775x2`
products. They have **not run locally**. A fresh-process replay of the captured
176-point NTX row through both original entries must then agree with caching
disabled. Full AD16/FD16, total solver timing, RSS and device peak memory remain
unverified. Fixed ordering alone is not an independent coefficient-accuracy
certificate, and this candidate must not be promoted on that claim.
