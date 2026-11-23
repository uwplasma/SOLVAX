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

## First CUDA run and API correction

The first remote run completed the NumPy check but all five CUDA checks
stopped during lowering: JAX 0.10 rejects the removed `pallas_call(backend=...)`
keyword. No CUDA arithmetic, derivative, or timing result was obtained.

The correction selects Triton through its existing `CompilerParams` type,
passes indexed references to Triton's masked load/store functions, and
expresses the same balanced reduction with supported equal-size `lax.split`
and `squeeze` operations instead of array slices. The loads/stores and split
signatures were checked against installed JAX 0.10 and upstream 0.7.2 source;
this is not an execution test on both versions.

A new local regression reads the actual installed API signatures without
importing JAX. It rejects unsupported keywords, wrong positional arguments,
unindexed loads/stores and ordinary sliced reductions. Both local tests pass;
the five CUDA checks remain pending. The correction changes no tile sizes,
reduction order, solver equations or derivative formulas.

## CUDA execution at `48d3698`

The second remote run passes the FP64 product and nested-JIT checks, ordinary
JVP/VJP and mapped operands, the generated solve against a dense reference,
and its parameter-gradient check. Both NumPy/API checks pass. The remaining
failure is the FP32 fallback versus a NumPy FP32 reference: its `(33,19)`
output identifies the last subcase, after the vector and empty cases complete.

NumPy rounding of the original inputs to TF32 reproduces all five printed
CUDA values within `2.384185792e-07`. The failing comparison mixed the GPU's
default product precision with the CPU reference's full FP32 arithmetic.
The correction uses a scoped `jax.default_matmul_precision("highest")`
only in that FP32 test. Its `3e-5` tolerance is unchanged; the FP64 vector/empty
checks are tightened to `3e-13`. No production source or precision default
changes, and each test subcase now has a label. The corrected FP32 test has
not yet rerun on CUDA.

| Right operand | Fixed median latency | Native median latency | Fixed/native | Fixed temporary bytes | Native temporary bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `775x775` | `1.690755002e-03 s` | `1.446174007e-03 s` | `1.169123` | 0 | 4194560 |
| `775x2` | `1.472404998e-03 s` | `1.400377994e-03 s` | `1.051434` | 0 | 4194560 |

These are three-call standalone wall-clock medians and compiled temporary
buffer estimates, not complete NTX timing or peak GPU memory. Process maximum
RSS for the entire seven-test run is `1315368 kB`; there is no comparable
native-only process measurement. Captured-row consistency across independent
compilations and both full NEOPAX AD16/FD16 comparisons remain unverified.
