---
orphan: true
---

# Experimental one-step LU application correction

The NEOPAX saved-point audit locates a raw-coefficient derivative error in
applying existing LU factors. JAX 0.7.2 and 0.10.0 reproduce the same error
under the same CUDA libraries. RHS-column batching does not provide a
reliable accuracy correction. The user approved testing one residual
correction on 2026-11-25 and requested forward/AD timing measurements.

`solvax._lu_refinement.refined_lu_solve` is an experimental helper, not an
enabled change to `direct.py`. It performs exactly:

1. Apply the existing LU factors to the RHS.
2. Evaluate the residual of the **factored** system.
3. Apply the same LU factors to that residual and add the correction.

It performs no new factorization, changes no pivots, and uses no iteration
or convergence threshold beyond that single correction. It targets LU
application error; it does not correct error already present in the factors
or upstream Schur matrices.

CUDA FP64 residuals use two-word triangular products, retaining the low word
between the U and L products. Explicit rounded FP64 multiplication, fused
multiply-add and two-sum operations preserve the small residual. The kernel
reads the packed LU directly, avoiding separate triangular matrix copies in
the residual calculation. See [NVIDIA PTX floating-point instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
and [LLVM early-clobber constraints](https://www.llvm.org/docs/LangRef.html#inline-assembler-constraint-strings).
CPU residual products retain ordinary precision; other dtypes fall back to
the existing native solve. CUDA accuracy is not inferred from CPU tests.

The helper differentiates the factored linear equation with
`jax.lax.custom_linear_solve`. Both the forward and transpose solves use the
same one-step correction. AD retains the factors and required solutions,
rather than a tape of the correction arithmetic. Forward execution adds
two compensated triangular products and one LU application per corrected
RHS; JVP and VJP execution also perform corrected solves. Actual overhead
must be measured rather than inferred from this operation count.

Four NumPy tests pass without importing JAX. They execute the actual
residual-kernel structure against host references, exercise cancellation,
low words, edge tiles and transposes, check one-step correction of an
injected LU-application error, and inspect supported API signatures.
Three opt-in CUDA tests check the compiled compensated arithmetic, native
and transpose solutions, JVP/VJP/FD consistency, and batching:

```bash
SOLVAX_TEST_LU_REFINEMENT_GPU=1 python -m unittest discover \
  -s tests -p test_lu_refinement.py -v
```

NEOPAX's saved-LU checker has an optional `--compare-lu-refinement` mode.
It requires `--columns-per-call 0`, gates the native control against the
capture, and compares the one-step correction at the original parameter
step. It reports error functionals, residual action norms, warm forward,
primal-plus-JVP and primal-plus-VJP timings, compilation times and compiled
buffer sizes. The dense AD probes test the isolated LU application, not the
full transport gradient or optimization workload. Compiled buffer sizes are
not measurements of host RSS. Neither a full transport accuracy improvement
nor a speed/memory claim is established until the CUDA results are obtained.

## First CUDA attempt: PTX syntax correction

The first remote run at `c48b96b` stopped during the JAX 0.7.2 CUDA tests.
All three CUDA tests hit the same assembler error: the inline `fma.rn.f64`
used an unsupported unary-negated register operand. JAX 0.10.0 tests and
the saved-point timing comparison were therefore not reached. The four
NumPy tests passed, but their original assembly mock evaluated the intended
formula without parsing each literal instruction, so it missed this error.

The product-error sequence now emits an explicit `neg.f64` before the FMA.
This preserves the intended arithmetic and changes only its valid PTX
encoding. The NumPy mock now interprets the actual instruction sequence.
An additional compiler-only regression test assembles both literal kernel
snippets for `sm_80` and verifies that the previous unary-operand sequence
is rejected. Set `SOLVAX_TEST_PTXAS` to an assembler path to run this check;
it requires no GPU or JAX import.

NVIDIA `ptxas` 12.9.86 was downloaded into the ignored local workspace,
verified against its published wheel checksum, and used without installing
or changing any runtime package. Both corrected snippets assemble, the
previous invalid snippet is rejected, and all five local checks pass.
The three full CUDA tests and the accuracy/timing comparison remain to be
rerun; no correction performance or transport-gradient claim follows from
this compiler-only check.
