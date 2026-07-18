# Block-tridiagonal solvers

Use this family when the unknown is partitioned into $N$ blocks of size $m$ and
only adjacent blocks couple:

$$
L_k x_{k-1}+D_k x_k+U_k x_{k+1}=b_k,
\qquad k=0,\ldots,N-1.
$$

Typical sources are one-dimensional multi-field transport, radial finite
volumes, line-implicit PDE methods, and spectral kinetic equations in which
mode $\ell$ couples only to $\ell\pm1$.

## Storage model

| Input | Shape | Meaning |
|---|---|---|
| `lower` | `(N, m, m)` | $L_k$; `lower[0]` is ignored |
| `diag` | `(N, m, m)` | $D_k$ |
| `upper` | `(N, m, m)` | $U_k$; `upper[-1]` is ignored |
| `rhs` | `(N, m)` or `(N, m, n_rhs)` | one or several right-hand sides |

The leading dimension is the structured direction. Dense physics coupling
belongs inside each $m\times m$ block.

## Derivation: block Thomas elimination

Starting from the last block, eliminate $x_{k+1}$. Define the Schur complements

$$
\Delta_{N-1}=D_{N-1},\qquad
\Delta_k=D_k-U_k\Delta_{k+1}^{-1}L_{k+1}.
$$

Apply the same elimination to the right-hand side:

$$
\sigma_{N-1}=b_{N-1},\qquad
\sigma_k=b_k-U_k\Delta_{k+1}^{-1}\sigma_{k+1}.
$$

The remaining lower-bidiagonal system is solved upward:

$$
x_0=\Delta_0^{-1}\sigma_0,
\qquad
x_k=\Delta_k^{-1}(\sigma_k-L_kx_{k-1}).
$$

SOLVAX factors every $\Delta_k$ with dense partial-pivoting LU and applies
triangular solves; it never forms $\Delta_k^{-1}$. The factorization costs
$O(Nm^3)$ and each right-hand side costs $O(Nm^2)$. Assembling and factoring
the dense $Nm\times Nm$ matrix would cost $O(N^3m^3)$ and use $O(N^2m^2)$
storage {cite}`golub2013,demmel1995`.

## One-shot solve

```python
x = sx.block_thomas(lower, diag, upper, rhs)
```

This is `block_thomas_factor` followed by `block_thomas_solve`. Use it when the
matrix is solved only once.

## Factor once, solve many

```python
factors = sx.block_thomas_factor(lower, diag, upper)
x_a = sx.block_thomas_solve(factors, rhs_a)
x_b = sx.block_thomas_solve(factors, rhs_b)
```

When blocks come from compact coefficients, avoid materializing the diagonal
band before factorization:

```python
def block_fn(k):
    return lower_block(k), diagonal_block(k), upper_block(k)

factors = sx.block_thomas_factor_fn(block_fn, n_blocks=N)
x = sx.block_thomas_solve(factors, rhs)
x_t = sx.block_thomas_solve(factors, adjoint_rhs, transpose=True)
```

`block_fn` is evaluated once per index. The reusable factors necessarily retain
`O(N m^2)` LU and off-diagonal state, but no full diagonal input band is kept in
addition to the Schur factors.

`BlockTridiagFactors` contains:

- `delta_lu`: LU storage for every Schur complement;
- `delta_piv`: corresponding dense pivot indices;
- `lower`, `upper`: off-diagonal bands used during substitution.

This split is useful for multiple forcing terms, repeated Newton corrections
with a frozen Jacobian, and direct preconditioning.

## Transposed solve and adjoints

```python
x_t = sx.block_thomas_solve(factors, rhs, transpose=True)
```

For the chosen elimination order, the Schur complements of $A^T$ are
$\Delta_k^T$. The same LU factors can therefore solve the transposed system by
transposed triangular substitution; no second factorization is needed. This is
the natural partner for implicit differentiation of $A(\theta)x=b$.

## Truncated low-mode solve

Suppose $b_k=0$ for $k\ge K$ and only $x_0,\ldots,x_{K-1}$ enter an observable.
The high blocks still modify the low Schur complements, so they cannot simply
be deleted. They can, however, be eliminated without storing all high-mode
right-hand-side intermediates. `block_thomas_truncated` returns only the first
`keep_lowest` solution blocks:

```python
x_low = sx.block_thomas_truncated(
    lower, diag, upper, rhs_low, keep_lowest=3
)
```

Here `rhs_low.shape[0]` must equal `keep_lowest`. Peak retained substitution
storage scales as $O(Km^2)$ rather than $O(Nm^2)$, while the necessary downward
elimination still visits every block {cite}`escoto2025`.

When even the bands are too large to store, assemble them on demand:

```python
def block_fn(k):
    return lower_block(k), diagonal_block(k), upper_block(k)

x_low = sx.block_thomas_truncated_fn(
    block_fn, n_blocks=N, rhs_low=rhs_low, keep_lowest=K
)
```

`n_blocks` and `keep_lowest` are static algorithm sizes under `jit`.
Each generated block index is assembled exactly once per solve. In the retained
head, the Schur update and all right-hand-side updates share one multi-column LU
solve. This matters when block assembly or dense triangular dispatch dominates.

Request a tail-aware algebraic residual without reconstructing another block:

```python
x_low, residual_l2 = sx.block_thomas_truncated_fn_with_residual(
    block_fn, n_blocks=N, rhs_low=rhs_low, keep_lowest=K,
    residual_rhs_index=0,
)
```

This evaluates the retained Schur equations from the pivoted LU factors. It
includes the eliminated tail and does not materialize the diagonal band. Omit
`residual_rhs_index` to combine all right-hand sides in one RMS diagnostic.

### Bounded-memory adjoint

The forward truncated solve is $O(Km^2)$ in memory, but differentiating it with
plain reverse mode tapes the downward sweep over every block, so the
*differentiated* solve costs $O(Nm^2)$ — the block-count independence is lost
exactly where gradient-based inversion needs it. Passing `adjoint_window`
selects a structure-preserving custom VJP that keeps the reverse pass bounded
too:

```python
x_low = sx.block_thomas_truncated(
    lower, diag, upper, rhs_low, keep_lowest=K, adjoint_window=w
)
```

Two facts make this exact where it can be and controlled where it cannot:

- **Right-hand-side gradient (exact).** The transpose of a block-tridiagonal
  operator is block-tridiagonal with the off-diagonals swapped and transposed,
  so the cotangent map $\bar b = P_K A^{-\top} E_K\,\bar x$ is *itself* a
  truncated solve of $A^\top$. It runs at $O(Km^2)$ and carries no truncation
  error, independent of `adjoint_window`.
- **Band gradient (windowed).** The full primal and adjoint spread over all
  blocks but decay geometrically away from the retained head for block
  diagonally dominant systems {cite}`demko1984,benzi2013`. Reconstructing the
  band gradients from a leading $(K+w)$-block re-solve therefore has error
  $O(\rho^{2w})$ with $\rho\in(0,1)$ set by the conditioning, at
  $O((K+w)m^2)$ memory. Setting `adjoint_window >= n_blocks` reproduces the
  exact gradient.

The result: forward *and* reverse run at memory independent of the block count,
so `jax.grad` through a truncated kinetic solve stays flat as $N$ grows while
the naive tape grows linearly. This is the differentiable counterpart of the
truncated forward solve and the tool for adjoint-based source/transport
inversion on tall block systems.

## Residual gate

Validate a solve with an operator action independent of the factorization:

```python
relative_residual = sx.block_tridiag_relative_residual(
    lower, diag, upper, x, rhs
)
```

The diagnostic evaluates every block row, including high-mode tails. It is a
numerical consistency gate, not a substitute for discretization convergence.

## Assembly, factors, and transpose scope

`block_thomas_truncated_fn` calls `block_fn` once for each block index during a
primal solve and retains factors only for the requested low blocks. Those
truncated factors live only for that call. Use `block_thomas_factor` followed
by `block_thomas_solve` when factors must survive across right-hand sides or
when an exact transposed solve is required. A full transpose generally
propagates through the discarded high-mode tail, so SOLVAX does not claim that
an O(K) truncated factorization can provide an exact transpose action.

## Mixed-precision variant

```python
x = sx.mixed_precision_block_thomas(
    lower, diag, upper, rhs,
    factor_dtype=jnp.float32,
    refine_steps=2,
)
```

The Schur-complement factors use low precision; working-precision residuals and
defect corrections recover accuracy when the conditioning permits. See
{doc}`mixed_precision` for the convergence condition and diagnostics.

## Comparison with alternatives

| Method | Prefer when | Difference from block Thomas |
|---|---|---|
| Scalar Thomas | $m=1$ or many independent scalar columns | lower block overhead; specialized accelerator path |
| Banded LU | scalar bandwidth is small but not naturally blocked | scalar band storage and static pivoting |
| Dense LU | only tiny systems or validation | ignores structure and scales cubically in total size |
| FGMRES + block preconditioner | extra nonlocal couplings perturb a block-tridiagonal core | matrix-free outer solve; direct method becomes approximate inverse |
| Cyclic reduction | massive parallelism dominates sequential sweep cost | more parallel work and different numerical/storage trade-offs |

## Failure modes

- A singular or ill-conditioned Schur complement makes the factorization
  unstable.
- Block LU stability is strongest for block-diagonally-dominant systems; test
  weakly dominant applications carefully {cite}`demmel1995`.
- Incorrect block ordering can turn a truly local operator into apparent dense
  coupling. Reorder by the structured direction before giving up the method.
- `factor_dtype=float16` or `bfloat16` is generally unsupported by the dense LU
  kernels used underneath; float32 is the practical low-precision path.

## API summary

- {func}`solvax.direct.block_thomas`
- {func}`solvax.direct.block_tridiag_matvec`
- {func}`solvax.direct.block_tridiag_relative_residual`
- {func}`solvax.direct.block_thomas_factor`
- {func}`solvax.direct.block_thomas_factor_fn`
- {func}`solvax.direct.block_thomas_solve`
- {func}`solvax.direct.block_thomas_truncated`
- {func}`solvax.direct.block_thomas_truncated_fn`
- {func}`solvax.direct.block_thomas_truncated_fn_with_residual`
- {func}`solvax.direct.mixed_precision_block_thomas`

Runnable counterparts: `examples/01_block_tridiagonal_kinetic.py`,
`examples/05_block_thomas_factor_solve.py`, and
`examples/16_mixed_precision_block_thomas.py`.

From a source checkout, reproduce the kinetic-shaped CPU or accelerator
benchmark with:

```bash
PYTHONPATH=src python benchmarks/benchmark_generated_block.py --output result.json
```

The JSON records the exact implementation hashes, JAX versions, device, cold
compile time, warm samples, executable memory, and error against the
materialized-band algorithm. Keep device families in separate result files;
cross-device timing comparisons are otherwise not meaningful.

Measured results for this change (float64; medians, not universal hardware
claims) are:

| Device and workload | v0.7.0 baseline | Fused head solve | Change |
|---|---:|---:|---:|
| Apple CPU, `13x15x32`, 2 RHS | 20.53 ms | 20.61 ms | +0.36% |
| RTX A4000, `13x15x63`, 8 RHS | 171.63 ms | 162.14 ms | -5.53% |

The GPU compile time increased from 0.53 s to 0.65 s. The checked-in JSON under
`benchmarks/results/` records raw samples, executable memory, software versions,
and source hashes. The change is therefore an accelerator/multi-RHS throughput
optimization with a cold-compile tradeoff, not a blanket speedup.
