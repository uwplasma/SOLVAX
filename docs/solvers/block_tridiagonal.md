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

`block_fn` is evaluated once per index. The reusable factors retain `O(N m^2)`
LU and off-diagonal state, but no full diagonal input band is kept in addition
to the Schur factors.

`BlockTridiagFactors` contains:

- `delta_lu`: LU storage for every Schur complement;
- `delta_piv`: corresponding dense pivot indices;
- `lower`, `upper`: off-diagonal bands used during substitution.

This split is useful for multiple forcing terms, repeated Newton corrections
with a frozen Jacobian, and direct preconditioning.

## Drop the off-diagonal bands from the factors

Of those three `(N, m, m)` arrays only the Schur LU is irrecoverable; `lower`
and `upper` are what `block_fn` returns. Ask for the factors without them:

```python
factors = sx.block_thomas_factor_fn(block_fn, n_blocks=N, store_offdiagonals=False)
x = sx.block_thomas_solve(factors, rhs)
x_t = sx.block_thomas_solve(factors, adjoint_rhs, transpose=True)
```

The result is a `GeneratedBlockTridiagFactors`, accepted by
`block_thomas_solve` on the same terms as `BlockTridiagFactors` and carrying its
own `block_fn`, so factors and generator cannot be paired up wrongly. Both
substitution sweeps rebuild the block they need, one at a time.

Retained state falls by a factor of three, and by six against float64 bands when
`factor_dtype=jnp.float32` puts the Schur LU in single precision — the two
options compose. To put the ratios on a scale where they decide something: a
family of chains whose three float64 bands come to 53.3 GB retains 17.8 GB as
Schur LU alone, and 8.9 GB with a float32 LU.

What it costs is generator evaluations: `block_fn` runs once per index to
factor, then **twice per index per solve**. Triangular-solve and matrix-product
counts are unchanged, and the elimination still happens exactly once, so this
remains a factor-once/solve-many path — unlike `block_thomas_checkpointed_fn`
below, which re-eliminates on every application and is priced accordingly. Use
it when the factors, not the sweep, are what does not fit.

`block_fn` must be a pure function of its index: the substitution assumes a
regenerated block equals the one the factorization saw.

## Checkpoint one generated solve

When the blocks are cheap to regenerate and the system is solved once, retain
only radial checkpoints and one segment:

```python
x = sx.block_thomas_checkpointed_fn(block_fn, N, rhs)
```

The default segment width is $\lceil\sqrt{N}\rceil$. The downward sweep stores
one Schur factor and transformed right-hand side per segment; substitution
regenerates and discards one segment at a time. This changes dense-factor
storage from $O(Nm^2)$ to $O(\sqrt{N}m^2)$ at the cost of generating every
block twice. The solve is exact to floating-point round-off, accepts multiple
right-hand sides and `transpose=True`, and uses the same checkpointed
transpose solve for implicit JVPs and VJPs. Use reusable factors instead when
the same matrix has several right-hand sides.

Reproduce the full-factor comparison with:

```bash
PYTHONPATH=src python benchmarks/benchmark_checkpointed_block.py --x64
```

For the default float64 `N=128`, `m=64` case, compiled temporary storage was
20.73 MB versus 1.99 MB on an Apple M4 CPU and 16.81 MB versus 2.95 MB on an
RTX A4000 (90.4% and 82.4% lower). Warm execution was about twice as long
because the saved factors are recomputed; the solutions agreed to round-off.

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

Both public entry points share this rule. `block_thomas_truncated` (stored
bands) routes its windowed reverse mode through the same generated
construction as `block_thomas_truncated_fn(..., params=...)`, handing the bands
over as the parameter pytree, so `adjoint_window` has one meaning across the
API and the two produce bitwise-identical gradients. Earlier releases closed
the array path at the window with a leading-principal subsystem, which is a
strictly weaker approximation; that closure is retained only as an ablation in
the test suite.

The rule is built as an *exact-window* adjoint, so most of it is exact and the
single approximation is explicit. Writing $W=\min(K+w,N)$ for the retained rows
and $M=\min(W+1,N)$ for the primal window:

- **The selected forward blocks are exact.** The elimination visits and
  eliminates *every* one of the $N$ rows — the tail sweep folds the whole
  remainder into the running Schur complement — so the returned blocks are
  blocks of the full $N$-row solution, not of a leading principal submatrix.
  Only storage and the upward substitution stop at the head. The method is
  memory-localized, **not** work-localized: arithmetic remains $O(Nm^3)$.
- **Both sweeps visit all rows.** The reverse pass solves the transposed
  generated chain with the same full-tail strategy, retaining exact adjoint
  blocks $\lambda_0,\dots,\lambda_{W-1}$.
- **Right-hand-side gradient (exact, at every window).** $\bar b = P_K\lambda$
  is exact and does not move with `adjoint_window`, including $w=0$. Only
  invertibility is used: no decay, dominance, or normality assumption.
- **Retained-row cotangents (exact).** For every $j<W$ the block cotangents
  $\bar L_j=-\lambda_j x_{j-1}^\top$, $\bar D_j=-\lambda_j x_j^\top$,
  $\bar U_j=-\lambda_j x_{j+1}^\top$ pair exact primal and exact adjoint
  blocks, so each retained row contributes exactly. Row $W-1$ needs $x_W$,
  which is why the primal window carries one halo block.
- **The only approximation is the omitted tail.** The gradient error is exactly
  $\sum_{j\ge W}$ of the omitted row contributions. Consequently
  `adjoint_window >= n_blocks` gives the exact gradient, and a parameter whose
  generator derivative vanishes above $W$ is differentiated exactly at *any*
  window.
- **Tail size.** Under uniform inverse localization
  $\lVert (A^{-1})_{ji}\rVert\le C_A\rho^{|j-i|}$ {cite}`demko1984,benzi2013`,
  the primal decays away from the source and the adjoint away from the output,
  and each row cotangent is an outer product of the two, giving the doubled
  exponent. The least-decayed of the three band terms is the lower block
  $\bar L_j = -\lambda_j x_{j-1}^\top$, which sets the rate: the error is
  $O(\rho^{2w})$ for bounded generator sensitivity and
  $O((K+w)^{s}\rho^{2w})$ when the row sensitivity grows like $(1+j)^{s}$ — the
  relevant case for kinetic operators whose collisional derivative scales with
  mode number. A spectral gap alone is **not** sufficient for a nonnormal
  operator.

Peak dense reverse workspace is $O((K+w)m^2)$ plus the primal halo and the
parameter storage. Total memory is independent of $N$ only when the blocks are
generated from compact parameters; a coefficient array of length $N$ is far
smaller than dense bands but still scales with $N$, and is reported separately.

The result: forward *and* reverse run at retained state independent of the
block count, so `jax.grad` through a selected-head kinetic solve stays flat as
$N$ grows while the naive tape grows linearly. No second-order guarantee is
implied — at finite $W$ the windowed gradient need not be the gradient of any
scalar surrogate, so differentiating it again does not yield a valid Hessian.

With array bands, the bands themselves still occupy $O(Nm^2)$; when blocks are
assembled from a low-dimensional parameterization, the generated path removes
that too:

```python
def block_fn(params, k):
    return lower_block(params, k), diagonal_block(params, k), upper_block(params, k)

x_low = sx.block_thomas_truncated_fn(
    block_fn, n_blocks=N, rhs_low=rhs_low, keep_lowest=K,
    params=params, adjoint_window=w,
)
```

The custom VJP then generates the *transposed* rows on the fly for the exact
right-hand-side gradient (three block assemblies per index) and pulls the
windowed band cotangents back through `block_fn`'s own derivative at each of
the leading $K+w$ indices, so `jax.grad` with respect to `params` runs at
$O((K+w)m^2)$ scratch — **no band arrays exist in either direction**, and the
reverse-mode footprint is measured flat from $N=32$ to $N=1024$ in the test
suite and the transport-inversion benchmark.

### Choosing the window, with a proof

The window is a tolerance dial, and `certified_adjoint_window` sets it from the
tolerance rather than from a rule of thumb. What makes a proof available here
is that the exact-window rule leaves *one* approximation to bound: the source
cotangent and every retained row cotangent are exact blocks of the full solve,
so

$$
\nabla_p J - g_W = \sum_{j \ge W} DB_j(p)^*[\bar L_j, \bar D_j, \bar U_j]
$$

holds with equality. Bounding each factor by its envelope --- the primal and
transposed transfer products, and the generator sensitivity
$\gamma_j = \|DB_j(p)\|_F$ --- gives a computable

$$
\|\nabla_p J - g_W\| \le S \sum_{j \ge W}
    \gamma_j \Lambda_j (X_{j-1} + X_j + X_{j+1}) =: B(W),
$$

with $S = \|\lambda_{K-1}\| \|x_{K-1}\|$ read off one selected-head solve.

Turning that into a *relative* statement needs a lower bound on the gradient,
which norms of the summands cannot give — they cannot see cancellation. One
windowed gradient supplies it through the reverse triangle inequality,
$\|\nabla_p J\| \ge \|g_{W_0}\| - B(W_0)$, and that bound is valid whichever
window produced it, so searching over windows costs no further solves.

```python
cert = sx.certified_adjoint_window(
    block_fn, n_blocks=N, keep_lowest=K, params=params,
    rhs_low=rhs_low, cotangent=cotangent, rtol=1e-8,
)
cert.window                     # smallest window that provably meets rtol
cert.certified_relative_error   # the proven bound, at or below rtol
cert.tail_bound                 # the absolute bound on ||grad - g_W||
```

Two things to expect. The bound is a norm bound, so it is conservative, and how
conservative depends strongly on the family: measured across the block-dominant,
weakly-dominant and kinetic chains in `tests/test_certified_window.py`, the
realized error runs between two and a half and seven orders of magnitude below
the certified one. The returned window is wider than strictly necessary by the
same token — the certificate buys a guarantee, not the minimal window, and
`check_localized_gradient` remains the way to find out how much narrower you
could have gone. And a chain that does not localize gets the full window with
`status="full-window"` and a zero tail bound — an over-estimate, never an
under-estimate.

The certificate depends on the *cotangent*, not on the operator alone. That is
not a wart: which window suffices genuinely depends on what is being
differentiated, and a window certified for one objective is not certified for
another.

Cost is two localization sweeps, one generator Jacobian per row, and one or two
differentiated solves. Pass `sensitivity=` when the per-row Jacobians are too
expensive or an analytic bound on $\gamma_j$ is known; overstating it only
widens the window, never invalidates the certificate.

### Batches whose chains localize differently

`adjoint_window` is static, so one `vmap` over a batch runs every chain at the
widest window any chain needs. On a collisionality scan that is expensive in
exactly the wrong place: the criterion gives up on the least collisional chain
and drags the whole batch to the full window, while most chains would localize
in a fraction of it.

Padding to the maximum does not fix this — it *is* the problem. A segmented
layout does: sort by window, cut into a few groups, trace one shape per group.
`plan_chain_windows` chooses the cuts optimally by dynamic programming over the
distinct window values, minimizing $\sum_b |b|(K + W_b)$.

```python
windows = [int(sx.certified_adjoint_window(chain, N, K, p, rhs, ct, rtol=1e-6))
           for chain in chains]
plan = sx.plan_chain_windows(windows, keep_lowest=K, max_buckets=4)
for window, members in plan.buckets:
    grads = jax.vmap(lambda c: chain_gradient(c, adjoint_window=window))(members)
```

Measured on a 24-chain scan spanning $\nu = 10^{-3.5}$ to $10^{-0.5}$ at
$N=80$, $K=3$, `rtol=1e-6`, where the certified windows run from 5 to the full
77:

| buckets | retained rows | vs. uniform |
|---|---:|---:|
| 1 (uniform) | 1920 | — |
| 2 | 1170 | 39.1% |
| 3 | 1015 | 47.1% |
| 4 | 935 | 51.3% |
| 8 | 833 | 56.6% |
| per-chain ideal | 785 | 59.1% |

Four buckets recover most of what per-chain windows could give, for four traced
shapes instead of twenty-four.

Inside a bucket, `chain_window=` gives each chain its own window as a traced
value. It changes which rows contribute to the gradient, not the retained
state, which the bucket's static window fixes — so use it for accuracy
bookkeeping, and the buckets for memory. With `chain_window` equal to the
static window the result is bit-identical to the uniform path.

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
- {class}`solvax.direct.GeneratedBlockTridiagFactors`
- {func}`solvax.direct.block_thomas_solve`
- {func}`solvax.direct.block_thomas_truncated`
- {func}`solvax.direct.block_thomas_truncated_fn`
- {func}`solvax.direct.block_thomas_truncated_fn_with_residual`
- {func}`solvax.direct.certified_adjoint_window`
- {func}`solvax.direct.plan_chain_windows`
- {func}`solvax.direct.localization_crossover_window`
- {func}`solvax.direct.check_localized_gradient`
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
