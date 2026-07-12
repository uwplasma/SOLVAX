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
- {func}`solvax.direct.block_thomas_factor`
- {func}`solvax.direct.block_thomas_solve`
- {func}`solvax.direct.block_thomas_truncated`
- {func}`solvax.direct.block_thomas_truncated_fn`
- {func}`solvax.direct.mixed_precision_block_thomas`

Runnable counterparts: `examples/01_block_tridiagonal_kinetic.py`,
`examples/05_block_thomas_factor_solve.py`, and
`examples/16_mixed_precision_block_thomas.py`.
