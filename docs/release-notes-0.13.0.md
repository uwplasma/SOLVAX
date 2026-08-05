# Release 0.13.0

## Reusable factors for chains whose bands do not fit

Reusable factors from `block_thomas_factor_fn` nominally hold three `(N, m, m)`
arrays: the Schur LU factors and the two off-diagonal bands the substitution
reads. Only the first cannot be recovered — the other two are exactly what the
block generator already returns.

`store_offdiagonals=False` therefore keeps the Schur LU alone, in a
`GeneratedBlockTridiagFactors` that carries its own `block_fn`, and has
`block_thomas_solve` rebuild `L_k` and `U_k` one block at a time during its two
substitution sweeps:

```python
factors = sx.block_thomas_factor_fn(
    row, n_blocks=N, store_offdiagonals=False, factor_dtype=jnp.float32
)
x = sx.block_thomas_solve(factors, rhs)                  # primal
y = sx.block_thomas_solve(factors, rhs, transpose=True)  # adjoint, same factors
```

Retained state falls to a third, and to a sixth against float64 bands when
`factor_dtype` puts the LU in single precision. The two options compose.
Measured at `N=48`, `m=24` as the summed byte count of the factor pytree's
leaves:

| policy | retained state | ratio |
|---|---:|---:|
| stored bands, float64 LU | 668160 B | 1.0000 |
| Schur LU only, float64 | 225792 B | 0.3379 |
| Schur LU only, float32 | 115200 B | 0.1724 |

The excess over exactly 1/3 and 1/6 is the pivot indices, which both policies
keep. On the scale where
this decides something rather than saves something: a family of chains whose
three float64 bands come to 53.3 GB retains 17.8 GB as Schur LU alone, and
8.9 GB with a float32 LU — the difference between fitting on a device and not.

## What it costs, and what it is not

The cost is generator evaluations, not arithmetic. `block_fn` runs once per
index to factor, and then **twice per index per solve**, once in the downward
sweep and once in the upward one. Triangular-solve and matrix-product counts
are unchanged.

The elimination still runs exactly once. That is the whole distinction from
`block_thomas_checkpointed_fn`, which re-eliminates on every application: here
an apply performs no factorization at all, only triangular solves against
factors that were built once. A one-shot solver cannot serve as a
preconditioner inside a Krylov method, because the method applies it tens of
times against the same operator; a factor-once/solve-many path can. This
release is what makes the low-storage route usable there.

`block_thomas_checkpointed_fn` is not superseded and is not deprecated. It
retains *no* band-sized state, where this route retains a third of one, so it
remains the answer where even the Schur LU does not fit.

## The constraint the sweeps are written to

Both sweeps thread the right-hand side through the scan **carry** and take only
the row index and that row's Schur factors as scan inputs. This is required,
not stylistic.

On every JAX release before 0.10.0 a `lax.scan` cannot be transposed if a value
linear in the differentiated input arrives as a scan *input*: the transpose
rule classifies every input of a plainly traced scan as a residual and then
asserts that no residual is an undefined primal, which a two-line cumulative-sum
scan is enough to trip. A linear carry has always been transposable. Reverse
mode is unaffected either way, because partial evaluation sets the
classification honestly there, so the symptom is confined to
`jax.linear_transpose` — which is exactly the call an implicit-differentiation
layer makes.

Each step is also written to touch only `Delta_k`, never row `k+1`'s Schur
factor. A shifted slice would materialize a second `(N, m, m)` array during the
solve and hand back most of the saving. XLA fuses the carry's per-step read and
write into an in-place slice update, so no sweep does work proportional to the
block count per step, and the solve holds one right-hand-side-shaped buffer
instead of three. `block_thomas_solve`'s unrolled sweeps exist for the same
underlying reason; the comment there now says so.

## Verification

The primal and the transposed solve are both exact under the new policy, and
gradients match the stored-band path in forward and reverse mode, with respect
to a scalar shift and to the bands themselves. Exactness is checked against the
materialized path and against dense reference solves, including a chain with
many short block rows and several right-hand sides.

Two of the tests are worth naming because they pin properties rather than
outputs. The adjoint property is stated directly as
`<A^-T u, v> == <u, A^-1 v>`, so it is held by arithmetic rather than by JAX's
transposition machinery. And on a deliberately near-singular pinned chain the
comparison is by **residual**, not by solution vector: a chain regularized by a
rank-one pin has a direction along which two exact factorizations legitimately
differ, so comparing solutions there would measure the conditioning instead of
the code.

There is also a structural assertion that no right-hand-side-shaped floating
input reappears in either scan — the property that keeps the saving real rather
than merely nominal.

## Upgrading

Nothing is removed, no signature changes incompatibly, and the default is
unchanged: `block_thomas_factor_fn` without `store_offdiagonals` returns the
same `BlockTridiagFactors` as before, bit for bit.

`GeneratedBlockTridiagFactors` carries its `block_fn` rather than taking it at
solve time, so factors and generator cannot be paired up wrongly — a mismatch
would be silently wrong rather than an error. As a pytree its two arrays are
the children and the generator is static, so the state crosses a `jit`
boundary; a generator that closes over arrays makes them compile-time constants
of the consuming trace, and a freshly created closure is a fresh static value.
Build the generator once if the factors are passed as an argument.
