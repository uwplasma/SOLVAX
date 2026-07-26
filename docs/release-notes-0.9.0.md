# Release 0.9.0

## Exact-window localized adjoint

Differentiating a generated block-tridiagonal solve with respect to the
parameters that build its rows is now an **exact-window** construction. This is
a behaviour change behind an unchanged API: code written against 0.8.x gets a
strictly stronger guarantee without modification.

Writing `K` for the source and output support, `w` for `adjoint_window`,
`W = min(K + w, N)` for the retained rows and `M = min(W + 1, N)` for the primal
window, the following are **exact at every window**:

- the selected forward blocks, which are blocks of the full `N`-row solution and
  not of a leading principal submatrix;
- the source cotangent `bar b = P_K lambda`, including at `w = 0` — only
  invertibility is used, no decay or dominance assumption;
- every retained row cotangent `j < W`, including the halo row `W - 1`;
- the derivative of any parameter whose generator derivative vanishes above `W`;
- the whole gradient when `w >= n_blocks`.

**The only approximation is the omission of operator rows `j >= W`.**

The previous release reconstructed the retained states from the leading
principal submatrix. Those states are not blocks of the full solution, so the
gradient error contained an interface term inside the retained window on top of
the omitted tail. Measured against that closure the advantage grows with the
window — 2x at `w = 0`, 24x at `w = 4` — because the interface error decays more
slowly than the true tail.

Both sweeps still visit every row: the method is **memory-localized, not
work-localized**, and arithmetic remains `O(N m^3)`.

## Choosing the window before the solve

```python
import solvax as sx

w = sx.suggest_adjoint_window(block_fn, n_blocks, keep_lowest=3)
x_low = sx.block_thomas_truncated_fn(
    block_fn, n_blocks, rhs_low, keep_lowest=3, params=p, adjoint_window=w
)
```

`localization_profile_fn` returns the exact per-row factors
`rho_k = ||Delta_k^{-1} L_k||`, read off the Schur recursion the forward solve
already performs. A single uniform decay rate is the wrong model when the
coupling and the diagonal scale differently with the row index — for a
Legendre-mode kinetic chain the collisional diagonal grows like `nu k(k+1)`
while the streaming coupling does not, so the chain is not localized at low `k`
and localizes above a crossover. Fitting one rate to the leading rows reports
the non-localized head.

## Complex arithmetic

The transposed chain and the row cotangents follow JAX's cotangent convention,
which propagates through the plain transpose; the incoming cotangent of a real
objective already carries the conjugation. This is pinned by explicit complex
tests against ordinary reverse mode rather than inferred from the real case.

## Limitations

- Total memory is independent of `n_blocks` only when the blocks are generated
  from compact parameters; a coefficient array of length `n_blocks` is far
  smaller than dense bands but still scales with `n_blocks`, and is reported
  separately.
- At finite `W` the windowed gradient need not be the gradient of any scalar
  surrogate, so differentiating it again does not yield a valid Hessian.
- A spectral gap alone does not imply the localization the tail bound needs for
  a nonnormal operator.
