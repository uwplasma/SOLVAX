# Release notes — 0.10.0

Two changes matter to anyone differentiating a generated block-tridiagonal
solve at a finite window, and one of them changes numbers.

## `adjoint_window` now means the same thing everywhere

`block_thomas_truncated` takes stored bands and `block_thomas_truncated_fn`
generates its rows on demand. Until this release the two disagreed about what
`adjoint_window` meant. The generated path used the exact-window construction:
it visits the whole chain, retains only the head, and every retained row is a
row of the *full* problem. The stored-band path instead closed the window with a
leading principal subsystem — a smaller, self-contained problem whose solution
blocks are not blocks of the full solution. That closure puts an interface error
*inside* the retained window, on top of the tail it omits.

Both paths now route through the same generated construction, so they produce
bitwise-identical finite-window gradients. The superseded closure is retained
privately and exercised only as an ablation in the test suite.

**This changes numbers** for `block_thomas_truncated(..., adjoint_window=w)`
with `w` short of the full chain: the gradient gets more accurate, and it is a
different number than 0.9.x returned. Full window is unaffected — it was exact
before and is exact now.

## The window advisor says what it is

`suggest_adjoint_window` has been renamed `localization_crossover_window`. It
now returns a `LocalizationWindow` carrying the per-row transfer profile, the
crossover row, and a `certified` field that is `False`.

The rename is not cosmetic. The old name invited reading the return value as a
window that had been checked against a tolerance. It was never that. The value
is read off the chain's own transfer norms — the row at which
`‖T_k‖₂` drops below one — and it tells you where the chain *starts* to
localize, not how accurate a gradient computed there will be. Decay past the
crossover is typically far faster than geometric, so a few rows beyond it often
buy several orders of magnitude. The honest way to use it:

```python
adv = sx.localization_crossover_window(generator, n_blocks, keep_lowest=3)
w = adv.window          # a starting point
adv.certified           # False, always: this is a diagnostic
adv.profile             # the per-row factors it was read from
```

then widen `w` until the gradient stops moving. On a problem small enough to
afford the comparison, `check_localized_gradient` does that check against the
full-window gradient directly.

`suggest_adjoint_window` still works, returns the plain integer, and warns. It
will be removed in 0.12.0.

## Optional chunking backend

The `adv-jax-math` chunking backend is optional and chosen at runtime;
`available_backends()` reports what is installed. SOLVAX pins no JAX version and
no Python version on its account, and the default install does not pull the
extra in. CI runs a dedicated job with the extra installed which asserts the
backend is reachable before exercising it, so a silently-skipped job cannot pass
as coverage.

## A note on 0.9.1

0.9.1 went out without a changelog entry, so two code states could both report
`0.9.1` while differing in their exact-window semantics and public API. This
release restores the correspondence between version and behaviour. Anything
depending on the finite-window gradient should pin `>=0.10.0`.
