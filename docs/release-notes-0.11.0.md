# Release notes — 0.11.0

One feature and one limitation, both found by putting the exact-window rule
into a production solver rather than by reading the code.

## The residual diagnostic survives the exact-window path

`block_thomas_truncated_fn_with_residual` returns the leading solution blocks
together with the RMS residual of the Schur system. Until now it could only
tape: a caller who wanted the residual *and* a bounded reverse pass had to run
the elimination twice, once through the exact-window rule for the solution and
once more for the residual.

That is the wrong trade, because the forward-only case — where the residual is
usually the only thing anyone wants — pays for it. Measured on a stellarator
drift-kinetic chain at `N_xi = 128`, `m = 81`, the two-sweep arrangement cost
**1.7×** on a forward solve.

The residual is a by-product of factors the elimination has already formed, so
it now comes back from the same sweep:

```python
solution, residual = sx.block_thomas_truncated_fn_with_residual(
    block_fn, n_blocks, rhs_low, keep_lowest=3,
    params=p, adjoint_window=advice,   # selects the exact-window reverse rule
    residual_rhs_index=0,
)
```

| | forward | gradient | reverse memory |
|---|---|---|---|
| taped (before) | 18.0 ms | 90.6 ms | 41.6 MiB |
| exact, full window | 18.5 ms | 50.6 ms | 33.2 MiB |
| exact, advised window (33) | — | 42.9 ms | 9.4 MiB |

At full window the solution and the residual are **bitwise identical** to the
taped path, so this is exact rather than an approximation, and there is no
forward-only regression.

The residual is returned through `stop_gradient`. It is a diagnostic; the
reverse rule computes cotangents from the solution alone and ignores the
residual's, which is sound *only* because that cotangent is structurally zero.
A test pins `d(residual)/dp == 0` — if the detachment were dropped, the rule
would silently return a wrong number instead.

## The exact-window rule is reverse-mode only

This is not new behaviour, but it was not written down, and it caught a real
integration.

The rule is registered as a `jax.custom_vjp`. JAX cannot push forward-mode
autodiff through one, so `jax.jacfwd` and `jax.jvp` through a windowed solve
**raise** rather than falling back to something slower:

```
TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function.
```

The arrangement this breaks is a common and sensible one: a solver whose
design gradients are taken in reverse, but whose derivative audits are taken
forward as an independent check. Such a code must keep an unwindowed path
available. In practice that means **applying the window is a decision made per
call site, not a global switch**, and a library wrapping this rule should not
make it the default.

`params=None` continues to differentiate the elimination directly, where both
modes work. `adjoint_window >= n_blocks` retains every row and is exact, so the
exactness is available without giving up anything except forward mode.

## Compatibility

No behaviour changes for existing callers: the new arguments are keyword-only
and default to the previous behaviour.
