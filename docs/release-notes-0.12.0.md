# Release 0.12.0

## A window you can prove, not just estimate

`certified_adjoint_window` returns the smallest adjoint window whose relative
gradient error is **provably** within a requested tolerance. Until now the
library was explicit that it could not do this: `localization_crossover_window`
accepts no tolerance and returns `certified=False`, because it reads only the
primal envelope and reports where the transfer norms cross one.

What makes a certificate available is a property of the exact-window rule
rather than new theory. Because the source cotangent and every retained row
cotangent are exact blocks of the full solve, the tail identity

$$
\nabla_p J - g_W = \sum_{j \ge W} DB_j(p)^*[\bar L_j, \bar D_j, \bar U_j]
$$

holds with *equality*. There is therefore exactly one thing to bound — the
omitted rows — and each of its factors has a computable envelope: the primal
and transposed transfer products, and the generator's own sensitivity
$\gamma_j = \|DB_j(p)\|_F$.

```python
cert = sx.certified_adjoint_window(
    block_fn, n_blocks=N, keep_lowest=K, params=params,
    rhs_low=rhs_low, cotangent=cotangent, rtol=1e-8,
)
cert.window                     # provably meets rtol
cert.certified_relative_error   # the proven bound
cert.tail_bound                 # absolute bound on ||grad - g_W||
```

The step that could have gone wrong is the *relative* statement. Dividing the
tail bound by the total bound looks natural and is invalid: the total
over-estimates the gradient, so a tolerance relative to it is weaker than one
relative to the gradient — the wrong direction for a certificate. Norms of the
summands cannot give a lower bound either, since they cannot see cancellation.
One windowed gradient can, through the reverse triangle inequality
$\|\nabla_p J\| \ge \|g_{W_0}\| - B(W_0)$, and that bound is valid whichever
window produced it, so searching over windows costs no further solves.

Verified against the full window, which the construction makes exact, on
block-dominant, weakly dominant, kinetic and complex chains:

| rtol | window | certified bound | realized error |
|---|---:|---:|---:|
| 1e-02 | 1 | 3.06e-04 | 9.35e-07 |
| 1e-04 | 2 | 1.12e-06 | 1.96e-10 |
| 1e-06 | 3 | 4.69e-09 | 6.31e-13 |
| 1e-10 | 4 | 3.50e-11 | 1.38e-15 |

Two things to expect. It is a norm bound, so it is conservative — between two
and a half and seven orders of magnitude across those families — and the
returned window is correspondingly wider than the minimum. The certificate buys
a guarantee, not the narrowest window; `check_localized_gradient` remains the
way to find out how much narrower you could have gone. And the certificate
depends on the *cotangent*, not the operator alone, because which window
suffices genuinely depends on what is being differentiated.

A chain that does not localize gets the full window with
`status="full-window"` and a zero tail bound — an over-estimate, never an
under-estimate.

## Batches whose chains localize differently

`adjoint_window` is static, so one `vmap` over a batch runs every chain at the
widest window any chain needs. On a collisionality scan that is expensive in
exactly the wrong place: the criterion gives up on the least collisional chain
and drags the whole batch to the full window, while most chains would localize
in a fraction of it.

Padding to the maximum does not fix that — it *is* the problem. A segmented
layout does. `plan_chain_windows` sorts the chains by window and cuts them into
at most `max_buckets` groups, choosing the cuts optimally by dynamic
programming over the distinct window values:

```python
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

Four traced shapes instead of twenty-four recover most of the available saving,
and every chain's gradient still matches its exact one to 6.6e-14.

`block_thomas_truncated_fn` also gains `chain_window=`, a traced per-chain
window under the static bound. It changes which rows contribute to the
gradient, not the retained state, so it is the accuracy primitive while the
buckets are the memory one. With `chain_window` equal to the static window the
result is bit-identical to the uniform path.

## Differentiating an externally solved eigenpair

Some eigenproblems are best solved by something that is not JAX — a sparse
factorization, a native Arnoldi, an application's own solver. `eigenpair_reverse`
differentiates such a solve implicitly, given a caller-certified simple right and
left eigenpair, without putting the eigensolver's iterations on the reverse tape.

For a normalized simple eigenpair the eigenvalue derivative is
$d\lambda = \ell^H (dA) r$; eigenvector-dependent objectives cost one bordered
reduced-resolvent solve. The eigensolver is injected rather than prescribed, so
the primal can be whatever the application already trusts.

A condition gate guards the exceptional-point case, and it is worth being
precise about what it does and does not see. On a deliberately near-defective
pair at $\varepsilon = 10^{-18}$ the true condition number was $5 \times 10^8$
while the gate measured $4.5 \times 10^7$, and the derivative came out 0.955
against an analytic 0.5. The gate catches the obvious degeneracy; it is not a
certificate of well-conditioning, and a caller near a genuine exceptional point
should not read a passing gate as one.

Alongside it:

- `estimate_rk4_timestep`, `propagator_eigenpairs` and `exponential_eigenpairs`
  for RK4-filtered and exponential-action extraction, with every candidate
  re-evaluated against the original continuous operator rather than the
  filtered one.
- `adaptive_eigenpair` for residual stopping, growth-defect rejection and
  operator-work accounting around an application-supplied restart.
- `sparse_operator_matrix`, `sparse_eigenpairs` and `SpluFactorization`, a
  bounded sparse bridge that converts each JAX column block straight to CSR
  rather than materializing a dense $n \times n$ host matrix, and lets one
  shifted LU serve the right and conjugate-transpose Arnoldi so implicit
  reverse mode does not refactor the same operator.

The sparse bridge is eager and CPU-factorized: JAX supplies the operator actions
and parameter derivatives, and the native primal is treated as an external
certified solve. It is not traceable, and says so.

## Scale and precision fixes

**The Thomas pivot guard was a fixed `1e-12`**, neither scale- nor
dtype-invariant: enormous beside a float32 system of order one, negligible
beside a float64 system scaled to `1e12`. It is now `sqrt(eps)` of the working
dtype times the coefficient scale.

The first version of that fix reduced globally and broke the sharding test,
which was right to catch it: a global maximum over a batch sharded across
devices is an all-reduce, and the batched tridiagonal solve is meant to have no
collectives. Reducing along the solve axis only is local, and is the better
quantity anyway — each system is guarded against its own coefficients rather
than the largest anywhere in the batch.

**The Fourier–Helmholtz solver hard-cast its geometry to `float64`**, silently
upgrading an x64-disabled program, or meaning nothing at all with x64 off. It
now infers the working precision from the caller and rejects complex geometry
explicitly.

**The Nyström null-direction test compared against exact zero.** On a
rank-deficient operator those eigenvalues emerge from an SVD as rounding noise,
so the result depended on the linear-algebra backend: the same zero operator
gave the identity under JAX 0.11.0 and arbitrary order-one values under 0.4.38.
The test is now relative to the largest retained eigenvalue with a floor at
`nu` — eigenvalues are formed as `singular**2 - nu` and cannot resolve below
that shift, so without the floor a zero operator has no scale to be relative
*to*.

## Diagnostics that answer the question they raise

**`nystrom_preconditioner` reports `spectrum_span`**, the smallest retained
eigenvalue over the largest. The convergence bound is stated in terms of the
`mu`-effective dimension, which a caller does not usually know; the span says
whether the sketch reached the decaying tail or sat on a flat plateau.
`nystrom_preconditioner_adaptive` acts on it, doubling the rank until it clears
a target and returning the rank used. A geometrically decaying spectrum stops
at rank 16; a flat one grows to the cap and reports span 1.0, which is the
honest answer, since no rank preconditions a flat spectrum.

**`RECYCLE_DRIFT_ADVISORY`** gives the drift diagnostic a threshold, calibrated
on a continuation of a 200×200 operator:

| drift | warm / cold iterations | verdict |
|---|---|---|
| 3.5e-05 | 12 / 21 | reuse helps |
| 3.5e-02 | 18 / 21 | reuse helps |
| 0.33 | 28 / 30 | reuse helps |
| 0.58 | 69 / 70 | reuse helps |
| 0.73 | 449 / 432 | reuse costs 4% |
| 0.87 | neither converges | operator is the problem |

The number matters less than the shape of the curve: reuse pays far further
into the drift range than caution would suggest, and the penalty for keeping a
stale pair is a few percent rather than a factor. A conservative threshold
throws away most of the benefit.

**Forward mode through the windowed rule** now raises an error that names
`block_thomas_checkpointed_fn`. JAX's own message — "can't apply forward-mode
autodiff (jvp) to a custom\_vjp function" — is true and tells you nothing, and a
code taking reverse-mode gradients and forward-mode audits needs both paths.

## Scope coverage

New tests pin tridiagonal behaviour across eight decades of scaling, the
complex banded path to `n=512` at three bandwidths including the real-RHS
specialisation, dtype promotion at the public entry points, and the
forward-mode message.

## Upgrading

Nothing is removed and no signature changes incompatibly. Two behaviour changes
are worth knowing about:

- Tridiagonal solves on systems scaled far from unity now use a different pivot
  floor. Results change only where the old fixed constant was the wrong scale,
  which is the point.
- `build_fourier_helmholtz_operator` returns the caller's precision rather than
  `float64`. Code that relied on the silent upgrade should pass float64
  geometry explicitly.
