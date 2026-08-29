# Release 0.19.0

## Exact selected spectral tails

`block_thomas_selected_tail_fn` complements the selected-head solver for
generated block-tridiagonal systems. For a right-hand side confined to its
supplied low-block prefix, it returns the exact highest requested blocks of the
full solution:

```python
import solvax as sx

x_tail = sx.block_thomas_selected_tail_fn(
    block_fn,
    n_blocks=N,
    rhs_low=rhs_low,
    keep_highest=2,
)
```

A selected-head solve followed by a high-to-low transfer-map sweep visits every
block while retaining only the source prefix and requested maps. Dense primal
workspace is therefore
$O((\mathtt{source\_blocks}+\mathtt{keep\_highest})m^2)$ and independent of the
chain length. This orientation also supports a singular leading diagonal block
when the complete system and its high-to-low Schur complements are nonsingular.
The returned blocks preserve ascending spectral order and support one or
multiple right-hand sides, JIT compilation, and ordinary autodiff. Ordinary
reverse mode tapes the generated sweeps; this release makes no bounded-adjoint
claim for the selected-tail entry point.

The intended first downstream use is a stopped-gradient kinetic convergence
diagnostic: transport moments can continue to use the exact selected head,
while the opposite spectral boundary is measured without allocating a full
distribution.

## Reusable continuation and fixed-work loops

Pseudo-arclength correctors now accept the tangent and predictor as dynamic
arguments to one compiled bordered solve. A mutually exclusive parameterized
bordered preconditioner can see those dynamic values without creating a new
closure at every branch step.

GMRES and Newton--Krylov also gain opt-in fixed-length scan control. Converged
slots are masked, early exit remains the default, happy breakdown carries
finite zero cotangents, and nonfinite Krylov norms fail closed rather than
being mistaken for a zero residual.

## Verification

The release candidate passed 759 tests with 6 optional-backend skips locally,
including 101 focused structured-direct and public-API tests. Hosted evidence
passed two exhaustive current-stack shards, the Python 3.10 minimum stack,
the optional advanced backend, macOS, Ruff, MyPy, warning-clean documentation,
and combined branch coverage. Isolated 0.19.0 artifacts measured 155,209 bytes
for the wheel and 500,581 bytes for the sdist and passed `twine check`.
