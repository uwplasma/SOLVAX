# Eigenvalue API

`harmonic_krylov_schur` finds individual interior modes without forming the
operator matrix. `block_harmonic_krylov` retains several competing branches,
accepts a recycled structured candidate subspace, and can generate a rational
subspace with an independently supplied shifted-inverse action. Both certify
convergence with residuals from the original operator.

`eigenvalue` differentiates a simple eigenvalue with the left/right
perturbation identity. `eigenpair` additionally differentiates the right
eigenvector through Nelson's bordered system, which supports phase-invariant
observables built from the mode structure. Both reject an ill-conditioned
simple-mode sensitivity near a cluster or exceptional point.

```{eval-rst}
.. automodule:: solvax.eigen
   :members:
   :member-order: bysource
```
