# Eigenpair API

`generalized_eigenpairs` is the targeted eigensolver for Hermitian block-tridiagonal
pencils with first-order implicit derivatives; see {doc}`../solvers/generalized_eigenpairs`.

## Application-supplied eigenpairs

`eigenpair_reverse` accepts application-supplied primal and left eigensolvers,
then differentiates their certified simple eigenpair implicitly. Eigenvalue
cotangents use the left/right perturbation identity. Eigenvector cotangents use
Nelson's bordered system, with optional application-supplied preconditioning
and a transposed reduced-resolvent solve.

The eigensolver iteration is deliberately outside SOLVAX and outside the
autodiff tape. This keeps the interface useful for propagator, rational,
Jacobi--Davidson, or external eigensolvers without duplicating them here. A
condition-number guard rejects clusters and exceptional points where a
single-mode derivative is not well defined.

```{eval-rst}
.. automodule:: solvax.eigen
   :members:
   :member-order: bysource
```
