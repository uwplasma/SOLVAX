# Release 0.20.0

## Matrix-free rectangular least squares

This release adds a differentiable solver for overdetermined nonlinear
residuals. `gauss_newton_least_squares` uses JAX JVP and VJP actions instead of
forming a dense Jacobian, solves damped normal equations with PCG, and accepts
steps by their measured trust ratio. Applications can provide an admissibility
predicate and a normal-equation preconditioner without weakening the residual
or stationarity gates.

The result includes accepted and rejected steps, nonlinear and linear work,
the final damping, convergence flags, and fixed-shape histories. These fields
make cold-start and outer-optimization costs explicit without retaining the
nonlinear iteration tape.

## Exact implicit stationarity derivatives

`implicit_least_squares` differentiates

$$
J(x, p)^T r(x, p)=0.
$$

The derivative includes the residual-weighted Hessian term as well as
`J.T J`; it is therefore the derivative of the solved least-squares problem,
not a zero-residual approximation. The primitive supports matrix-free tangent
and adjoint solves and composes with application-level custom VJPs.

## Verification

The new module has complete statement and branch coverage in its focused test
set. Tests cover square and rectangular problems, nonzero-residual implicit
derivatives, JIT execution, PyTrees, preconditioning, admissibility,
globalization, and invalid controls. The complete hosted compatibility, type,
documentation, macOS, and coverage matrix passed on the feature commit before
this release candidate was cut.
