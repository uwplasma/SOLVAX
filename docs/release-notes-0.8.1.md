# Release 0.8.1

SOLVAX 0.8.1 adds `solvax.elliptic`, a spectral Fourier--Helmholtz elliptic
solve for separable Helmholtz-type problems on a periodic axis and a bounded
axis (`build_fourier_helmholtz_operator`, `solve_fourier_helmholtz`,
`FourierHelmholtzOperator`). The periodic axis is Fourier-transformed and the
remaining per-mode tridiagonal system in the bounded axis is solved in one
batched call; the whole inversion is `jit`/`grad`/`vmap` transparent.

This is the $\nabla_\perp^2\phi=\rho$ inversion used by reduced drift-plane and
vorticity models: build the operator once per geometry, reuse it every
timestep. See {doc}`solvers/elliptic`.
