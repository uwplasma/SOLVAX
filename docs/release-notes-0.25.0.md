# Release 0.25.0

## MUMPS as an explicit, memory-admitted factorization

`SpluFactorization` accepts an optional MUMPS backend alongside the default
SuperLU (#118). Numerical factorization is refused after symbolic analysis if
the estimated memory (`INFOG(16)`, `INFOG(21)`) exceeds a required byte budget
with a configurable margin, and the internal workspace is capped through
`ICNTL(23)`. Real and complex `N`, `T` and `H` substitutions, and multiple
right-hand sides, reuse one factorization. Execution is single-rank CPU through
`MPI.COMM_SELF`; there is no distributed factorization, no automatic backend
selection and no JAX tracing. The workspace cap is not a process memory cap, so
callers still need their own headroom. PyMUMPS and compatible MUMPS/MPI
libraries are optional dependencies, and a missing binding raises an
actionable error. SuperLU remains the default.

## Reusable scalar tridiagonal factors

Public array-only factor and solve functions let callers whose coefficient
bands stay fixed skip repeated Thomas elimination (#117). The checked solve
keeps the existing pivot, backward-residual and fallback contract, and the
factor object retains the exact input bands so certification never relies on
reconstructing them. On a warmed CPU benchmark of shape `(257, 64)` over 200
checked solves, reuse takes a solve from 0.228 ms to 0.176 ms (1.30x) with
identical solutions and diagnostics; the unchecked path is 2.60x faster.

## Complex matrices keep their phase under equilibration

`equilibrate` cast its input to float64 before computing magnitudes, which
silently discarded the imaginary part of a complex matrix and returned the
scaling of a different operator (#116). It now keeps complex entries, with the
row and column scales still real. On `[[1e-6+1j, 2j], [0, 3000-4000j]]` the
previous release's diagonal-scaling identity was off by up to 1e6.

## Least-squares reports its inner solve

`gauss_newton_least_squares` exposes aggregate and per-step PCG convergence and
relative residuals (#119), which its documentation already told callers to
inspect. Accepting an inexact step on a good trust ratio remains the default;
`require_linear_convergence=True` opts into rejecting a step whose inner solve
missed its tolerance.
