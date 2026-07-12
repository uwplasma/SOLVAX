# Choosing a method

Choose from mathematical structure first, then hardware and differentiation
requirements. No solver is uniformly best.

## Decision table

| Problem structure | First choice | Useful alternative | Main caveat |
|---|---|---|---|
| Scalar tridiagonal, many columns | `tridiagonal_solve` | banded LU | no pivoting; requires a stable tridiagonal system |
| Dense block-tridiagonal | block Thomas | GMRES + block-Thomas preconditioner | block elimination assumes nonsingular Schur complements |
| Narrow nonperiodic band | banded LU | GMRES + banded preconditioner | static rather than dynamic pivoting |
| Narrow periodic band | periodic banded LU | GMRES with core inverse | low-rank corner model must match the operator |
| General Hermitian positive definite | PCG | FGMRES | PCG fails explicitly on nonpositive curvature |
| General nonsymmetric or indefinite | FGMRES | native SuperLU on CPU | memory grows with restart size |
| Slowly varying sequence | GCROT | warm-started FGMRES | recycle storage and startup QR cost |
| Contractive nonlinear partitioned map | Aitken | Anderson mixing | neither method makes a noncontractive map globally convergent |
| General sparse CPU solve outside JAX | native SuperLU | FGMRES | no `jit`, `vmap`, or `grad` |

## Direct versus iterative

A direct structured method is attractive when the structure is exact and the
factors fit in memory. It has predictable work, handles several right-hand
sides efficiently, and can reuse a factorization. An iterative method is
attractive when only an operator action is available, fill-in would be large,
or a good approximate inverse is substantially cheaper than an exact solve.

For $N$ blocks of size $m$, block Thomas costs approximately $O(Nm^3)$ to
factor and $O(Nm^2 n_{rhs})$ to solve. Dense LU on the assembled $Nm$ system
costs $O(N^3m^3)$ and discards the radial or spectral structure. FGMRES avoids
factorization, but stores $O(nm_r)$ basis data for restart size $m_r$ and pays
one operator and preconditioner application per Arnoldi step.

## PCG versus GMRES

Use PCG only when $A$ is Hermitian positive definite and the preconditioner is
positive definite. Under those assumptions it uses short recurrences and much
less memory than GMRES {cite}`hestenes1952,saad2003`. If the assumptions are
uncertain, FGMRES is safer: it accepts nonsymmetry, indefiniteness, and changing
preconditioners, at the price of orthogonalization and restart storage.

MINRES is often preferable to CG for Hermitian indefinite problems, but SOLVAX
does not currently implement MINRES {cite}`paige1975`. BiCGSTAB uses short
recurrences for nonsymmetric systems, but its convergence can be irregular;
SOLVAX instead provides the more memory-intensive, residual-minimizing FGMRES
family {cite}`vorst1992,saad2003`.

## FGMRES versus GCROT

Use FGMRES for unrelated systems. Use GCROT when solving
$A(\mu_i)x_i=b_i$ across continuation steps, optimization iterations, time
steps, or repeated right-hand sides. Recycling pays when difficult spectral
directions persist between systems. It can hurt when the operator changes
abruptly or when the system is already cheap.

SOLVAX retains one normalized cycle correction per restart in a FIFO recycle
space. This is simpler than harmonic-Ritz GCRO-DR and should not be described as
an identical implementation of that algorithm {cite}`parks2006,morgan2002`.

## Aitken versus Anderson

Aitken acceleration stores one previous residual and chooses a safeguarded
scalar relaxation. It is inexpensive and suitable when a single relaxation
parameter captures most of the coupling stiffness. Anderson mixing uses a
history of residuals and solves a small regularized least-squares problem; it
can capture several slow coupling directions at higher storage and dense
history cost {cite}`anderson1965,walker2011`.

## Differentiation strategy

There are two distinct derivatives:

1. **Algorithmic derivative:** differentiate every executed iteration. This is
   useful when the finite iteration is itself the model.
2. **Implicit derivative:** differentiate the converged equation. This is the
   usual choice for equilibrium, steady-state, and linear-response models.

Use {func}`solvax.implicit.linear_solve`,
{func}`solvax.implicit.root_solve`, or
{func}`solvax.pcg.pcg_linear_solve` for the second interpretation. The backward
solve must be at least as accurate as the gradient application requires.

## Precision and hardware

- `tridiagonal_solve(method="auto")` uses the reproducible Thomas path on CPU
  and the fused JAX/XLA path on accelerators.
- Mixed-precision refinement is valuable only when low-precision factorization
  is materially faster and $\kappa(A)u_{low}<1$.
- Native SuperLU is a host-only escape hatch, not an accelerator solver.
- Larger Krylov restart spaces often reduce iteration counts but increase
  memory, compilation size, and orthogonalization work.
