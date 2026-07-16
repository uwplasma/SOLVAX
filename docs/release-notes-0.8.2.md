# Release 0.8.2

SOLVAX 0.8.2 adds `galerkin_deflation`, a balanced symmetry-preserving Galerkin
coarse correction $S+(I-SA)PA_c^{-1}P^T(I-AS)$ for fixed SPD preconditioners
used with PCG. Given a symmetric smoother, a prolongation, and the Galerkin
coarse operator $A_c=P^TAP$, the two-level inverse stays symmetric positive
definite, so it is safe inside the strict PCG preconditioner contract. See the
Galerkin-deflation section of {doc}`preconditioners`.
