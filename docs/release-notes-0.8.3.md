# Release 0.8.3

SOLVAX 0.8.3 adds `additive_preconditioner`, a positive weighted combination
$B=\sum_iw_iB_i$ of self-adjoint positive-definite inverse actions. Unlike
sequential (multiplicative) composition, the additive sum preserves symmetry,
so line, block, and Schwarz-style pieces can be combined for PCG on arrays or
arbitrary matching pytrees. See the symmetric additive composition section of
{doc}`preconditioners`.
