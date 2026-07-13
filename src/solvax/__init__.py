"""solvax: differentiable structured linear solvers and matrix-free methods in JAX.

Generic solver infrastructure for kinetic and PDE codes: batched structured
direct solves, preconditioned/recycled Krylov methods, memory-chunked
autodiff, and implicit differentiation — everything jit/vmap/grad-transparent
unless explicitly marked as a host-side native bridge.
"""

from solvax.autodiff import (
    auto_chunk_size,
    chunk_map,
    chunked_jacfwd,
    chunked_jacobian,
    chunked_jacrev,
)
from solvax.banded import (
    BandedLUFactors,
    PeriodicBandedLUFactors,
    banded_matvec,
    lu_factor_banded,
    lu_factor_banded_periodic,
    lu_solve_banded,
    lu_solve_banded_periodic,
)
from solvax.direct import (
    BlockTridiagFactors,
    block_thomas,
    block_thomas_factor,
    block_thomas_factor_fn,
    block_thomas_solve,
    block_thomas_truncated,
    block_thomas_truncated_fn,
    block_tridiag_matvec,
    block_tridiag_relative_residual,
    mixed_precision_block_thomas,
)
from solvax.fixed_point import (
    FixedPointSolution,
    aitken_fixed_point,
    aitken_relaxation,
    anderson_mixing,
)
from solvax.implicit import linear_solve, root_solve
from solvax.krylov import KrylovSolution, gcrot, gmres
from solvax.native import SpluFactorization, splu_solve
from solvax.operators import (
    BlockTridiagonalOperator,
    BorderedOperator,
    KroneckerOperator,
    MatrixFreeOperator,
    SumOperator,
    schur_projected_precond,
)
from solvax.pcg import PCGDiagnostics, PCGSolution, pcg, pcg_linear_solve, status_name
from solvax.precond import (
    block_jacobi,
    coarse_operator,
    jacobi,
    kronecker_nkp,
    line_smoother,
    mixed_precision,
    nearest_kronecker,
    p_multigrid,
)
from solvax.refine import as_low_precision, iterative_refinement
from solvax.tridiagonal import tridiagonal_solve

__version__ = "0.7.2"

__all__ = [
    "BandedLUFactors",
    "PeriodicBandedLUFactors",
    "banded_matvec",
    "lu_factor_banded",
    "lu_factor_banded_periodic",
    "lu_solve_banded",
    "lu_solve_banded_periodic",
    "BlockTridiagFactors",
    "block_tridiag_matvec",
    "block_tridiag_relative_residual",
    "block_thomas",
    "block_thomas_factor",
    "block_thomas_factor_fn",
    "block_thomas_solve",
    "block_thomas_truncated",
    "block_thomas_truncated_fn",
    "mixed_precision_block_thomas",
    "tridiagonal_solve",
    "FixedPointSolution",
    "aitken_fixed_point",
    "aitken_relaxation",
    "anderson_mixing",
    "KrylovSolution",
    "gmres",
    "gcrot",
    "PCGSolution",
    "PCGDiagnostics",
    "pcg",
    "pcg_linear_solve",
    "status_name",
    "linear_solve",
    "root_solve",
    "MatrixFreeOperator",
    "SumOperator",
    "KroneckerOperator",
    "BlockTridiagonalOperator",
    "BorderedOperator",
    "schur_projected_precond",
    "jacobi",
    "block_jacobi",
    "coarse_operator",
    "line_smoother",
    "p_multigrid",
    "mixed_precision",
    "kronecker_nkp",
    "nearest_kronecker",
    "iterative_refinement",
    "as_low_precision",
    "chunk_map",
    "auto_chunk_size",
    "chunked_jacfwd",
    "chunked_jacrev",
    "chunked_jacobian",
    "SpluFactorization",
    "splu_solve",
    "__version__",
]
