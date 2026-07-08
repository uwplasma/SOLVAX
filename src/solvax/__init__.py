"""solvax: differentiable structured linear solvers and matrix-free methods in JAX.

Generic solver infrastructure for kinetic and PDE codes: batched structured
direct solves, preconditioned/recycled Krylov methods, and implicit
differentiation — everything jit/vmap/grad-transparent unless explicitly
marked as a host-side native bridge.
"""

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
    block_thomas_solve,
    block_thomas_truncated,
)
from solvax.implicit import linear_solve, root_solve
from solvax.krylov import KrylovSolution, gcrot, gmres
from solvax.native import SpluFactorization, splu_solve
from solvax.refine import as_low_precision, iterative_refinement

__version__ = "0.1.0.dev0"

__all__ = [
    "BandedLUFactors",
    "PeriodicBandedLUFactors",
    "banded_matvec",
    "lu_factor_banded",
    "lu_factor_banded_periodic",
    "lu_solve_banded",
    "lu_solve_banded_periodic",
    "BlockTridiagFactors",
    "block_thomas",
    "block_thomas_factor",
    "block_thomas_solve",
    "block_thomas_truncated",
    "KrylovSolution",
    "gmres",
    "gcrot",
    "linear_solve",
    "root_solve",
    "iterative_refinement",
    "as_low_precision",
    "SpluFactorization",
    "splu_solve",
    "__version__",
]
