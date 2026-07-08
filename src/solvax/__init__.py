"""solvax: differentiable structured linear solvers and matrix-free methods in JAX.

Generic solver infrastructure for kinetic and PDE codes: batched structured
direct solves, preconditioned/recycled Krylov methods, and implicit
differentiation — everything jit/vmap/grad-transparent unless explicitly
marked as a host-side native bridge.
"""

from solvax.direct import (
    BlockTridiagFactors,
    block_thomas,
    block_thomas_factor,
    block_thomas_solve,
    block_thomas_truncated,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "BlockTridiagFactors",
    "block_thomas",
    "block_thomas_factor",
    "block_thomas_solve",
    "block_thomas_truncated",
    "__version__",
]
