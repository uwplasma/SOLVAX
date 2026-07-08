"""Host-side sparse-direct bridge: SuperLU factorization via SciPy.

A thin, *non-differentiable* escape hatch to battle-tested sparse LU
(SuperLU, through :func:`scipy.sparse.linalg.splu`) for general sparse
systems that fall outside the structured solvers in ``solvax.direct``. The
factorization and triangular solves run on the host CPU, entirely outside
the JAX trace machinery — these functions must **not** be called under
``jit``, ``vmap``, or ``grad``. A guard raises a clear :class:`RuntimeError`
if a traced value is passed; if you need staging, wrap the call in
:func:`jax.pure_callback` yourself, and for gradients combine with
``solvax.implicit.linear_solve`` outside jit.

SciPy is an optional dependency, imported lazily; install it with
``pip install solvax[native]``.

References
----------
- X. S. Li, *An Overview of SuperLU*, ACM Trans. Math. Softw. 31(3), 302
  (2005), DOI 10.1145/1089014.1089017.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _import_scipy_sparse():
    """Import scipy.sparse lazily with an actionable error message."""
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except ImportError as err:
        raise ImportError(
            "solvax.native requires SciPy for the SuperLU bridge; install it "
            "with `pip install solvax[native]` (or `pip install scipy`)."
        ) from err
    return sparse, sparse_linalg


def _check_not_traced(b, name: str) -> None:
    """Raise if ``b`` is a JAX tracer (i.e. we are under jit/vmap/grad)."""
    if isinstance(b, jax.core.Tracer):
        raise RuntimeError(
            f"solvax.native.{name} runs SciPy's SuperLU on the host and is "
            "not traceable: it must not be called under jit, vmap, or grad. "
            "Call it eagerly on concrete arrays, or wrap it in "
            "jax.pure_callback if staging is required."
        )


class SpluFactorization:
    """Reusable SuperLU factorization of a scipy sparse matrix.

    Factor once, solve many times::

        lu = SpluFactorization(A_csr)
        x1 = lu.solve(b1)
        x2 = lu.solve(b2)

    Attributes:
        shape: shape of the factored matrix.
    """

    def __init__(self, matrix):
        """Factor ``matrix`` with SuperLU.

        Args:
            matrix: scipy sparse matrix in CSR or CSC format (anything with
                ``.tocsc()``); converted to CSC as SuperLU requires.
        """
        sparse, sparse_linalg = _import_scipy_sparse()
        if not sparse.issparse(matrix):
            raise TypeError(
                "SpluFactorization expects a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        self.shape = matrix.shape
        self._lu = sparse_linalg.splu(matrix.tocsc())

    def solve(self, b) -> jax.Array:
        """Solve ``A x = b`` with the stored factors.

        Args:
            b: concrete (non-traced) right-hand side, shape ``(n,)`` or
                ``(n, n_rhs)``.

        Returns:
            The solution as a jax array.

        Raises:
            RuntimeError: if called with a traced value (under jit/vmap/grad).
        """
        _check_not_traced(b, "SpluFactorization.solve")
        return jnp.asarray(self._lu.solve(np.asarray(b)))


def splu_solve(matrix, b) -> jax.Array:
    """One-shot host-side sparse-direct solve of ``matrix @ x = b``.

    Convenience wrapper: :class:`SpluFactorization` then a single solve. For
    repeated solves with the same matrix, construct the factorization once
    and reuse it.

    Args:
        matrix: scipy sparse matrix (CSR or CSC).
        b: concrete (non-traced) right-hand side, shape ``(n,)`` or
            ``(n, n_rhs)``.

    Returns:
        The solution as a jax array.

    Raises:
        RuntimeError: if called with a traced value (under jit/vmap/grad).
        ImportError: if SciPy is not installed.
    """
    _check_not_traced(b, "splu_solve")
    return SpluFactorization(matrix).solve(b)
