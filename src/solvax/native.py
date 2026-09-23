"""Host-side sparse-direct bridges for SciPy SuperLU and optional MUMPS.

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
``pip install solvax[native]``. MUMPS is selected explicitly and additionally
requires PyMUMPS plus compatible MUMPS and MPI libraries.

References
----------
- X. S. Li, *An Overview of SuperLU*, ACM Trans. Math. Softw. 31(3), 302
  (2005), DOI 10.1145/1089014.1089017.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_MUMPS_MB = 1_000_000


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


def _is_tracer(value) -> bool:
    """Whether ``value`` is a JAX tracer, without depending on where it lives.

    ``jax.core.Tracer`` is the documented check, but ``jax.core`` has been
    shrinking toward private status across JAX releases and already resolves
    into ``jax._src``. Using it while it exists and degrading to the tracer
    protocol otherwise keeps this working across an upgrade that moves it,
    instead of raising ``AttributeError`` from inside a guard whose whole job
    is to produce a clear error.
    """
    tracer = getattr(jax, "core", None)
    tracer = getattr(tracer, "Tracer", None) if tracer is not None else None
    if tracer is not None:
        return isinstance(value, tracer)
    # A tracer carries an abstract value and no concrete buffer; a committed
    # array carries both a shape and its data.
    return hasattr(value, "aval") and not hasattr(value, "addressable_shards")


def _check_not_traced(b, name: str) -> None:
    """Raise if ``b`` is a JAX tracer (i.e. we are under jit/vmap/grad)."""
    if _is_tracer(b):
        raise RuntimeError(
            f"solvax.native.{name} runs a native sparse solver on the host and is "
            "not traceable: it must not be called under jit, vmap, or grad. "
            "Call it eagerly on concrete arrays, or wrap it in "
            "jax.pure_callback if staging is required."
        )


def _import_mumps(dtype: np.dtype):
    """Import a PyMUMPS context matching ``dtype`` and MPI.COMM_SELF."""
    try:
        import mumps
        from mpi4py import MPI
    except ImportError as err:
        raise ImportError(
            "backend='mumps' requires PyMUMPS, mpi4py, and compatible MUMPS/MPI "
            "libraries; install the Python dependencies with `pip install "
            "solvax[mumps]` after providing the system libraries."
        ) from err

    context_names = {
        np.dtype(np.float32): "SMumpsContext",
        np.dtype(np.float64): "DMumpsContext",
        np.dtype(np.complex64): "CMumpsContext",
        np.dtype(np.complex128): "ZMumpsContext",
    }
    try:
        context_name = context_names[dtype]
    except KeyError as err:
        raise TypeError(
            "the MUMPS backend supports float32, float64, complex64, and complex128 "
            f"matrices, got {dtype}"
        ) from err
    try:
        return getattr(mumps, context_name), MPI.COMM_SELF
    except AttributeError as err:
        raise ImportError(
            f"the installed PyMUMPS binding does not provide {context_name}, "
            f"required for {dtype} matrices"
        ) from err


def _mumps_memory_bytes(value: int, field: str) -> int:
    """Convert a MUMPS memory INFOG value from decimal MB to bytes."""
    value = int(value)
    if value < 0:
        raise RuntimeError(
            f"MUMPS returned invalid negative {field}={value}; memory INFOG fields "
            "are measured directly in millions of bytes"
        )
    return value * _MUMPS_MB


class _MumpsFactorization:
    """Owned PyMUMPS context with analysis-before-factor memory admission."""

    def __init__(self, matrix, memory_limit_bytes: int, memory_safety_factor: float):
        self._context = None
        self._size = matrix.shape[0]
        self._dtype = np.dtype(matrix.dtype)
        context_type, comm = _import_mumps(self._dtype)
        context = context_type(par=1, sym=0, comm=comm)
        self._context = context
        try:
            context.set_silent()
            context.set_centralized_sparse(matrix.tocoo())
            context.run(job=1)

            self.symbolic_memory_bytes = _mumps_memory_bytes(
                context.get_infog(16), "INFOG(16)"
            )
            admitted_bytes = int(
                np.ceil(self.symbolic_memory_bytes * memory_safety_factor)
            )
            if admitted_bytes > memory_limit_bytes:
                raise MemoryError(
                    "MUMPS factorization refused after symbolic analysis: "
                    f"estimated {self.symbolic_memory_bytes} bytes, "
                    f"{admitted_bytes} bytes with safety factor "
                    f"{memory_safety_factor:g}, budget {memory_limit_bytes} bytes"
                )

            # ICNTL(23) caps MUMPS internal workspace in integer decimal MB.
            context.set_icntl(23, memory_limit_bytes // _MUMPS_MB)
            context.run(job=2)
            self.effective_memory_bytes = _mumps_memory_bytes(
                context.get_infog(21), "INFOG(21)"
            )
        except BaseException:
            self.close()
            raise

    def solve(self, b: np.ndarray, trans: str) -> np.ndarray:
        """Solve one or more right-hand sides with the stored factors."""
        if self._context is None:
            raise RuntimeError("MUMPS factorization is closed")
        rhs = np.asarray(b)
        if np.iscomplexobj(rhs) and not np.issubdtype(self._dtype, np.complexfloating):
            raise TypeError("a real MUMPS factorization cannot solve a complex right-hand side")
        rhs = np.asarray(rhs, dtype=self._dtype)
        if rhs.ndim not in {1, 2} or rhs.shape[0] != self._size:
            raise ValueError(
                f"b must have shape ({self._size},) or ({self._size}, n_rhs)"
            )
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            return np.empty_like(rhs)

        conjugate = trans == "H" and np.issubdtype(self._dtype, np.complexfloating)
        context = self._context

        def solve_column(column: np.ndarray) -> np.ndarray:
            solution = np.array(np.conjugate(column) if conjugate else column, copy=True)
            context.set_rhs(solution)
            context.run(job=3)
            return np.conjugate(solution) if conjugate else solution

        def solve_block(block: np.ndarray) -> np.ndarray:
            # One MUMPS solve phase for every column: PyMUMPS's ``set_rhs`` takes
            # a single vector, so the right-hand-side count and leading dimension
            # are set on the MUMPS structure directly, and reset afterwards
            # because ``set_rhs`` does not reset them.
            solution = np.array(
                np.conjugate(block) if conjugate else block, order="F", copy=True
            )
            try:
                context._refs.update(rhs=solution)
                context.id.nrhs = solution.shape[1]
                context.id.lrhs = self._size
                context.id.rhs = context.cast_array(solution)
                context.run(job=3)
            finally:
                context.id.nrhs = 1
            return np.conjugate(solution) if conjugate else solution

        multi_rhs = all(hasattr(context, name) for name in ("id", "_refs", "cast_array"))
        try:
            context.set_icntl(9, 1 if trans == "N" else 0)
            if rhs.ndim == 1:
                return solve_column(rhs)
            if multi_rhs:
                return solve_block(rhs)
            return np.column_stack(
                [solve_column(rhs[:, index]) for index in range(rhs.shape[1])]
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release native MUMPS storage, once."""
        context, self._context = self._context, None
        if context is not None:
            context.destroy()


class SpluFactorization:
    """Reusable host-side sparse LU factorization.

    Factor once, solve many times::

        lu = SpluFactorization(A_csr)
        x1 = lu.solve(b1)
        x2 = lu.solve(b2)

    Attributes:
        shape: shape of the factored matrix.
    """

    def __init__(
        self,
        matrix,
        *,
        backend: str = "superlu",
        memory_limit_bytes: int | None = None,
        memory_safety_factor: float | None = None,
    ):
        """Factor ``matrix`` with SuperLU or explicit MUMPS.

        Args:
            matrix: scipy sparse matrix in CSR or CSC format (anything with
                ``.tocsc()``).
            backend: ``"superlu"`` (the default) or ``"mumps"``.
            memory_limit_bytes: required MUMPS per-process memory budget.
            memory_safety_factor: multiplier applied to the MUMPS symbolic
                estimate before admission. Defaults to 1.2 and must be at
                least one. Only supported by MUMPS.
        """
        sparse, sparse_linalg = _import_scipy_sparse()
        if not sparse.issparse(matrix):
            raise TypeError(
                "SpluFactorization expects a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        if backend not in {"superlu", "mumps"}:
            raise ValueError("backend must be 'superlu' or 'mumps'")
        self.shape = matrix.shape
        self.backend = backend
        self.symbolic_memory_bytes: int | None = None
        self.effective_memory_bytes: int | None = None
        self._lu = None
        self._mumps = None
        if backend == "superlu":
            if memory_limit_bytes is not None or memory_safety_factor is not None:
                raise ValueError("memory controls are only supported by backend='mumps'")
            self._lu = sparse_linalg.splu(matrix.tocsc())
            return

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("the MUMPS backend requires a square matrix")
        if memory_limit_bytes is None:
            raise ValueError("memory_limit_bytes is required for backend='mumps'")
        if not isinstance(memory_limit_bytes, (int, np.integer)) or isinstance(
            memory_limit_bytes, (bool, np.bool_)
        ):
            raise TypeError("memory_limit_bytes must be an integer")
        memory_limit_bytes = int(memory_limit_bytes)
        if memory_limit_bytes < _MUMPS_MB:
            raise ValueError("memory_limit_bytes must be at least 1,000,000 for MUMPS")
        memory_limit_mb = memory_limit_bytes // _MUMPS_MB
        if memory_limit_mb > np.iinfo(np.int32).max:
            raise ValueError("memory_limit_bytes exceeds the supported ICNTL(23) range")
        if memory_safety_factor is None:
            memory_safety_factor = 1.2
        if not np.isfinite(memory_safety_factor) or memory_safety_factor < 1:
            raise ValueError("memory_safety_factor must be finite and at least 1")

        self._mumps = _MumpsFactorization(
            matrix, memory_limit_bytes, float(memory_safety_factor)
        )
        self.symbolic_memory_bytes = self._mumps.symbolic_memory_bytes
        self.effective_memory_bytes = self._mumps.effective_memory_bytes

    def _solve_numpy(self, b, *, trans: str = "N") -> np.ndarray:
        """Apply stored factors without a host/device round trip."""

        if trans not in {"N", "T", "H"}:
            raise ValueError("trans must be 'N', 'T', or 'H'")

        if self.backend == "mumps":
            if self._mumps is None:
                raise RuntimeError("factorization is closed")
            return self._mumps.solve(np.asarray(b), trans)
        if self._lu is None:
            raise RuntimeError("factorization is closed")
        return self._lu.solve(np.asarray(b), trans=trans)

    def close(self) -> None:
        """Release native factorization storage; repeated calls are harmless."""
        if self._mumps is not None:
            self._mumps.close()
            self._mumps = None
        self._lu = None

    def __enter__(self) -> SpluFactorization:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def solve(self, b, *, trans: str = "N") -> jax.Array:
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
        if trans not in {"N", "T", "H"}:
            raise ValueError("trans must be 'N', 'T', or 'H'")
        return jnp.asarray(self._solve_numpy(b, trans=trans))


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
