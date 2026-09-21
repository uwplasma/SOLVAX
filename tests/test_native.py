"""Tests for solvax.native host-side sparse-direct bridges."""

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import SpluFactorization, splu_solve
from solvax import native as native_module

jax.config.update("jax_enable_x64", True)

scipy_sparse = pytest.importorskip("scipy.sparse")


def make_sparse_system(n=80, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = scipy_sparse.random(
        n, n, density=density, random_state=rng, format="csr"
    ) + 10.0 * scipy_sparse.eye(n, format="csr")
    b = rng.standard_normal(n)
    return a.tocsr(), jnp.asarray(b)


def test_splu_solve_matches_dense():
    a, b = make_sparse_system()
    x = splu_solve(a, b)
    assert isinstance(x, jax.Array)
    x_dense = np.linalg.solve(a.toarray(), np.asarray(b))
    assert np.allclose(np.asarray(x), x_dense, atol=1e-10)


def test_splu_solve_accepts_csc_and_multiple_rhs():
    a, _ = make_sparse_system(seed=1)
    rng = np.random.default_rng(2)
    rhs = jnp.asarray(rng.standard_normal((a.shape[0], 3)))
    x = splu_solve(a.tocsc(), rhs)
    assert np.allclose(a.toarray() @ np.asarray(x), np.asarray(rhs), atol=1e-10)


def test_factorization_reuse_identical():
    a, b = make_sparse_system(seed=3)
    lu = SpluFactorization(a)
    x_once = splu_solve(a, b)
    x1 = lu.solve(b)
    x2 = lu.solve(b)
    assert np.array_equal(np.asarray(x1), np.asarray(x2))
    assert np.array_equal(np.asarray(x1), np.asarray(x_once))
    # A second right-hand side reuses the same factors.
    x3 = lu.solve(2.0 * b)
    assert np.allclose(np.asarray(x3), 2.0 * np.asarray(x1), atol=1e-10)
    xh = lu.solve(b, trans="H")
    assert np.allclose(a.toarray().conj().T @ np.asarray(xh), np.asarray(b), atol=1e-10)
    with pytest.raises(ValueError, match="trans"):
        lu.solve(b, trans="bad")


def test_splu_solve_raises_under_jit():
    a, b = make_sparse_system(seed=4)

    with pytest.raises(RuntimeError, match="must not be called under jit"):
        jax.jit(lambda v: splu_solve(a, v))(b)

    lu = SpluFactorization(a)
    with pytest.raises(RuntimeError, match="must not be called under jit"):
        jax.jit(lambda v: lu.solve(v))(b)


def test_rejects_dense_input():
    _, b = make_sparse_system(seed=5)
    with pytest.raises(TypeError, match="scipy sparse"):
        splu_solve(np.eye(b.shape[0]), b)


class FakeMumpsContext:
    """Small PyMUMPS protocol fake; numerical solves still use the supplied matrix."""

    instances = []
    symbolic_mb = 4
    effective_mb = 3
    fail_solve = False

    def __init__(self, *, par, sym, comm):
        assert (par, sym) == (1, 0)
        self.calls = []
        self.icntl = {}
        self.destroyed = False
        self.__class__.instances.append(self)

    def set_silent(self):
        self.calls.append(("silent",))

    def set_centralized_sparse(self, matrix):
        self.matrix = matrix.toarray()
        self.calls.append(("matrix",))

    def run(self, *, job):
        self.calls.append(("run", job))
        if job == 3:
            if self.fail_solve:
                raise RuntimeError("native solve failed")
            operator = self.matrix if self.icntl[9] == 1 else self.matrix.T
            self.rhs[...] = np.linalg.solve(operator, self.rhs)

    def get_infog(self, index):
        return {16: self.symbolic_mb, 21: self.effective_mb}[index]

    def set_icntl(self, index, value):
        self.icntl[index] = value
        self.calls.append(("icntl", index, value))

    def set_rhs(self, rhs):
        self.rhs = rhs
        self.calls.append(("rhs",))

    def destroy(self):
        self.destroyed = True
        self.calls.append(("destroy",))


@pytest.fixture
def fake_mumps(monkeypatch):
    FakeMumpsContext.instances = []
    FakeMumpsContext.symbolic_mb = 4
    FakeMumpsContext.effective_mb = 3
    FakeMumpsContext.fail_solve = False
    monkeypatch.setattr(
        native_module, "_import_mumps", lambda dtype: (FakeMumpsContext, "COMM_SELF")
    )
    return FakeMumpsContext


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("trans", ["N", "T", "H"])
def test_mumps_transpose_modes_and_reuse(fake_mumps, dtype, trans):
    if np.issubdtype(dtype, np.complexfloating):
        matrix = np.array(
            [[4.0 + 0.5j, 1.0 - 0.25j], [2.0 + 0.75j, 3.0 - 0.5j]], dtype=dtype
        )
    else:
        matrix = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=dtype)
    sparse_matrix = scipy_sparse.csr_matrix(matrix)
    rhs = np.array([[1.0 + 0.25j, 2.0], [-1.0j, 3.0 - 0.5j]], dtype=np.complex128)
    if not np.issubdtype(dtype, np.complexfloating):
        rhs = rhs.real.astype(dtype)

    factor = SpluFactorization(
        sparse_matrix, backend="mumps", memory_limit_bytes=8_000_000
    )
    solution = np.asarray(factor.solve(rhs, trans=trans))
    operator = {"N": matrix, "T": matrix.T, "H": matrix.conj().T}[trans]
    assert np.allclose(operator @ solution, rhs)
    assert factor.symbolic_memory_bytes == 4_000_000
    assert factor.effective_memory_bytes == 3_000_000

    context = fake_mumps.instances[-1]
    assert context.calls.count(("run", 1)) == 1
    assert context.calls.count(("run", 2)) == 1
    assert context.calls.count(("run", 3)) == rhs.shape[1]
    assert context.icntl[23] == 8
    factor.close()
    factor.close()
    assert context.calls.count(("destroy",)) == 1
    with pytest.raises(RuntimeError, match="closed"):
        factor.solve(rhs)


def test_mumps_symbolic_memory_refusal_destroys_context(fake_mumps):
    matrix = scipy_sparse.eye(3, dtype=np.float64, format="csr")
    with pytest.raises(MemoryError, match="safety factor 1.5"):
        SpluFactorization(
            matrix,
            backend="mumps",
            memory_limit_bytes=5_000_000,
            memory_safety_factor=1.5,
        )
    context = fake_mumps.instances[-1]
    assert ("run", 2) not in context.calls
    assert context.destroyed


def test_mumps_memory_controls_and_negative_infog(fake_mumps):
    matrix = scipy_sparse.eye(3, dtype=np.float64, format="csr")
    with pytest.raises(ValueError, match="memory_limit_bytes is required"):
        SpluFactorization(matrix, backend="mumps")
    with pytest.raises(ValueError, match="at least 1,000,000"):
        SpluFactorization(matrix, backend="mumps", memory_limit_bytes=999_999)
    with pytest.raises(ValueError, match=r"ICNTL\(23\) range"):
        SpluFactorization(
            matrix,
            backend="mumps",
            memory_limit_bytes=(np.iinfo(np.int32).max + 1) * 1_000_000,
        )
    with pytest.raises(ValueError, match="safety_factor"):
        SpluFactorization(
            matrix,
            backend="mumps",
            memory_limit_bytes=8_000_000,
            memory_safety_factor=0.9,
        )

    fake_mumps.symbolic_mb = -4
    with pytest.raises(RuntimeError, match=r"negative INFOG\(16\)"):
        SpluFactorization(matrix, backend="mumps", memory_limit_bytes=8_000_000)
    assert fake_mumps.instances[-1].destroyed

    fake_mumps.symbolic_mb = 4
    fake_mumps.effective_mb = -3
    with pytest.raises(RuntimeError, match=r"negative INFOG\(21\)"):
        SpluFactorization(matrix, backend="mumps", memory_limit_bytes=8_000_000)
    assert fake_mumps.instances[-1].destroyed


def test_mumps_context_manager_and_rhs_validation(fake_mumps):
    matrix = scipy_sparse.eye(3, dtype=np.float64, format="csr")
    with SpluFactorization(
        matrix, backend="mumps", memory_limit_bytes=8_000_000
    ) as factor:
        context = fake_mumps.instances[-1]
        solve_calls = list(context.calls)
        with pytest.raises(ValueError, match="shape"):
            factor.solve(np.ones(2))
        assert context.calls == solve_calls
        with pytest.raises(TypeError, match="complex right-hand side"):
            factor.solve(np.ones(3, dtype=np.complex128))
        assert context.calls == solve_calls
        empty = factor.solve(np.empty((3, 0)))
        assert empty.shape == (3, 0)
        assert context.calls == solve_calls
    assert fake_mumps.instances[-1].destroyed


def test_native_solve_validates_trans_and_superlu_memory_controls(fake_mumps):
    matrix = scipy_sparse.eye(3, dtype=np.float64, format="csr")
    factor = SpluFactorization(matrix, backend="mumps", memory_limit_bytes=8_000_000)
    context = fake_mumps.instances[-1]
    calls = list(context.calls)
    with pytest.raises(ValueError, match="trans"):
        factor._solve_numpy(np.ones(3), trans="bad")
    assert context.calls == calls
    factor.close()

    with pytest.raises(ValueError, match="only supported"):
        SpluFactorization(matrix, memory_limit_bytes=8_000_000)
    with pytest.raises(ValueError, match="only supported"):
        SpluFactorization(matrix, memory_safety_factor=1.2)

    superlu = SpluFactorization(matrix)
    superlu.close()
    with pytest.raises(RuntimeError, match="closed"):
        superlu._solve_numpy(np.ones(3))


def test_mumps_validation_and_native_failure_cleanup(fake_mumps):
    matrix = scipy_sparse.eye(3, dtype=np.float64, format="csr")
    with pytest.raises(ValueError, match="backend"):
        SpluFactorization(matrix, backend="unknown")
    with pytest.raises(ValueError, match="square"):
        SpluFactorization(
            scipy_sparse.csr_matrix(np.ones((2, 3))),
            backend="mumps",
            memory_limit_bytes=8_000_000,
        )
    with pytest.raises(TypeError, match="integer"):
        SpluFactorization(matrix, backend="mumps", memory_limit_bytes=True)

    factor = SpluFactorization(matrix, backend="mumps", memory_limit_bytes=8_000_000)
    context = fake_mumps.instances[-1]
    assert np.allclose(factor.solve(np.ones(3)), np.ones(3))
    fake_mumps.fail_solve = True
    with pytest.raises(RuntimeError, match="native solve failed"):
        factor.solve(np.ones(3))
    assert context.destroyed


def test_mumps_import_selects_only_requested_context(monkeypatch):
    contexts = {
        np.dtype(np.float32): "SMumpsContext",
        np.dtype(np.float64): "DMumpsContext",
        np.dtype(np.complex64): "CMumpsContext",
        np.dtype(np.complex128): "ZMumpsContext",
    }
    fake_module = SimpleNamespace(**{name: object() for name in contexts.values()})
    fake_mpi = SimpleNamespace(MPI=SimpleNamespace(COMM_SELF="self"))
    monkeypatch.setitem(sys.modules, "mumps", fake_module)
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi)
    for dtype, name in contexts.items():
        context, communicator = native_module._import_mumps(dtype)
        assert context is getattr(fake_module, name)
        assert communicator == "self"

    with pytest.raises(TypeError, match="supports float32"):
        native_module._import_mumps(np.dtype(np.int64))
    del fake_module.DMumpsContext
    with pytest.raises(ImportError, match="does not provide DMumpsContext"):
        native_module._import_mumps(np.dtype(np.float64))


def test_mumps_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mumps", None)
    with pytest.raises(ImportError, match=r"pip install solvax\[mumps\]"):
        native_module._import_mumps(np.dtype(np.float64))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_mumps_real_binding_when_available(dtype):
    pytest.importorskip("mumps")
    matrix = np.array([[5.0 + 1.0j, 1.0], [2.0 - 0.5j, 4.0 + 2.0j]])
    rhs = np.array([[1.0 + 2.0j, 3.0], [-1.0j, 2.0 - 0.5j]])
    if not np.issubdtype(dtype, np.complexfloating):
        matrix, rhs = matrix.real, rhs.real
    matrix = scipy_sparse.csr_matrix(matrix.astype(dtype))
    rhs = rhs.astype(dtype)
    tolerance = 100 * np.finfo(dtype).eps
    with SpluFactorization(
        matrix, backend="mumps", memory_limit_bytes=128_000_000
    ) as factor, SpluFactorization(matrix) as reference:
        for trans, operator in (
            ("N", matrix.toarray()),
            ("T", matrix.toarray().T),
            ("H", matrix.toarray().conj().T),
        ):
            solution = np.asarray(factor.solve(rhs, trans=trans))
            np.testing.assert_allclose(operator @ solution, rhs, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(
                solution, reference._solve_numpy(rhs, trans=trans), rtol=tolerance, atol=tolerance
            )
        assert factor.symbolic_memory_bytes is not None
        assert factor.effective_memory_bytes is not None
