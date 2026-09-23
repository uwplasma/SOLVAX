"""Tests for solvax.sparse_direct: traced host sparse-direct solves and eigenvalues."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import column_groups
from solvax import sparse_direct as sd

jax.config.update("jax_enable_x64", True)
scipy_sparse = pytest.importorskip("scipy.sparse")


@pytest.fixture(autouse=True)
def fresh_cache():
    sd.set_factor_cache_size(2)
    sd.clear_factor_cache()
    yield
    sd.set_factor_cache_size(2)
    sd.clear_factor_cache()


def random_system(n=40, density=0.08, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    a = scipy_sparse.random(n, n, density=density, random_state=rng, format="csr")
    if complex_:
        a = a + 1j * scipy_sparse.random(n, n, density=density, random_state=rng, format="csr")
    a = a + 6.0 * scipy_sparse.identity(n, format="csr")
    pattern, values = sd.CsrPattern.from_scipy(a, include_diagonal=True)
    return a.tocsr(), pattern, jnp.asarray(values)


def dense(pattern, values):
    return jnp.zeros(pattern.shape, values.dtype).at[pattern.rows, pattern.indices].add(values)


# --------------------------------------------------------------------------
# Pattern, product and assembly
# --------------------------------------------------------------------------


def test_pattern_roundtrip_and_diagonal():
    a = scipy_sparse.csr_matrix(np.array([[0.0, 2.0], [3.0, 0.0]]))
    pattern, values = sd.CsrPattern.from_scipy(a)
    assert pattern.nnz == 2 and pattern.shape == (2, 2)
    np.testing.assert_array_equal(pattern.to_scipy(values).toarray(), a.toarray())
    with pytest.raises(ValueError, match="diagonal"):
        pattern.diagonal_positions()
    full, full_values = sd.CsrPattern.from_scipy(a, include_diagonal=True)
    assert full.nnz == 4
    np.testing.assert_array_equal(full_values[full.diagonal_positions()], [0.0, 0.0])
    np.testing.assert_array_equal(full.to_scipy(full_values).toarray(), a.toarray())
    assert hash(full) != hash(pattern) and full != pattern


def test_pattern_validation():
    with pytest.raises(ValueError, match="indptr must have"):
        sd.CsrPattern([1, 1], [0], (1, 1))
    with pytest.raises(ValueError, match="non-decreasing"):
        sd.CsrPattern([0, 2, 1], [0], (2, 2))
    with pytest.raises(ValueError, match="column indices"):
        sd.CsrPattern([0, 1], [3], (1, 2))
    with pytest.raises(TypeError, match="scipy sparse"):
        sd.CsrPattern.from_scipy(np.eye(2))
    with pytest.raises(ValueError, match="square"):
        sd.CsrPattern.from_scipy(scipy_sparse.csr_matrix(np.ones((2, 3))), include_diagonal=True)
    rect = sd.CsrPattern([0, 1], [2], (1, 3))
    with pytest.raises(ValueError, match="square pattern"):
        rect.diagonal_positions()
    with pytest.raises(ValueError, match="values must have shape"):
        rect.to_scipy(np.ones(2))
    with pytest.raises(ValueError, match="backend"):
        sd.HostFactorOptions(backend="pardiso")


def test_csr_matvec_matches_dense():
    a, pattern, values = random_system(complex_=True)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(a.shape[0])
    xs = rng.standard_normal((a.shape[0], 3))
    np.testing.assert_allclose(sd.csr_matvec(pattern, values, x), a @ x, atol=1e-13)
    jitted = jax.jit(sd.csr_matvec, static_argnums=0)
    np.testing.assert_allclose(jitted(pattern, values, xs), a @ xs, atol=1e-13)
    with pytest.raises(ValueError, match="values must have shape"):
        sd.csr_matvec(pattern, values[:-1], x)
    with pytest.raises(ValueError, match="x must have shape"):
        sd.csr_matvec(pattern, values, x[:-1])


def test_csr_data_from_products_recovers_values_and_is_differentiable():
    a, pattern, values = random_system(seed=2)
    groups = column_groups(pattern.to_scipy(np.ones(pattern.nnz)))
    n = a.shape[0]
    seeds = np.zeros((len(groups), n))
    for g, cols in enumerate(groups):
        seeds[g, cols] = 1.0

    def assemble(scale):
        products = jax.vmap(lambda s: scale * sd.csr_matvec(pattern, values, s))(jnp.asarray(seeds))
        return sd.csr_data_from_products(pattern, groups, products)

    np.testing.assert_allclose(jax.jit(assemble)(2.0), 2.0 * values, atol=1e-13)
    np.testing.assert_allclose(jax.grad(lambda s: assemble(s).sum())(1.0), values.sum(), rtol=1e-12)
    with pytest.raises(ValueError, match="products must have shape"):
        sd.csr_data_from_products(pattern, groups, jnp.zeros((1, n)))
    with pytest.raises(ValueError, match="more than one group"):
        sd.csr_data_from_products(pattern, [np.arange(n), np.array([0])], jnp.zeros((2, n)))
    with pytest.raises(ValueError, match="belong to a group"):
        sd.csr_data_from_products(pattern, [np.arange(1, n)], jnp.zeros((1, n)))


# --------------------------------------------------------------------------
# sparse_solve
# --------------------------------------------------------------------------


@pytest.mark.parametrize("complex_", [False, True])
def test_sparse_solve_forward_and_multiple_rhs(complex_):
    a, pattern, values = random_system(seed=3, complex_=complex_)
    rng = np.random.default_rng(4)
    b = rng.standard_normal(a.shape[0])
    bs = rng.standard_normal((a.shape[0], 4))
    x = jax.jit(sd.sparse_solve, static_argnums=0)(pattern, values, b)
    np.testing.assert_allclose(a @ np.asarray(x), b, atol=1e-11)
    xs = sd.sparse_solve(pattern, values, bs)
    np.testing.assert_allclose(a @ np.asarray(xs), bs, atol=1e-11)
    assert sd.factor_cache_info()["misses"] == 1  # one factorization for both calls


def test_sparse_solve_integer_values_promote():
    pattern, values = sd.CsrPattern.from_scipy(scipy_sparse.csr_matrix(np.diag([2, 4])))
    x = sd.sparse_solve(pattern, jnp.asarray(values), jnp.asarray([2, 4]))
    assert jnp.issubdtype(x.dtype, jnp.floating)
    np.testing.assert_allclose(x, [1.0, 1.0])


def test_sparse_solve_gradients_match_dense_and_factor_once():
    a, pattern, values = random_system(seed=5, complex_=True)
    rng = np.random.default_rng(6)
    b = jnp.asarray(rng.standard_normal(a.shape[0]) + 1j * rng.standard_normal(a.shape[0]))
    w = jnp.asarray(rng.standard_normal(a.shape[0]))

    def loss(solver, vals, rhs):
        x = solver(vals, rhs)
        return jnp.sum(jnp.abs(x) ** 2 * w)

    def host(vals, rhs):
        return sd.sparse_solve(pattern, vals, rhs)

    def reference(vals, rhs):
        return jnp.linalg.solve(dense(pattern, vals), rhs)

    value, grads = jax.value_and_grad(lambda v, r: loss(host, v, r), argnums=(0, 1))(values, b)
    assert sd.factor_cache_info()["misses"] == 1  # the transpose solve reused the factor
    assert sd.factor_cache_info()["hits"] >= 1
    ref_value, ref_grads = jax.value_and_grad(lambda v, r: loss(reference, v, r), argnums=(0, 1))(
        values, b
    )
    np.testing.assert_allclose(value, ref_value, rtol=1e-11)
    for got, want in zip(grads, ref_grads, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)

    tangent = jnp.asarray(rng.standard_normal(values.shape))
    _, jvp = jax.jvp(lambda v: host(v, b), (values,), (tangent.astype(values.dtype),))
    _, ref_jvp = jax.jvp(lambda v: reference(v, b), (values,), (tangent.astype(values.dtype),))
    np.testing.assert_allclose(jvp, ref_jvp, rtol=1e-9, atol=1e-12)


def test_sparse_solve_vmap_rhs_is_one_factorization_and_vmap_values_factor_each():
    a, pattern, values = random_system(seed=7)
    rng = np.random.default_rng(8)
    bs = jnp.asarray(rng.standard_normal((5, a.shape[0])))
    xs = jax.vmap(lambda r: sd.sparse_solve(pattern, values, r))(bs)
    np.testing.assert_allclose(a @ np.asarray(xs).T, np.asarray(bs).T, atol=1e-11)
    assert sd.factor_cache_info()["misses"] == 1

    sd.clear_factor_cache()
    scales = jnp.asarray([1.0, 2.0, 3.0])
    ys = jax.vmap(lambda s: sd.sparse_solve(pattern, s * values, bs[0]))(scales)
    for s, y in zip(scales, ys, strict=True):
        np.testing.assert_allclose(float(s) * (a @ np.asarray(y)), bs[0], atol=1e-11)
    assert sd.factor_cache_info()["misses"] == 3


def test_sparse_solve_validation_and_cache_controls():
    a, pattern, values = random_system(seed=9)
    b = jnp.ones(a.shape[0])
    with pytest.raises(ValueError, match="square"):
        sd.sparse_solve(sd.CsrPattern([0, 1], [0], (1, 2)), jnp.ones(1), jnp.ones(1))
    with pytest.raises(ValueError, match="values must have shape"):
        sd.sparse_solve(pattern, values[:-1], b)
    with pytest.raises(ValueError, match="b must have shape"):
        sd.sparse_solve(pattern, values, b[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        sd.set_factor_cache_size(-1)
    sd.set_factor_cache_size(0)
    sd.sparse_solve(pattern, values, b)
    sd.sparse_solve(pattern, values, b)
    info = sd.factor_cache_info()
    assert info == {"size": 0, "entries": 0, "hits": 0, "misses": 2}
    sd.set_factor_cache_size(1)
    sd.sparse_solve(pattern, values, b)
    sd.sparse_solve(pattern, 2.0 * values, b)
    assert sd.factor_cache_info()["entries"] == 1


def test_sparse_solve_mumps_options_reach_the_factorization(monkeypatch):
    seen = []

    class Recorder(sd.SpluFactorization):
        def __init__(self, matrix, **kwargs):
            seen.append(kwargs)
            super().__init__(matrix)

    monkeypatch.setattr(sd, "SpluFactorization", Recorder)
    a, pattern, values = random_system(seed=10)
    options = sd.HostFactorOptions(backend="mumps", memory_limit_bytes=10**8)
    x = sd.sparse_solve(pattern, values, jnp.ones(a.shape[0]), options=options)
    np.testing.assert_allclose(a @ np.asarray(x), np.ones(a.shape[0]), atol=1e-11)
    assert seen == [{"backend": "mumps", "memory_limit_bytes": 10**8, "memory_safety_factor": None}]


# --------------------------------------------------------------------------
# sparse_eigenvalue
# --------------------------------------------------------------------------


def parametric_family(n=60, seed=11):
    """A(p) = A0 + p0 A1 + p1 A2 with one isolated, rightmost eigenvalue."""
    rng = np.random.default_rng(seed)

    def rand():
        m = scipy_sparse.random(n, n, density=0.06, random_state=rng, format="csr")
        return m + 1j * scipy_sparse.random(n, n, density=0.06, random_state=rng, format="csr")

    diag = -np.linspace(1.0, 4.0, n)
    diag[7] = 1.5
    a0 = rand() * 0.3 + scipy_sparse.diags(diag)
    union = abs(a0) + abs(rand()) + scipy_sparse.identity(n)
    pattern, _ = sd.CsrPattern.from_scipy(union.tocsr(), include_diagonal=True)
    mats = [a0, rand(), rand()]
    blocks = [
        jnp.asarray(np.asarray(m.tocsr()[pattern.rows, pattern.indices]).reshape(-1)) for m in mats
    ]

    def values_at(p):
        return blocks[0] + p[0] * blocks[1] + p[1] * blocks[2]

    def operator(p, x):
        return sd.csr_matvec(pattern, values_at(p), x)

    return pattern, values_at, operator


def test_sparse_eigenvalue_value_and_gradient_match_dense():
    pattern, values_at, operator = parametric_family()
    p0 = jnp.asarray([0.3, -0.2])

    def host(p):
        result = sd.sparse_eigenvalue(operator, p, pattern, values_at(p), 1.3 + 0.1j)
        return result

    def dense_rightmost(p):
        lam = jnp.linalg.eigvals(dense(pattern, values_at(p)))
        return lam[jnp.argmax(lam.real)]

    result = jax.jit(host)(p0)
    np.testing.assert_allclose(result.value, dense_rightmost(p0), rtol=1e-11)
    assert result.residual < 1e-10 and result.left_residual < 1e-10
    matrix = np.asarray(dense(pattern, values_at(p0)))
    y, x = np.asarray(result.left), np.asarray(result.right)
    np.testing.assert_allclose(matrix.conj().T @ y, np.conj(result.value) * y, atol=1e-9)
    np.testing.assert_allclose(matrix @ x, result.value * x, atol=1e-9)

    # Reference: JAX's own eigenvalue derivative of the dense matrix, and
    # central differences as an independent check of both.
    for part in (jnp.real, jnp.imag):
        grad = jax.grad(lambda p, part=part: part(host(p).value))(p0)
        want = jax.grad(lambda p, part=part: part(dense_rightmost(p)))(p0)
        np.testing.assert_allclose(grad, want, rtol=1e-8, atol=1e-12)
        step = 1e-6
        fd = [
            (part(dense_rightmost(p0.at[i].add(step))) - part(dense_rightmost(p0.at[i].add(-step))))
            / (2 * step)
            for i in range(2)
        ]
        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)
    _, tangent = jax.jvp(lambda p: host(p).value, (p0,), (jnp.asarray([1.0, 0.0]),))
    want = jax.jvp(dense_rightmost, (p0,), (jnp.asarray([1.0, 0.0]),))[1]
    np.testing.assert_allclose(tangent, want, rtol=1e-8)


def test_sparse_eigenvalue_nearest_vmap_and_fail_closed():
    pattern, values_at, operator = parametric_family(seed=12)
    p = jnp.asarray([0.1, 0.1])
    lam = np.linalg.eigvals(np.asarray(dense(pattern, values_at(p))))
    target = lam[np.argsort(lam.real)[-3]]
    near = sd.sparse_eigenvalue(
        operator, p, pattern, values_at(p), target + 1e-3, select="nearest", candidates=3
    )
    np.testing.assert_allclose(near.value, target, rtol=1e-10)

    ps = jnp.asarray([[0.0, 0.0], [0.2, -0.1]])
    batched = jax.vmap(
        lambda q: sd.sparse_eigenvalue(operator, q, pattern, values_at(q), 1.3).value
    )(ps)
    for q, value in zip(ps, batched, strict=True):
        ref = np.linalg.eigvals(np.asarray(dense(pattern, values_at(q))))
        np.testing.assert_allclose(value, ref[np.argmax(ref.real)], rtol=1e-10)

    strict = sd.sparse_eigenvalue(
        operator, p, pattern, values_at(p), 1.3, residual_tolerance=1e-300
    )
    assert np.isnan(complex(strict.value))
    capped = sd.sparse_eigenvalue(
        operator, p, pattern, values_at(p), 1.3, tolerance=1e-15, maxiter=1, candidates=8
    )
    assert np.isnan(complex(capped.value)) and np.isinf(float(capped.residual))


@pytest.mark.parametrize("failure", ["raises", "residual"])
def test_sparse_eigenvalue_left_failure_is_fail_closed(monkeypatch, failure):
    pattern, values_at, operator = parametric_family(seed=13)
    original = sd.sparse_eigenpairs

    def right_only(matrix, **kwargs):
        if not kwargs.get("adjoint"):
            return original(matrix, **kwargs)
        if failure == "raises":
            raise sd._import_scipy_sparse()[1].ArpackNoConvergence("no", [], [])
        solution = original(matrix, **kwargs)
        return solution._replace(residuals=solution.residuals + 1.0)

    monkeypatch.setattr(sd, "sparse_eigenpairs", right_only)
    p = jnp.zeros(2)
    out = sd.sparse_eigenvalue(operator, p, pattern, values_at(p), 1.3)
    assert np.isnan(complex(out.value))
    if failure == "residual":
        assert float(out.residual) < 1e-8 and float(out.left_residual) > 1.0


def test_real_mumps_backend_solve_gradient_and_eigenvalue():
    pytest.importorskip("mumps")
    options = sd.HostFactorOptions(backend="mumps", memory_limit_bytes=10**9)
    a, pattern, values = random_system(n=120, seed=15, complex_=True)
    rng = np.random.default_rng(16)
    bs = jnp.asarray(rng.standard_normal((a.shape[0], 6)) + 0j)
    xs = sd.sparse_solve(pattern, values, bs, options=options)
    np.testing.assert_allclose(a @ np.asarray(xs), bs, atol=1e-11)

    def loss(vals):
        return jnp.sum(jnp.abs(sd.sparse_solve(pattern, vals, bs[:, 0], options=options)) ** 2)

    def reference(vals):
        return jnp.sum(jnp.abs(jnp.linalg.solve(dense(pattern, vals), bs[:, 0])) ** 2)

    np.testing.assert_allclose(jax.grad(loss)(values), jax.grad(reference)(values), rtol=1e-9)
    fam_pattern, values_at, operator = parametric_family(seed=17)
    p = jnp.asarray([0.1, -0.1])
    got = sd.sparse_eigenvalue(operator, p, fam_pattern, values_at(p), 1.3, options=options).value
    lam = np.linalg.eigvals(np.asarray(dense(fam_pattern, values_at(p))))
    np.testing.assert_allclose(got, lam[np.argmax(lam.real)], rtol=1e-10)


def test_sparse_eigenvalue_validation():
    pattern, values_at, operator = parametric_family(n=12, seed=14)
    p = jnp.zeros(2)
    with pytest.raises(ValueError, match="select"):
        sd.sparse_eigenvalue(operator, p, pattern, values_at(p), 0.0, select="largest")
    with pytest.raises(ValueError, match="candidates"):
        sd.sparse_eigenvalue(operator, p, pattern, values_at(p), 0.0, candidates=11)
    with pytest.raises(ValueError, match="values must have shape"):
        sd.sparse_eigenvalue(operator, p, pattern, values_at(p)[:-1], 0.0)
    bare, _ = sd.CsrPattern.from_scipy(scipy_sparse.csr_matrix(np.eye(12)[::-1]))
    with pytest.raises(ValueError, match="diagonal"):
        sd.sparse_eigenvalue(operator, p, bare, jnp.ones(12), 0.0)
