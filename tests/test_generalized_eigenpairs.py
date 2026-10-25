"""Targeted generalized eigenpairs of Hermitian block-tridiagonal pencils.

Every case is checked against a dense ``scipy.linalg.eigh(F, G)``; derivatives against
central finite differences, the closed-form first-order perturbation sum, and the
``jvp``/``vjp`` dot-product identity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import solvax as sx
from solvax.eigen import _dense, _hermitian

jax.config.update("jax_enable_x64", True)

N_BLOCKS, M, COUNT, TARGET = 10, 3, 3, 0.2


def _band(rng, n, m, cplx, shift=0.0):
    """Random Hermitian band triple (``lower`` is junk on purpose: it is ignored)."""

    def draw(*shape):
        return rng.normal(size=shape) + (1j * rng.normal(size=shape) if cplx else 0.0)

    diag = draw(n, m, m)
    diag = diag + np.conj(np.swapaxes(diag, -1, -2)) + shift * np.eye(m)
    return tuple(jnp.asarray(a) for a in (draw(n, m, m), diag, draw(n, m, m)))


def _along(a, b, t):
    return tuple(x + t * y for x, y in zip(a, b, strict=True))


def _dense_pencil(F, G):
    return np.asarray(_dense(_hermitian(F))), np.asarray(_dense(_hermitian(G)))


@pytest.fixture(params=[False, True], ids=["real", "complex"])
def pencil(request):
    """``F(t) = F0 + t F1``, ``G(t) = G0 + t G1`` with ``G`` positive definite for ``|t| < 1``."""
    rng = np.random.default_rng(1)
    n, m, c = N_BLOCKS, M, request.param
    return (
        _band(rng, n, m, c),
        _band(rng, n, m, c),
        _band(rng, n, m, c, 4 * m),
        _band(rng, n, m, c, 0.3),
    )


def _solve(pencil, t=0.0, **kw):
    F0, F1, G0, G1 = pencil
    return sx.generalized_eigenpairs(_along(F0, F1, t), _along(G0, G1, t), TARGET, COUNT, **kw)


def _nearest(values, target, count):
    return np.sort(values[np.argsort(np.abs(values - target))[:count]])


def test_matches_dense_reference(pencil):
    F, G = _along(pencil[0], pencil[1], 0.1), _along(pencil[2], pencil[3], 0.1)
    result = sx.generalized_eigenpairs(F, G, TARGET, COUNT)
    Fd, Gd = _dense_pencil(F, G)
    values = sla.eigh(Fd, Gd, eigvals_only=True)
    assert bool(result.converged) and not bool(result.fallback)
    np.testing.assert_allclose(result.values, _nearest(values, TARGET, COUNT), rtol=0, atol=1e-12)
    Y = np.asarray(result.vectors).reshape(-1, COUNT)
    assert np.abs(Fd @ Y - Gd @ Y * np.asarray(result.values)).max() < 1e-9
    assert float(result.orthogonality) < 1e-12 and float(result.residuals.max()) <= 1e-10
    d = np.sort(np.abs(values - TARGET))
    np.testing.assert_allclose(result.gap, d[COUNT] - d[COUNT - 1], atol=1e-4)
    assert result.vectors.dtype == F[1].dtype and result.values.dtype == np.float64


def test_residual_reports_unconverged_iteration(pencil):
    result = _solve(pencil, max_iter=1)
    assert int(result.iterations) == 1 and not bool(result.converged)
    assert float(result.residuals.max()) > 1e-10


def test_larger_problem_interior_target_against_dense():
    rng = np.random.default_rng(5)
    F, G = _band(rng, 40, 6, True), _band(rng, 40, 6, True, 24)
    result = sx.generalized_eigenpairs(F, G, -0.5, 6)
    values = sla.eigh(*_dense_pencil(F, G), eigvals_only=True)
    assert bool(result.converged)
    np.testing.assert_allclose(result.values, _nearest(values, -0.5, 6), rtol=0, atol=1e-11)


def test_shift_equal_to_an_eigenvalue_is_singular_but_works():
    n, m = 14, 1
    diag = jnp.asarray(np.arange(n, dtype=float).reshape(n, m, m) - 5.0)
    zero = jnp.zeros((n, m, m))
    F, G = (zero, diag, zero), (zero, jnp.ones((n, m, m)), zero)
    result = sx.generalized_eigenpairs(F, G, 0.0, 3)  # 0 is an eigenvalue: F - 0 G is singular
    np.testing.assert_allclose(result.values, [-1.0, 0.0, 1.0], atol=1e-12)
    assert bool(result.converged)


def test_bad_pivot_falls_back_to_dense_and_reports():
    """Block Thomas breaks down on a singular Schur pivot of an invertible matrix."""
    n = 14
    rng = np.random.default_rng(2)
    diag = rng.normal(size=(n, 1, 1))
    diag[-1] = 0.7  # the last Schur pivot is F - sigma G = 0 exactly at sigma = 0.7
    upper = jnp.asarray(rng.normal(size=(n, 1, 1)))
    F = (jnp.zeros((n, 1, 1)), jnp.asarray(diag), upper)
    G = (jnp.zeros((n, 1, 1)), jnp.ones((n, 1, 1)), jnp.zeros((n, 1, 1)))
    values = sla.eigh(*_dense_pencil(F, G), eigvals_only=True)
    result = sx.generalized_eigenpairs(F, G, 0.7, 2)
    assert bool(result.fallback) and bool(result.converged)
    np.testing.assert_allclose(result.values, _nearest(values, 0.7, 2), atol=1e-12)
    # derivative through the dense path
    dF = (jnp.zeros_like(upper), jnp.asarray(rng.normal(size=(n, 1, 1))), upper * 0)
    _, dv = jax.jvp(
        lambda t: sx.generalized_eigenpairs(_along(F, dF, t), G, 0.7, 2).values, (0.0,), (1.0,)
    )
    h = 1e-6
    fd = (
        sx.generalized_eigenpairs(_along(F, dF, h), G, 0.7, 2).values
        - sx.generalized_eigenpairs(_along(F, dF, -h), G, 0.7, 2).values
    ) / (2 * h)
    np.testing.assert_allclose(dv, fd, atol=1e-8)
    # without the dense path the breakdown is reported, not hidden
    off = sx.generalized_eigenpairs(F, G, 0.7, 2, dense_limit=0)
    assert not bool(off.fallback) and not bool(off.converged)


def test_invertible_indefinite_two_by_two_pencil():
    """``[[0, 1], [1, 0]]`` has eigenvalues +-1 although its first Schur pivot is zero."""
    F = (jnp.zeros((2, 1, 1)), jnp.zeros((2, 1, 1)), jnp.array([[[1.0]], [[0.0]]]))
    G = (jnp.zeros((2, 1, 1)), jnp.ones((2, 1, 1)), jnp.zeros((2, 1, 1)))
    result = sx.generalized_eigenpairs(F, G, 0.0, 2)
    np.testing.assert_allclose(result.values, [-1.0, 1.0], atol=1e-14)
    assert bool(result.fallback) and bool(result.converged)


def test_lower_is_ignored_and_diag_is_hermitized():
    rng = np.random.default_rng(3)
    F, G = _band(rng, 8, 2, True), _band(rng, 8, 2, True, 8)
    skew = rng.normal(size=F[1].shape) + 1j * rng.normal(size=F[1].shape)
    skew = jnp.asarray(skew - np.conj(np.swapaxes(skew, -1, -2)))  # anti-Hermitian
    messy = (F[0] * 7 + 1, F[1] + 0.3 * skew, F[2])
    clean = sx.generalized_eigenpairs((jnp.zeros_like(F[0]), F[1], F[2]), G, 0.0, 2).values
    np.testing.assert_allclose(
        sx.generalized_eigenpairs(messy, G, 0.0, 2).values, clean, atol=1e-12
    )


def test_count_is_validated():
    rng = np.random.default_rng(3)
    F, G = _band(rng, 2, 2, False), _band(rng, 2, 2, False, 8)
    for count in (0, 5):
        with pytest.raises(ValueError):
            sx.generalized_eigenpairs(F, G, 0.0, count)


# --------------------------------------------------------------------------- derivatives


def _outputs(pencil, t):
    F0, F1, G0, G1 = pencil
    r = sx.generalized_eigenpairs(_along(F0, F1, t[0]), _along(G0, G1, t[1]), TARGET, COUNT)
    return r.values, r.vectors


def test_value_jvp_matches_finite_difference_and_perturbation_sum(pencil):
    t0, dt = jnp.array([0.1, 0.05]), jnp.array([0.3, -0.7])
    (values, vectors), (dvalues, _) = jax.jvp(lambda t: _outputs(pencil, t), (t0,), (dt,))
    F0, F1, G0, G1 = pencil
    dF, dG = (np.asarray(_dense(_hermitian(a))) for a in (F1, G1))
    Y = np.asarray(vectors).reshape(-1, COUNT)
    exact = [
        np.real(Y[:, i].conj() @ ((dt[0] * dF - values[i] * dt[1] * dG) @ Y[:, i]))
        for i in range(COUNT)
    ]
    np.testing.assert_allclose(dvalues, exact, atol=1e-10)
    h = 1e-6
    fd = (_outputs(pencil, t0 + h * dt)[0] - _outputs(pencil, t0 - h * dt)[0]) / (2 * h)
    np.testing.assert_allclose(dvalues, fd, atol=2e-9)


def test_vector_jvp_gauge_normalization_and_finite_difference(pencil):
    t0, dt = jnp.array([0.1, 0.05]), jnp.array([0.3, -0.7])
    (_, vectors), (_, dvectors) = jax.jvp(lambda t: _outputs(pencil, t), (t0,), (dt,))
    F0, F1, G0, G1 = pencil
    G, dG = (np.asarray(_dense(_hermitian(a))) for a in (_along(G0, G1, 0.05), G1))
    Y, dY = (np.asarray(a).reshape(-1, COUNT) for a in (vectors, dvectors))
    overlap = np.einsum("ki,kl,li->i", Y.conj(), G, dY)  # y^H G dy
    expected = -0.5 * dt[1] * np.einsum("ki,kl,li->i", Y.conj(), dG, Y)  # Re = -y^H dG y / 2
    np.testing.assert_allclose(overlap, expected, atol=1e-10)  # and Im = 0: the phase gauge

    def aligned(t):  # parallel-transport the phase back to the reference vectors
        y = np.asarray(_outputs(pencil, t)[1]).reshape(-1, COUNT)
        phase = np.einsum("ki,kl,li->i", Y.conj(), G, y)
        return y * (phase.conj() / np.abs(phase))

    h = 1e-6
    fd = (aligned(t0 + h * dt) - aligned(t0 - h * dt)) / (2 * h)
    np.testing.assert_allclose(dY, fd, atol=2e-8)


def test_vector_jvp_matches_dense_perturbation_theory(pencil):
    """``dy_i = sum_{j != i} w_j (-w_j^H dA w_i) / (lam_j - lam_i) - w_i w_i^H dG w_i / 2``."""
    F0, F1, G0, G1 = pencil
    t0, dt = jnp.zeros(2), jnp.array([1.0, 0.5])
    (_, vectors), (_, dvectors) = jax.jvp(lambda t: _outputs(pencil, t), (t0,), (dt,))
    Fd, Gd = _dense_pencil(F0, G0)
    dF, dG = (np.asarray(_dense(_hermitian(a))) for a in (F1, G1))
    dF, dG = dt[0] * dF, dt[1] * dG
    lam, W = sla.eigh(Fd, Gd)
    near = np.argsort(np.abs(lam - TARGET))[:COUNT]
    for k, i in enumerate(near[np.argsort(lam[near])]):
        j = np.array([q for q in range(len(lam)) if q != i])
        dA = dF - lam[i] * dG
        c = -(W[:, j].conj().T @ dA @ W[:, i]) / (lam[j] - lam[i])
        dy = W[:, j] @ c - 0.5 * (W[:, i].conj() @ dG @ W[:, i]).real * W[:, i]
        phase = W[:, i].conj() @ Gd @ np.asarray(vectors).reshape(-1, COUNT)[:, k]
        np.testing.assert_allclose(
            np.asarray(dvectors).reshape(-1, COUNT)[:, k], dy * phase / abs(phase), atol=1e-8
        )


def test_vjp_is_the_transpose_of_jvp(pencil):
    rng = np.random.default_rng(9)
    t0, dt = jnp.array([0.1, 0.05]), jnp.array([0.3, -0.7])
    f = lambda t: _outputs(pencil, t)  # noqa: E731
    _, (dv, dy) = jax.jvp(f, (t0,), (dt,))
    out, pullback = jax.vjp(f, t0)
    cv = jnp.asarray(rng.normal(size=COUNT))
    cy = jnp.asarray(
        rng.normal(size=out[1].shape)
        + (1j * rng.normal(size=out[1].shape) if pencil[0][1].dtype.kind == "c" else 0)
    )
    (g,) = pullback((cv, cy))
    lhs = jnp.real(jnp.sum(cv * dv) + jnp.sum(cy * dy))
    np.testing.assert_allclose(lhs, g @ dt, rtol=1e-10)
    # a scalar objective through jit, with the gradient of the cluster mean
    grad = jax.jit(jax.grad(lambda t: _outputs(pencil, t)[0].mean()))(t0)
    h = 1e-6
    fd = [
        (_outputs(pencil, t0 + h * e)[0].mean() - _outputs(pencil, t0 - h * e)[0].mean()) / (2 * h)
        for e in jnp.eye(2)
    ]
    np.testing.assert_allclose(grad, fd, atol=2e-9)


def test_finite_difference_step_sweep(pencil):
    """Central differences approach the implicit derivative as ``h^2`` down to the noise floor."""
    t0, dt = jnp.array([0.1, 0.05]), jnp.array([0.3, -0.7])
    mean = lambda t: _outputs(pencil, t)[0].mean()  # noqa: E731
    exact = jax.jvp(mean, (t0,), (dt,))[1]
    errors = [
        abs((mean(t0 + h * dt) - mean(t0 - h * dt)) / (2 * h) - exact)
        for h in 10.0 ** -np.arange(1, 8)
    ]
    assert errors[1] < errors[0] / 20 and errors[2] < errors[1] / 20
    assert min(errors) < 1e-9


def test_second_derivative_of_values_matches_finite_difference_of_gradient(pencil):
    f = lambda t: _outputs(pencil, t)[0].sum()  # noqa: E731
    t0 = jnp.array([0.1, 0.05])
    hessian = jax.hessian(f)(t0)
    h = 1e-5
    fd = np.array(
        [(jax.grad(f)(t0 + h * e) - jax.grad(f)(t0 - h * e)) / (2 * h) for e in jnp.eye(2)]
    )
    np.testing.assert_allclose(hessian, fd, atol=1e-7)


@pytest.mark.parametrize("split", [0.0, 1e-4], ids=["degenerate", "near-degenerate"])
def test_cluster_mean_derivative_survives_degeneracy(split):
    """The mean over a degenerate pair is smooth; the individual values and vectors are not."""
    rng = np.random.default_rng(4)
    n = 12
    scalar = lambda shift: jnp.asarray(np.kron(rng.normal(size=(n, 1, 1)) + shift, np.eye(2)))  # noqa: E731
    zero = jnp.zeros((n, 2, 2))
    F = (
        zero,
        scalar(0.0) + split * jnp.diag(jnp.array([1.0, -1.0])),
        jnp.asarray(np.kron(rng.normal(size=(n, 1, 1)), np.eye(2))),
    )
    G = (zero, scalar(6.0), zero)
    dF, dG = _band(rng, n, 2, True), _band(rng, n, 2, True, 0.0)
    levels = sla.eigh(*_dense_pencil(F, G), eigvals_only=True)
    target = float(levels[8] + 1e-3)  # next to a doubly degenerate level

    def solve(t):
        return sx.generalized_eigenpairs(_along(F, dF, t[0]), _along(G, dG, 0.1 * t[1]), target, 2)

    t0, dt = jnp.zeros(2), jnp.array([1.0, 1.0])
    assert bool(solve(t0).converged) and float(solve(t0).gap) > 0.01
    mean = lambda t: solve(t).values.mean()  # noqa: E731
    h = 1e-6
    fd = (mean(t0 + h * dt) - mean(t0 - h * dt)) / (2 * h)
    np.testing.assert_allclose(jax.jvp(mean, (t0,), (dt,))[1], fd, atol=1e-8)
    if split == 0.0:  # a degenerate eigenvector is gauge fixed: no rotation inside the pair
        Y, dY = (
            np.asarray(a).reshape(-1, 2) for a in jax.jvp(lambda t: solve(t).vectors, (t0,), (dt,))
        )
        Gm, dGm = (np.asarray(_dense(_hermitian(B))) for B in (G, dG))
        np.testing.assert_allclose(Y.conj().T @ Gm @ dY, -0.05 * Y.conj().T @ dGm @ Y, atol=1e-8)


def _degenerate_pair(cplx):
    """``F = blockdiag(I2, diag(3, 4))``, ``G = I``: the pair nearest ``1.0`` is degenerate."""
    z = jnp.zeros((2, 2, 2))
    d = jnp.stack([jnp.eye(2), jnp.diag(jnp.array([3.0, 4.0]))]).astype(complex if cplx else float)
    eye = (z, jnp.stack([jnp.eye(2)] * 2).astype(d.dtype), z)
    dF = _band(np.random.default_rng(7), 2, 2, cplx)
    return (z, d, z), eye, dF


@pytest.mark.parametrize("cplx", [False, True], ids=["real", "complex"])
def test_reverse_mode_through_exact_degeneracy_is_finite(cplx):
    """Reverse mode transposes the whole tangent map, so a NaN anywhere in it (even one multiplied
    by a zero cotangent) leaks: ``grad`` of the cluster mean and of a subspace-invariant function
    of the vectors must be finite, correct, and free of non-finite constants."""
    F, G, dF = _degenerate_pair(cplx)
    solve = lambda t: sx.generalized_eigenpairs(_along(F, dF, t), G, 1.0, 2)  # noqa: E731
    mean = lambda t: solve(t).values.mean()  # noqa: E731
    r = solve(0.0)
    assert bool(r.converged) and float(r.gap) > 0.5
    Y = np.asarray(r.vectors).reshape(-1, 2)
    projected_trace = np.trace(Y.conj().T @ np.asarray(_dense(_hermitian(dF))) @ Y).real / 2
    h = 1e-6
    fd = (mean(h) - mean(-h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(mean)(0.0), projected_trace, atol=1e-12)
    np.testing.assert_allclose(fd, projected_trace, atol=1e-8)

    weights = jax.random.normal(jax.random.PRNGKey(0), (4, 4))

    def invariant(t):  # the projector on the pair does not depend on the gauge
        Yt = solve(t).vectors.reshape(4, 2)
        return jnp.sum(weights * jnp.real(Yt @ jnp.conj(Yt.T)))

    pullback = jax.vjp(invariant, 0.0)[1]
    np.testing.assert_allclose(
        pullback(1.0)[0], (invariant(h) - invariant(-h)) / (2 * h), atol=1e-7
    )
    for fn in (jax.vjp(mean, 0.0)[1], pullback):
        consts = jax.make_jaxpr(fn)(1.0).consts
        assert all(
            np.isfinite(np.asarray(c)).all() for c in consts if np.asarray(c).dtype.kind in "fc"
        )


def test_jit_and_vmap(pencil):
    F0, F1, G0, G1 = pencil
    run = jax.jit(lambda t: sx.generalized_eigenpairs(_along(F0, F1, t), G0, TARGET, COUNT))
    batched = jax.vmap(run)(jnp.array([0.0, 0.1]))
    np.testing.assert_allclose(batched.values[1], run(0.1).values, atol=1e-12)
    assert bool(batched.converged.all())


def test_block_thomas_adjoint_and_transpose_semantics():
    rng = np.random.default_rng(6)
    n, m = 5, 3
    lower, diag, upper = (
        jnp.asarray(rng.normal(size=(n, m, m)) + 1j * rng.normal(size=(n, m, m))) for _ in range(3)
    )
    diag = diag + 6 * jnp.eye(m)
    rhs = jnp.asarray(rng.normal(size=(n, m, 2)) + 1j * rng.normal(size=(n, m, 2)))
    factors = sx.block_thomas_factor(lower, diag, upper)
    A = np.asarray(_dense((lower, diag, upper)))
    flat = np.asarray(rhs).reshape(n * m, 2)
    for kwargs, op in (({}, A), ({"transpose": True}, A.T), ({"adjoint": True}, A.conj().T)):
        x = np.asarray(sx.block_thomas_solve(factors, rhs, **kwargs)).reshape(n * m, 2)
        np.testing.assert_allclose(op @ x, flat, atol=1e-12)
    with pytest.raises(ValueError):
        sx.block_thomas_solve(factors, rhs, transpose=True, adjoint=True)
