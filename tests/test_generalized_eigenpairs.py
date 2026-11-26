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
from solvax.eigen import _dense, _gap_resolved, _hermitian, _ritz

jax.config.update("jax_enable_x64", True)

N_BLOCKS, M, COUNT, TARGET = 10, 3, 3, 0.2


def _band(rng, n, m, cplx, shift=0.0):
    """Random Hermitian band triple, ``lower=None`` (not provided: only ``diag`` and ``upper``)."""

    def draw(*shape):
        return rng.normal(size=shape) + (1j * rng.normal(size=shape) if cplx else 0.0)

    diag = draw(n, m, m)
    diag = diag + np.conj(np.swapaxes(diag, -1, -2)) + shift * np.eye(m)
    return None, jnp.asarray(diag), jnp.asarray(draw(n, m, m))


def _along(a, b, t):
    return tuple(None if x is None else x + t * y for x, y in zip(a, b, strict=True))


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
    F = (None, jnp.asarray(diag), upper)
    G = (jnp.zeros((n, 1, 1)), jnp.ones((n, 1, 1)), jnp.zeros((n, 1, 1)))
    values = sla.eigh(*_dense_pencil(F, G), eigvals_only=True)
    t = 0.7
    result = sx.generalized_eigenpairs(F, G, t, 2, dense_limit=n)
    assert bool(result.fallback) and bool(result.converged)
    np.testing.assert_allclose(result.values, _nearest(values, t, 2), atol=1e-12)
    # derivative through the dense path
    dF = (jnp.zeros_like(upper), jnp.asarray(rng.normal(size=(n, 1, 1))), upper * 0)
    _, dv = jax.jvp(
        lambda s: sx.generalized_eigenpairs(_along(F, dF, s), G, t, 2, dense_limit=n).values,
        (0.0,),
        (1.0,),
    )
    h = 1e-6
    fd = (
        sx.generalized_eigenpairs(_along(F, dF, h), G, t, 2, dense_limit=n).values
        - sx.generalized_eigenpairs(_along(F, dF, -h), G, t, 2, dense_limit=n).values
    ) / (2 * h)
    np.testing.assert_allclose(dv, fd, atol=1e-8)
    # without the dense path (the default) the failed probe is reported, not hidden; the shift,
    # sqrt(eps) off the singular pivot, leaves block Thomas accurate enough here to converge
    off = sx.generalized_eigenpairs(F, G, t, 2)
    assert bool(off.fallback)
    if bool(off.converged):
        np.testing.assert_allclose(off.values, _nearest(values, t, 2), atol=1e-12)


def test_invertible_indefinite_two_by_two_pencil():
    """``[[0, 1], [1, 0]]`` has eigenvalues +-1 although its first Schur pivot is zero."""
    F = (None, jnp.zeros((2, 1, 1)), jnp.array([[[1.0]], [[0.0]]]))
    G = (jnp.zeros((2, 1, 1)), jnp.ones((2, 1, 1)), jnp.zeros((2, 1, 1)))
    result = sx.generalized_eigenpairs(F, G, 0.0, 2, dense_limit=2)
    np.testing.assert_allclose(result.values, [-1.0, 1.0], atol=1e-14)
    assert bool(result.fallback) and bool(result.converged)
    assert float(result.excluded_residual) == 0.0 and float(result.gap) == np.inf  # all selected


def test_lower_none_reads_diag_and_upper_and_reports_diag_defect():
    rng = np.random.default_rng(3)
    F, G = _band(rng, 8, 2, True), _band(rng, 8, 2, True, 8)
    skew = rng.normal(size=F[1].shape) + 1j * rng.normal(size=F[1].shape)
    skew = jnp.asarray(skew - np.conj(np.swapaxes(skew, -1, -2)))  # anti-Hermitian
    messy = sx.generalized_eigenpairs((None, F[1] + 0.3 * skew, F[2]), G, 0.0, 2)
    clean = sx.generalized_eigenpairs(F, G, 0.0, 2)
    np.testing.assert_allclose(messy.values, clean.values, atol=1e-12)
    assert float(messy.hermitian_defect) > 0.01 and not bool(messy.converged)
    assert float(clean.hermitian_defect) == 0.0 and bool(clean.converged)


def test_count_is_validated():
    rng = np.random.default_rng(3)
    F, G = _band(rng, 2, 2, False), _band(rng, 2, 2, False, 8)
    for count in (0, 5):
        with pytest.raises(ValueError):
            sx.generalized_eigenpairs(F, G, 0.0, count)


# --------------------------------------------------------------------------- derivatives


def _outputs(pencil, t):
    F0, F1, G0, G1 = pencil
    # tol far below the default: finite differences amplify the 1e-10 stopping residual
    F, G = _along(F0, F1, t[0]), _along(G0, G1, t[1])
    r = sx.generalized_eigenpairs(F, G, TARGET, COUNT, tol=1e-13)
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
    # FD floor ~ tol / (gap h) ~ 1e-7; the exact check is the dense perturbation-theory test below
    np.testing.assert_allclose(dY, fd, atol=1e-7)


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
        None,
        scalar(0.0) + split * jnp.diag(jnp.array([1.0, -1.0])),
        jnp.asarray(np.kron(rng.normal(size=(n, 1, 1)), np.eye(2))),
    )
    G = (None, scalar(6.0), zero)
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
    eye = (None, jnp.stack([jnp.eye(2)] * 2).astype(d.dtype), z)
    dF = _band(np.random.default_rng(7), 2, 2, cplx)
    return (None, d, z), eye, dF


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


def _accumulating_pencil(n=60):
    """Scalar Jacobi matrix (Lanczos on ``diag(lam)``): eigenvalue 1, a set accumulating at 0.6."""
    lam = np.sort(np.concatenate([[1.0], 0.6 - 0.2 / np.arange(1, n) ** 2]))
    Q, a, b = [np.ones(n) / np.sqrt(n)], [], []
    for k in range(n):
        w = lam * Q[-1]
        a.append(Q[-1] @ w)
        w = w - a[-1] * Q[-1]
        for _ in range(2):
            for q in Q:
                w = w - q * (q @ w)
        if k < n - 1:
            b.append(np.linalg.norm(w))
            Q.append(w / b[-1])
    z = jnp.zeros((n, 1, 1))
    F = (None, jnp.asarray(a).reshape(n, 1, 1), jnp.asarray(np.append(b, 0.0)).reshape(n, 1, 1))
    return F, (z, jnp.ones((n, 1, 1)), z), lam


def test_target_between_an_isolated_and_an_accumulating_eigenvalue_converges():
    """Rayleigh-Ritz on ``F`` itself picks ghost pairs (mixtures of eigenvectors on both sides of
    the target, with ``theta`` near it): the iteration then stalls at residual ~1e-1 and eigenvalue
    error 0.3. Rayleigh-Ritz for the shift-invert operator ranks by ``|mu|`` and converges."""
    F, G, lam = _accumulating_pencil()
    result = sx.generalized_eigenpairs(F, G, 0.8, 3, 1e-10, 100, 0)
    want = np.sort(lam[np.argsort(np.abs(lam - 0.8))[:3]])
    assert bool(result.converged) and int(result.iterations) < 100
    np.testing.assert_allclose(np.asarray(result.values), want, rtol=1e-12)
    assert float(result.gap) > 0.0


def _kronecker_pencil(n, m):
    """``F = T_n (x) I_m + M_n (x) T_m``, ``G = M_n (x) I_m`` with its closed-form spectrum.

    ``T = tridiag(-1, 2, -1)`` and the linear finite-element mass ``M = tridiag(1, 4, 1) / 6``
    share the eigenvectors ``s_k = sin(i k pi / (n + 1))``, with eigenvalues
    ``lam_k = 2 - 2 cos(k pi / (n + 1))`` and ``mu_k = (4 + 2 cos(k pi / (n + 1))) / 6``
    (LeVeque, Finite Difference Methods, 2007, section 2.10), so that
    ``F (s_j (x) s_k) = (lam_j / mu_j + lam_k) G (s_j (x) s_k)``.
    """
    theta_n, theta_m = (np.arange(1, k + 1) * np.pi / (k + 1) for k in (n, m))
    t_m = 2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
    ones = np.ones((n, 1, 1))
    F = (ones * (-np.eye(m) + t_m / 6), ones * (2 * np.eye(m) + 4 * t_m / 6))
    G = (ones * np.eye(m) / 6, ones * 4 * np.eye(m) / 6)
    mu = (4 + 2 * np.cos(theta_n)) / 6
    spectrum = ((2 - 2 * np.cos(theta_n)) / mu)[:, None] + (2 - 2 * np.cos(theta_m))[None, :]
    s_n = np.sin(np.outer(np.arange(1, n + 1), theta_n))
    s_m = np.sin(np.outer(np.arange(1, m + 1), theta_m))
    vectors = np.einsum("ij,ck->icjk", s_n, s_m)  # (block i, component c, pair j, k)
    triple = [tuple(jnp.asarray(a) for a in (off, d, off)) for off, d in (F, G)]
    return *triple, spectrum, mu, vectors


@pytest.mark.parametrize("n,m", [(200, 1), (40, 3)])
def test_kronecker_pencil_eigenvalues_match_closed_form(n, m):
    F, G, spectrum, _, _ = _kronecker_pencil(n, m)
    result = sx.generalized_eigenpairs(F, G, 3.0, 4)
    want = _nearest(spectrum.ravel(), 3.0, 4)
    # Hermitian pencil: |lam - theta| <= ||F y - theta G y||_2 / (lambda_min(G) ||y||_2), with
    # lambda_min(G) > 1/3; the residual is reported in the infinity norm, which bounds the
    # 2-norm ratio within a factor sqrt(n m).
    Fd, Gd = _dense_pencil(F, G)
    scale = np.abs(Fd).sum(1).max() + np.abs(want) * np.abs(Gd).sum(1).max()
    bound = 3 * np.sqrt(n * m) * np.asarray(result.residuals) * scale
    assert bool(result.converged)
    assert np.all(np.abs(np.asarray(result.values) - want) <= bound)


@pytest.mark.parametrize("n,m", [(200, 1), (40, 3)])
def test_block_thomas_on_shifted_kronecker_pencil_matches_closed_form(n, m):
    # (F - sigma G) v = mu_j (Lam - sigma) v for every Kronecker eigenvector v.
    F, G, spectrum, mu, vectors = _kronecker_pencil(n, m)
    shifted = tuple(f - 3.0 * g for f, g in zip(F, G, strict=True))
    eig = mu[:, None] * (spectrum - 3.0)
    x = sx.block_thomas(*shifted, jnp.asarray(vectors.reshape(n, m, n * m)))
    want = (vectors / eig).reshape(n, m, n * m)
    # Forward error of a backward-stable solve: kappa eps per column, with a margin of 100.
    kappa = np.abs(eig).max() / np.abs(eig).min()
    err = np.abs(np.asarray(x) - want).max((0, 1)) / np.abs(want).max((0, 1))
    assert err.max() < 100 * kappa * np.finfo(float).eps


def _diagonal(lam):
    """``F = diag(lam)``, ``G = I`` as 1 x 1 blocks: the eigenvalues are ``lam`` exactly."""
    z = jnp.zeros((lam.size, 1, 1))
    return (z, jnp.asarray(lam).reshape(-1, 1, 1), z), (z, jnp.ones((lam.size, 1, 1)), z)


def test_converged_certifies_the_nearest_excluded_pair():
    """Diagonal pencil of plan section 8.9, eigenvalues ``1e-3`` and ``linspace(1, 10, 191)``: the
    selected pair converges in the first sweep and the excluded one at ``lam = 1`` does not reach
    ``tol``. S8: its residual bound ``||r||_{G^-1}`` (here ``||r||``, ``G = I``) already resolves
    the gap of about 1, so the first sweep is certified; the reported gap is within that bound."""
    lam = np.concatenate([[1e-3], np.linspace(1.0, 10.0, 191)])
    F, G = _diagonal(lam)
    r = sx.generalized_eigenpairs(F, G, 0.0, 1, 1e-10, 100, 0)
    assert float(r.residuals.max()) <= 1e-10 < float(r.excluded_residual)
    assert bool(r.converged) and int(r.iterations) == 0
    np.testing.assert_allclose(np.asarray(r.values), [1e-3], rtol=1e-12)
    # |theta - 1| <= ||r||_2 <= ||r||_inf sqrt(n), ||r||_inf <= excluded_residual (||F|| + |theta|)
    bound = float(r.excluded_residual) * (10.0 + 1.0) * np.sqrt(lam.size)
    assert abs(float(r.gap) - (1.0 - 1e-3)) <= bound < float(r.gap) / 2


def test_excluded_pair_in_a_cluster_stops_once_it_resolves_the_gap():
    """Three values at the target, then 397 in ``[0.6, 0.61]``: the nearest excluded Ritz pair sits
    in the cluster and converges at a ratio near one, so the selected pairs pass ``tol`` after any
    number of sweeps while the excluded one needs about 12 (SOLVAX 0.36.2 reported ``converged``
    at sweep 1 with an excluded residual of 2.9e-4). S8: that pair resolves the gap of 0.1 after
    the first sweep, so it stops there (it ran 12 sweeps to ``tol`` before) and is certified."""
    F, G = _diagonal(np.concatenate([[0.4999, 0.5, 0.5001], np.linspace(0.6, 0.61, 397)]))
    r = sx.generalized_eigenpairs(F, G, 0.50003, 3, 1e-10, 100, 0)
    assert float(r.residuals.max()) <= 1e-10 < float(r.excluded_residual)
    assert bool(r.converged) and int(r.iterations) == 0
    np.testing.assert_allclose(np.asarray(r.values), [0.4999, 0.5, 0.5001], rtol=1e-12)
    np.testing.assert_allclose(float(r.gap), 0.6 - 0.5001, rtol=1e-3)


def test_excluded_pair_that_does_not_resolve_the_gap_is_not_certified():
    """Near tie: the selected value 0.4 and the excluded cluster starting at ``0.6 + 1e-7`` are
    ``1e-7`` apart in distance from the target 0.5, so the gap is resolved only once the excluded
    error bound is below ``5e-8``: the early sweeps must not report ``converged``."""
    F, G = _diagonal(np.concatenate([[0.4], np.linspace(0.6 + 1e-7, 0.61, 399)]))
    for max_iter in range(3):
        r = sx.generalized_eigenpairs(F, G, 0.5, 1, 1e-10, max_iter, 0)
        assert float(r.excluded_residual) > 1e-10 and not bool(r.converged)
    r = sx.generalized_eigenpairs(F, G, 0.5, 1, 1e-10, 3, 0)  # selected at tol, excluded is not
    assert float(r.residuals.max()) <= 1e-10 < float(r.excluded_residual)
    assert not bool(r.converged)


def test_excluded_residual_under_jit_vmap_and_derivatives(pencil):
    """The new field rides through ``jit`` and ``vmap``; value derivatives are unchanged in both
    modes; positional access to the earlier fields is unchanged."""
    F0, F1, G0, G1 = pencil
    run = jax.jit(lambda t: sx.generalized_eigenpairs(_along(F0, F1, t), G0, TARGET, COUNT))
    batched = jax.vmap(run)(jnp.array([0.0, 0.1]))
    assert batched.excluded_residual.shape == (2,)
    assert bool(batched.converged.all())
    # The rule itself, from an independent bound: |theta - lam| <= ||r||_{G^-1} <= ||r||_2 /
    # sqrt(s_min) <= sqrt(N) ||r||_inf / sqrt(s_min), ||x||_inf <= ||x||_2 <= 1 / sqrt(s_min),
    # s_min the smallest eigenvalue of the dense G; ||r||_inf = res (||F|| + |theta| ||G||) ||x||.
    Gd = _dense_pencil(F0, G0)[1]
    s_min, N = np.linalg.eigvalsh(Gd)[0], Gd.shape[0]
    nG = np.abs(Gd).sum(1).max()
    for i, t in enumerate([0.0, 0.1]):
        Fd = _dense_pencil(_along(F0, F1, t), G0)[0]
        nF, far = np.abs(Fd).sum(1).max(), np.abs(np.asarray(batched.values[i]) - TARGET).max()
        theta = abs(TARGET) + far + float(batched.gap[i])  # |theta| bound of the excluded pair
        e = lambda res, th, nF=nF: res * (nF + th * nG) * np.sqrt(N) / s_min  # noqa: E731
        bound = e(float(batched.excluded_residual[i]), theta)
        bound += e(float(batched.residuals[i].max()), abs(TARGET) + far)
        assert float(batched.excluded_residual[i]) <= 1e-10 or bound <= float(batched.gap[i]) / 2
    r = run(0.0)
    # positions 0-7 keep their meaning: compare with a separate eager call, read by name
    eager = sx.generalized_eigenpairs(F0, G0, TARGET, COUNT)
    fields = sx.EigenpairResult._fields
    assert fields[:8] == ("values", "vectors", "residuals", "orthogonality", "iterations",
                          "converged", "gap", "fallback")  # fmt: skip
    assert {"excluded_residual", "probe_residual", "hermitian_defect"} <= set(fields)
    for i, name in enumerate(fields[:8]):
        np.testing.assert_allclose(r[i], getattr(eager, name), rtol=1e-9, atol=1e-12, err_msg=name)
    total = lambda t: run(t).values.sum()  # noqa: E731
    h = 1e-6
    fd = (total(h) - total(-h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(total)(0.0), fd, rtol=1e-6)
    np.testing.assert_allclose(jax.jvp(total, (0.0,), (1.0,))[1], fd, rtol=1e-6)


def _lu_shapes(jaxpr):
    """Operand shapes of every ``lu`` primitive in a closed jaxpr, sub-jaxprs included."""
    shapes = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "lu":
            shapes.append(tuple(eqn.invars[0].aval.shape))
        for value in eqn.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                sub = getattr(sub, "jaxpr", sub)  # a ClosedJaxpr holds its Jaxpr
                if hasattr(sub, "eqns"):
                    shapes += _lu_shapes(sub)
    return shapes


def test_default_compiles_no_dense_lu_under_vmap():
    """Gate of S4: ``vmap`` over 8 targets with the default ``dense_limit`` compiles no LU of
    size ``n_blocks * m``; a ``cond`` under ``vmap`` would run it on every call."""
    n, m = 100, 2  # 200 unknowns, below the former default of 2048
    rng = np.random.default_rng(4)
    F, G = _band(rng, n, m, False), _band(rng, n, m, False, 4 * m)
    run = lambda t: sx.generalized_eigenpairs(F, G, t, 2).values  # noqa: E731
    jaxpr = jax.make_jaxpr(jax.vmap(run))(jnp.linspace(-1.0, 1.0, 8)).jaxpr
    assert (n * m, n * m) not in [s[-2:] for s in _lu_shapes(jaxpr)]
    explicit = lambda t: sx.generalized_eigenpairs(F, G, t, 2, dense_limit=n * m).values  # noqa: E731
    jaxpr = jax.make_jaxpr(jax.vmap(explicit))(jnp.linspace(-1.0, 1.0, 8)).jaxpr
    assert (n * m, n * m) in [s[-2:] for s in _lu_shapes(jaxpr)]  # opt-in still compiles it


def test_probe_failure_is_reported_at_any_size():
    """``fallback`` is the probe's verdict and ``probe_residual`` its value, with or without the
    dense path."""
    n = 14
    rng = np.random.default_rng(2)
    diag = rng.normal(size=(n, 1, 1))
    diag[-1] = 0.7  # singular last Schur pivot at sigma = 0.7 (as in the dense fallback test)
    F = (None, jnp.asarray(diag), jnp.asarray(rng.normal(size=(n, 1, 1))))
    G = (jnp.zeros((n, 1, 1)), jnp.ones((n, 1, 1)), jnp.zeros((n, 1, 1)))
    bound = np.finfo(float).eps ** 0.75  # the probe threshold
    for limit in (0, n):
        r = sx.generalized_eigenpairs(F, G, 0.7, 2, dense_limit=limit)
        assert bool(r.fallback) and not float(r.probe_residual) <= bound
        assert bool(r.converged) or limit == 0
    F, G = _band(rng, 10, 3, False), _band(rng, 10, 3, False, 12)
    r = sx.generalized_eigenpairs(F, G, 0.2, 2)
    assert not bool(r.fallback) and float(r.probe_residual) <= bound


def test_target_exactly_on_an_eigenvalue_of_a_large_pencil_converges():
    """Plan step S5: ``tridiag(1, 0, 1)`` with 401 unknowns has the eigenvalue
    ``2 cos(201 pi / 402) = 0``. With the target on it, ``|mu| ~ 1e14``, a Krylov block loses
    rank and the Cholesky factor of the Rayleigh-Ritz Gram matrix was NaN."""
    n = 401
    ones, zero = jnp.ones((n, 1, 1)), jnp.zeros((n, 1, 1))
    F, G = (None, zero, ones), (zero, ones, zero)
    exact = 2 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))
    r = sx.generalized_eigenpairs(F, G, 0.0, 3)
    assert bool(r.converged) and np.isfinite(np.asarray(r.residuals)).all()
    np.testing.assert_allclose(np.asarray(r.values), np.sort(_nearest(exact, 0.0, 3)), atol=1e-12)


def test_ritz_cuts_a_rank_deficient_basis():
    """A repeated basis column makes the Gram matrix singular, where a Cholesky factor is NaN:
    the cut direction comes back as a zero vector with ``theta = 0`` and the others are exact."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(20, 20))
    A = jnp.asarray(A + A.T)
    V = jnp.asarray(np.linalg.qr(rng.normal(size=(20, 4)))[0])
    V = jnp.concatenate([V, V[:, :1]], 1)
    theta, C = _ritz(V.T @ A @ V, V.T @ V)
    assert np.isfinite(np.asarray(theta)).all() and np.isfinite(np.asarray(C)).all()
    cut = np.linalg.norm(np.asarray(C), axis=0) < 1e-12
    assert cut.sum() == 1 and abs(float(theta[cut][0])) < 1e-12
    Q = np.linalg.qr(np.asarray(V[:, :4]))[0]
    np.testing.assert_allclose(
        np.sort(np.asarray(theta)[~cut]), np.linalg.eigvalsh(Q.T @ np.asarray(A) @ Q), atol=1e-12
    )


def _skew(F, eps):
    """``F`` with ``eps`` times a fixed anti-Hermitian matrix added to ``diag``: its Hermitian part
    is ``F``."""
    k = jnp.triu(jnp.ones(F[1].shape[-2:]), 1)
    return F[0], F[1] + eps * (k - k.T).astype(F[1].dtype), F[2]


def test_non_hermitian_input_is_refused(pencil):
    """S6: a non-Hermitian ``diag`` is not silently replaced by its Hermitian part. The defect is
    returned and ``converged`` is False above ``tol``."""
    F0, _, G0, _ = pencil
    clean = sx.generalized_eigenpairs(F0, G0, TARGET, COUNT)
    assert float(clean.hermitian_defect) <= 1e-15 and bool(clean.converged)
    bent = _skew(F0, 1e-3)
    r = sx.generalized_eigenpairs(bent, G0, TARGET, COUNT)
    assert float(r.hermitian_defect) > 1e-5
    assert not bool(r.converged)
    rg = sx.generalized_eigenpairs(F0, _skew(G0, 1e-3), TARGET, COUNT)
    assert float(rg.hermitian_defect) > 1e-5 and not bool(rg.converged)
    # a defect below tol passes
    small = sx.generalized_eigenpairs(_skew(F0, 1e-13), G0, TARGET, COUNT)
    assert float(small.hermitian_defect) <= 1e-10 and bool(small.converged)


def test_hermitian_defect_under_jit_vmap_and_grad(pencil):
    """The new last field rides through ``jit`` and ``vmap`` and has zero, finite derivative."""
    F0, F1, G0, _ = pencil
    run = jax.jit(lambda e: sx.generalized_eigenpairs(_skew(F0, e), G0, TARGET, COUNT))
    batched = jax.vmap(run)(jnp.array([0.0, 1e-3]))
    assert sx.EigenpairResult._fields[-1] == "hermitian_defect"
    assert batched.hermitian_defect.shape == (2,)
    assert [bool(c) for c in batched.converged] == [True, False]
    g = jax.grad(lambda t: run(1e-3 * t).hermitian_defect)(1.0)
    assert np.isfinite(float(g)) and float(g) == 0.0
    g = jax.grad(lambda t: sx.generalized_eigenpairs(_along(F0, F1, t), G0, TARGET, COUNT)
                 .hermitian_defect)(0.0)  # fmt: skip
    assert float(g) == 0.0


def test_non_hermitian_lower_is_refused():
    """S6: a real non-symmetric tridiagonal (diag 2, upper -1, lower -0.2, n=40) has its spectrum
    in [1.106, 2.894]; its Hermitian part's eigenvalues near 0.5 are not eigenvalues of it."""
    n = 40
    F = (jnp.full((n, 1, 1), -0.2), jnp.full((n, 1, 1), 2.0), jnp.full((n, 1, 1), -1.0))
    G = (None, jnp.ones((n, 1, 1)), jnp.zeros((n, 1, 1)))
    r = sx.generalized_eigenpairs(F, G, 0.5, 3)
    assert not bool(r.converged)
    assert 0.1 < float(r.hermitian_defect) < 10  # ||lower - upper^H|| / ||A_h|| = 0.8 / 4
    consistent = (jnp.full((n, 1, 1), -1.0), F[1], F[2])
    ok = sx.generalized_eigenpairs(consistent, G, 0.5, 3)
    assert float(ok.hermitian_defect) == 0.0 and bool(ok.converged)


def test_default_tol_follows_the_dtype():
    """S6: ``tol=None`` is ``max(1e-10, 100 eps)``; a float32 pencil converges in a few sweeps
    instead of running ``max_iter`` sweeps to ``converged=False`` at a residual of about 1.9e-7."""
    rng = np.random.default_rng(1)
    F, G = _band(rng, N_BLOCKS, M, False), _band(rng, N_BLOCKS, M, False, 4 * M)
    f32 = lambda b: tuple(None if a is None else a.astype(jnp.float32) for a in b)  # noqa: E731
    r = sx.generalized_eigenpairs(f32(F), f32(G), TARGET, COUNT)
    assert bool(r.converged) and int(r.iterations) < 20
    assert float(r.residuals.max()) <= 100 * float(jnp.finfo(jnp.float32).eps)
    r64 = sx.generalized_eigenpairs(F, G, TARGET, COUNT)
    assert bool(r64.converged) and float(r64.residuals.max()) <= 1e-10


def _laplacian(n, t):
    """``F = tridiag(-1, 2 + t sin(k), -1)``, ``G = I``: a dense spectrum near 0, so a target below
    it converges at a ratio near one and needs several sweeps."""
    z = jnp.zeros((n, 1, 1))
    upper = -jnp.ones((n, 1, 1)).at[-1].set(0)
    return (None, jnp.asarray(2 + t * np.sin(np.arange(n))).reshape(n, 1, 1), upper), (z, z + 1, z)


def test_warm_start_from_a_nearby_pencil_takes_fewer_sweeps():
    """S8 warm start: the vectors of ``t = 0`` start the solve at ``t = 0.01``; same values as a
    cold start and as ``eigvalsh`` to 1e-12 (tol 1e-10 times a condition of order 1)."""
    n, target = 400, -0.2
    F0, G = _laplacian(n, 0.0)
    F, _ = _laplacian(n, 1e-2)
    previous = sx.generalized_eigenpairs(F0, G, target, 3, 1e-10)
    cold = sx.generalized_eigenpairs(F, G, target, 3, 1e-10)
    warm = sx.generalized_eigenpairs(F, G, target, 3, 1e-10, start=previous.vectors)
    assert int(cold.iterations) >= 2 and bool(cold.converged) and bool(warm.converged)
    assert int(warm.iterations) < int(cold.iterations)
    A = np.diag(np.asarray(F[1]).ravel()) - np.eye(n, k=1) - np.eye(n, k=-1)
    np.testing.assert_allclose(warm.values, _nearest(np.linalg.eigvalsh(A), target, 3), atol=1e-12)
    np.testing.assert_allclose(warm.values, cold.values, atol=1e-12)
    # a partial start fills the rest with the random block; a wrong shape is refused
    part = sx.generalized_eigenpairs(F, G, target, 3, 1e-10, start=previous.vectors[..., :1])
    np.testing.assert_allclose(part.values, cold.values, atol=1e-12)
    with pytest.raises(ValueError, match="start"):
        sx.generalized_eigenpairs(F, G, target, 3, start=previous.vectors[1:])
    with pytest.raises(ValueError, match="start"):
        sx.generalized_eigenpairs(F, G, target, 3, start=jnp.zeros((n, 1, 15)))


def test_start_has_zero_derivative_and_none_is_unchanged(pencil):
    """``start=None`` is the cold start; values carry the implicit derivative, ``start`` none."""
    F0, F1, G0, G1 = pencil
    cold = _solve(pencil)
    for name in ("values", "residuals", "iterations"):
        again = getattr(_solve(pencil, start=None), name)
        np.testing.assert_array_equal(again, getattr(cold, name))
    loss = lambda s: _solve(pencil, start=s).values.sum()  # noqa: E731
    np.testing.assert_array_equal(jax.grad(loss)(cold.vectors), 0.0)


def test_max_iter_is_traced_and_does_not_recompile():
    """S8: a static ``max_iter`` compiled once per value (1.1 to 2.6 s each on CPU)."""
    F, G = _laplacian(50, 0.0)
    run = jax.jit(lambda mi: sx.generalized_eigenpairs(F, G, -0.2, 2, 1e-10, mi).iterations)
    assert int(run(0)) == 0 and int(run(30)) > 0
    assert run._cache_size() == 1
    before = sx.generalized_eigenpairs._cache_size()
    sx.generalized_eigenpairs(F, G, -0.2, 2, 1e-10, 7)
    sx.generalized_eigenpairs(F, G, -0.2, 2, 1e-10, 9)
    assert sx.generalized_eigenpairs._cache_size() - before <= 1


def test_gap_rule_margin_and_selected_term():
    """S8 rule ``e_excluded + max e_selected <= gap / 2``: distances 0.1 and 0.2 (gap 0.1). With
    radii 0.02 and 0.04 the sum 0.06 exceeds 0.05: not resolved, although 0.06 <= gap and the
    excluded radius alone (0.04) is below gap / 2. Radii 0.01 and 0.03 resolve it."""
    distance = jnp.array([0.1, 0.2, 0.5])
    assert not bool(_gap_resolved(distance, jnp.array([0.02, 0.04, 9.0]), 1))
    assert bool(_gap_resolved(distance, jnp.array([0.01, 0.03, 9.0]), 1))


@pytest.mark.parametrize("cplx", [False, True], ids=["real", "complex"])
def test_gap_rule_uses_the_g_inverse_norm(cplx):
    """``(s F, s G)`` has the eigenpairs of ``(F, G)``; with ``x^H G x = 1`` the residual scales by
    ``sqrt(s)`` and ``||r||_{G^-1}`` not at all, so the cluster pencil (gap 0.1, excluded residual
    2.9e-4 after one sweep) stops after one sweep for any ``s``. A 2-norm bound would be
    ``sqrt(s) = 1e4`` times larger at ``s = 1e8`` and would not resolve the gap. ``G`` is a
    non-identity diagonal (entries 1 to 4); ``cplx`` runs the same pencil in complex128."""
    lam = np.concatenate([[0.4999, 0.5, 0.5001], np.linspace(0.6, 0.61, 397)])
    g = 1.0 + 3.0 * np.random.default_rng(0).random(lam.size)
    z = np.zeros((lam.size, 1, 1))
    for s in (1.0, 1e8):
        F = (z, jnp.asarray(s * lam * g).reshape(-1, 1, 1), z)
        G = (z, jnp.asarray(s * g).reshape(-1, 1, 1), z)
        if cplx:
            F, G = [tuple(a.astype(complex) for a in b) for b in (F, G)]
        r = sx.generalized_eigenpairs(F, G, 0.50003, 3, 1e-10, 100, 0)
        assert float(r.residuals.max()) <= 1e-10 < float(r.excluded_residual)
        assert bool(r.converged) and int(r.iterations) == 0
        np.testing.assert_allclose(np.asarray(r.values), [0.4999, 0.5, 0.5001], rtol=1e-12)
