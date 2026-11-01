"""Tests for ``solvax.separable`` and ``solvax.inverse_iteration``.

References are independent of the code under test: a dense face-by-face assembly, closed-form
series-resistance conduction, a manufactured Poisson solution, Taylor remainders and dense
generalized eigenproblems.
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

import solvax as sx

jax.config.update("jax_enable_x64", True)

KINDS = [
    ("dirichlet", "dirichlet"),
    ("neumann", "periodic"),
    ("periodic", "dirichlet"),
    ("neumann", "neumann"),
]


def _widths(shape, seed=0):
    rng = np.random.default_rng(seed)
    return [0.1 + rng.random(n) for n in shape]


def _volumes(widths):
    return functools.reduce(np.multiply.outer, widths)


def dense_flux_form(widths, kinds, k, s, c):
    """Assemble ``A`` face by face: outward flux through each face, area times conductance."""
    shape, volumes = tuple(len(w) for w in widths), _volumes(widths)
    n = int(np.prod(shape))
    matrix = np.diag((volumes * s).ravel())
    for idx in itertools.product(*map(range, shape)):
        row = np.ravel_multi_index(idx, shape)
        for axis, kind in enumerate(kinds):
            w, size = widths[axis], shape[axis]
            area = volumes[idx] / w[idx[axis]]
            for step in (-1, 1):
                j = idx[axis] + step
                if 0 <= j < size or kind == "periodic":
                    nb = idx[:axis] + (j % size,) + idx[axis + 1 :]
                    g = 1.0 / (w[idx[axis]] / 2 / k[idx] + w[j % size] / 2 / k[nb])
                    matrix[row, row] += c * area * g
                    matrix[row, np.ravel_multi_index(nb, shape)] -= c * area * g
                elif kind == "dirichlet":
                    matrix[row, row] += c * area * k[idx] / (w[idx[axis]] / 2)
    assert matrix.shape == (n, n)
    return matrix


def _problem(shape, kinds, contrast=20.0, seed=0, shift=0.5):
    rng = np.random.default_rng(seed + 1)
    widths = _widths(shape, seed)
    k = np.exp(rng.standard_normal(shape) * np.log(contrast) / 3.0)
    s = shift * (1.0 + rng.random(shape))
    return widths, kinds, k, s, 0.7


@pytest.mark.parametrize("kinds", KINDS)
def test_flux_form_matches_dense_face_assembly_and_is_spd(kinds):
    widths, kinds, k, s, c = _problem((5, 4), kinds)
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    dense = dense_flux_form(widths, kinds, k, s, c)
    x = np.random.default_rng(3).standard_normal(5 * 4)
    assert np.allclose(op.apply(x.reshape(5, 4)).ravel(), dense @ x, rtol=1e-13, atol=1e-13)
    assert np.allclose(dense, dense.T, rtol=1e-13, atol=1e-13)
    assert np.linalg.eigvalsh(dense).min() > 0.0


def test_flux_form_three_axes_matches_dense():
    widths, kinds, k, s, c = _problem((3, 4, 3), ("dirichlet", "periodic", "neumann"))
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    x = np.random.default_rng(4).standard_normal((3, 4, 3))
    reference = dense_flux_form(widths, kinds, k, s, c) @ x.ravel()
    assert np.allclose(op.apply(x).ravel(), reference, rtol=1e-13, atol=1e-13)


def test_two_cell_conductance_is_the_series_resistance():
    # Two unit-volume cells: the coupling is 1 / (d1/k1 + d2/k2), the area and volume entering once.
    widths, k = [np.array([0.4, 1.2])], np.array([3.0, 0.5])
    op = sx.flux_form_operator(widths, ("neumann",), k, 1.0, 1.0)
    coupling = 1.0 / (0.2 / 3.0 + 0.6 / 0.5)
    expected = np.array([[0.4 + coupling, -coupling], [-coupling, 1.2 + coupling]])
    assert np.allclose(op.apply(jnp.eye(2)[0]), expected[:, 0], rtol=1e-14)
    assert np.allclose(op.apply(jnp.eye(2)[1]), expected[:, 1], rtol=1e-14)


@pytest.mark.parametrize("kinds", KINDS)
def test_solve_matches_dense_and_reports_status(kinds):
    widths, kinds, k, s, c = _problem((6, 5), kinds, contrast=100.0)
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    rhs = np.random.default_rng(5).standard_normal((6, 5))
    x, info = jax.jit(lambda o, r: o.solve(r))(op, rhs)
    reference = np.linalg.solve(dense_flux_form(widths, kinds, k, s, c), rhs.ravel())
    assert bool(info.converged) and int(info.iterations) > 0 and float(info.residual) < 1e-11
    assert np.allclose(x.ravel(), reference, rtol=1e-9, atol=1e-11)


def test_one_material_takes_one_iteration():
    widths, kinds, _, _, _ = _problem((7, 6), ("dirichlet", "neumann"))
    op = sx.flux_form_operator(widths, kinds, 2.5, 0.8, 0.3)
    rhs = np.random.default_rng(6).standard_normal((7, 6))
    x, info = op.solve(rhs)
    assert int(info.iterations) == 1
    assert np.allclose(op.apply(x), rhs, rtol=1e-11, atol=1e-11)


def test_layered_conduction_is_exact_with_dirichlet_lift():
    # Two materials in series, x = 0 at one end and 1 at the other: the harmonic mean makes the
    # cell-centre values lie on the exact piecewise-linear profile, to rounding.
    left, right, k1, k2 = np.array([0.1, 0.25, 0.15]), np.array([0.3, 0.2]), 7.0, 0.3
    widths = [np.concatenate([left, right])]
    k = np.concatenate([np.full(3, k1), np.full(2, k2)])
    op = sx.flux_form_operator(widths, ("dirichlet",), k, 0.0, 1.0)
    x, info = op.solve(op.lift([(0.0, 1.0)]))
    flux = 1.0 / (0.5 / k1 + 0.5 / k2)
    centres = np.cumsum(widths[0]) - widths[0] / 2
    exact = np.where(
        centres < 0.5, flux * centres / k1, flux * 0.5 / k1 + flux * (centres - 0.5) / k2
    )
    assert bool(info.converged)
    assert np.allclose(x, exact, rtol=1e-11, atol=1e-12)


def test_neumann_gradient_lift_reproduces_a_linear_profile():
    widths, kinds = _widths((6, 4)), ("neumann", "dirichlet")
    op = sx.flux_form_operator(widths, kinds, 1.0, 0.0, 1.0)
    # x = 2 + 3 a + 5 b with a the first and b the second coordinate: gradient 3 along axis 0,
    # Dirichlet values 2 + 3 a + 0 and 2 + 3 a + 5 L2 along axis 1 at the two faces.
    a = (np.cumsum(widths[0]) - widths[0] / 2)[:, None]
    length = widths[1].sum()
    b = (np.cumsum(widths[1]) - widths[1] / 2)[None, :]
    exact = 2.0 + 3.0 * a + 5.0 * b + 0 * (a + b)
    data = [(3.0, 3.0), (2.0 + 3.0 * a, 2.0 + 3.0 * a + 5.0 * length)]
    x, info = op.solve(op.lift(data))
    assert bool(info.converged) and np.allclose(x, exact, rtol=1e-10)


def test_lift_refuses_a_periodic_axis():
    op = sx.flux_form_operator(_widths((4,)), ("periodic",), 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="periodic"):
        op.lift([(0.0, 0.0)])


def test_flux_is_conserved_across_interfaces():
    # Neumann everywhere with a shift: sum(A x) equals sum(V s x), so the fluxes telescope.
    widths, kinds, k, s, c = _problem((6, 5), ("neumann", "periodic"), contrast=1e3)
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    x = np.random.default_rng(7).standard_normal((6, 5))
    assert abs(jnp.sum(op.apply(x)) - jnp.sum(_volumes(widths) * s * x)) < 1e-11


def test_singular_variant_gauge_defect_and_incompatible_source():
    widths, kinds, k, _, c = _problem((6, 5), ("periodic", "neumann"), contrast=50.0)
    volumes = _volumes(widths)
    op = sx.flux_form_operator(widths, kinds, k, 0.0, c)
    rng = np.random.default_rng(8)
    raw = rng.standard_normal((6, 5))
    compatible = raw - raw.sum() / volumes.sum() * volumes  # sum(rhs) = 0
    x, info = op.solve(compatible)
    assert bool(info.converged) and abs(float(info.defect)) < 1e-12
    assert abs(float(jnp.sum(volumes * x))) < 1e-11  # zero volume-mean gauge
    assert np.allclose(op.apply(x), compatible, atol=1e-10)
    # rounding-level incompatibility is a compatible right-hand side
    _, info = op.solve(compatible + 1e-14 * volumes)
    assert bool(info.converged)
    # a real net source is reported and refused, not absorbed by the gauge
    x, info = op.solve(compatible + 0.3 * volumes)
    assert not bool(info.converged) and np.isnan(x).all()
    assert abs(float(info.defect) - 0.3) < 1e-12
    # transpose test of the autodiff pair, for any direction: <J dr, ct> = <dr, J^T ct>
    ct, dr = rng.standard_normal((6, 5)), rng.standard_normal((6, 5))
    _, tangent = jax.jvp(lambda r: op.solve(r)[0], (compatible,), (dr,))
    _, pullback = jax.vjp(lambda r: op.solve(r)[0], compatible)
    assert abs(jnp.sum(tangent * ct) - jnp.sum(dr * pullback(ct)[0])) < 1e-10
    # on compatible directions the derivative is the zero-mean pseudo-inverse
    dr = dr - dr.sum() / volumes.sum() * volumes
    dense = scipy.linalg.pinvh(dense_flux_form(widths, kinds, k, np.zeros((6, 5)), c))
    gauge = np.eye(30) - np.outer(np.ones(30), volumes.ravel()) / volumes.sum()
    _, tangent = jax.jvp(lambda r: op.solve(r)[0], (compatible,), (dr,))
    assert np.allclose(tangent.ravel(), gauge @ dense @ dr.ravel(), atol=1e-9)


def test_unconverged_solve_is_nan_with_nan_derivatives_in_every_input():
    widths, kinds, k, s, c = _problem((8, 8), ("dirichlet", "dirichlet"), contrast=1e4)
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    rhs = np.random.default_rng(9).standard_normal((8, 8))

    def total(rhs, k, s, c):
        x, info = op.with_coefficients(k, s, c).solve(rhs, max_steps=1)
        return jnp.sum(x**2), info

    (value, info), grads = jax.value_and_grad(total, argnums=(0, 1, 2, 3), has_aux=True)(
        rhs, k, s, c
    )
    assert not bool(info.converged) and np.isnan(value)
    assert all(np.isnan(np.asarray(g)).all() for g in grads)
    # the same call converged has finite derivatives
    (value, info), grads = jax.value_and_grad(
        lambda *a: (lambda x, i: (jnp.sum(x**2), i))(*op.with_coefficients(*a[1:]).solve(a[0])),
        argnums=(0, 1, 2, 3),
        has_aux=True,
    )(rhs, k, s, c)
    assert bool(info.converged) and all(np.isfinite(np.asarray(g)).all() for g in grads)


def test_invalid_coefficients_fail_closed():
    widths, kinds, k, s, c = _problem((5, 5), ("dirichlet", "neumann"))
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    rhs = np.ones((5, 5))
    for bad in ({"conductivity": -k}, {"shift": -s}, {"coefficient": -1.0}):
        x, info = op.with_coefficients(**bad).solve(rhs)
        assert not bool(info.converged) and np.isnan(x).all()


def test_derivatives_match_dense_implicit_derivatives():
    widths, kinds, k, s, c = _problem((5, 4), ("dirichlet", "periodic"))
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    rng = np.random.default_rng(10)
    rhs, weight = rng.standard_normal((5, 4)), rng.standard_normal((5, 4))

    def objective(rhs, k, s, c):
        return jnp.sum(weight * op.with_coefficients(k, s, c).solve(rhs)[0])

    def dense(rhs, k, s, c):
        return jnp.sum(
            weight.ravel() * jnp.linalg.solve(_jax_dense(widths, kinds, k, s, c), rhs.ravel())
        )

    args = (jnp.asarray(rhs), jnp.asarray(k), jnp.asarray(s), jnp.asarray(c))
    got = jax.grad(objective, argnums=(0, 1, 2, 3))(*args)
    want = jax.grad(dense, argnums=(0, 1, 2, 3))(*args)
    for g, w in zip(got, want, strict=True):
        assert np.allclose(g, w, rtol=1e-7, atol=1e-9)


def _jax_dense(widths, kinds, k, s, c):
    """The dense operator as a differentiable function: columns by applying the stencil."""
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    shape = tuple(len(w) for w in widths)
    return jax.vmap(lambda e: op.apply(e.reshape(shape)).ravel(), out_axes=1)(
        jnp.eye(int(np.prod(shape)))
    )


def test_conductivity_taylor_remainder_is_second_order():
    widths, kinds, k, s, c = _problem((6, 5), ("neumann", "dirichlet"))
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    rhs = np.random.default_rng(11).standard_normal((6, 5))
    direction = np.random.default_rng(12).standard_normal((6, 5))

    def f(h):
        return jnp.sum(jnp.sin(op.with_coefficients(k * (1 + h * direction)).solve(rhs)[0]))

    slope = jax.grad(f)(0.0)
    remainders = [abs(f(h) - f(0.0) - h * slope) for h in (1e-2, 5e-3, 2.5e-3)]
    assert 3.6 < remainders[0] / remainders[1] < 4.4 and 3.6 < remainders[1] / remainders[2] < 4.4


def test_new_leaf_values_do_not_retrace_and_vmap_over_coefficients():
    widths, kinds, k, s, c = _problem((5, 4), ("dirichlet", "neumann"))
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    traces = []

    @jax.jit
    def run(op, rhs):
        traces.append(1)
        return op.solve(rhs)[0]

    rhs = np.ones((5, 4))
    first = run(op, rhs)
    second = run(op.with_coefficients(conductivity=2 * k, shift=3 * s), rhs)
    assert len(traces) == 1 and not np.allclose(first, second)
    batched = jax.vmap(lambda scale: op.with_coefficients(conductivity=scale * k).solve(rhs)[0])
    assert batched(jnp.array([1.0, 2.0])).shape == (2, 5, 4)


def test_float32_operator_stays_float32():
    # the working precision is that of the leaves; the default tolerance follows it
    widths, kinds, k, s, c = _problem((5, 4), ("dirichlet", "neumann"), contrast=3.0)
    op = sx.flux_form_operator(widths, kinds, k, s, c)
    op = jax.tree.map(lambda a: a.astype(jnp.float32), op)
    x, info = op.solve(np.ones((5, 4), np.float32))
    assert x.dtype == jnp.float32 and bool(info.converged)


# ---------------------------------------------------------------------------------------------
# FastDiagonal


@pytest.mark.parametrize("kinds", KINDS)
def test_fast_diagonal_matches_dense_and_its_transpose(kinds):
    widths = _widths((5, 4), seed=2)
    fd = sx.fast_diagonal(widths, kinds, shift=0.6, coefficient=1.9)
    volumes = _volumes(widths)
    unit = np.ones((5, 4))
    dense = dense_flux_form(widths, kinds, unit, 0.6 * unit, 1.9) / volumes.ravel()[:, None]
    rng = np.random.default_rng(13)
    x, y = rng.standard_normal(20), rng.standard_normal(20)
    assert np.allclose(fd.matvec(x.reshape(5, 4)).ravel(), dense @ x, rtol=1e-12)
    assert np.allclose(fd.solve(x.reshape(5, 4)).ravel(), np.linalg.solve(dense, x), rtol=1e-11)
    assert np.allclose(
        fd.solve(x.reshape(5, 4), transpose=True).ravel(), np.linalg.solve(dense.T, x), rtol=1e-11
    )
    # dot-product test: <M^-1 x, y> = <x, M^-T y>; M is not symmetric in the plain inner product
    left = fd.solve(x.reshape(5, 4)).ravel() @ y
    right = x @ fd.solve(y.reshape(5, 4), transpose=True).ravel()
    assert abs(left - right) < 1e-12 * abs(left) and not np.allclose(dense, dense.T)


def test_fast_diagonal_poisson_converges_at_second_order_to_a_manufactured_solution():
    # -lap u = f with u = sin(pi x) cos(pi y) cos(2 pi z): Dirichlet, Neumann, periodic on (0, 1)^3.
    def error(n):
        widths = [np.diff(np.linspace(0, 1, n + 1) ** 1.2)] * 3  # smoothly stretched
        centres = [np.cumsum(w) - w / 2 for w in widths]
        x, y, z = np.meshgrid(*centres, indexing="ij")
        u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(2 * np.pi * z)
        fd = sx.fast_diagonal(widths, ("dirichlet", "neumann", "periodic"))
        return float(np.max(np.abs(fd.solve(6 * np.pi**2 * u) - u)))

    coarse, fine = error(12), error(24)
    assert 3.3 < coarse / fine < 4.7


def test_singular_fast_diagonal_is_the_pseudo_inverse_with_zero_mean():
    widths = _widths((5, 4), seed=3)
    fd = sx.fast_diagonal(widths, ("periodic", "neumann"))
    assert fd.singular
    volumes = _volumes(widths)
    rhs = np.random.default_rng(15).standard_normal((5, 4))
    x = fd.solve(rhs)
    assert abs(float(jnp.sum(volumes * x))) < 1e-12
    unit = np.ones((5, 4))
    dense = (
        dense_flux_form(widths, ("periodic", "neumann"), unit, 0 * unit, 1.0)
        / volumes.ravel()[:, None]
    )
    removed = rhs - np.sum(volumes * rhs) / volumes.sum()  # the volume-mean part is dropped
    assert np.allclose(dense @ x.ravel(), removed.ravel(), atol=1e-12)
    assert np.allclose(fd.solve(rhs + 5.0), x, atol=1e-12)


def test_fast_diagonal_derivatives_in_shift_and_coefficient_follow_taylor():
    widths = _widths((5, 4), seed=4)
    rhs = np.random.default_rng(16).standard_normal((5, 4))

    def f(h):
        fd = sx.fast_diagonal(
            widths, ("dirichlet", "neumann"), shift=0.5 + h, coefficient=1.0 + 2 * h
        )
        return jnp.sum(jnp.sin(fd.solve(rhs)))

    fd0 = sx.fast_diagonal(widths, ("dirichlet", "neumann"), shift=0.5)
    slope = jax.grad(f)(0.0)
    remainders = [abs(f(h) - f(0.0) - h * slope) for h in (1e-2, 5e-3, 2.5e-3)]
    assert fd0.solve(rhs).shape == (5, 4) and 3.6 < remainders[0] / remainders[1] < 4.4


def test_fast_diagonal_rejects_bad_input():
    with pytest.raises(ValueError, match="kind"):
        sx.fast_diagonal([np.ones(3)], ("robin",))
    with pytest.raises(ValueError, match="positive"):
        sx.fast_diagonal([np.array([1.0, -1.0])], ("neumann",))
    with pytest.raises(ValueError, match="one entry per axis"):
        sx.fast_diagonal([np.ones(3)], ("neumann", "neumann"))


# ---------------------------------------------------------------------------------------------
# inverse_iteration


def _pencil(seed=0, n=12):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, n))
    stiffness = q @ q.T + n * np.eye(n)
    weight = 0.5 + rng.random(n)
    return stiffness, weight


def test_inverse_iteration_matches_the_dense_generalized_eigenpair():
    stiffness, weight = _pencil()
    lam_all, vec_all = scipy.linalg.eigh(stiffness, np.diag(weight))
    inverse = np.linalg.inv(stiffness)
    lam, v, info = sx.inverse_iteration(
        lambda b: inverse @ b, weight, np.ones(12), tol=1e-12, max_steps=500
    )
    assert bool(info.converged) and float(info.residual) <= 1e-12
    assert abs(lam - lam_all[0]) < 1e-12 * lam_all[0]
    assert abs(abs(vec_all[:, 0] @ (weight * v))) == pytest.approx(1.0, abs=1e-10)
    assert float(v @ (weight * v)) == pytest.approx(1.0)
    assert np.linalg.norm(stiffness @ v - lam * weight * v) < 1e-10


def test_residual_certifies_the_eigenvalue_error():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    stiffness = q @ np.diag(np.arange(1.0, 11.0) ** 2) @ q.T
    weight = 0.5 + rng.random(10)
    spectrum = scipy.linalg.eigh(stiffness, np.diag(weight), eigvals_only=True)
    inverse = np.linalg.inv(stiffness)
    for tol in (1e-1, 1e-2, 1e-4):
        lam, _, info = sx.inverse_iteration(
            lambda b: inverse @ b, weight, np.ones(10), tol=tol, max_steps=500
        )
        assert bool(info.converged) and float(info.residual) <= tol
        assert np.min(np.abs(lam - spectrum) / spectrum) <= float(info.residual)
    assert abs(lam - spectrum[0]) < 1e-3 * spectrum[0]


def test_square_duct_eigenvalue_is_the_sum_of_the_axis_minima():
    # -lap theta = Lambda theta on a Dirichlet box: Lambda is the sum of the 1-D lowest eigenvalues.
    widths = _widths((7, 6), seed=5)
    fd = sx.fast_diagonal(widths, ("dirichlet", "dirichlet"), shift=0.0)
    volumes = _volumes(widths)
    lam, v, info = sx.inverse_iteration(
        lambda b: fd.solve(b / volumes), volumes, np.ones((7, 6)), tol=1e-12, max_steps=500
    )
    expected = fd.eigenvalues[0][0] + fd.eigenvalues[1][0]
    assert bool(info.converged) and abs(lam - expected) < 1e-12 * expected
    assert np.all(v > 0.0) or np.all(v < 0.0)  # the ground state has no node


def test_eigenvalue_first_derivative_exact_scaling_and_dense_finite_differences():
    widths, kinds, k, _, _ = _problem((5, 4), ("dirichlet", "dirichlet"), contrast=10.0)
    volumes = _volumes(widths)
    weight0 = 0.5 + np.random.default_rng(17).random((5, 4))

    def eigenvalue(scale_k, scale_f, k=k):
        op = sx.flux_form_operator(widths, kinds, scale_k * k, 0.0, 1.0)
        solve = lambda b: op.solve(b, rtol=1e-13)[0]  # noqa: E731
        lam, _, info = sx.inverse_iteration(
            solve, volumes * scale_f * weight0, np.ones((5, 4)), tol=1e-11, max_steps=500
        )
        return lam

    lam = eigenvalue(1.0, 1.0)
    # K is linear in k and the weight enters inversely: lam ~ k / f, so both derivatives are exact
    assert jax.grad(eigenvalue, 0)(1.0, 1.0) == pytest.approx(float(lam), rel=1e-8)
    assert jax.grad(eigenvalue, 1)(1.0, 1.0) == pytest.approx(float(-lam), rel=1e-8)
    # a nonuniform perturbation against dense central differences of the generalized eigenvalue
    direction = np.random.default_rng(18).standard_normal((5, 4))
    grad = jax.grad(lambda h: eigenvalue(1.0, 1.0, k * (1 + h * direction)))(0.0)

    def dense(h):
        stiffness = dense_flux_form(widths, kinds, k * (1 + h * direction), np.zeros((5, 4)), 1.0)
        return scipy.linalg.eigh(
            stiffness, np.diag((volumes * weight0).ravel()), eigvals_only=True
        )[0]

    assert grad == pytest.approx((dense(1e-5) - dense(-1e-5)) / 2e-5, rel=1e-6)
    # forward mode agrees with reverse mode
    assert jax.jvp(lambda h: eigenvalue(1.0 + h, 1.0), (0.0,), (1.0,))[1] == pytest.approx(
        float(lam), rel=1e-8
    )


def test_eigenvector_derivative_is_documented_unsupported_and_returned_frozen():
    stiffness, weight = _pencil(2)
    ground = scipy.linalg.eigh(stiffness, np.diag(weight))[1][:, 0]

    def head(scale):
        inverse = jnp.linalg.inv(scale * stiffness)
        _, v, _ = sx.inverse_iteration(
            lambda b: inverse @ b, weight, np.ones(12), tol=1e-12, max_steps=500
        )
        return jnp.sum(v * ground)

    assert jax.grad(head)(1.0) == 0.0  # stop_gradient: zero by construction, not computed


def test_unconverged_or_invalid_eigenproblem_is_nan_with_nan_derivative():
    stiffness, weight = _pencil(3)

    def eigenvalue(scale, steps=2, weight=weight):
        inverse = jnp.linalg.inv(scale * stiffness)
        return sx.inverse_iteration(
            lambda b: inverse @ b, weight, np.arange(1.0, 13.0), tol=1e-13, max_steps=steps
        )

    lam, _, info = eigenvalue(1.0)
    assert not bool(info.converged) and np.isnan(lam)
    assert np.isnan(jax.grad(lambda s: eigenvalue(s)[0])(1.0))
    lam, _, info = eigenvalue(1.0, steps=500, weight=-weight)
    assert not bool(info.converged) and np.isnan(lam)
    lam, _, info = eigenvalue(1.0, steps=500)
    assert bool(info.converged) and np.isfinite(jax.grad(lambda s: eigenvalue(s, 500)[0])(1.0))


# ---------------------------------------------------------------------------------------------
# axis weights and the Schur complement of a constraint stage


def _one_dimensional(w, kind):
    """``M = W^{-1} K`` of one axis from the face-by-face reference assembly."""
    return dense_flux_form([w], [kind], np.ones(len(w)), np.zeros(len(w)), 1.0) / w[:, None]


def test_axis_weights_match_the_dense_weighted_kronecker_sum():
    widths, weights = _widths((5, 4, 3), seed=6), np.array([0.3, 2.0, 0.7])
    kinds = ("dirichlet", "periodic", "neumann")
    mats = [_one_dimensional(w, k) for w, k in zip(widths, kinds, strict=True)]
    eye = [np.eye(len(w)) for w in widths]
    dense = 0.4 * np.eye(60) + 1.1 * (
        weights[0] * np.kron(np.kron(mats[0], eye[1]), eye[2])
        + weights[1] * np.kron(np.kron(eye[0], mats[1]), eye[2])
        + weights[2] * np.kron(np.kron(eye[0], eye[1]), mats[2])
    )
    fd = sx.fast_diagonal(widths, kinds, shift=0.4, coefficient=1.1, weights=weights)
    x = np.random.default_rng(20).standard_normal((5, 4, 3))
    assert np.allclose(fd.matvec(x).ravel(), dense @ x.ravel(), rtol=1e-12)
    assert np.allclose(fd.solve(x).ravel(), np.linalg.solve(dense, x.ravel()), rtol=1e-10)
    assert np.allclose(
        fd.solve(x, transpose=True).ravel(), np.linalg.solve(dense.T, x.ravel()), rtol=1e-10
    )


def test_axis_weighted_poisson_converges_at_second_order_to_a_manufactured_solution():
    # w_x u_xx + w_y u_yy = f for u = sin(pi x) cos(2 pi y): Dirichlet in x, Neumann in y.
    def error(n):
        widths = [np.diff(np.linspace(0, 1, n + 1) ** 1.3)] * 2
        x, y = np.meshgrid(*[np.cumsum(w) - w / 2 for w in widths], indexing="ij")
        u = np.sin(np.pi * x) * np.cos(2 * np.pi * y)
        fd = sx.fast_diagonal(widths, ("dirichlet", "neumann"), weights=[0.25, 4.0])
        return float(np.max(np.abs(fd.solve((0.25 * np.pi**2 + 16 * np.pi**2) * u) - u)))

    assert 3.3 < error(16) / error(32) < 4.7


def test_axis_weight_derivative_follows_taylor():
    widths, rhs = _widths((5, 4), seed=7), np.random.default_rng(21).standard_normal((5, 4))

    def f(h):
        fd = sx.fast_diagonal(widths, ("neumann", "dirichlet"), weights=[1.0 + h, 2.0 - h])
        return jnp.sum(jnp.sin(fd.solve(rhs)))

    slope = jax.grad(f)(0.0)
    r = [abs(f(h) - f(0.0) - h * slope) for h in (1e-2, 5e-3, 2.5e-3)]
    assert 3.6 < r[0] / r[1] < 4.4 and 3.6 < r[1] / r[2] < 4.4
    with pytest.raises(ValueError, match="one scalar per axis"):
        sx.fast_diagonal(widths, ("neumann", "dirichlet"), weights=[1.0])


def _stage(widths, pressure_kinds, shifts, nu):
    """Dense staggered ``G_c``, ``D_c`` and ``H_c`` of a 2-D stage: no-slip walls, periodic wraps.

    Component ``c`` lives on the lower face of every cell along axis ``c`` (interior faces only
    on a wall axis); ``H_c = s_c + nu (-laplacian_c)`` with the normal second difference over
    the dual cell and the tangential cell-centred one with a wall half a cell away.
    """
    sizes = [len(w) for w in widths]
    pieces = []
    for c in range(2):
        w, n, periodic = widths[c], sizes[c], pressure_kinds[c] == "periodic"
        faces = list(range(n)) if periodic else list(range(1, n))
        g = np.zeros((len(faces), n))
        d = np.zeros((n, len(faces)))
        m = np.zeros((len(faces), len(faces)))
        for a, f in enumerate(faces):
            lo, hi = (f - 1) % n, f
            dist = (w[lo] + w[hi]) / 2
            g[a, hi] += 1 / dist
            g[a, lo] -= 1 / dist
            d[hi, a] -= 1 / w[hi]
            d[lo, a] += 1 / w[lo]
            m[a, a] += (1 / w[hi] + 1 / w[lo]) / dist
            for b, width in ((a + 1, w[hi]), (a - 1, w[lo])):
                if periodic:
                    m[a, b % len(faces)] -= 1 / (width * dist)
                elif 0 <= b < len(faces):
                    m[a, b] -= 1 / (width * dist)
        o = 1 - c
        tangential = _one_dimensional(
            widths[o], "periodic" if pressure_kinds[o] == "periodic" else "dirichlet"
        )
        eye_f, eye_o = np.eye(len(faces)), np.eye(sizes[o])
        order = (lambda a, b: np.kron(a, b)) if c == 0 else (lambda a, b: np.kron(b, a))
        h = shifts[c] * np.eye(len(faces) * sizes[o]) + nu * (
            order(m, eye_o) + order(eye_f, tangential)
        )
        pieces.append(
            (order(g, eye_o), order(d, eye_o), h, (len(faces), sizes[o])[:: 1 if c == 0 else -1])
        )
    return pieces


def _dense_schur(pieces):
    return -sum(d @ np.linalg.solve(h, g) for g, d, h, _ in pieces)


def _callables(pieces, shape):
    def gradient(p):
        return [(g @ p.ravel()).reshape(s) for g, _, _, s in pieces]

    def divergence(fields):
        return sum(d @ f.ravel() for (_, d, _, _), f in zip(pieces, fields, strict=True)).reshape(
            shape
        )

    helmholtz = [
        lambda f, h=h, s=s: jnp.linalg.solve(h, f.ravel()).reshape(s) for _, _, h, s in pieces
    ]
    return gradient, divergence, helmholtz


def _kkt(pieces, b, cells):
    """Dense saddle point ``[[H, G], [D, 0]] [u; q] = [b; 0]`` by minimum-norm least squares.

    ``q`` is determined up to a constant, so it is returned with its plain mean removed.
    """
    sizes = [len(p[0]) for p in pieces]
    top = np.hstack(
        [scipy.linalg.block_diag(*[p[2] for p in pieces]), np.vstack([p[0] for p in pieces])]
    )
    bottom = np.hstack([*[p[1] for p in pieces], np.zeros((cells, cells))])
    sol = np.linalg.lstsq(
        np.vstack([top, bottom]), np.concatenate([*b, np.zeros(cells)]), rcond=None
    )[0]
    q = sol[sum(sizes) :]
    return np.split(sol[: sum(sizes)], np.cumsum(sizes)[:-1]), q - q.mean()


def _stage_rhs(pieces, shape, seed):
    b = [np.random.default_rng(seed + c).standard_normal(len(p[0])) for c, p in enumerate(pieces)]
    gradient, divergence, helmholtz = _callables(pieces, shape)
    fields = [
        h(jnp.asarray(bc.reshape(p[3]))) for h, bc, p in zip(helmholtz, b, pieces, strict=True)
    ]
    return b, -divergence(fields), (gradient, divergence, helmholtz)


def test_schur_multiplier_is_exact_where_the_viscous_operator_commutes():
    # periodic, uniform: H_c commutes with G, so the multiplier inverts -D H^-1 G to rounding
    widths, kinds = [np.full(8, 0.25), np.full(6, 0.4)], ("periodic", "periodic")
    shifts, nu = np.array([1.3, 3.5]), 0.07
    pieces = _stage(widths, kinds, shifts, nu)
    fd = sx.fast_diagonal_schur(widths, kinds, shifts, nu)
    r = np.random.default_rng(22).standard_normal((8, 6))
    r = r - r.mean()
    q = fd.solve(r)
    assert np.allclose(_dense_schur(pieces) @ q.ravel(), r.ravel(), atol=1e-11)
    assert abs(q.mean()) < 1e-13
    # with nu = 0 the multiplier is the axis-weighted Kronecker sum, weights 1 / s_c
    weighted = sx.fast_diagonal(widths, kinds, weights=1 / shifts)
    assert np.allclose(
        sx.fast_diagonal_schur(widths, kinds, shifts, 0.0).solve(r), weighted.solve(r)
    )
    # the exact stage agrees with a dense saddle-point solve, in one or two iterations
    b, rhs, (gradient, divergence, helmholtz) = _stage_rhs(pieces, (8, 6), 23)
    q, info = fd.solve_exact(rhs, gradient, divergence, helmholtz)
    u, q_dense = _kkt(pieces, b, 48)
    assert bool(info.converged) and int(info.iterations) <= 2
    assert np.allclose(q.ravel(), q_dense, atol=1e-9)
    velocity = [
        h(jnp.asarray(bc.reshape(p[3])) - g)
        for h, bc, p, g in zip(helmholtz, b, pieces, gradient(q), strict=True)
    ]
    assert all(np.allclose(v.ravel(), uc, atol=1e-9) for v, uc in zip(velocity, u, strict=True))
    assert np.max(np.abs(divergence(velocity))) < 1e-9


def test_schur_exact_stage_with_no_slip_walls_needs_the_inner_iteration():
    widths, kinds = _widths((7, 6), seed=8), ("neumann", "neumann")
    shifts, nu = np.array([1.5, 4.0]), 0.05
    pieces = _stage(widths, kinds, shifts, nu)
    volumes = _volumes(widths)
    schur = _dense_schur(pieces)
    assert np.allclose(
        volumes.ravel()[:, None] * schur, (volumes.ravel()[:, None] * schur).T, atol=1e-10
    )
    fd = sx.fast_diagonal_schur(widths, kinds, shifts, nu)
    b, rhs, (gradient, divergence, helmholtz) = _stage_rhs(pieces, (7, 6), 30)
    rhs = rhs - jnp.sum(volumes * rhs) / volumes.sum()  # compatible
    q, info = fd.solve_exact(rhs, gradient, divergence, helmholtz)
    # with walls the multiplier alone is only a preconditioner
    assert np.max(np.abs(schur @ fd.solve(rhs).ravel() - rhs.ravel())) > 1e-3 * np.max(np.abs(rhs))
    assert bool(info.converged) and 3 <= int(info.iterations) < 40
    assert np.allclose(schur @ q.ravel(), rhs.ravel(), atol=1e-9)
    assert abs(float(jnp.sum(volumes * q))) < 1e-10
    u, q_dense = _kkt(pieces, b, 42)
    assert np.allclose(q.ravel() - np.mean(q), q_dense, atol=1e-8)
    # the preconditioner helps: unpreconditioned CG on V S needs more iterations
    plain = sx.pcg(
        lambda p: volumes * (schur @ p.reshape(-1).copy()).reshape(7, 6),
        volumes * rhs,
        rtol=1e-11,
        max_steps=200,
    )
    assert int(plain.iterations) > int(info.iterations)


def _gauged_solve(schur, volumes, rhs):
    """Dense ``S q = rhs`` with the volume-mean-zero gauge, by bordering ``V S`` with ``V``."""
    n = len(volumes)
    bordered = jnp.block(
        [[volumes[:, None] * schur, volumes[:, None]], [volumes[None, :], jnp.zeros((1, 1))]]
    )
    return jnp.linalg.solve(bordered, jnp.concatenate([volumes * rhs, jnp.zeros(1)]))[:n]


def test_schur_solve_exact_derivatives_and_failure_semantics():
    widths, kinds = _widths((5, 4), seed=9), ("neumann", "neumann")
    shifts, volumes = np.array([1.5, 2.5]), _volumes(widths)
    weight = np.random.default_rng(31).standard_normal((5, 4))
    rhs = np.random.default_rng(32).standard_normal((5, 4))
    rhs = rhs - np.sum(volumes * rhs) / volumes.sum()
    fd = sx.fast_diagonal_schur(widths, kinds, shifts, 0.05)

    def exact(nu, steps=200):
        pieces = _stage(widths, kinds, shifts, nu)
        q, info = fd.solve_exact(jnp.asarray(rhs), *_callables(pieces, (5, 4)), max_steps=steps)
        return jnp.sum(weight * q), info

    def dense(nu):
        schur = _dense_schur_jax(_stage(widths, kinds, shifts, nu))
        return jnp.sum(weight.ravel() * _gauged_solve(schur, volumes.ravel(), rhs.ravel()))

    assert abs(exact(0.05)[0] - dense(0.05)) < 1e-10
    assert jax.grad(lambda n: exact(n)[0])(0.05) == pytest.approx(
        float(jax.grad(dense)(0.05)), rel=1e-7
    )
    # an unconverged exact stage is NaN with NaN derivatives in the closed-over parameter
    value, info = exact(0.05, steps=1)
    assert not bool(info.converged) and np.isnan(value)
    assert np.isnan(jax.grad(lambda n: exact(n, 1)[0])(0.05))
    # an operator of the wrong sign is reported, not solved
    gradient, divergence, helmholtz = _callables(_stage(widths, kinds, shifts, 0.05), (5, 4))
    q, info = fd.solve_exact(jnp.asarray(rhs), gradient, lambda f: -divergence(f), helmholtz)
    assert not bool(info.converged) and np.isnan(q).all()


def _dense_schur_jax(pieces):
    return -sum(
        jnp.asarray(d) @ jnp.linalg.solve(jnp.asarray(h), jnp.asarray(g)) for g, d, h, _ in pieces
    )


def test_schur_singular_pressure_reports_a_net_source_and_rejects_bad_shifts():
    widths, kinds = _widths((5, 4), seed=10), ("neumann", "neumann")
    pieces = _stage(widths, kinds, np.array([1.2, 1.7]), 0.1)
    fd = sx.fast_diagonal_schur(widths, kinds, [1.2, 1.7], 0.1)
    gradient, divergence, helmholtz = _callables(pieces, (5, 4))
    q, info = fd.solve_exact(0.4 + jnp.zeros((5, 4)), gradient, divergence, helmholtz)
    assert not bool(info.converged) and np.isnan(q).all() and abs(float(info.defect) - 0.4) < 1e-12
    with pytest.raises(ValueError, match="one entry per axis"):
        sx.fast_diagonal_schur(widths, kinds, [1.0])
    bad = sx.fast_diagonal_schur(widths, kinds, [1.2, -1.0], 0.1)
    rhs = jnp.asarray(np.random.default_rng(33).standard_normal((5, 4)))
    q, info = bad.solve_exact(rhs - rhs.mean(), gradient, divergence, helmholtz)
    assert not bool(info.converged)
