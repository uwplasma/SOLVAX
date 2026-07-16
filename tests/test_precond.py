"""Tests for solvax.precond: preconditioner builders vs GMRES iteration counts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import lu_factor

from solvax import (
    additive_preconditioner,
    block_jacobi,
    block_thomas_factor,
    block_thomas_solve,
    coarse_operator,
    galerkin_deflation,
    gmres,
    jacobi,
    kronecker_nkp,
    line_smoother,
    linear_solve,
    lu_factor_banded,
    lu_solve_banded,
    mixed_precision,
    nearest_kronecker,
    p_multigrid,
    pcg,
)

jax.config.update("jax_enable_x64", True)


def gmres_iters(matvec, b, precond=None, rtol=1e-8, restart=40, max_restarts=50):
    """Run GMRES and return (iterations, solution)."""
    sol = gmres(
        matvec, b, precond=precond, rtol=rtol, restart=restart,
        max_restarts=max_restarts,
    )
    return int(sol.iterations), sol


def block_scaled_system(n_blocks=20, m=4, seed=0):
    """Block-diagonally-dominant system with widely varying block scales.

    Strong dense diagonal blocks (scaled per block over 3 decades) plus a
    weak off-block coupling, so (block-)diagonal scaling repairs the
    conditioning.
    """
    rng = np.random.default_rng(seed)
    n = n_blocks * m
    dense = 0.05 * rng.standard_normal((n, n)) / np.sqrt(n)
    blocks = np.zeros((n_blocks, m, m))
    scales = np.logspace(0.0, 3.0, n_blocks)
    for k in range(n_blocks):
        blk = scales[k] * (np.eye(m) + 0.3 * rng.standard_normal((m, m)))
        blocks[k] = blk
        dense[k * m : (k + 1) * m, k * m : (k + 1) * m] = blk
    b = rng.standard_normal(n)
    return jnp.asarray(dense), jnp.asarray(blocks), jnp.asarray(b)


def laplacian_1d(n):
    """Dirichlet 1-D Laplacian tridiag(-1, 2, -1) as a dense array."""
    return np.diag(2.0 * np.ones(n)) - np.diag(np.ones(n - 1), 1) - np.diag(
        np.ones(n - 1), -1
    )


def test_jacobi_and_block_jacobi_beat_unpreconditioned():
    a, blocks, b = block_scaled_system()
    matvec = lambda v: a @ v  # noqa: E731

    plain_iters, plain = gmres_iters(matvec, b)
    jac_iters, jac = gmres_iters(matvec, b, precond=jacobi(jnp.diag(a)))
    blk_iters, blk = gmres_iters(matvec, b, precond=block_jacobi(blocks))

    assert bool(jac.converged) and bool(blk.converged)
    assert jac_iters < plain_iters
    assert blk_iters < plain_iters
    assert blk_iters <= jac_iters  # exact blocks beat their diagonal

    x_ref = np.linalg.solve(np.asarray(a), np.asarray(b))
    for sol in (jac, blk):
        err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
        assert err <= 1e-7


def test_block_jacobi_full_matrix_is_direct_solve():
    a, _, b = block_scaled_system(n_blocks=10, m=3, seed=1)
    matvec = lambda v: a @ v  # noqa: E731
    precond = block_jacobi(a[None])  # one block = the whole matrix
    iters, sol = gmres_iters(matvec, b, precond=precond, rtol=1e-10)
    assert bool(sol.converged)
    assert iters <= 2


def test_coarse_operator_block_thomas_preconditioning():
    n_blocks, m = 25, 4
    n = n_blocks * m
    rng = np.random.default_rng(2)

    # Hard operator: 1-D Poisson (viewed block-tridiagonally) times a
    # well-conditioned coupling term the coarse solve knows nothing about.
    lap = laplacian_1d(n)
    coupling = np.eye(n) + 0.1 * rng.standard_normal((n, n)) / np.sqrt(n)
    dense = jnp.asarray(coupling @ lap)
    matvec = lambda v: dense @ v  # noqa: E731
    b = jnp.asarray(rng.standard_normal(n))

    plain_iters, plain = gmres_iters(matvec, b, max_restarts=10)
    assert (not bool(plain.converged)) or plain_iters > 60

    # Simplified operator: the unperturbed block-tridiagonal part, solved
    # exactly with the block-Thomas factorization.
    lower = np.zeros((n_blocks, m, m))
    diag = np.zeros((n_blocks, m, m))
    upper = np.zeros((n_blocks, m, m))
    for k in range(n_blocks):
        s = slice(k * m, (k + 1) * m)
        diag[k] = lap[s, s]
        if k > 0:
            lower[k] = lap[s, slice((k - 1) * m, k * m)]
        if k < n_blocks - 1:
            upper[k] = lap[s, slice((k + 1) * m, (k + 2) * m)]
    factors = block_thomas_factor(*map(jnp.asarray, (lower, diag, upper)))
    precond = coarse_operator(
        lambda v: block_thomas_solve(factors, v.reshape(n_blocks, m)).reshape(-1)
    )

    pre_iters, pre = gmres_iters(matvec, b, precond=precond)
    assert bool(pre.converged)
    assert pre_iters < 20

    x_ref = np.linalg.solve(np.asarray(dense), np.asarray(b))
    err = np.linalg.norm(np.asarray(pre.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-7


def anisotropic_2d(nx=24, ny=24, cx=1.0, cy=100.0):
    """2-D anisotropic operator (I - cx d_xx - cy d_yy, Dirichlet, unit h).

    Returns the dense matrix (row-major over the (nx, ny) grid) and the
    per-direction line-solve callables built from banded factors.
    """
    n = nx * ny
    diag_val = 1.0 + 2.0 * cx + 2.0 * cy
    dense = np.zeros((n, n))
    for i in range(nx):
        for j in range(ny):
            p = i * ny + j
            dense[p, p] = diag_val
            if i > 0:
                dense[p, p - ny] = -cx
            if i < nx - 1:
                dense[p, p + ny] = -cx
            if j > 0:
                dense[p, p - 1] = -cy
            if j < ny - 1:
                dense[p, p + 1] = -cy

    def tridiag_bands(size, off):
        bands = np.zeros((3, size))
        bands[0, 1:] = off
        bands[1, :] = diag_val
        bands[2, :-1] = off
        return jnp.asarray(bands)

    fac_x = lu_factor_banded(tridiag_bands(nx, -cx), 1, 1)
    fac_y = lu_factor_banded(tridiag_bands(ny, -cy), 1, 1)

    def solve_x_lines(v):
        # Lines along x: one tridiagonal system per y-column, solved as a
        # multi-rhs banded solve on the (nx, ny) reshape.
        return lu_solve_banded(fac_x, v.reshape(nx, ny)).reshape(-1)

    def solve_y_lines(v):
        return lu_solve_banded(fac_y, v.reshape(nx, ny).T).T.reshape(-1)

    return jnp.asarray(dense), solve_x_lines, solve_y_lines


def test_line_smoother_alternating_beats_single_direction():
    dense, solve_x, solve_y = anisotropic_2d()
    matvec = lambda v: dense @ v  # noqa: E731
    rng = np.random.default_rng(3)
    b = jnp.asarray(rng.standard_normal(dense.shape[0]))

    # Single direction: only the weakly coupled x-lines are solved.
    single = line_smoother(matvec, [solve_x], omega=0.9)
    # Alternating directions include the strong y-coupling.
    alternating = line_smoother(matvec, [solve_x, solve_y], omega=0.9)

    single_iters, single_sol = gmres_iters(matvec, b, precond=single)
    alt_iters, alt_sol = gmres_iters(matvec, b, precond=alternating)

    assert bool(alt_sol.converged)
    assert alt_iters <= single_iters
    assert bool(single_sol.converged)  # weaker, but must still converge

    x_ref = np.linalg.solve(np.asarray(dense), np.asarray(b))
    err = np.linalg.norm(np.asarray(alt_sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-7


def test_line_smoother_omega_sequence_and_sweeps():
    dense, solve_x, solve_y = anisotropic_2d(nx=12, ny=12, cy=20.0)
    matvec = lambda v: dense @ v  # noqa: E731
    rng = np.random.default_rng(4)
    b = jnp.asarray(rng.standard_normal(dense.shape[0]))

    precond = line_smoother(
        matvec, [solve_x, solve_y], omega=[1.0, 0.8], sweeps=2
    )
    iters, sol = gmres_iters(matvec, b, precond=precond)
    assert bool(sol.converged)
    assert iters < 20


def test_additive_preconditioner_is_spd_and_pcg_safe():
    first = jnp.diag(jnp.asarray([0.5, 0.25, 0.2]))
    second = jnp.asarray(
        [[0.4, 0.1, 0.0], [0.1, 0.5, 0.1], [0.0, 0.1, 0.3]]
    )
    components = [lambda value: first @ value, lambda value: second @ value]
    residual = jnp.asarray([0.2, -0.4, 0.8])

    mean = additive_preconditioner(components)
    np.testing.assert_allclose(mean(residual), 0.5 * (first + second) @ residual)

    weighted = additive_preconditioner(components, weights=[0.25, 0.75])
    dense = jax.vmap(weighted)(jnp.eye(3)).T
    np.testing.assert_allclose(dense, 0.25 * first + 0.75 * second)
    np.testing.assert_allclose(dense, dense.T)
    assert np.linalg.eigvalsh(np.asarray(dense)).min() > 0.0

    operator = jnp.linalg.inv(dense)
    solution = pcg(
        lambda value: operator @ value,
        residual,
        precond=weighted,
        rtol=1e-12,
        max_steps=4,
    )
    assert bool(solution.converged)
    assert int(solution.iterations) == 1
    np.testing.assert_allclose(solution.x, dense @ residual, rtol=1e-12, atol=1e-12)


def test_additive_preconditioner_supports_pytree_jit_grad_and_placement():
    tree = {"field": jnp.asarray([0.5, -1.0]), "gauge": (jnp.asarray(0.25),)}

    def scaled(scale):
        scale_tree = lambda value: jax.tree_util.tree_map(  # noqa: E731
            lambda leaf: scale * leaf, value
        )
        double_tree = lambda value: jax.tree_util.tree_map(  # noqa: E731
            lambda leaf: 2.0 * leaf, value
        )
        return additive_preconditioner([scale_tree, double_tree])(tree)

    applied = jax.jit(scaled)(jnp.asarray(1.0))
    expected = jax.tree_util.tree_map(lambda leaf: 1.5 * leaf, tree)
    jax.tree_util.tree_map(np.testing.assert_allclose, applied, expected)

    objective = lambda scale: sum(  # noqa: E731
        jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(scaled(scale))
    )
    assert jax.jit(jax.grad(objective))(1.0) == pytest.approx(1.96875)

    device = jax.devices()[0]
    placed = jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf, device), tree)
    preconditioner = additive_preconditioner(
        [
            lambda value: value,
            lambda value: jax.tree_util.tree_map(lambda leaf: 2.0 * leaf, value),
        ]
    )
    leaves = jax.tree_util.tree_leaves(jax.jit(preconditioner)(placed))
    assert all(leaf.devices() == {device} for leaf in leaves)


def poisson_hierarchy():
    """Three-level 1-D Poisson hierarchy on grids of 129, 65 and 33 points.

    Vectors hold all unknowns of tridiag(-1, 2, -1) (Dirichlet values
    beyond both ends); coarse point i sits at fine point 2i, with
    full-weighting restriction, linear-interpolation prolongation, and
    Galerkin coarse operators ``A_{l+1} = R_l A_l P_l``.
    """
    def prolong_dense(nf, nc):
        p = np.zeros((nf, nc))
        p[2 * np.arange(nc), np.arange(nc)] = 1.0
        p[2 * np.arange(nc - 1) + 1, np.arange(nc - 1)] = 0.5
        p[2 * np.arange(nc - 1) + 1, np.arange(1, nc)] = 0.5
        return p

    p_dense = [prolong_dense(129, 65), prolong_dense(65, 33)]
    r_dense = [0.5 * p.T for p in p_dense]  # full weighting

    a0 = laplacian_1d(129) * 130.0**2
    a1 = r_dense[0] @ a0 @ p_dense[0]
    a2 = r_dense[1] @ a1 @ p_dense[1]
    mats = [jnp.asarray(a) for a in (a0, a1, a2)]

    def make_restrict(nc):
        def restrict(r):
            padded = jnp.concatenate([jnp.zeros(1), r, jnp.zeros(1)])
            fine_idx = 2 * jnp.arange(nc) + 1  # positions in padded array
            return 0.25 * (
                padded[fine_idx - 1] + 2.0 * padded[fine_idx] + padded[fine_idx + 1]
            )

        return restrict

    def make_prolong(nf, nc):
        def prolong(e):
            out = jnp.zeros(nf).at[2 * jnp.arange(nc)].set(e)
            mid = 0.5 * (e[:-1] + e[1:])
            return out.at[2 * jnp.arange(nc - 1) + 1].set(mid)

        return prolong

    restricts = [make_restrict(65), make_restrict(33)]
    prolongs = [make_prolong(129, 65), make_prolong(65, 33)]
    coarse_inv = jnp.asarray(np.linalg.inv(a2))
    coarse_solve = lambda b: coarse_inv @ b  # noqa: E731
    matvecs = [lambda v: mats[0] @ v, lambda v: mats[1] @ v]
    diags = [jnp.diag(mats[0]), jnp.diag(mats[1])]
    return mats, matvecs, restricts, prolongs, coarse_solve, diags


def test_p_multigrid_standalone_residual_reduction():
    mats, matvecs, restricts, prolongs, coarse_solve, diags = poisson_hierarchy()

    # Damped-Jacobi smoothers as user callables (two sweeps, omega = 2/3).
    def make_smoother(diag):
        inv = 1.0 / diag

        def smooth(matvec, x, b):
            for _ in range(2):
                x = x + (2.0 / 3.0) * inv * (b - matvec(x))
            return x

        return smooth

    cycle = p_multigrid(
        matvecs, restricts, prolongs, coarse_solve,
        smoothers=[make_smoother(d) for d in diags],
    )

    rng = np.random.default_rng(5)
    b = jnp.asarray(rng.standard_normal(129))
    x = jnp.zeros(129)
    norms = [float(jnp.linalg.norm(b))]
    for _ in range(5):
        x = x + cycle(b - matvecs[0](x))
        norms.append(float(jnp.linalg.norm(b - matvecs[0](x))))

    avg_reduction = (norms[0] / norms[-1]) ** (1.0 / 5.0)
    assert avg_reduction >= 10.0


def test_p_multigrid_as_gmres_preconditioner():
    mats, matvecs, restricts, prolongs, coarse_solve, diags = poisson_hierarchy()
    rng = np.random.default_rng(6)
    b = jnp.asarray(rng.standard_normal(129))
    matvec = matvecs[0]

    plain_iters, plain = gmres_iters(matvec, b, max_restarts=10)
    assert (not bool(plain.converged)) or plain_iters > 60

    # Per-level diagonals: the built-in damped-Jacobi smoother path.
    precond = p_multigrid(
        matvecs, restricts, prolongs, coarse_solve, smoothers=diags, cycles=2
    )
    pre_iters, pre = gmres_iters(matvec, b, precond=precond)
    assert bool(pre.converged)
    assert pre_iters < 15

    x_ref = np.linalg.solve(np.asarray(mats[0]), np.asarray(b))
    err = np.linalg.norm(np.asarray(pre.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-7


def test_galerkin_deflation_is_spd_jittable_and_accelerates_pcg():
    n_fine, n_coarse = 65, 33
    dense = jnp.asarray(laplacian_1d(n_fine))
    transfer = np.zeros((n_fine, n_coarse))
    transfer[2 * np.arange(n_coarse), np.arange(n_coarse)] = 1.0
    for index in range(n_coarse - 1):
        transfer[2 * index + 1, index : index + 2] = 0.5
    transfer = jnp.asarray(transfer)

    matvec = lambda value: dense @ value  # noqa: E731
    prolong = lambda value: transfer @ value  # noqa: E731
    coarse_dense = transfer.T @ dense @ transfer
    coarse_inverse = jnp.linalg.inv(coarse_dense)
    coarse_solve = lambda value: coarse_inverse @ value  # noqa: E731
    smoother = lambda value: (2.0 / 3.0) * value / jnp.diag(dense)  # noqa: E731
    precond = galerkin_deflation(
        matvec,
        smoother,
        prolong,
        coarse_solve,
        jnp.zeros(n_coarse),
    )

    rng = np.random.default_rng(7)
    left = jnp.asarray(rng.standard_normal(n_fine))
    right = jnp.asarray(rng.standard_normal(n_fine))
    assert jnp.vdot(left, precond(right)) == pytest.approx(
        jnp.vdot(precond(left), right), abs=1e-12
    )

    # Materializing this small test operator verifies positive definiteness;
    # production applications remain matrix-free.
    precond_dense = jax.vmap(precond)(jnp.eye(n_fine)).T
    assert np.linalg.eigvalsh(np.asarray(precond_dense)).min() > 0.0
    np.testing.assert_allclose(jax.jit(precond)(right), precond(right), rtol=1e-13)

    rhs = jnp.asarray(rng.standard_normal(n_fine))
    plain = pcg(matvec, rhs, rtol=1e-10, max_steps=100)
    accelerated = pcg(
        matvec, rhs, precond=precond, rtol=1e-10, max_steps=100
    )
    assert bool(plain.converged) and bool(accelerated.converged)
    assert int(accelerated.iterations) < int(plain.iterations) // 3
    np.testing.assert_allclose(
        accelerated.x,
        np.linalg.solve(np.asarray(dense), np.asarray(rhs)),
        rtol=1e-9,
        atol=1e-9,
    )


def kron_system(na=8, nb=6, seed=7):
    rng = np.random.default_rng(seed)
    a = np.eye(na) + 0.5 * rng.standard_normal((na, na)) / np.sqrt(na)
    b = np.eye(nb) + 0.5 * rng.standard_normal((nb, nb)) / np.sqrt(nb)
    return jnp.asarray(a), jnp.asarray(b)


def test_nearest_kronecker_recovers_exact_factors():
    a, b = kron_system()
    m = jnp.kron(a, b)
    a2, b2 = nearest_kronecker(m, 8, 6)
    err = float(jnp.linalg.norm(jnp.kron(a2, b2) - m))
    assert err < 1e-10

    rng = np.random.default_rng(8)
    rhs = jnp.asarray(rng.standard_normal(48))
    precond = kronecker_nkp(lu_factor(a2), lu_factor(b2))
    iters, sol = gmres_iters(lambda v: m @ v, rhs, precond=precond, rtol=1e-10)
    assert bool(sol.converged)
    assert iters <= 2


def test_kronecker_nkp_preconditions_near_kronecker_operator():
    a, b = kron_system(seed=9)
    n = 48
    rng = np.random.default_rng(10)
    m = jnp.kron(a, b) + 0.01 * jnp.asarray(rng.standard_normal((n, n))) / np.sqrt(n)
    a2, b2 = nearest_kronecker(m, 8, 6)
    precond = kronecker_nkp(lu_factor(a2), lu_factor(b2))

    rhs = jnp.asarray(rng.standard_normal(n))
    iters, sol = gmres_iters(lambda v: m @ v, rhs, precond=precond)
    assert bool(sol.converged)
    assert iters < 10

    x_ref = np.linalg.solve(np.asarray(m), np.asarray(rhs))
    err = np.linalg.norm(np.asarray(sol.x) - x_ref) / np.linalg.norm(x_ref)
    assert err <= 1e-7


def test_mixed_precision_preconditioner():
    a, blocks, b = block_scaled_system(seed=11)
    matvec = lambda v: a @ v  # noqa: E731

    full = block_jacobi(blocks)
    low = mixed_precision(full)  # float32 inside

    v = jnp.asarray(np.random.default_rng(0).standard_normal(a.shape[0]))
    assert low(v).dtype == jnp.float64  # cast back up

    full_iters, full_sol = gmres_iters(matvec, b, precond=full)
    low_iters, low_sol = gmres_iters(matvec, b, precond=low)

    assert bool(full_sol.converged) and bool(low_sol.converged)
    assert low_iters <= 2 * full_iters + 5  # at most a small factor
    # Final residual still meets the float64 rtol despite the f32 inside.
    assert float(low_sol.residual_norm) <= 1e-8 * float(jnp.linalg.norm(b))


def test_grad_through_preconditioned_gmres():
    n_blocks, m = 5, 3
    n = n_blocks * m
    rng = np.random.default_rng(12)
    a0 = jnp.asarray(np.eye(n) + 0.2 * rng.standard_normal((n, n)) / np.sqrt(n))
    a1 = jnp.asarray(rng.standard_normal((n, n)) / np.sqrt(n))
    b = jnp.asarray(rng.standard_normal(n))
    blocks = jnp.stack(
        [a0[k * m : (k + 1) * m, k * m : (k + 1) * m] for k in range(n_blocks)]
    )
    precond = block_jacobi(blocks)

    def solver(matvec, rhs):
        return gmres(matvec, rhs, precond=precond, rtol=1e-12, restart=20).x

    def loss(theta):
        matvec = lambda v: a0 @ v + theta * (a1 @ v)  # noqa: E731
        x = linear_solve(matvec, b, solver)
        return jnp.sum(x**2)

    theta0 = 0.3
    grad = float(jax.grad(loss)(theta0))
    eps = 1e-6
    fd = float((loss(theta0 + eps) - loss(theta0 - eps)) / (2 * eps))
    assert np.isclose(grad, fd, rtol=1e-5, atol=1e-8)


def test_builder_validation():
    with pytest.raises(ValueError, match="n_blocks, m, m"):
        block_jacobi(jnp.eye(3))
    with pytest.raises(ValueError, match="at least one"):
        line_smoother(lambda v: v, [])
    with pytest.raises(ValueError, match="at least one"):
        additive_preconditioner([])
    with pytest.raises(ValueError, match="match len"):
        additive_preconditioner([lambda value: value], weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="finite and positive"):
        additive_preconditioner([lambda value: value], weights=[0.0])
    with pytest.raises(ValueError, match="pytree structure"):
        additive_preconditioner([lambda value: (value,)])(jnp.ones(2))
    with pytest.raises(ValueError, match="match len"):
        line_smoother(lambda v: v, [lambda v: v], omega=[0.5, 0.5])
    with pytest.raises(ValueError, match="equal length"):
        p_multigrid(
            [lambda v: v], [], [], lambda v: v, smoothers=[jnp.ones(3)]
        )
    with pytest.raises(ValueError, match="cycles"):
        p_multigrid(
            [lambda v: v],
            [lambda v: v],
            [lambda v: v],
            lambda v: v,
            smoothers=[jnp.ones(3)],
            cycles=0,
        )
    with pytest.raises(ValueError, match="shape"):
        nearest_kronecker(jnp.eye(5), 2, 3)
