"""Forward and reverse derivatives of the public differentiable functions.

Each case is a function of one flat array on a small well-conditioned problem. The
forward derivative ``J v`` from ``jax.jvp`` is compared with the central difference
``(f(x + h v) - f(x - h v)) / (2 h)``, and the reverse derivative with the forward one
through the transpose identity ``<w, J v> = <J^T w, v>``.

Complex inputs are perturbed along a real direction ``v`` and along ``i v``. JAX
differentiates every function as a real-linear map of (real, imaginary) parts, so the
same two checks hold whether or not the function is holomorphic; the pairing is the
real bilinear form ``<a, b> = Re sum(a b)`` (no conjugate), under which ``jax.vjp`` is
the transpose of ``jax.jvp``. Cases marked holomorphic in the table are also checked
for ``J (i v) = i J v``. A reverse-only function (a custom VJP) is checked by the
identity with the central difference in place of ``J v``, to the difference's tolerance.

Tolerances follow from the step, not from a run. With ``h = eps^(1/3)`` and a unit
direction, the central difference error is at most ``h^2 M3 / 6 + eps M0 / h``, where
``M0`` and ``M3`` bound the output and its third derivative along ``v`` (Nocedal and
Wright, Numerical Optimization, 2006, section 8.1). The problems have condition numbers
below 10, so ``M3 <= 6000 |J v|`` with a wide margin; the bound used is
``1000 h^2 max(|J v|, |fd|) + 10 (eps + r) |y| / h``, where ``r`` is the relative
tolerance a solver is run to (its error divided by the step). In float64 both terms are
below ``2e-8`` relative, so a relative error of ``1e-6`` in a derivative rule is caught.
The transpose identity holds to rounding: ``100 eps + r`` relative to
``|w| |J v| + |J^T w| |v|``.
"""

import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx
from solvax.least_squares import LeastSquaresConfig

jax.config.update("jax_enable_x64", True)

N = 6
SEED = 0  # every case draws from its own generator, seeded by SEED and the case id
# Inputs are standard normal draws clipped to [-1, 1] (real and imaginary parts), so
# |entry| <= sqrt(2). With a diagonal shift of 6 every operator below is strictly
# diagonally dominant for any draw (off-diagonal row sums <= 3.6 against a diagonal
# >= 4.5), with condition number below 10.
SHIFT = 6.0


def _tri(x):
    lower, diag, upper, rhs = jnp.split(x, 4)
    return lower, diag + SHIFT, upper, rhs


def _matrix(x, n=N):
    return x[: n * n].reshape(n, n) / n + SHIFT * jnp.eye(n), x[n * n :]


def _newton(f, x0):
    x = x0
    for _ in range(40):
        x = x - jnp.linalg.solve(jax.jacobian(f, holomorphic=jnp.iscomplexobj(x0))(x), f(x))
    return x


def _banded(x):
    lower, diag, upper, rhs = _tri(x)
    bands = jnp.stack([jnp.roll(upper, 1), diag, lower])  # rows: upper, diagonal, lower
    return sx.lu_solve_banded(sx.lu_factor_banded(bands, 1, 1), rhs)


def _block(x, m=2):
    blocks = x[: 3 * N * m * m].reshape(3, N, m, m) / m
    rhs = x[3 * N * m * m :].reshape(N, m)
    return blocks[0], blocks[1] + SHIFT * jnp.eye(m), blocks[2], rhs


def _block_factored(x):
    lower, diag, upper, rhs = _block(x)
    return sx.block_thomas_solve(sx.block_thomas_factor(lower, diag, upper), rhs)


def _dense_solve(matvec, rhs):
    """Black-box direct solver: assembles the operator it is handed, column by column."""
    eye = jnp.eye(rhs.shape[0], dtype=rhs.dtype)
    return jnp.linalg.solve(jax.vmap(matvec, in_axes=1, out_axes=1)(eye), rhs)


def _root(x):
    # x^3 + 2 x = p: monotone for real p; holomorphic for complex p, Newton from p / 2.
    return sx.root_solve(lambda z: z**3 + 2 * z - x, x / 2, _newton)


def _recycled(x):
    a, b = _matrix(x)
    return sx.recycled_linear_solve(
        lambda p, v: p @ v, a, b, lambda mv, rhs, recycle: (_dense_solve(mv, rhs), recycle)
    )


def _fixed_point(x):
    a, b = _matrix(x)
    m = jnp.eye(N) - a / SHIFT  # entries <= sqrt(2) / 36: the map v -> m v + b contracts
    return sx.affine_fixed_point_gmres(lambda v: m @ v + b, jnp.zeros_like(b), rtol=1e-15).x


def _pcg(x):
    a, b = _matrix(x)
    spd = (a + a.T) / 2
    return sx.pcg_linear_solve(lambda v: spd @ v, b, rtol=1e-15, max_steps=4 * N).x


def _sparse(x):
    from solvax.sparse_direct import CsrPattern

    a, b = _matrix(x)
    if not hasattr(_sparse, "pattern"):
        _sparse.pattern = CsrPattern(np.arange(N + 1) * N, np.tile(np.arange(N), N), (N, N))
    return sx.sparse_solve(_sparse.pattern, a.ravel(), b)


def _exponential(x):
    a, v = _matrix(x)
    a = a - SHIFT * jnp.eye(N)  # ||a|| <= sqrt(2)
    plan = sx.plan_exponential_action(horizon=1.0, norm_bound=2.0, shift=0.0, tolerance=1e-15)
    return sx.exponential_action(lambda u: a @ u, v, plan).value


def _helmholtz(x):
    # metric coefficients and right-hand side are perturbed; exp keeps the metric positive
    g11, g33, rhs = jnp.exp(x[:N] / 4), jnp.exp(x[N : 2 * N] / 4), x[2 * N :].reshape(N, 4)
    half = jnp.full(N, 0.5)
    op = sx.build_fourier_helmholtz_operator(
        dx=half, dz=half, g11=g11, g33=g33, rhs_scale=jnp.ones(N), nz=4
    )
    return sx.solve_fourier_helmholtz(rhs, operator=op)


def _poisson_spacing(x):
    rhs = (x[2:] - x[2:].mean()).reshape(4, 3)
    return sx.solve_periodic_poisson(rhs, spacing=tuple(jnp.exp(x[:2] / 4)))


def _linear(x):
    a, b = _matrix(x)
    return sx.linear_solve(lambda v: a @ v, b, _dense_solve)


def _gmres(x):
    a, b = _matrix(x)
    return sx.gmres(lambda v: a @ v, b, restart=N, max_restarts=3, rtol=1e-15, fixed_work=True).x


def _iteration(x):
    contraction = lambda v: jnp.cos(v) / 2 + x  # noqa: E731  Lipschitz constant 1/2
    return sx.fixed_point_iteration(contraction, x, max_steps=80, fixed_steps=True).x


def _least_squares(x):
    # residual M(p) v - b(p) with M of 2N x 3: twice the first columns of I plus a perturbation
    m = 2 * jnp.eye(2 * N, 3) + x[: 6 * N].reshape(2 * N, 3) / (4 * N)
    # a damping floor near eps: with the default floor the primal stops about 1e-9 short of
    # the least-squares solution, which a central difference cannot resolve
    config = LeastSquaresConfig(
        rtol=1e-15,
        max_steps=30,
        initial_damping=1e-14,
        minimum_damping=1e-15,
        linear_rtol=1e-15,
        linear_max_steps=20,
    )
    return sx.implicit_least_squares(lambda v: m @ v - x[6 * N :], jnp.zeros(3), config=config)


def _alfven(x):
    alpha_v, alpha_b = 1 + jnp.exp(x[:N] / 4), 1 + jnp.exp(x[N : 2 * N] / 4)
    rhs = (x[3 * N : 4 * N] + 1j * x[4 * N : 5 * N], x[5 * N :] + 0.5j)
    return jnp.concatenate(sx.alfven_block(alpha_v, alpha_b, x[2 * N : 3 * N])(rhs))


def _schur(x):
    a, b, c, d = (x[4 * i : 4 * i + 4].reshape(2, 2) / 2 for i in range(4))
    a, d = a + SHIFT * jnp.eye(2), d - SHIFT * jnp.eye(2)
    schur = c @ jnp.linalg.solve(a, b) - d
    apply = sx.schur_complement_precond(
        lambda r: jnp.linalg.solve(a, r),
        lambda y: b @ y,
        lambda u: c @ u,
        lambda r: jnp.linalg.solve(schur, r),
    )
    return jnp.concatenate(apply((x[16:18], x[18:20])))


def _lines(x):
    diag, lower, upper, rhs = (x[12 * i : 12 * i + 12].reshape(4, 3) for i in range(4))
    precond = sx.additive_tridiagonal_line_preconditioner(
        diag + SHIFT, ((0, lower.at[0].set(0.0), upper.at[-1].set(0.0)),)
    )
    return precond(rhs)


def _checkpointed(x):
    body = lambda i, y: y + 0.1 * jnp.sin(x * y + i)  # noqa: E731
    return sx.checkpointed_fori_loop(0, 12, body, x, checkpoint_size=4)


def _case(name, f, n, kinds, r=0.0, holomorphic=False):
    return name, f, n, kinds, r, holomorphic


BLOCK, DENSE = 3 * N * 4 + 2 * N, N * N + N  # input sizes
# name, function, input size, dtypes (f: float64, c: complex128), solver tolerance r,
# holomorphic in the perturbed input. No docstring claims float32 derivatives.
CASES = [
    _case("tridiagonal_solve", lambda x: sx.tridiagonal_solve(*_tri(x)), 4 * N, "fc", 0, True),
    _case("tridiagonal_solve:lax", lambda x: sx.tridiagonal_solve(*_tri(x), method="lax"), 24, "f"),
    _case(
        "cyclic_tridiagonal_solve",
        lambda x: sx.cyclic_tridiagonal_solve(*_tri(x)),
        24,
        "fc",
        0,
        True,
    ),
    _case("lu_solve_banded", _banded, 4 * N, "f"),
    _case("block_thomas", lambda x: sx.block_thomas(*_block(x)), BLOCK, "fc", 0, True),
    _case("block_thomas_factor+solve", _block_factored, BLOCK, "fc", 0, True),
    _case(
        "mixed_precision_block_thomas",
        lambda x: sx.mixed_precision_block_thomas(*_block(x)),
        BLOCK,
        "f",
    ),
    _case("linear_solve", _linear, DENSE, "fc", 0, True),
    _case("recycled_linear_solve", _recycled, DENSE, "f"),
    _case("root_solve", _root, N, "fc", 0, True),
    _case("pcg_linear_solve", _pcg, DENSE, "f", 1e-15),
    _case("affine_fixed_point_gmres", _fixed_point, DENSE, "f", 1e-15),
    _case("gmres:fixed_work", _gmres, DENSE, "fc", 1e-15, True),
    _case("fixed_point_iteration:fixed_steps", _iteration, N, "f"),
    _case("checkpointed_fori_loop", _checkpointed, N, "f"),
    _case("sparse_solve", _sparse, DENSE, "f"),
    _case("exponential_action", _exponential, DENSE, "fc", 1e-15, True),
    _case(
        "solve_periodic_poisson",
        lambda x: sx.solve_periodic_poisson(x - x.mean()),
        8,
        "fc",
        0,
        True,
    ),
    _case("solve_periodic_poisson:spacing", _poisson_spacing, 14, "f"),
    _case(
        "periodic_poisson_eigenvalues",
        lambda x: sx.periodic_poisson_eigenvalues((4, 3), tuple(jnp.exp(x / 4))),
        2,
        "f",
    ),
    _case("solve_fourier_helmholtz", _helmholtz, 6 * N, "f"),
    _case("implicit_least_squares", _least_squares, 8 * N, "f", 1e-15),
    _case("alfven_block", _alfven, 6 * N, "f"),
    _case("schur_complement_precond", _schur, 20, "f"),
    _case("additive_tridiagonal_line_preconditioner", _lines, 48, "f"),
]
# Documented as a custom VJP (reverse mode only): no forward rule exists to compare.
REVERSE_ONLY = {_recycled}
DTYPES = {"f": jnp.float64, "c": jnp.complex128}
# Measured failures of a documented contract. Not strict: the case passes silently once
# the source is fixed.
PARAMS = [
    pytest.param(
        f,
        n,
        DTYPES[k],
        r,
        hol,
        case,
        id=case,
    )
    for name, f, n, kinds, r, hol in CASES
    for k in kinds
    for case in [f"{name}-{jnp.dtype(DTYPES[k]).name}"]
]


def _pair(a, b):
    return float(jnp.real(jnp.sum(a * b)))


def _norm(a):
    return float(jnp.linalg.norm(jnp.ravel(a)))


@pytest.mark.parametrize("f, n, dtype, r, holomorphic, case", PARAMS)
def test_jvp_matches_central_difference_and_vjp_is_its_transpose(f, n, dtype, r, holomorphic, case):
    rng = np.random.default_rng([SEED, zlib.crc32(case.encode())])
    complex_ = jnp.issubdtype(dtype, jnp.complexfloating)
    eps = float(jnp.finfo(dtype).eps)
    h = eps ** (1 / 3)

    def draw(shape, unit=False, cplx=complex_):
        z = np.clip(rng.standard_normal(shape), -1, 1)
        z = z + 1j * np.clip(rng.standard_normal(shape), -1, 1) if cplx else z
        return jnp.asarray(z / np.linalg.norm(z) if unit else z)

    x = draw(n).astype(dtype)
    directions = [draw(n, unit=True, cplx=False).astype(dtype)]
    directions += [1j * directions[0]] if complex_ else []
    y = f(x)
    for v in directions:
        fd = (f(x + h * v) - f(x - h * v)) / (2 * h)
        jv = fd if f in REVERSE_ONLY else jax.jvp(f, (x,), (v,))[1]
        bound = 1000 * h**2 * max(_norm(jv), _norm(fd)) + 10 * (eps + r) * _norm(y) / h
        assert _norm(fd - jv) <= bound, (_norm(fd - jv), bound)
        w = draw(y.shape, cplx=jnp.iscomplexobj(y)).astype(y.dtype)
        (jtw,) = jax.vjp(f, x)[1](w)
        scale = _norm(w) * _norm(jv) + _norm(jtw) * _norm(v)
        slack = bound / _norm(jv) if f in REVERSE_ONLY else 100 * eps + r
        assert abs(_pair(w, jv) - _pair(jtw, v)) <= slack * scale
    if holomorphic and complex_:
        jv, jiv = (jax.jvp(f, (x,), (d,))[1] for d in directions)
        assert _norm(jiv - 1j * jv) <= 100 * eps * _norm(jv)
