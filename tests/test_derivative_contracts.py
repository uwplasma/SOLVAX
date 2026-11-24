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

Tolerances follow from the step, not from a run. With ``h = eps^(1/3)``, the central
difference error is at most ``h^2 M3 / 6 + eps M0 / h`` where ``M0`` and ``M3`` bound
the output and its third derivative along ``v`` (Nocedal and Wright, Numerical
Optimization, 2006, section 8.1); the problems are built with condition
numbers below 10 and unit-sized data so that ``M0 <= 10`` and ``M3 <= 60``, giving
``10 (h^2 + eps / h)`` relative to ``max(1, |J v|)``. A function that solves to a
tolerance ``r`` adds ``r / h`` (its error divided by the step). The transpose identity
holds to rounding: ``100 eps`` (plus ``r``) relative to ``|w| |J v| + |J^T w| |v|``.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx

jax.config.update("jax_enable_x64", True)

N = 6
RNG = np.random.default_rng(7)
DIAG_SHIFT = 4.0  # diagonally dominant: condition number below 10 for N = 6


def _tri(x):
    lower, diag, upper, rhs = jnp.split(x, 4)
    return lower, diag + DIAG_SHIFT, upper, rhs


def _matrix(x, n=N):
    return x[: n * n].reshape(n, n) / n + DIAG_SHIFT * jnp.eye(n), x[n * n :]


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
    return blocks[0], blocks[1] + DIAG_SHIFT * jnp.eye(m), blocks[2], rhs


def _dense_solve(matvec, rhs):
    """Black-box direct solver: assembles the operator it is handed, column by column."""
    eye = jnp.eye(rhs.shape[0], dtype=rhs.dtype)
    return jnp.linalg.solve(jax.vmap(matvec, in_axes=1, out_axes=1)(eye), rhs)


def _root(x):
    # x^3 + 2 x = p (real) or z^3 + 2 z = p (holomorphic, root near p / 2).
    return sx.root_solve(lambda z: z**3 + 2 * z - x, x / 2, _newton)


def _recycled(x):
    a, b = _matrix(x)

    def solver(matvec, rhs, recycle):
        return _dense_solve(matvec, rhs), recycle

    return sx.recycled_linear_solve(lambda p, v: p @ v, a, b, solver)


def _fixed_point(x):
    a, b = _matrix(x)
    m = jnp.eye(N) - a / (2 * DIAG_SHIFT)  # ||m|| < 1/2: the affine map x -> m x + b contracts
    return sx.affine_fixed_point_gmres(lambda v: m @ v + b, jnp.zeros_like(b), rtol=1e-13).x


def _pcg(x):
    a, b = _matrix(x)
    spd = (a + a.T) / 2
    return sx.pcg_linear_solve(lambda v: spd @ v, b, rtol=1e-13).x


def _sparse(x):
    a, b = _matrix(x)
    return sx.sparse_solve(_sparse.pattern, a.ravel(), b)


_sparse.pattern = None


def _exponential(x):
    a, v = _matrix(x)
    a = a - DIAG_SHIFT * jnp.eye(N)  # ||a|| <= 1
    plan = sx.plan_exponential_action(horizon=1.0, norm_bound=float(N), shift=0.0, tolerance=1e-14)
    return sx.exponential_action(lambda u: a @ u, v, plan).value


def _helmholtz(x):
    half, one = jnp.full(N, 0.5), jnp.ones(N)
    op = sx.build_fourier_helmholtz_operator(
        dx=half, dz=half, g11=one, g33=one, rhs_scale=one, nz=4
    )
    return sx.solve_fourier_helmholtz(x.reshape(N, 4), operator=op)


def _linear(x):
    a, b = _matrix(x)
    return sx.linear_solve(lambda v: a @ v, b, _dense_solve)


def _gmres(x):
    a, b = _matrix(x)
    return sx.gmres(lambda v: a @ v, b, restart=N, max_restarts=3, rtol=1e-13, fixed_work=True).x


def _iteration(x):
    contraction = lambda v: jnp.cos(v) / 2 + x  # noqa: E731  Lipschitz constant 1/2
    return sx.fixed_point_iteration(contraction, x, max_steps=80, fixed_steps=True).x


BLOCK, DENSE = 3 * N * 4 + 2 * N, N * N + N  # input sizes
# (name, function, input size, dtypes, solver tolerance r, holomorphic)
CASES = [
    ("tridiagonal_solve", lambda x: sx.tridiagonal_solve(*_tri(x)), 4 * N, "fc", 0, False),
    (
        "tridiagonal_solve:lax",
        lambda x: sx.tridiagonal_solve(*_tri(x), method="lax"),
        4 * N,
        "f",
        0,
        0,
    ),
    (
        "cyclic_tridiagonal_solve",
        lambda x: sx.cyclic_tridiagonal_solve(*_tri(x)),
        4 * N,
        "fc",
        0,
        0,
    ),
    ("lu_solve_banded", _banded, 4 * N, "f", 0, False),
    ("block_thomas", lambda x: sx.block_thomas(*_block(x)), BLOCK, "fc", 0, False),
    (
        "mixed_precision_block_thomas",
        lambda x: sx.mixed_precision_block_thomas(*_block(x)),
        BLOCK,
        "f",
        0,
        False,
    ),
    ("linear_solve", _linear, DENSE, "fc", 0, True),
    ("recycled_linear_solve", _recycled, DENSE, "f", 0, False),
    ("root_solve", _root, N, "fc", 0, True),
    ("pcg_linear_solve", _pcg, DENSE, "f", 1e-13, False),
    ("affine_fixed_point_gmres", _fixed_point, DENSE, "f", 1e-13, False),
    ("gmres:fixed_work", _gmres, DENSE, "fc", 1e-13, False),
    ("fixed_point_iteration:fixed_steps", _iteration, N, "f", 0, False),
    ("sparse_solve", _sparse, DENSE, "f", 0, False),
    ("exponential_action", _exponential, DENSE, "fc", 1e-14, True),
    ("solve_periodic_poisson", lambda x: sx.solve_periodic_poisson(x - x.mean()), 8, "fc", 0, True),
    (
        "periodic_poisson_eigenvalues",
        lambda x: sx.periodic_poisson_eigenvalues((4, 3), tuple(1 + x)),
        2,
        "f",
        0,
        False,
    ),
    ("solve_fourier_helmholtz", _helmholtz, 4 * N, "f", 0, False),
]
# Documented as a custom VJP (reverse mode only): no forward rule exists to compare.
REVERSE_ONLY = {_recycled}
DTYPES = {"f": jnp.float64, "c": jnp.complex128, "s": jnp.float32}
# Measured failures of a documented contract, kept visible until the source is fixed.
DEFECTS = {
    "root_solve-complex128": "default tangent_solve calls jax.jacobian without holomorphic=True "
    "and raises TypeError on a complex root (implicit.py, _default_tangent_solve)",
}
PARAMS = [
    pytest.param(
        f,
        n,
        DTYPES[k],
        r,
        hol,
        id=(case := f"{name}-{jnp.dtype(DTYPES[k]).name}"),
        marks=[pytest.mark.xfail(strict=True, reason=DEFECTS[case])] if case in DEFECTS else [],
    )
    for name, f, n, kinds, r, hol in CASES
    for k in kinds
]


def _pair(a, b):
    return float(jnp.real(jnp.sum(a * b)))


def _norm(a):
    return float(jnp.linalg.norm(jnp.ravel(a)))


@pytest.mark.parametrize("f, n, dtype, r, holomorphic", PARAMS)
def test_jvp_matches_central_difference_and_vjp_is_its_transpose(f, n, dtype, r, holomorphic):
    if f is _sparse:
        pytest.importorskip("scipy")
        from solvax.sparse_direct import CsrPattern

        _sparse.pattern = CsrPattern(np.arange(N + 1) * N, np.tile(np.arange(N), N), (N, N))
    start = time.perf_counter()
    complex_ = jnp.issubdtype(dtype, jnp.complexfloating)
    real = jnp.finfo(dtype).dtype
    eps = float(jnp.finfo(real).eps)
    h = eps ** (1 / 3)

    def draw():
        z = RNG.standard_normal(n) + (1j * RNG.standard_normal(n) if complex_ else 0)
        return jnp.asarray(z / np.sqrt(n), dtype)

    x = draw()
    v = draw().real.astype(dtype)
    directions = [v, 1j * draw().real.astype(dtype)] if complex_ else [v]
    tol = 10 * (h**2 + eps / h) + 10 * r / h
    for v in directions:
        fd = (f(x + h * v) - f(x - h * v)) / (2 * h)
        if f in REVERSE_ONLY:  # the transpose identity is checked against the difference itself
            y, jv = f(x), fd
        else:
            y, jv = jax.jvp(f, (x,), (v,))
            assert _norm(fd - jv) <= tol * max(1.0, _norm(jv)), (_norm(fd - jv), tol)

        w = jnp.asarray(
            RNG.standard_normal(y.shape)
            + (1j * RNG.standard_normal(y.shape) if jnp.iscomplexobj(y) else 0),
            y.dtype,
        )
        (jtw,) = jax.vjp(f, x)[1](w)
        scale = _norm(w) * _norm(jv) + _norm(jtw) * _norm(v)
        bound = tol if f in REVERSE_ONLY else 100 * eps + r
        assert abs(_pair(w, jv) - _pair(jtw, v)) <= bound * scale
    if complex_ and holomorphic:
        v = directions[0]
        _, jv = jax.jvp(f, (x,), (v,))
        _, jiv = jax.jvp(f, (x,), (1j * v,))
        assert _norm(jiv - 1j * jv) <= 100 * eps * _norm(jv)
    assert time.perf_counter() - start < 30
