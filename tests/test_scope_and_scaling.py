"""Behaviour at the edges of the documented scope.

Each test here corresponds to a scope question that was open as an issue: what
the solvers do under extreme scaling, at sizes larger than the unit tests
exercise, across dtypes, and when a caller reaches for a transform a rule does
not implement. None of these is exotic; they are the first things a user hits
outside the happy path, and the answers were previously unrecorded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import solvax as sx
from solvax.banded import lu_factor_banded, lu_solve_banded

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Tridiagonal behaviour under extreme scaling (issue #61)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0, 1e4, 1e8])
def test_tridiagonal_is_scale_invariant(scale: float) -> None:
    """Scaling the whole system must scale the solution, not change it.

    The pivot guard used to be a fixed ``1e-12``, which is enormous beside a
    system of size ``1e-8`` and negligible beside one of size ``1e8``: the same
    constant was simultaneously too aggressive and too timid. The guard is now
    ``sqrt(eps)`` times the coefficient scale, so this holds across decades.
    """
    n = 24
    rng = np.random.default_rng(0)
    diag = jnp.asarray(4.0 + rng.uniform(0, 1, n)) * scale
    lower = jnp.asarray(-1.0 + rng.uniform(-0.1, 0.1, n)) * scale
    upper = jnp.asarray(-1.0 + rng.uniform(-0.1, 0.1, n)) * scale
    rhs = jnp.asarray(rng.normal(size=n))

    solution = sx.tridiagonal_solve(lower, diag, upper, rhs)
    unscaled = sx.tridiagonal_solve(lower / scale, diag / scale, upper / scale, rhs)
    assert np.allclose(np.asarray(solution) * scale, np.asarray(unscaled), rtol=1e-9)


def test_tridiagonal_records_its_floor_relative_to_the_problem() -> None:
    """A near-singular row must be caught at every scale, not only near one."""
    for scale in (1e-6, 1.0, 1e6):
        n = 12
        diag = jnp.full((n,), 2.0) * scale
        diag = diag.at[5].set(0.0)          # exactly singular row
        lower = jnp.full((n,), -1.0) * scale
        upper = jnp.full((n,), -1.0) * scale
        rhs = jnp.ones((n,))
        solution = sx.tridiagonal_solve(lower, diag, upper, rhs)
        assert jnp.all(jnp.isfinite(solution)), f"non-finite at scale {scale}"


# --------------------------------------------------------------------------
# Banded complex path at realistic sizes (issue #60)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("n", "kl", "ku"), [(64, 2, 2), (256, 3, 1), (512, 1, 4)])
def test_complex_banded_matches_dense_at_scale(n: int, kl: int, ku: int) -> None:
    """The complex path was only covered on small systems.

    A defect that depends on bandwidth or on the number of elimination steps
    would not show up at ``n = 8``. This compares against a dense solve at
    sizes and bandwidths a user would actually reach for.
    """
    rng = np.random.default_rng(n * 31 + kl)
    dense = np.zeros((n, n), dtype=np.complex128)
    for offset in range(-kl, ku + 1):
        length = n - abs(offset)
        values = rng.normal(size=length) + 1j * rng.normal(size=length)
        dense += np.diag(values, offset)
    dense += np.diag(np.full(n, 4.0 * (kl + ku + 1) + 0j))   # keep it dominant

    bands = np.zeros((kl + ku + 1, n), dtype=np.complex128)
    for i in range(n):
        for offset in range(-kl, ku + 1):
            j = i + offset
            if 0 <= j < n:
                bands[ku - offset, j] = dense[i, j]

    rhs = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    got = lu_solve_banded(lu_factor_banded(jnp.asarray(bands), kl, ku), rhs)
    expected = np.linalg.solve(dense, np.asarray(rhs))
    relative = np.linalg.norm(np.asarray(got) - expected) / np.linalg.norm(expected)
    assert relative < 1e-9, f"n={n} kl={kl} ku={ku}: relative error {relative:.2e}"


def test_complex_banded_real_rhs_specialisation_agrees_at_scale() -> None:
    """The real-right-hand-side specialisation must match the general path."""
    n, kl, ku = 256, 2, 2
    rng = np.random.default_rng(7)
    bands = rng.normal(size=(kl + ku + 1, n)) + 1j * rng.normal(size=(kl + ku + 1, n))
    bands[ku] += 4.0 * (kl + ku + 1)
    bands = jnp.asarray(bands)
    factors = lu_factor_banded(bands, kl, ku)

    real_rhs = jnp.asarray(rng.normal(size=n))
    via_real = lu_solve_banded(factors, real_rhs)
    via_complex = lu_solve_banded(factors, real_rhs.astype(jnp.complex128))
    assert jnp.allclose(via_real, via_complex, rtol=1e-11, atol=1e-11)


# --------------------------------------------------------------------------
# dtype promotion (issue #62)
# --------------------------------------------------------------------------
def test_tridiagonal_promotes_like_jax_does() -> None:
    """Mixed inputs promote by the usual rules; the result says which won."""
    n = 16
    f32 = jnp.full((n,), 4.0, dtype=jnp.float32)
    f64 = jnp.full((n,), -1.0, dtype=jnp.float64)
    rhs32 = jnp.ones((n,), dtype=jnp.float32)

    assert sx.tridiagonal_solve(f32, f32, f32, rhs32).dtype == jnp.float32
    # A float64 band promotes the whole solve, exactly as `jnp.result_type` says.
    assert sx.tridiagonal_solve(f64, f64, f64, rhs32).dtype == jnp.float64
    # A complex band promotes a real right-hand side.
    c64 = f32.astype(jnp.complex64)
    assert jnp.issubdtype(sx.tridiagonal_solve(c64, c64, c64, rhs32).dtype, jnp.complexfloating)


def test_fourier_helmholtz_follows_the_caller_precision() -> None:
    """The elliptic solver used to hard-cast to float64 regardless of input.

    That silently upgraded an x64-disabled program, or -- with x64 off -- meant
    nothing at all while claiming otherwise in the signature.
    """
    nx, nz = 8, 12
    for dtype in (jnp.float32, jnp.float64):
        ones = jnp.ones((nx,), dtype=dtype)
        operator = sx.build_fourier_helmholtz_operator(
            dx=ones * 0.1, dz=ones * 0.2, g11=ones, g33=ones, rhs_scale=ones, nz=nz
        )
        assert operator.rhs_scale.dtype == dtype, dtype
        rhs = jnp.ones((nx, nz), dtype=dtype)
        assert jnp.all(jnp.isfinite(sx.solve_fourier_helmholtz(rhs, operator=operator)))


def test_fourier_helmholtz_rejects_complex_geometry() -> None:
    ones = jnp.ones((8,), dtype=jnp.complex128)
    with pytest.raises(TypeError, match="must be real"):
        sx.build_fourier_helmholtz_operator(
            dx=ones, dz=ones, g11=ones, g33=ones, rhs_scale=ones, nz=12
        )


# --------------------------------------------------------------------------
# Forward mode through the windowed rule (issue #58)
# --------------------------------------------------------------------------
def test_forward_mode_error_names_the_alternative() -> None:
    """`custom_vjp` gives a true but useless message; ours says what to do."""
    n_blocks, m, keep, window = 8, 2, 2, 3
    eye = jnp.eye(m)

    def block_fn(params, j):
        return (
            eye * 0.2 * (j > 0),
            eye * (4.0 + params[0] * (1.0 + j)),
            eye * 0.2 * (j < n_blocks - 1),
        )

    rhs = jnp.ones((keep, m))
    with pytest.raises(TypeError) as caught:
        jax.jacfwd(
            lambda q: sx.block_thomas_truncated_fn(
                block_fn, n_blocks, rhs, keep, params=q, adjoint_window=window
            )
        )(jnp.array([0.7]))

    message = str(caught.value)
    assert "block_thomas_checkpointed_fn" in message, "does not name the way forward"
    assert "reverse mode only" in message
