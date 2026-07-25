"""Generated block-tridiagonal families for the localized-adjoint study.

Each family exposes the same interface -- a row generator ``block_fn(params, j)``
plus a dense assembler for exact references -- and each one isolates a distinct
ingredient of the weighted tail bound

    ||grad J - g_W|| <= C_A^2 B_rho Q_rho (1+rho+rho^2)
                        sum_{j>=W} gamma_j rho^{2(j-K)+1}.

* :func:`spd_block_toeplitz` and :func:`scalar_toeplitz` control the inverse
  localization rate ``rho`` directly, and the scalar case has a closed-form
  ``rho`` so the doubled exponent can be checked against theory rather than a
  fit.
* :func:`dominant_chain` controls strict block dominance
  ``delta = max_j(||D_j^-1 L_j|| + ||D_j^-1 U_j||) < 1``, whose Neumann series
  gives an independent estimate of ``rho``.
* :func:`nonnormal_shift` fixes the spectrum while varying the localization, so
  an eigenvalue-only argument is visibly insufficient.
* :func:`polynomial_envelope` grows the row sensitivity like ``(1+j)^s``, the
  case that matters for kinetic operators whose collisional derivative scales
  with mode number.
* :func:`near_breakdown` weakens dominance toward the failure edge.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Family(NamedTuple):
    """A generated chain plus the references a benchmark needs.

    Attributes:
        name: family identifier recorded with every measurement.
        block_fn: ``(params, j) -> (L_j, D_j, U_j)``.
        dense: ``params -> (N m, N m)`` assembly of the same generator.
        params: base parameter vector.
        n_blocks: chain length ``N``.
        block_size: block size ``m``.
        keep: source/output support ``K``.
        rho_theory: analytic localization rate when known, else ``None``.
        gamma_exponent: ``s`` in the row envelope ``gamma_j ~ (1+j)^s``.
        meta: family-specific settings recorded verbatim.
    """

    name: str
    block_fn: object
    dense: object
    params: jax.Array
    n_blocks: int
    block_size: int
    keep: int
    rho_theory: float | None
    gamma_exponent: float
    meta: dict


def _assembler(block_fn, n_blocks: int, block_size: int):
    def dense(params):
        idx = jnp.arange(n_blocks)
        lo, di, up = jax.vmap(lambda j: block_fn(params, j))(idx)
        m = block_size
        a = jnp.zeros((n_blocks * m, n_blocks * m), dtype=di.dtype)
        for j in range(n_blocks):
            a = a.at[j * m : (j + 1) * m, j * m : (j + 1) * m].add(di[j])
            if j + 1 < n_blocks:
                a = a.at[j * m : (j + 1) * m, (j + 1) * m : (j + 2) * m].add(up[j])
                a = a.at[(j + 1) * m : (j + 2) * m, j * m : (j + 1) * m].add(lo[j + 1])
        return a

    return dense


def scalar_toeplitz(n_blocks: int = 40, keep: int = 1, offdiag: float = 0.3) -> Family:
    """Scalar SPD Toeplitz chain ``tridiag(-c, 1, -c)`` with closed-form decay.

    For ``|2c| < 1`` the inverse entries behave like ``r^{|i-j|}`` with
    ``r = (1 - sqrt(1 - 4c^2)) / (2c)``, so the predicted parameter-tail rate
    ``r^{2w}`` can be compared against theory without fitting. A single shared
    diagonal parameter makes the omitted tail genuinely nonzero.
    """
    c = float(offdiag)
    r = (1.0 - np.sqrt(max(1.0 - 4.0 * c * c, 0.0))) / (2.0 * c)

    def block_fn(params, j):
        del j
        one = jnp.ones((1, 1), dtype=jnp.float64)
        return -c * one, one * (1.0 + params[0]), -c * one

    return Family(
        name="scalar_toeplitz",
        block_fn=block_fn,
        dense=_assembler(block_fn, n_blocks, 1),
        params=jnp.asarray([0.05]),
        n_blocks=n_blocks,
        block_size=1,
        keep=keep,
        rho_theory=r,
        gamma_exponent=0.0,
        meta={"offdiag": c},
    )


def spd_block_toeplitz(
    n_blocks: int = 24, block_size: int = 3, keep: int = 2, coupling: float = 0.25
) -> Family:
    """SPD block Toeplitz chain with a shared diagonal scale parameter."""
    m = block_size
    eye = jnp.eye(m, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    off = jnp.asarray(rng.standard_normal((m, m)) * coupling)
    off = 0.5 * (off + off.T)

    def block_fn(params, j):
        del j
        return off, eye * (2.0 + params[0]), off

    return Family(
        name="spd_block_toeplitz",
        block_fn=block_fn,
        dense=_assembler(block_fn, n_blocks, m),
        params=jnp.asarray([0.1]),
        n_blocks=n_blocks,
        block_size=m,
        keep=keep,
        rho_theory=None,
        gamma_exponent=0.0,
        meta={"coupling": coupling},
    )


def dominant_chain(
    n_blocks: int = 24,
    block_size: int = 3,
    keep: int = 2,
    delta: float = 0.5,
    seed: int = 0,
) -> Family:
    """Nonsymmetric chain with controlled strict block dominance ``delta < 1``.

    Off-diagonal blocks are scaled so that
    ``||D^-1 L|| + ||D^-1 U|| ~ delta``; the Neumann series then predicts
    ``rho ~ delta``, giving an independent estimate of the localization rate.
    """
    m = block_size
    rng = np.random.default_rng(seed)
    lo = rng.standard_normal((m, m))
    up = rng.standard_normal((m, m))
    lo = lo / np.linalg.norm(lo, 2) * (delta / 2.0)
    up = up / np.linalg.norm(up, 2) * (delta / 2.0)
    lo_j, up_j = jnp.asarray(lo), jnp.asarray(up)
    eye = jnp.eye(m, dtype=jnp.float64)

    def block_fn(params, j):
        del j
        return lo_j, eye * (1.0 + params[0]), up_j

    return Family(
        name="dominant_chain",
        block_fn=block_fn,
        dense=_assembler(block_fn, n_blocks, m),
        params=jnp.asarray([0.05]),
        n_blocks=n_blocks,
        block_size=m,
        keep=keep,
        rho_theory=float(delta),
        gamma_exponent=0.0,
        meta={"delta": delta, "seed": seed},
    )


def nonnormal_shift(n_blocks: int = 30, keep: int = 1, a: float = 0.7) -> Family:
    """``A = I - a S`` with the upper shift ``S``: spectrum fixed at one.

    Every eigenvalue equals one for any ``a``, yet the inverse entries behave
    like ``a^{|i-j|}``: decaying for ``|a| < 1``, non-decaying at ``|a| = 1``,
    growing beyond. This is the counterexample to an eigenvalue-gap-only claim.
    """

    def block_fn(params, j):
        del j
        one = jnp.ones((1, 1), dtype=jnp.float64)
        return jnp.zeros((1, 1), dtype=jnp.float64), one * (1.0 + params[0]), -a * one

    return Family(
        name="nonnormal_shift",
        block_fn=block_fn,
        dense=_assembler(block_fn, n_blocks, 1),
        params=jnp.asarray([0.0]),
        n_blocks=n_blocks,
        block_size=1,
        keep=keep,
        rho_theory=abs(float(a)),
        gamma_exponent=0.0,
        meta={"a": a, "eigenvalues": "all equal to 1 + params[0]"},
    )


def polynomial_envelope(
    n_blocks: int = 30, keep: int = 2, offdiag: float = 0.3, exponent: float = 2.0
) -> Family:
    """Row sensitivity growing like ``(1+j)^s`` at fixed localization.

    The shared parameter multiplies the diagonal by ``(1+j)^s``, mimicking a
    collisional derivative that grows with mode number, so the tail is governed
    by ``(K+w)^s rho^{2w}`` rather than by ``rho^{2w}`` alone.
    """
    c = float(offdiag)
    r = (1.0 - np.sqrt(max(1.0 - 4.0 * c * c, 0.0))) / (2.0 * c)

    def block_fn(params, j):
        jf = jnp.asarray(j).astype(jnp.float64)
        weight = (1.0 + jf) ** exponent
        one = jnp.ones((1, 1), dtype=jnp.float64)
        return -c * one, one * (1.0 + params[0] * weight), -c * one

    return Family(
        name="polynomial_envelope",
        block_fn=block_fn,
        dense=_assembler(block_fn, n_blocks, 1),
        params=jnp.asarray([1e-3]),
        n_blocks=n_blocks,
        block_size=1,
        keep=keep,
        rho_theory=r,
        gamma_exponent=float(exponent),
        meta={"offdiag": c, "exponent": exponent},
    )


def near_breakdown(
    n_blocks: int = 24, block_size: int = 3, keep: int = 2, delta: float = 0.95
) -> Family:
    """Weakly dominant chain approaching the localization/stability edge."""
    family = dominant_chain(
        n_blocks=n_blocks, block_size=block_size, keep=keep, delta=delta, seed=7
    )
    return family._replace(name="near_breakdown")


FAMILIES = {
    "scalar_toeplitz": scalar_toeplitz,
    "spd_block_toeplitz": spd_block_toeplitz,
    "dominant_chain": dominant_chain,
    "nonnormal_shift": nonnormal_shift,
    "polynomial_envelope": polynomial_envelope,
    "near_breakdown": near_breakdown,
}


def _fit_decay(mags: np.ndarray) -> float:
    """Geometric rate fitted to a block-magnitude profile, ignoring zeros."""
    finite = mags[np.isfinite(mags)]
    if finite.size == 0 or finite.max() <= 0.0:
        return float("nan")
    good = mags > finite.max() * 1e-14
    idx = np.arange(mags.size)[good]
    if idx.size < 3:
        return float("nan")
    slope = np.polyfit(idx, np.log(mags[good]), 1)[0]
    return float(np.exp(slope))


def measured_rho(family: Family) -> float:
    """Localization rate fitted from the dense inverse of the assembled chain.

    The theorem's assumption bounds ``||(A^-1)_{ji}||`` by ``C_A rho^{|j-i|}``,
    a two-sided statement, so the rate reported here is the *slower* of the
    decay along the first block column and the first block row. Measuring only
    one direction is not enough: for the nonnormal shift family ``A = I - aS``
    the inverse is upper triangular, so its first block column has no decay to
    fit at all while its first block row decays like ``a^j``.
    """
    a = np.asarray(family.dense(family.params))
    m, n = family.block_size, family.n_blocks
    inv = np.linalg.inv(a)
    col = np.array([np.linalg.norm(inv[j * m : (j + 1) * m, 0:m], 2) for j in range(n)])
    row = np.array([np.linalg.norm(inv[0:m, j * m : (j + 1) * m], 2) for j in range(n)])
    rates = [r for r in (_fit_decay(col), _fit_decay(row)) if np.isfinite(r)]
    return max(rates) if rates else float("nan")
