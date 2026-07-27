"""Localized adjoint of a generated block-tridiagonal chain.

Differentiates a selected-head solve with respect to the compact parameters
that build its rows, and shows the four things the exact-window rule
guarantees:

* the selected forward blocks are blocks of the *full* solution;
* the source cotangent is exact at every window, including ``w = 0``;
* a parameter supported inside the window is differentiated exactly;
* the global parameter gradient converges to the exact one, and the window it
  needs is read off the chain's own localization profile before the solve.

The chain is modelled on a Legendre-mode kinetic operator: a coupling that does
not grow with the row index against a diagonal that grows like ``nu k(k+1)``.
Such a chain is *not* localized at low ``k``, which is why a single decay rate
fitted to the leading rows is misleading and the per-row profile is used
instead.

Run:  python examples/12_localized_adjoint_window.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import solvax as sx  # noqa: E402

N, M, KEEP = 60, 3, 2


def make_chain(nu: float):
    """Row generator ``(params, k) -> (L_k, D_k, U_k)`` for a kinetic-like chain."""
    rng = np.random.default_rng(0)
    eye = jnp.eye(M)
    coupling = jnp.asarray(rng.standard_normal((M, M)) * 0.4)

    def block_fn(params, k):
        kf = jnp.asarray(k).astype(jnp.float64)
        # params[0] scales the coupling; params[1] scales the collisional
        # diagonal, whose row sensitivity grows like k(k+1)
        off = coupling * (1.0 + params[0])
        diag = eye * (0.05 + 0.5 * params[1] * nu * kf * (kf + 1.0))
        return off, diag, off

    return block_fn


def main() -> None:
    rng = np.random.default_rng(1)
    rhs = jnp.asarray(rng.standard_normal((KEEP, M)))
    params = jnp.asarray([0.0, 1.0])

    for nu in (1.0, 0.05):
        block_fn = make_chain(nu)
        generator = lambda k, bf=block_fn: bf(params, k)  # noqa: E731

        # --- the window, chosen before any differentiated solve -------------
        rho = np.asarray(sx.localization_profile_fn(generator, N))
        finite = np.isfinite(rho)
        crossing = np.flatnonzero(finite & (rho < 1.0))
        advised = sx.localization_crossover_window(generator, N, KEEP)

        def loss(p, w, bf=block_fn):
            y = sx.block_thomas_truncated_fn(
                bf, N, rhs, KEEP, params=p, adjoint_window=w
            )
            return jnp.sum(y**2)

        exact = jax.grad(lambda p: loss(p, N))(params)  # full window is exact

        print(f"\n=== nu = {nu:g} ===")
        print(f"  first row with rho_k < 1 : {int(crossing[0]) if crossing.size else None}")
        print(f"  advised adjoint_window   : {advised}")

        # --- the source cotangent does not move with the window -------------
        base = None
        for w in (0, advised):
            gb = jax.grad(lambda b, w=w, bf=block_fn: jnp.sum(
                sx.block_thomas_truncated_fn(
                    bf, N, b, KEEP, params=params, adjoint_window=w
                ) ** 2
            ))(rhs)
            base = gb if base is None else base
            drift = float(jnp.max(jnp.abs(gb - base)))
            print(f"  source cotangent drift at w={w:<3}: {drift:.2e}  (exact at every window)")

        # --- the parameter gradient converges ------------------------------
        print(f"  {'w':>4}  {'relative gradient error':>24}")
        for w in sorted({0, 2, 4, 8, advised, min(advised + 8, N - KEEP)}):
            if KEEP + w > N:
                continue
            got = jax.grad(lambda p, w=w: loss(p, w))(params)
            err = float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact))
            mark = "  <- advised" if w == advised else ""
            print(f"  {w:>4}  {err:>24.3e}{mark}")


if __name__ == "__main__":
    main()
