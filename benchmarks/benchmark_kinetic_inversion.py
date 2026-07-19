"""Transport inversion through the bounded-memory truncated adjoint.

The application the truncated block solve exists for: a spectral kinetic
system whose block index is a velocity-space mode, forced and observed only in
the lowest moments but coupled upward through streaming, so the whole ladder
participates in every solve. The inverse problem recovers a collisionality
profile ``nu_k = nu0 (1 + a t)``, ``t = k / n_blocks``, from the observed
moments under four independent forcings, by damped Newton on the misfit — the
gradient *and* the 2x2 Hessian both flow through
``block_thomas_truncated(adjoint_window=w)``.

Three records come out:

1. **Inversion**: quadratic convergence of the Newton iteration to the exact
   profile, and a finite-difference validation of the adjoint gradient.
2. **Identifiability**: the eigenvalues of the extended (nu0, a, b) misfit
   Hessian, showing the quadratic coefficient is practically unidentifiable
   from truncated low-moment observations — its sensitivity decays with mode
   number like the block-inverse decay bound itself. The machinery that makes
   the inversion cheap also diagnoses what the data can and cannot determine.
3. **Memory**: compiled reverse-mode scratch versus the mode count for three
   paths — the naive tape, the array-band bounded adjoint (removes the solve
   tape; the O(N m^2) band arrays remain), and the generated-block bounded
   adjoint (``block_thomas_truncated_fn(params=...)``), which materializes no
   band arrays in either direction and is flat in the mode count.

Deterministic and JSON-serializable for the reproducibility package.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from solvax import __version__, block_thomas_truncated, block_thomas_truncated_fn

jax.config.update("jax_enable_x64", True)

M = 36  # block size (spatial points per mode)
KEEP = 3  # observed lowest moments
N_RHS = 4  # independent forcings
WINDOW = 10  # adjoint window (block-dominant regime: band-grad error ~ rho^2w)


def _structure(m: int):
    rng = np.random.default_rng(3)
    eye = np.eye(m)
    neighbor = np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    collision = jnp.asarray(0.1 * rng.standard_normal((m, m)) + eye)
    streaming = jnp.asarray(0.25 * m * neighbor)
    rhs_low = jnp.asarray(rng.standard_normal((KEEP, m, N_RHS)))
    return collision, streaming, rhs_low


COLLISION, STREAMING, RHS_LOW = _structure(M)


def _observe(profile: jax.Array, n_blocks: int) -> jax.Array:
    """Observed low moments for profile coefficients ``(nu0, a[, b])``."""
    t = jnp.arange(n_blocks) / n_blocks
    poly = jnp.ones_like(t) + profile[1] * t
    if profile.shape[0] > 2:
        poly = poly + profile[2] * t**2
    nu = profile[0] * poly
    diag = 3.0 * M * (nu[:, None, None] * COLLISION[None])
    lower = jnp.broadcast_to(STREAMING, (n_blocks, M, M))
    upper = jnp.broadcast_to(-STREAMING, (n_blocks, M, M))
    return block_thomas_truncated(
        lower, diag, upper, RHS_LOW, KEEP, adjoint_window=WINDOW
    )


def run_kinetic_inversion_benchmark(
    *,
    n_blocks: int = 96,
    newton_steps: int = 8,
    memory_block_counts: tuple[int, ...] = (32, 64, 128, 256, 512),
) -> dict[str, object]:
    """Return JSON-safe inversion, identifiability, and memory records."""
    true_params = jnp.asarray([1.0, 0.6])
    data = _observe(true_params, n_blocks)

    def loss(params):
        return jnp.sum((_observe(params, n_blocks) - data) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    hessian = jax.jit(jax.hessian(loss))

    params = jnp.asarray([0.6, 0.1])
    trajectory = []
    for _ in range(newton_steps):
        value, grad = value_and_grad(params)
        trajectory.append(float(value))
        step = jnp.linalg.solve(hessian(params) + 1e-10 * jnp.eye(2), grad)
        params = params - step

    probe = jnp.asarray([0.6, 0.1])
    _, grad0 = value_and_grad(probe)
    eps = 1e-6
    fd0 = (
        loss(probe + jnp.array([eps, 0.0])) - loss(probe - jnp.array([eps, 0.0]))
    ) / (2 * eps)
    gradient_fd_match = bool(np.isclose(float(grad0[0]), float(fd0), rtol=1e-5))

    # Identifiability of the extended profile (nu0, a, b): the misfit Hessian
    # spectrum shows what truncated low-moment data can determine.
    extended = jnp.asarray([1.0, 0.6, -0.3])
    data3 = _observe(extended, n_blocks)

    def loss3(params):
        return jnp.sum((_observe(params, n_blocks) - data3) ** 2)

    spectrum = np.linalg.eigvalsh(np.asarray(jax.hessian(loss3)(extended)))

    def temp_bytes(count: int, window: int | None) -> int:
        def objective(params_):
            t = jnp.arange(count) / count
            nu = params_[0] * (1.0 + params_[1] * t)
            diag = 3.0 * M * (nu[:, None, None] * COLLISION[None])
            lower = jnp.broadcast_to(STREAMING, (count, M, M))
            upper = jnp.broadcast_to(-STREAMING, (count, M, M))
            solution = block_thomas_truncated(
                lower, diag, upper, RHS_LOW, KEEP, adjoint_window=window
            )
            return jnp.sum(solution**2)

        compiled = jax.jit(jax.grad(objective)).lower(true_params).compile()
        return int(compiled.memory_analysis().temp_size_in_bytes)

    def temp_bytes_generated(count: int) -> int:
        def block_fn(params_, index):
            nu = params_[0] * (1.0 + params_[1] * index / count)
            return STREAMING, 3.0 * M * nu * COLLISION, -STREAMING

        def objective(params_):
            solution = block_thomas_truncated_fn(
                block_fn, count, RHS_LOW, KEEP,
                params=params_, adjoint_window=WINDOW,
            )
            return jnp.sum(solution**2)

        compiled = jax.jit(jax.grad(objective)).lower(true_params).compile()
        return int(compiled.memory_analysis().temp_size_in_bytes)

    memory = [
        {
            "n_blocks": count,
            "naive_temp_bytes": temp_bytes(count, None),
            "bounded_temp_bytes": temp_bytes(count, WINDOW),
            "generated_temp_bytes": temp_bytes_generated(count),
        }
        for count in memory_block_counts
    ]

    return {
        "solvax_version": __version__,
        "params": {
            "n_blocks": n_blocks, "m": M, "keep": KEEP, "n_rhs": N_RHS,
            "window": WINDOW, "newton_steps": newton_steps,
        },
        "true_params": [float(v) for v in true_params],
        "recovered_params": [float(v) for v in params],
        "loss_trajectory": trajectory,
        "gradient_fd_match": gradient_fd_match,
        "extended_hessian_eigenvalues": [float(v) for v in spectrum],
        "memory_scaling": memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = run_kinetic_inversion_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}  params={result['params']}")
    losses = result["loss_trajectory"]
    print(f"newton loss trajectory: {' -> '.join(f'{v:.2e}' for v in losses)}")
    print(f"true params:      {result['true_params']}")
    print(f"recovered params: {[round(v, 8) for v in result['recovered_params']]}")
    print(f"adjoint gradient matches finite differences: {result['gradient_fd_match']}")
    eigs = result["extended_hessian_eigenvalues"]
    print(f"(nu0, a, b) misfit Hessian eigenvalues: {[f'{v:.2e}' for v in eigs]}")
    print("  -> the quadratic coefficient is unidentifiable from truncated moments")
    print(f"\n{'N':>6} {'naive_KiB':>12} {'bounded_KiB':>12} {'generated_KiB':>14}")
    for row in result["memory_scaling"]:
        naive, bounded = row["naive_temp_bytes"], row["bounded_temp_bytes"]
        generated = row["generated_temp_bytes"]
        print(f"{row['n_blocks']:>6} {naive / 1024:>12.1f} {bounded / 1024:>12.1f}"
              f" {generated / 1024:>14.1f}")


if __name__ == "__main__":
    main()
