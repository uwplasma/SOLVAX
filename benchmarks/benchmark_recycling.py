"""Joint primal+adjoint Krylov recycling along a continuation trajectory.

Simulates the solver workload of a gradient-based optimization loop: a slowly
varying operator ``A(theta_j) = A0 + theta_j dA`` where each step needs the
primal solve ``A x = b`` (the model) and the adjoint solve ``A^T lam = g``
(the gradient). Compares, at identical tolerance:

- cold FGMRES per step (no reuse);
- cold GCROT per step (deflation built, then discarded);
- recycled GCROT carrying one pair per operator *direction* — the primal pair
  across primal solves and a second, independent pair across adjoint solves.

Also records the ``recycle_drift`` diagnostic per step next to the operator
step size, exhibiting the linear drift/step-size relation that governs when
recycling stays effective.

Deterministic and JSON-serializable for the reproducibility package.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from solvax import __version__, gcrot, gmres

jax.config.update("jax_enable_x64", True)


def _family(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = jnp.asarray(np.eye(n) + 0.5 * rng.standard_normal((n, n)) / np.sqrt(n))
    direction = jnp.asarray(rng.standard_normal((n, n)) / np.sqrt(n))
    b = jnp.asarray(rng.standard_normal(n))
    g = jnp.asarray(rng.standard_normal(n))
    return base, direction, b, g


def run_recycling_benchmark(
    *,
    n: int = 200,
    steps: int = 10,
    step_size: float = 2.0e-3,
    m: int = 30,
    k: int = 10,
    rtol: float = 1.0e-10,
) -> dict[str, object]:
    """Return JSON-safe per-step iteration counts and drift diagnostics."""
    base, direction, b, g = _family(n)

    def matvecs(theta):
        matrix = base + theta * direction
        return (lambda v: matrix @ v), (lambda v: matrix.T @ v)

    rows = []
    primal_pair = adjoint_pair = None
    for j in range(steps):
        theta = j * step_size
        matvec, matvec_t = matvecs(theta)

        cold_gmres = gmres(matvec, b, restart=m, rtol=rtol)
        cold_gcrot = gcrot(matvec, b, m=m, k=k, rtol=rtol)
        primal = gcrot(matvec, b, m=m, k=k, rtol=rtol, recycle=primal_pair)
        adjoint = gcrot(matvec_t, g, m=m, k=k, rtol=rtol, recycle=adjoint_pair)
        adjoint_cold = gcrot(matvec_t, g, m=m, k=k, rtol=rtol)
        primal_pair, adjoint_pair = primal.recycle, adjoint.recycle

        rows.append(
            {
                "step": j,
                "cold_gmres_iterations": int(cold_gmres.iterations),
                "cold_gcrot_iterations": int(cold_gcrot.iterations),
                "recycled_primal_iterations": int(primal.iterations),
                "recycled_adjoint_iterations": int(adjoint.iterations),
                "cold_adjoint_iterations": int(adjoint_cold.iterations),
                "primal_drift": float(primal.recycle_drift),
                "adjoint_drift": float(adjoint.recycle_drift),
                "all_converged": bool(
                    cold_gmres.converged & cold_gcrot.converged
                    & primal.converged & adjoint.converged & adjoint_cold.converged
                ),
            }
        )

    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "cold_gmres_iterations", "cold_gcrot_iterations",
            "recycled_primal_iterations", "recycled_adjoint_iterations",
            "cold_adjoint_iterations",
        )
    }
    return {
        "solvax_version": __version__,
        "params": {"n": n, "steps": steps, "step_size": step_size, "m": m, "k": k,
                   "rtol": rtol},
        "per_step": rows,
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = run_recycling_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}  params={result['params']}")
    print(
        f"\n{'step':>4} {'gmres':>6} {'gcrot':>6} {'recyc':>6}"
        f" {'adj_cold':>8} {'adj_recyc':>9} {'drift':>10}"
    )
    for row in result["per_step"]:
        print(
            f"{row['step']:>4} {row['cold_gmres_iterations']:>6}"
            f" {row['cold_gcrot_iterations']:>6} {row['recycled_primal_iterations']:>6}"
            f" {row['cold_adjoint_iterations']:>8} {row['recycled_adjoint_iterations']:>9}"
            f" {row['primal_drift']:>10.2e}"
        )
    totals = result["totals"]
    saved = totals["cold_gmres_iterations"] - totals["recycled_primal_iterations"]
    print(f"\ntotals: {totals}")
    print(f"primal matvecs saved vs cold GMRES: {saved}")


if __name__ == "__main__":
    main()
