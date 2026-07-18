"""Parameter-sweep robustness of SOLVAX solvers on the research problem suite.

For each family in :mod:`benchmarks.problems`, sweeps the hardness parameter
(cell Peclet number, Helmholtz wavenumber, anisotropy ratio, mesh size) and
records iterations-to-tolerance, convergence, achieved relative residual, and
warm wall time for the SOLVAX solver+preconditioner combination appropriate to
the family (Jacobi-preconditioned PCG for SPD, Jacobi-preconditioned FGMRES
otherwise), next to the ``jax.scipy.sparse.linalg`` baseline at identical
tolerance and preconditioner. Baselines report wall time and achieved residual
only — they do not expose iteration counts.

``--verify`` solves every family at small size against a dense reference and
exits nonzero on mismatch; CI smoke-runs this mode so the generators and the
driver can never rot.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg as jax_cg
from jax.scipy.sparse.linalg import gmres as jax_gmres

from benchmarks.problems import FAMILIES, Problem
from solvax import __version__, gmres, jacobi, pcg

jax.config.update("jax_enable_x64", True)

TOL = 1.0e-8
SWEEPS = {
    "poisson": [{"grid": g} for g in (16, 32, 48, 64)],
    "convection_diffusion": [
        {"grid": 32, "peclet": p} for p in (0.1, 1.0, 10.0, 100.0)
    ],
    "helmholtz": [{"grid": 32, "wavenumber": k} for k in (1.0, 5.0, 10.0, 20.0)],
    "anisotropic_diffusion": [
        {"grid": 32, "epsilon": e} for e in (1.0, 0.1, 0.01, 0.001)
    ],
}


def _relative_residual(problem: Problem, x: jax.Array) -> float:
    return float(
        jnp.linalg.norm(problem.matvec(x) - problem.rhs) / jnp.linalg.norm(problem.rhs)
    )


def _timed(fn, *args) -> tuple[jax.Array, float]:
    compiled = jax.jit(fn)
    result = compiled(*args)
    jax.block_until_ready(result)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    return result, float(np.median(times) * 1e3)


def _solve_case(problem: Problem) -> dict[str, object]:
    precond = jacobi(problem.diagonal)
    max_steps = 4 * problem.rhs.shape[0]

    if problem.spd:
        def solvax_solve(rhs):
            return pcg(problem.matvec, rhs, precond=precond, rtol=TOL, max_steps=max_steps)

        def baseline_solve(rhs):
            return jax_cg(problem.matvec, rhs, M=precond, tol=TOL, maxiter=max_steps)[0]
    else:
        def solvax_solve(rhs):
            return gmres(
                problem.matvec, rhs, precond=precond, restart=40,
                rtol=TOL, max_restarts=max_steps // 40 + 1,
            )

        def baseline_solve(rhs):
            return jax_gmres(
                problem.matvec, rhs, M=precond, restart=40,
                tol=TOL, maxiter=max_steps, solve_method="batched",
            )[0]

    solution, solvax_ms = _timed(solvax_solve, problem.rhs)
    baseline_x, baseline_ms = _timed(baseline_solve, problem.rhs)
    return {
        "family": problem.name,
        "params": problem.params,
        "solver": "pcg+jacobi" if problem.spd else "fgmres+jacobi",
        "iterations": int(solution.iterations),
        "converged": bool(solution.converged),
        "relative_residual": _relative_residual(problem, solution.x),
        "warm_ms": solvax_ms,
        "baseline": "jax.scipy cg" if problem.spd else "jax.scipy gmres",
        "baseline_relative_residual": _relative_residual(problem, baseline_x),
        "baseline_warm_ms": baseline_ms,
    }


def run_sweep_benchmark() -> dict[str, object]:
    """Return JSON-safe sweep results for every family."""
    rows = []
    for family, points in SWEEPS.items():
        for params in points:
            rows.append(_solve_case(FAMILIES[family](**params)))
    return {"solvax_version": __version__, "tolerance": TOL, "results": rows}


def verify(grid: int = 12) -> int:
    """Solve every family small against a dense reference; 0 on success."""
    failures = 0
    cases = [
        FAMILIES["poisson"](grid),
        FAMILIES["convection_diffusion"](grid, peclet=10.0),
        FAMILIES["helmholtz"](grid, wavenumber=5.0),
        FAMILIES["anisotropic_diffusion"](grid, epsilon=0.01),
    ]
    for problem in cases:
        expected = np.linalg.solve(problem.dense(), np.asarray(problem.rhs))
        # The matvec must agree with the dense assembly...
        action = np.asarray(problem.matvec(jnp.asarray(expected)))
        matvec_ok = np.allclose(action, np.asarray(problem.rhs), atol=1e-8)
        # ...and the routed SOLVAX solver must reach the dense solution.
        row = _solve_case(problem)
        solve_ok = row["relative_residual"] < 1e-6 and row["converged"]
        status = "ok" if (matvec_ok and solve_ok) else "FAIL"
        print(f"{problem.name:>24} matvec={matvec_ok} solve={solve_ok} {status}")
        failures += 0 if (matvec_ok and solve_ok) else 1
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--verify", action="store_true", help="dense verification mode")
    args = parser.parse_args()
    if args.verify:
        raise SystemExit(verify())
    result = run_sweep_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}  rtol={result['tolerance']}")
    print(
        f"{'family':>24} {'param':>14} {'iters':>6} {'conv':>5}"
        f" {'rel_res':>9} {'ms':>8} {'base_ms':>8}"
    )
    for row in result["results"]:
        param = {k: v for k, v in row["params"].items() if k != "grid"} or {
            "grid": row["params"]["grid"]
        }
        key, value = next(iter(param.items()))
        print(
            f"{row['family']:>24} {key + '=' + str(value):>14} {row['iterations']:>6}"
            f" {str(row['converged']):>5} {row['relative_residual']:>9.1e}"
            f" {row['warm_ms']:>8.2f} {row['baseline_warm_ms']:>8.2f}"
        )


if __name__ == "__main__":
    main()
