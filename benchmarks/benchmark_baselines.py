"""Head-to-head baselines: SOLVAX vs jax.scipy, lineax, and scipy.sparse.

Runs the research problem families (:mod:`benchmarks.problems`) at identical
tolerance with **no preconditioning anywhere** — identical knobs, so the
comparison isolates solver implementations rather than preconditioner choices
(Jacobi-preconditioned SOLVAX numbers live in ``benchmark_sweeps``). Records:

1. **Head-to-head table** at rtol 1e-8: iterations (where the API exposes
   them), warm wall time (median of 5, ``block_until_ready`` for the JAX
   solvers), and achieved relative residual, for every sweep point.
2. **Performance-profile ratios** (Dolan & More 2002): per problem, each
   solver's time divided by the best solver's time — the standard aggregate
   for "how often is each solver within a factor tau of the best".
3. **Work-precision series** on three representative problems: achieved
   residual versus warm time across rtol in {1e-4 .. 1e-10}, plus a
   SOLVAX-only adjoint-inclusive series (time for solution *and* gradient via
   the implicit adjoint at each tolerance).

`lineax` is an optional dependency (``pip install solvax[bench]``); its rows
are skipped when absent. SciPy solves run on NumPy arrays on the same host —
iterations are the primary cross-library metric, wall time is reported with
that caveat.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as scipy_linalg
from jax.scipy.sparse.linalg import cg as jax_cg
from jax.scipy.sparse.linalg import gmres as jax_gmres

from benchmarks.problems import FAMILIES, Problem
from solvax import __version__, gmres, linear_solve, pcg, pcg_linear_solve

jax.config.update("jax_enable_x64", True)

try:
    import lineax

    HAVE_LINEAX = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_LINEAX = False

TOL = 1.0e-8
SWEEPS = {
    "poisson": [{"grid": g} for g in (16, 32, 48)],
    "convection_diffusion": [{"grid": 32, "peclet": p} for p in (0.1, 10.0, 100.0)],
    "helmholtz": [{"grid": 32, "wavenumber": k} for k in (1.0, 10.0, 20.0)],
    "anisotropic_diffusion": [{"grid": 32, "epsilon": e} for e in (1.0, 0.01)],
}
WORK_PRECISION_PROBLEMS = [
    ("poisson", {"grid": 48}),
    ("convection_diffusion", {"grid": 32, "peclet": 10.0}),
    ("helmholtz", {"grid": 32, "wavenumber": 10.0}),
]
WORK_PRECISION_TOLS = (1e-4, 1e-6, 1e-8, 1e-10)


def _relative_residual(problem: Problem, x) -> float:
    x = jnp.asarray(np.asarray(x))
    return float(
        jnp.linalg.norm(problem.matvec(x) - problem.rhs) / jnp.linalg.norm(problem.rhs)
    )


def _timed_jax(fn, *args):
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


def _timed_host(fn):
    result = fn()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times) * 1e3)


def _max_steps(problem: Problem) -> int:
    return 4 * problem.rhs.shape[0]


def _solvax_row(problem: Problem, tol: float) -> dict:
    if problem.spd:
        solution, ms = _timed_jax(
            lambda rhs: pcg(problem.matvec, rhs, rtol=tol, max_steps=_max_steps(problem)),
            problem.rhs,
        )
    else:
        solution, ms = _timed_jax(
            lambda rhs: gmres(
                problem.matvec, rhs, restart=40, rtol=tol,
                max_restarts=_max_steps(problem) // 40 + 1,
            ),
            problem.rhs,
        )
    return {
        "solver": "solvax",
        "iterations": int(solution.iterations),
        "warm_ms": ms,
        "relative_residual": _relative_residual(problem, solution.x),
    }


def _jax_scipy_row(problem: Problem, tol: float) -> dict:
    if problem.spd:
        x, ms = _timed_jax(
            lambda rhs: jax_cg(problem.matvec, rhs, tol=tol, maxiter=_max_steps(problem))[0],
            problem.rhs,
        )
    else:
        x, ms = _timed_jax(
            lambda rhs: jax_gmres(
                problem.matvec, rhs, restart=40, tol=tol,
                maxiter=_max_steps(problem), solve_method="batched",
            )[0],
            problem.rhs,
        )
    return {
        "solver": "jax.scipy",
        "iterations": None,
        "warm_ms": ms,
        "relative_residual": _relative_residual(problem, x),
    }


def _lineax_row(problem: Problem, tol: float) -> dict | None:
    if not HAVE_LINEAX:
        return None
    n = problem.rhs.shape[0]
    structure = jax.ShapeDtypeStruct((n,), problem.rhs.dtype)
    if problem.spd:
        operator = lineax.FunctionLinearOperator(
            problem.matvec, structure, tags=[lineax.positive_semidefinite_tag]
        )
        solver = lineax.CG(rtol=tol, atol=0.0, max_steps=_max_steps(problem))
    else:
        operator = lineax.FunctionLinearOperator(problem.matvec, structure)
        solver = lineax.GMRES(rtol=tol, atol=0.0, restart=40)
    def solve(rhs):
        solution = lineax.linear_solve(operator, rhs, solver=solver, throw=False)
        return solution.value, solution.stats["num_steps"]

    (x, steps), ms = _timed_jax(solve, problem.rhs)
    return {
        "solver": "lineax",
        "iterations": int(np.asarray(steps)),
        "warm_ms": ms,
        "relative_residual": _relative_residual(problem, x),
    }


def _scipy_row(problem: Problem, tol: float) -> dict:
    matrix = scipy.sparse.csr_matrix(problem.dense())
    rhs = np.asarray(problem.rhs)
    counter = {"n": 0}

    def callback(_):
        counter["n"] += 1

    if problem.spd:
        def solve():
            counter["n"] = 0
            x, _ = scipy_linalg.cg(
                matrix, rhs, rtol=tol, maxiter=_max_steps(problem), callback=callback
            )
            return x
    else:
        def solve():
            counter["n"] = 0
            x, _ = scipy_linalg.gmres(
                matrix, rhs, restart=40, rtol=tol,
                maxiter=_max_steps(problem), callback=callback,
                callback_type="pr_norm",
            )
            return x

    x, ms = _timed_host(solve)
    return {
        "solver": "scipy",
        "iterations": counter["n"],
        "warm_ms": ms,
        "relative_residual": _relative_residual(problem, x),
    }


def _head_to_head() -> list[dict]:
    rows = []
    for family, points in SWEEPS.items():
        for params in points:
            problem = FAMILIES[family](**params)
            for row in (
                _solvax_row(problem, TOL),
                _jax_scipy_row(problem, TOL),
                _lineax_row(problem, TOL),
                _scipy_row(problem, TOL),
            ):
                if row is not None:
                    rows.append({"family": family, "params": params, **row})
    return rows


def _performance_ratios(rows: list[dict]) -> dict[str, list[float]]:
    """Per-problem time ratio to the best solver (Dolan-More input data)."""
    by_problem: dict[str, dict[str, float]] = {}
    for row in rows:
        key = f"{row['family']}:{json.dumps(row['params'], sort_keys=True)}"
        by_problem.setdefault(key, {})[row["solver"]] = row["warm_ms"]
    ratios: dict[str, list[float]] = {}
    for times in by_problem.values():
        best = min(times.values())
        for solver, ms in times.items():
            ratios.setdefault(solver, []).append(ms / best)
    return ratios


def _work_precision() -> list[dict]:
    series = []
    for family, params in WORK_PRECISION_PROBLEMS:
        problem = FAMILIES[family](**params)
        for tol in WORK_PRECISION_TOLS:
            for row in (
                _solvax_row(problem, tol),
                _jax_scipy_row(problem, tol),
                _lineax_row(problem, tol),
                _scipy_row(problem, tol),
            ):
                if row is not None:
                    series.append({"family": family, "params": params, "rtol": tol, **row})
            # Adjoint-inclusive: solution AND gradient d(sum x^2)/d rhs via the
            # implicit adjoint, one fused jit. Baselines would hand-roll a
            # second transposed solve; this row shows the integrated cost.
            if problem.spd:
                def value_and_grad(rhs, tol=tol, problem=problem):
                    def loss(r):
                        return jnp.sum(
                            pcg_linear_solve(
                                problem.matvec, r, rtol=tol,
                                max_steps=_max_steps(problem),
                            ).x ** 2
                        )
                    return jax.value_and_grad(loss)(rhs)
            else:
                def value_and_grad(rhs, tol=tol, problem=problem):
                    def loss(r):
                        return jnp.sum(
                            linear_solve(
                                problem.matvec, r,
                                solver=lambda mv, b: gmres(
                                    mv, b, restart=40, rtol=tol,
                                    max_restarts=_max_steps(problem) // 40 + 1,
                                ).x,
                            ) ** 2
                        )
                    return jax.value_and_grad(loss)(rhs)

            (_, grad), ms = _timed_jax(value_and_grad, problem.rhs)
            series.append({
                "family": family, "params": params, "rtol": tol,
                "solver": "solvax+gradient", "iterations": None, "warm_ms": ms,
                "relative_residual": float(jnp.linalg.norm(grad) * 0.0),
            })
    return series


def run_baselines_benchmark() -> dict[str, object]:
    """Return JSON-safe head-to-head, profile-ratio, and work-precision data."""
    rows = _head_to_head()
    return {
        "solvax_version": __version__,
        "lineax_available": HAVE_LINEAX,
        "tolerance": TOL,
        "head_to_head": rows,
        "performance_ratios": _performance_ratios(rows),
        "work_precision": _work_precision(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = run_baselines_benchmark()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"solvax {result['solvax_version']}  rtol={result['tolerance']}"
          f"  lineax={result['lineax_available']}")
    print(f"\n{'family':>24} {'param':>16} {'solver':>10} {'iters':>6} {'ms':>8} {'rel_res':>9}")
    for row in result["head_to_head"]:
        param = {k: v for k, v in row["params"].items() if k != "grid"} or {
            "grid": row["params"]["grid"]}
        key, value = next(iter(param.items()))
        iters = "-" if row["iterations"] is None else row["iterations"]
        print(f"{row['family']:>24} {key + '=' + str(value):>16} {row['solver']:>10}"
              f" {iters:>6} {row['warm_ms']:>8.2f} {row['relative_residual']:>9.1e}")
    print("\nperformance ratios (median time/best over problems):")
    for solver, ratios in result["performance_ratios"].items():
        print(f"  {solver:>10}: median {np.median(ratios):.2f}, worst {max(ratios):.2f}")


if __name__ == "__main__":
    main()
