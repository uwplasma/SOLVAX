"""Exactness decomposition and weighted tail decay of the localized adjoint.

Produces the record behind the paper's two central claims:

1. **Exactness decomposition.** The source cotangent and every retained row
   cotangent are exact at *every* window, so the only quantity that moves with
   ``w`` is the global parameter gradient. The superseded leading-principal
   closure is measured alongside as an ablation; its error carries an
   interface term inside the window in addition to the omitted tail.
2. **Weighted decay.** The parameter-gradient error follows the weighted tail
   bound, with the doubled exponent ``rho^{2w}`` for bounded row sensitivity
   and ``(K+w)^s rho^{2w}`` when the row sensitivity grows polynomially.

Every reference is an exact dense differentiation of the *same* generator, so
what is measured is the adjoint rule and not the discretization. The realized
error is reported without assuming monotonicity: row contributions can cancel,
so the theorem constrains an envelope, not each adjacent pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from solvax.direct import (  # noqa: E402
    _leading_principal_params_bar,
    block_thomas_truncated_fn,
)

from .problems import FAMILIES, Family, measured_rho  # noqa: E402
from .provenance import provenance  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def _objective(y):
    return jnp.real(jnp.sum(y * jnp.conj(y)))


def _exact_reference(family: Family, rhs):
    """Exact parameter and source gradients by dense reverse mode."""
    m, n, k = family.block_size, family.n_blocks, family.keep

    def loss(params, b):
        a = family.dense(params)
        full = jnp.zeros((n * m,), dtype=a.dtype).at[: k * m].set(b.reshape(-1))
        x = jnp.linalg.solve(a, full).reshape(n, m)
        return _objective(x[:k])

    return jax.grad(loss, argnums=(0, 1))(family.params, rhs)


def _windowed(family: Family, rhs, window: int):
    def loss(params):
        return _objective(
            block_thomas_truncated_fn(
                family.block_fn,
                family.n_blocks,
                rhs,
                family.keep,
                params=params,
                adjoint_window=window,
            )
        )

    return jax.grad(loss)(family.params)


def run_family(family: Family, windows: list[int], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rhs = jnp.asarray(rng.standard_normal((family.keep, family.block_size)))

    ref_p, ref_b = _exact_reference(family, rhs)
    ref_p_norm = float(jnp.linalg.norm(ref_p))

    # cotangent of the objective at the selected head, for the ablation call
    y = block_thomas_truncated_fn(
        family.block_fn,
        family.n_blocks,
        rhs,
        family.keep,
        params=family.params,
        adjoint_window=family.n_blocks,
    )
    ct = 2.0 * y

    rows = []
    for w in windows:
        if family.keep + w > family.n_blocks:
            continue
        grad_p = _windowed(family, rhs, w)
        grad_b = jax.grad(
            lambda b, w=w: _objective(
                block_thomas_truncated_fn(
                    family.block_fn,
                    family.n_blocks,
                    b,
                    family.keep,
                    params=family.params,
                    adjoint_window=w,
                )
            )
        )(rhs)
        closure_p = _leading_principal_params_bar(
            family.block_fn, family.n_blocks, family.params, family.keep, w, rhs, ct
        )
        abs_err = float(jnp.linalg.norm(grad_p - ref_p))
        rows.append(
            {
                "window": int(w),
                "retained_rows": int(min(family.keep + w, family.n_blocks)),
                # exact at every window -> these two must sit at roundoff
                "source_cotangent_abs_error": float(jnp.linalg.norm(grad_b - ref_b)),
                "params_abs_error": abs_err,
                "params_rel_error": abs_err / max(ref_p_norm, 1e-300),
                "leading_principal_abs_error": float(
                    jnp.linalg.norm(closure_p - ref_p)
                ),
            }
        )

    rho = measured_rho(family)
    # fitted slope of log10(error) per window block; the doubled exponent
    # predicts 2*log10(rho) once the geometric regime is reached
    usable = [
        r for r in rows if r["params_abs_error"] > 1e-13 * max(ref_p_norm, 1e-300)
    ]
    fitted = None
    compensated = None
    if len(usable) >= 3:
        w_arr = np.array([r["window"] for r in usable], dtype=float)
        e_arr = np.log10([r["params_abs_error"] for r in usable])
        fitted = float(np.polyfit(w_arr, e_arr, 1)[0])
        # Weighted-theorem compensation: dividing by the polynomial row-weight
        # factor (1 + K + w)^s should recover the pure geometric rate, which is
        # what distinguishes the weighted statement from a bare rho^{2w} claim.
        if family.gamma_exponent:
            weight = np.log10((1.0 + family.keep + w_arr) ** family.gamma_exponent)
            compensated = float(np.polyfit(w_arr, e_arr - weight, 1)[0])

    return {
        "family": family.name,
        "n_blocks": family.n_blocks,
        "block_size": family.block_size,
        "keep": family.keep,
        "parameter_dimension": int(family.params.size),
        "meta": family.meta,
        "rho_theory": family.rho_theory,
        "rho_measured": rho,
        "gamma_exponent": family.gamma_exponent,
        "exact_gradient_norm": ref_p_norm,
        "fitted_log10_slope_per_window": fitted,
        "weight_compensated_log10_slope_per_window": compensated,
        "predicted_log10_slope_per_window": (
            float(2.0 * np.log10(rho)) if np.isfinite(rho) and rho > 0 else None
        ),
        "windows": rows,
    }


def run(quick: bool = False) -> dict:
    windows = list(range(0, 6 if quick else 11))
    names = (
        ["scalar_toeplitz", "polynomial_envelope"]
        if quick
        else list(FAMILIES)
    )
    return {
        "provenance": provenance(__file__, "src/solvax/direct.py"),
        "windows_requested": windows,
        "families": [run_family(FAMILIES[n](), windows) for n in names],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run(quick=args.quick)
    text = json.dumps(result, indent=2)
    if args.out:
        path = Path(args.out)
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing record: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")

    for fam in result["families"]:
        print(f"\n=== {fam['family']}  (rho_measured={fam['rho_measured']:.4f}, "
              f"s={fam['gamma_exponent']:.0f}) ===")
        print(f"{'w':>3} {'src cotangent':>15} {'params err':>13} "
              f"{'closure err':>13} {'gain':>7}")
        for r in fam["windows"]:
            gain = r["leading_principal_abs_error"] / max(r["params_abs_error"], 1e-300)
            print(
                f"{r['window']:>3} {r['source_cotangent_abs_error']:>15.2e} "
                f"{r['params_abs_error']:>13.2e} "
                f"{r['leading_principal_abs_error']:>13.2e} {gain:>6.1f}x"
            )
        if fam["fitted_log10_slope_per_window"] is not None:
            print(
                f"    fitted slope {fam['fitted_log10_slope_per_window']:.3f} "
                f"vs predicted {fam['predicted_log10_slope_per_window']:.3f} "
                "log10/window"
            )
            if fam["weight_compensated_log10_slope_per_window"] is not None:
                print(
                    "    after dividing by (1+K+w)^s: "
                    f"{fam['weight_compensated_log10_slope_per_window']:.3f} "
                    "log10/window"
                )


if __name__ == "__main__":
    main()
