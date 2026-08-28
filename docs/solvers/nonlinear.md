# Globalized nonlinear solves and continuation

`pseudo_transient_continuation` is the nonlinear path for a square residual
$F(x)=0$ when an undamped Newton step is not globally safe. It retains the
matrix-free JVP and FGMRES structure of `newton_krylov`, but solves

$$
\left(\frac{M}{\Delta\tau_k}+J_k\right)\delta_k=-F_k
$$

and adapts the pseudo-time step. For small $\Delta\tau$, the positive metric
$M$ regularizes a difficult or poorly initialized problem. As the true
residual falls, switched evolution relaxation grows $\Delta\tau$ and the
method becomes inexact Newton {cite}`kelley1998`.

## A bounded solve

```python
import jax.numpy as jnp
import solvax as sx

def residual(x):
    return diffusion(x) + x**3 - forcing

config = sx.PseudoTransientConfig(
    initial_dt=1e-2,
    max_dt=1e10,
    rtol=1e-9,
)

solution = sx.pseudo_transient_continuation(
    residual,
    initial,
    mass=mass_action,
    precond=lambda rhs, dt: shifted_preconditioner(rhs, dt),
    admissible=lambda x: jnp.all(jnp.isfinite(x)) & (minimum_jacobian(x) > 0),
    config=config,
)
```

`admissible` is a hard predicate. A candidate that violates bounds,
nestedness, a Jacobian floor, or another application invariant is backtracked
and never becomes the iterate. A valid step must also decrease the **true**
nonlinear residual by the configured Armijo fraction. No condition is turned
into a penalty weight.

The result reports accepted and rejected steps, linear failures, nonlinear
residual evaluations, total Krylov work, the final pseudo-time and forcing
term, and fixed-size histories of residual, pseudo-time, forcing, step length,
acceptance, and Krylov iterations. Inspect `converged` and `linear_converged`;
neither is inferred from a stale model residual.

## Inexact Newton forcing

Each correction uses a safeguarded Eisenstat--Walker tolerance
{cite}`eisenstat1996`:

$$
\eta_{k+1}=\operatorname{clip}\left(
\max\left[\gamma\left(\frac{\|F_{k+1}\|}{\|F_k\|}\right)^p,
          \gamma\eta_k^p\right],\eta_{\min},\eta_{\max}\right).
$$

Early linear systems are not solved to the final nonlinear tolerance. The
history records the forcing term and Krylov iterations, while
`residual_evaluations` makes the nonlinear work explicit. On the pinned
32-variable nonsymmetric test, adaptive forcing reaches the same root with
1170 Krylov iterations versus 1352 for a fixed `1e-6` inner tolerance.

## Adaptive residual homotopy

`adaptive_continuation` follows roots of `residual(x, alpha)` with host-side
stage orchestration and JIT-able pseudo-transient stages:

```python
branch = sx.adaptive_continuation(
    lambda x, alpha: low_residual(x)
    + alpha * (strong_residual(x) - low_residual(x)),
    low_order_root,
    continuation_config=sx.ContinuationConfig(initial_step=0.1),
    accept_stage=validation_gate,
)
```

Fast stages grow the parameter step; nonlinear failure or a validation-gate
rejection leaves `(x, alpha)` unchanged and shrinks it. Every attempt is a
`ContinuationStep` containing the old/new parameter, acceptance, nonlinear and
linear work, final residual, and minimum pseudo-time. If the step crosses
`min_step`, the driver returns an unconverged result rather than changing
solver families or silently accepting a branch jump.

This orchestration is deliberately host-side because `accept_stage` may read a
separate de-aliased certificate or other application diagnostics. The stage
solve itself remains compatible with `jax.jit`.

## Folds and pseudo-arclength

Near a fold, parameter continuation can become singular. The bordered helper
augments the physical residual with

$$
\langle t_x,x-x_{\mathrm{pred}}\rangle
+t_\alpha(\alpha-\alpha_{\mathrm{pred}})=0.
$$

`pseudo_arclength_residual` exposes this square residual for any nonlinear
solver. `pseudo_arclength_corrector` applies the pseudo-transient solver when
the chosen tangent orientation gives a stable pseudo-time evolution. Tangent
sign is mathematically arbitrary but matters to that evolution; reverse it if
the bordered Jacobian has negative-real modes. Applications with an explicit
bordered Schur preconditioner may pass the residual directly to their own
Newton--Krylov corrector.

## Selection boundary

| Situation | Use |
|---|---|
| good initial guess, safe full steps | `newton_krylov` |
| difficult initial guess, hard physical bounds | `pseudo_transient_continuation` |
| known family of roots | `adaptive_continuation` |
| fold or branch singularity | pseudo-arclength bordered corrector |
| derivatives of the converged selected root | `root_solve` with a tangent solver |

Pseudo-transient iteration is a primal globalization method. Differentiate the
converged equation implicitly; do not reverse-differentiate the accept/reject
history.
