"""Globalize a bounded nonlinear root and continue it along a branch."""

import jax
import jax.numpy as jnp

import solvax as sx

jax.config.update("jax_enable_x64", True)


def residual(x, alpha):
    """A two-scale branch whose positive root moves with alpha."""

    return jnp.array(
        [
            x[0] ** 2 + 0.05 * x[1] - (1.0 + alpha),
            20.0 * (x[1] - 0.25 * x[0]),
        ]
    )


nonlinear = sx.PseudoTransientConfig(
    initial_dt=1.0,
    max_dt=1.0e10,
    rtol=1.0e-9,
    linear_restart=4,
)
continuation = sx.ContinuationConfig(initial_step=0.2, max_step=0.5)

solution = sx.adaptive_continuation(
    residual,
    jnp.array([1.0, 0.25]),
    nonlinear_config=nonlinear,
    continuation_config=continuation,
    admissible=lambda x, alpha: (x[0] > 0.0) & (alpha <= 1.0),
)

print("converged:", solution.converged)
print("alpha:", solution.alpha)
print("root:", solution.x)
print("continuation attempts:", len(solution.steps))
print("accepted:", sum(step.accepted for step in solution.steps))
print("rejected:", sum(not step.accepted for step in solution.steps))
