# Release 0.17.0

## Implicit derivatives for affine multiphysics coupling

For an affine fixed-point map $G(x)=Lx+c$,
`affine_fixed_point_gmres` solves

$$
(I-L)\delta=G(x_0)-x_0,\qquad x^*=x_0+\delta,
$$

without assembling $L$. The same public function now registers this as a
linear solve, so reverse mode solves the transposed fixed-point equation rather
than recording FGMRES iterations:

```python
import jax
import jax.numpy as jnp
import solvax as sx


def response(source):
    damping = jnp.asarray([0.2, 0.6, 0.85])
    solution = sx.affine_fixed_point_gmres(
        lambda state: damping * state + source,
        jnp.zeros_like(source),
        restart=3,
        rtol=1.0e-10,
        transpose_rtol=1.0e-10,
    )
    return jnp.mean(solution.x**2)


value, gradient = jax.jit(jax.value_and_grad(response))(jnp.ones(3))
```

The primal and derivative retain matrix-free array/PyTree maps, custom inner
products, right preconditioning, convergence diagnostics, and the existing
`KrylovSolution` result. `transpose_precond`, `transpose_rtol`,
`transpose_atol`, and `transpose_max_restarts` control the adjoint solve
independently; primal settings remain their defaults.

The affine contract is essential. A genuinely nonlinear map should use a
nonlinear primal solver wrapped by `root_solve`, whose implicit derivative is
the nonlinear fixed-point analogue.

## Verification

Analytical gradients of a multimode affine map agree to $2\times10^{-9}$
relative tolerance. The compiled reverse program contains
`custom_linear_solve`, and its temporary memory is unchanged when the maximum
restart count grows from 2 to 20. The previous raw-GMRES path fails reverse
mode at its dynamic Krylov loop. On an 8,192-unknown float32 map, the new primal
warm median is within 1.3% of the previous implementation.

The complete test matrix covers Python 3.10 and 3.12, minimum and current JAX
stacks, the optional chunking backend, Linux and macOS, lint, typing, package
builds, solver benchmarks, and executable documentation. Combined line/branch
coverage is 97.28% locally and remains above the 95% hosted gate.
