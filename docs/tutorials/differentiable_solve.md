# Tutorial: differentiate a converged matrix-free solve

We solve

$$
A(\theta)x(\theta)=b
$$

and differentiate the scalar objective

$$
J(\theta)=\frac12x(\theta)^H W x(\theta).
$$

The primal algorithm is FGMRES, but the gradient is defined by the converged
linear equation through `linear_solve`.

## 1. Parameterized operator

```python
import jax
import jax.numpy as jnp
import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 64
grid = jnp.arange(1, n + 1, dtype=jnp.float64)
b = jnp.sin(jnp.pi * grid / (n + 1))
W = 1.0 + grid / n

def make_operator(theta):
    diagonal = 2.0 + theta + 0.1 * jnp.sin(grid)

    def matvec(x):
        padded = jnp.pad(x, (1, 1))
        return diagonal * x - padded[:-2] - padded[2:]

    return matvec, diagonal
```

For positive `theta`, this shifted tridiagonal operator is SPD, although we use
FGMRES here to demonstrate the general wrapper.

## 2. Primal solver policy

```python
def solve_with_gmres(operator, rhs):
    solution = sx.gmres(
        operator,
        rhs,
        restart=30,
        rtol=1e-11,
        atol=1e-13,
        max_restarts=20,
    )
    return solution.x
```

`linear_solve` expects the solver to return the array, not the result object.
In production, run a separate diagnostic solve during validation or use
`pcg_linear_solve` when its HPD assumptions hold and retained diagnostics are
needed.

## 3. Implicit objective

```python
def objective(theta):
    matvec, _ = make_operator(theta)
    x = sx.linear_solve(matvec, b, solve_with_gmres)
    return 0.5 * jnp.vdot(x, W * x).real

theta = 0.7
value, gradient = jax.value_and_grad(objective)(theta)
print(value, gradient)
```

The reverse pass solves the JAX linear-transpose system for the cotangent. JAX differentiates the closed-over
operator coefficients to assemble the parameter cotangent; it does not reverse
through every FGMRES iteration.

## 4. Validate with finite differences

```python
step = 1e-5
finite_difference = (
    objective(theta + step) - objective(theta - step)
) / (2.0 * step)

relative_error = jnp.abs(gradient - finite_difference) / jnp.maximum(
    jnp.abs(finite_difference), 1e-14
)
print("relative gradient error:", relative_error)
```

Central differences have $O(h^2)$ truncation error and $O(u/h)$ cancellation
error. Repeat over several step sizes; agreement over a plateau is stronger
evidence than a single step.

## 5. Use structure in the solver

This operator is tridiagonal, so a production primal/adjoint policy could close
over a structured direct solve. For a parameter that changes the diagonal,
factorization must be recomputed inside the parameterized function. For a fixed
operator and changing right-hand sides, factor once outside.

For an HPD operator, the higher-level alternative is:

```python
def objective_pcg(theta):
    matvec, diagonal = make_operator(theta)
    precond = sx.jacobi(diagonal)
    solution = sx.pcg_linear_solve(
        matvec,
        b,
        precond=precond,
        rtol=1e-11,
        transpose_rtol=1e-12,
    )
    return 0.5 * jnp.vdot(solution.x, W * solution.x).real
```

## 6. Nonlinear extension

For a root $f(x,\theta)=0$, retain the application-specific Newton, Aitken, or
Anderson primal solve and use `root_solve`. For a large state, provide a
matrix-free `tangent_solve`; otherwise the default vector tangent path
materializes a dense Jacobian.

## 7. Common mistakes

- Closing over a Python value rather than a JAX parameter prevents the desired
  gradient path.
- Returning `KrylovSolution` instead of `.x` from the generic solver callback
  violates the `linear_solve` callback contract.
- A loose primal or adjoint tolerance appears as gradient error.
- Discontinuous branch selection is not made smooth by `root_solve`.
- Complex objectives should be real-valued; validate the Hermitian adjoint
  convention with directional finite differences.
