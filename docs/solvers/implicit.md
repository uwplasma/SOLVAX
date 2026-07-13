# Matrix-free nonlinear solves and implicit differentiation

## Newton--Krylov root solve

`newton_krylov` solves $F(x)=0$ without materializing the Jacobian. Each
Newton update uses `jax.linearize` for the Jacobian-vector action and restarted
flexible GMRES for the correction {cite}`knoll2004`. The state and residual may
be arbitrary matching JAX PyTrees.

```python
solution = sx.newton_krylov(
    residual,
    initial,
    precond=approximate_inverse,
    inner_product=weighted_product,
    rtol=1e-8,
    max_steps=10,
    linear_restart=20,
)
```

The nonlinear stopping test uses the true residual relative to the initial
residual. Inspect both `solution.converged` and `solution.linear_converged`, as
well as `newton_iterations`, `linear_iterations`, and `residual_norm`.
`inner_product=` controls GMRES orthogonalization and supports weighted or
distributed reductions; `norm=` may independently define the nonlinear
residual norm.

For large PDE or kinetic systems, `precond=` should approximate the inverse
Jacobian action using inexpensive problem structure. Jacobian-free products
remove matrix storage, but do not by themselves make Krylov convergence
mesh-independent {cite}`knoll2004`.

Implicit differentiation computes derivatives of a converged equation rather
than derivatives of the iterations used to solve it. This makes the gradient
cost independent of the number of primal iterations and avoids storing their
history {cite}`blondel2022,skene2026`.

## Linear solve derivative

Let

$$
A(\theta)x(\theta)=b(\theta).
$$

Differentiating gives

$$
A\,dx=db-(dA)x,
$$

so

$$
dx=A^{-1}[db-(dA)x].
$$

For a reverse-mode cotangent $\bar x$, solve the JAX linear-transpose system

$$
A^T\lambda=\bar x.
$$

Then

$$
\bar b=\lambda,
\qquad
\bar A=-\lambda x^T.
$$

The adjoint requires one transposed/adjoint solve regardless of how many steps
the forward solver used.

## `linear_solve`

```python
def solver(operator, rhs):
    return sx.gmres(operator, rhs, rtol=1e-11).x

x = sx.linear_solve(
    matvec,
    b,
    solver,
    transpose_matvec=adjoint_matvec,  # optional
)
```

Inputs:

- `matvec`: linear action closing over differentiable parameters;
- `b`: right-hand side;
- `solver`: callable `(operator, rhs) -> x` traceable by JAX;
- `transpose_matvec`: optional explicit adjoint action.

If no explicit transpose is supplied, JAX constructs the linear transpose of
`matvec`. The same solver callable is used for the transposed action, so it must
be suitable for both primal and adjoint systems. For nonsymmetric problems,
preconditioners may need separate adjoint forms; close that policy into a
solver or supply a dedicated wrapper.

For complex programs, `linear_solve` follows JAX's transpose/cotangent
semantics rather than independently imposing a Hermitian-adjoint convention.
Use a real scalar objective and validate directional derivatives, as in
`examples/18_complex_krylov_gradient.py`.

## Root derivative

Let $x^*(\theta)$ satisfy

$$
f(x^*,\theta)=0.
$$

The implicit function theorem gives

$$
\frac{dx^*}{d\theta}
=-left(\frac{\partial f}{\partial x}\right)^{-1}
\frac{\partial f}{\partial\theta}.
$$

```python
root = sx.root_solve(
    f,
    x0,
    solver=newton_solver,
    tangent_solve=tangent_linear_solver,
)
```

`solver(f, x0)` computes the primal root. `tangent_solve(g, y)` solves the
linearized system $g(v)=y$, where `g` is a Jacobian-vector product at the root.
The default uses scalar division for scalar roots and materializes a dense
Jacobian for vectors, so large roots should always supply a matrix-free tangent
solver.

```python
def tangent_solve(jvp, rhs):
    return sx.gmres(jvp, rhs, rtol=1e-10).x
```

## PCG-specific wrapper

When the operator is HPD, `pcg_linear_solve` combines implicit gradients with
forward `PCGSolution` diagnostics. It permits independent transpose tolerances,
iteration limits, and preconditioners. See {doc}`pcg`.

## Accuracy

The computed derivative is accurate only if:

1. the primal residual is sufficiently small;
2. the tangent/adjoint residual is sufficiently small;
3. the solution branch is locally differentiable;
4. the supplied transpose action matches the primal operator.

Gradient error is not controlled by primal tolerance alone. Tighten the
adjoint solve and compare selected directional derivatives with central finite
differences.

## What implicit differentiation does not smooth

- branch selection in a multi-root problem;
- clipping, contact, or active-set changes;
- discontinuous remeshing or discrete model selection;
- a singular Jacobian at a bifurcation;
- an unconverged primal solve.

Implicit differentiation gives the local derivative of the selected regular
branch. It should not be used to claim differentiability across a branch jump.

## Comparison with unrolling

| Strategy | Memory | Derivative meaning | Best use |
|---|---|---|---|
| differentiate iterations | grows with unrolled work unless rematerialized | finite algorithm | learned optimizers, deliberately truncated solves |
| implicit differentiation | one tangent/adjoint solve | converged equation | equilibrium, steady state, linear response |
| finite differences | one or more extra primal solves per direction | numerical directional derivative | validation, not high-dimensional gradients |

## API summary

- {func}`solvax.implicit.newton_krylov`
- {func}`solvax.implicit.linear_solve`
- {func}`solvax.implicit.root_solve`
- {func}`solvax.pcg.pcg_linear_solve`

Runnable counterparts: `examples/06_implicit_differentiation.py`,
`examples/17_pcg.py`, and `examples/18_complex_krylov_gradient.py`.
