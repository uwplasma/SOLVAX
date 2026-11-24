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
    linear_rtol=0.5,
    forcing="eisenstat_walker",
    fixed_work=False,
)
```

The nonlinear stopping test uses the true residual relative to the initial
residual. Inspect both `solution.converged` and `solution.linear_converged`, as
well as `newton_iterations`, `linear_iterations`, and `residual_norm`.
`inner_product=` controls GMRES orthogonalization and supports weighted or
distributed reductions; `norm=` may independently define the nonlinear
residual norm.

Nonfinite residual norms or stopping thresholds are failures, including an infinite
initial norm with relative stopping. A successful linear flag alone says
nothing about the nonlinear root. The `root_solve` wrapper trusts its supplied
solver; check convergence before using an implicit derivative and propagate
failure rather than returning an uncertified iterate.

For bounded-cost composition inside `jax.lax.scan`, opt in with
`fixed_work=True`. Newton and its inner GMRES then use fixed-length scan
control and mask updates after convergence; the default early-exit path is
unchanged. This unrolled derivative is useful when the algorithm itself is the
object of differentiation. For derivatives of a converged root, prefer
`root_solve`, whose implicit rule avoids retaining or differentiating through
the Newton/Krylov iteration history.

The default `forcing="constant"` passes `linear_rtol` to every inner GMRES
solve. Opt-in `forcing="eisenstat_walker"` treats `linear_rtol` as the initial
forcing term and uses the safeguarded Eisenstat--Walker choice 2 residual ratio
for later Newton steps {cite}`eisenstat1996`. This avoids solving early Newton
systems to terminal accuracy while tightening the inner tolerance as the
nonlinear residual contracts. `forcing_gamma`, `forcing_alpha`, and
`forcing_max` control the choice-2 formula; the maximum must remain below one.

`line_search=True` backtracks each step on the residual norm: the step `s`
of forcing `eta` is halved, and `eta <- 1 - (1 - eta)/2`, until
`norm(F(x + s)) <= [1 - t (1 - eta)] norm(F(x))` with `t = armijo = 1e-4`
(Algorithm INB of Eisenstat and Walker, SIAM J. Optim. 4, 393 (1994); the
condition is eq. (2.5) of Pawlowski, Shadid, Simonis and Walker, SIAM Rev. 48,
700 (2006)). A full step that passes is taken unchanged, so a problem whose
full steps all pass gives the same iterates bit for bit. After
`max_backtracks` halvings without a pass the iteration stops unconverged with
`newton_iterations < max_steps`. On `arctan(x - b) = 0` with 50 unknowns
started 2 to the right of the root, plain Newton diverges (final residual norm
11.1) and the line search converges in 5 steps. It is not available with
`fixed_work=True`.

`newton_krylov_root` returns a `RootResult` whose root is differentiable by the
implicit rule of `root_solve` (the Newton iterations are not differentiated;
`jax.grad` through `newton_krylov` itself raises in its `while_loop`):

```python
res = sx.newton_krylov_root(residual, x0, precond=m, precond_transpose=m_t,
                            atol=1e-8, rtol=0.0, line_search=True)
```

A forward `precond` also preconditions the GMRES tangent, so reverse mode
needs `precond_transpose` (`v -> M^{-T} v`), or `jax.linear_transpose(precond)`
when `precond` is linear. `tangent_precond` overrides it. Status, residual
`max |F|` and the NaN derivative of an unconverged root follow
`certified_root`. On the 1-D Bratu problem `-u'' = lam exp(u) + c` with 1000
unknowns and `M` the inverse Laplacian, the gradient in `(lam, c)` agrees with
central differences to 6.4e-9 and with the dense implicit derivative to 4.6e-13.
Arrays only, no pytrees.

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
    solution = sx.gmres(operator, rhs, rtol=1e-11)
    return solution.x, (
        solution.residual_norm,
        solution.iterations,
        solution.converged,
    )

x, diagnostics = sx.linear_solve(
    matvec,
    b,
    solver,
    transpose_matvec=adjoint_matvec,  # optional
    transpose_solver=solver,           # optional; defaults to solver
    has_aux=True,
)
```

Inputs:

- `matvec`: linear action closing over differentiable parameters;
- `b`: right-hand side;
- `solver`: callable `(operator, rhs) -> x` traceable by JAX, or
  `(operator, rhs) -> (x, aux)` when `has_aux=True`;
- `transpose_matvec`: optional explicit adjoint action;
- `transpose_solver`: optional dedicated solver for the transposed system;
- `has_aux`: return the forward solver's auxiliary diagnostics with `x`.

If no explicit transpose is supplied, JAX constructs the linear transpose of
`matvec`. By default the same solver callable is used for the transposed action.
For nonsymmetric problems, pass `transpose_solver` when the adjoint needs a
different initial guess, preconditioner, or stopping policy. With
`has_aux=True`, both callbacks return `(x, aux)`; the public result retains the
forward diagnostics.

For complex programs, `linear_solve` follows JAX's transpose/cotangent
semantics rather than independently imposing a Hermitian-adjoint convention.
Use a real scalar objective and validate directional derivatives, as in
`examples/18_complex_krylov_gradient.py`.

## Recycling the forward Krylov space in the adjoint

The adjoint system $A^T\lambda=\bar x$ has the same spectrum as the forward
one, so the near-invariant subspace the forward solve paid to discover still
governs adjoint convergence. `recycled_linear_solve` carries the forward
solve's recycle pair into the adjoint solve instead of discarding it:

```python
def operator(params, x):          # linear in x, parameters explicit
    return params["shift"] * x + laplacian(x)

def solver(matvec, rhs, recycle):
    solution = sx.gcrot(matvec, rhs, m=15, k=12, rtol=1e-10,
                        recycle=recycle, recycle_strategy="harmonic")
    return solution.x, solution.recycle

x = sx.recycled_linear_solve(operator, params, b, solver)
```

The parameters are explicit rather than closed over, because the forward pair
has to reach the backward pass through the residual of a custom VJP; that is
the only difference from `linear_solve`. The adjoint remains exactly one
transposed solve — it is simply a cheaper one. On a spectrum with eight
eigenvalues decades below the bulk, the test suite pins the adjoint at 11
inner iterations with reuse against 47 without, a saving that survives
charging the warm start its $k$ operator applications.

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
The default uses scalar division for scalar roots, a dense Jacobian for vectors
of at most `dense_limit` unknowns (1024, `solvax.implicit.DENSE_TANGENT_LIMIT`),
and above that unpreconditioned restarted GMRES (cycle of 50, 40 restarts,
relative residual `max(1e-10, 100 eps)`), so no `n x n` buffer is formed. The
limit is not a cost crossover; it keeps the dense solve, which never gives NaN
on a nonsingular Jacobian, where it is affordable: on a well-conditioned
Jacobian GMRES is already cheaper at 512 unknowns, but unpreconditioned GMRES
fails on hard Jacobians (a random non-normal one of condition 1e2 is enough),
and then the tangent is NaN. The remedy is `dense_limit=n` or a preconditioned
tangent solver. Complex vector roots are solved as the real system of size
`2n`, so `f` may contain `conj`; complex scalar roots need a holomorphic `f`.
Reverse mode transposes a custom tangent solver, so wrap a Krylov solve in
`jax.lax.custom_linear_solve`:

```python
def tangent_solve(jvp, rhs):
    def solve(matvec, b):
        return sx.gmres(matvec, b, precond=precond, rtol=1e-10).x

    return jax.lax.custom_linear_solve(jvp, rhs, solve, transpose_solve=solve)
```

To keep the dense tangent above the limit, pass `dense_limit=n`.

The default GMRES tangent also takes a right preconditioner and its options,
which avoids writing the wrapper above:

```python
root = sx.root_solve(f, x0, newton_solver, tangent_precond=(M, M_T))
root = sx.root_solve(f, x0, newton_solver,
                     tangent_precond=lambda x_root: build_pair(x_root),
                     tangent_rtol=1e-10, tangent_restart=50,
                     tangent_max_restarts=40)
```

`tangent_precond` is a pair of callables `v -> M^{-1} v` and
`v -> M^{-T} v`, or a factory called once with the root, so a preconditioner
built from the Jacobian at the root needs no second primal solve. Given, it
selects the GMRES tangent at every size. `M_T` is used for the transposed
system of reverse mode; `M_T=None` uses `jax.linear_transpose(M)`, so `M` must
then be linear. For a complex root `M` acts on the complex vector, and `M_T` is
its adjoint (conjugate transpose), because the tangent is the real `2n` system.
On a nonsymmetric advection-diffusion root of 1200 unknowns with
`cond(df/dx) = 7e4`, the unpreconditioned tangent is NaN, and
`tangent_precond` with the LU of the linear part matches the dense tangent to
2e-13 and central differences to 3e-10.

With `has_aux=True` the solver returns `(root, aux)` and so does `root_solve`;
`aux` (iterations, residual, status) has a zero derivative. Before JAX 0.7
give it a floating dtype, as `certified_root` does.

`certified_root` takes the same tangent keywords (`tangent_solve`,
`dense_limit`, `tangent_precond`, `tangent_rtol`, `tangent_restart`,
`tangent_max_restarts`) and passes them to `root_solve`. They apply to vector
roots: a scalar root's tangent is a division, so `tangent_precond` with a
scalar start raises. The status and residual do not depend on them.

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
- {func}`solvax.implicit.newton_krylov_root`
- {func}`solvax.implicit.linear_solve`
- {func}`solvax.implicit.recycled_linear_solve`
- {func}`solvax.implicit.root_solve`
- {func}`solvax.pcg.pcg_linear_solve`

Runnable counterparts: `examples/06_implicit_differentiation.py`,
`examples/17_pcg.py`, and `examples/18_complex_krylov_gradient.py`.
