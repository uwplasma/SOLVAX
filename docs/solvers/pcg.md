# Preconditioned conjugate gradients

PCG solves $Ax=b$ when $A$ is Hermitian positive definite (HPD) and the
preconditioner represents a positive-definite inverse action. SOLVAX operates
matrix-free on arrays or arbitrary JAX pytrees.

## Algorithm

Initialize

$$
r_0=b-Ax_0,\qquad z_0=M^{-1}r_0,\qquad p_0=z_0.
$$

For $k=0,1,\ldots$ compute

$$
\alpha_k=\frac{r_k^Hz_k}{p_k^HAp_k},
$$

$$
x_{k+1}=x_k+\alpha_kp_k,
\qquad
r_{k+1}=r_k-\alpha_kAp_k,
$$

$$
z_{k+1}=M^{-1}r_{k+1},\qquad
\beta_k=\frac{r_{k+1}^Hz_{k+1}}{r_k^Hz_k},
$$

$$
p_{k+1}=z_{k+1}+\beta_kp_k.
$$

For exact arithmetic and an HPD operator, search directions are $A$-conjugate
and the method terminates in at most $n$ steps. In practice, convergence is
governed by the spectrum of $M^{-1}A$, with the classical bound depending on
its condition number {cite}`hestenes1952,saad2003`.

## Pytree model

```python
rhs = {
    "velocity": jnp.array([1.0, -2.0]),
    "potential": jnp.array([3.0]),
}

def matvec(state):
    return jax.tree.map(apply_block, state)

solution = sx.pcg(matvec, rhs, precond=precondition)
```

SOLVAX computes a global inner product by summing `jnp.vdot` over all leaves.
All input and output trees must have identical structure. Integer right-hand
sides are promoted to a floating dtype.

## Inputs

```python
sx.pcg(
    matvec,
    b,
    x0=None,
    precond=None,
    rtol=1e-8,
    atol=0.0,
    max_steps=500,
    single_reduction=False,
)
```

- `matvec`: HPD operator action on the `b` pytree.
- `x0`: optional matching pytree; zeros by default.
- `precond`: positive-definite inverse action; identity by default.
- `rtol`, `atol`: residual stopping tolerances.
- `max_steps`: fixed compiled iteration and history size.
- `single_reduction`: use the algebraically equivalent single-reduction PCG
  recurrence. XLA can combine its independent scalar products into one tuple
  all-reduce, improving strong scaling at the cost of two extra work vectors.

The single-reduction recurrence follows the same rearrangement exposed by
PETSc's `KSPCGUseSingleReduction`. It retains the unpreconditioned residual norm
and stopping contract of the classical implementation. Because finite-precision
recurrences can drift differently, it is opt-in and should be qualified on the
target operator before production use.

## Outputs and statuses

`PCGSolution` contains:

| Field | Meaning |
|---|---|
| `x` | solution pytree |
| `residual_norm` | final absolute norm |
| `relative_residual_norm` | final norm divided by $\lVert b\rVert$ |
| `iterations` | executed iterations |
| `converged` | status is convergence |
| `status` | integer termination code |
| `residual_history` | shape `(max_steps + 1,)`; final value repeated after exit |

Outside a trace:

```python
print(sx.status_name(solution.status))
```

Possible names are:

- `converged`;
- `max_iterations`;
- `non_positive_curvature` when $p^HAp\le0$;
- `preconditioner_breakdown` when $r^HM^{-1}r\le0$;
- `nonfinite`.

The breakdown statuses are valuable model checks: they expose violated HPD
assumptions rather than silently returning an apparent solution.

## Implicitly differentiable PCG

`pcg_linear_solve` retains forward diagnostics but differentiates the converged
linear system with `jax.lax.custom_linear_solve`:

```python
solution = sx.pcg_linear_solve(
    matvec,
    b,
    precond=precondition,
    rtol=1e-10,
    transpose_precond=transpose_precondition,
    transpose_rtol=1e-11,
)
```

The transpose/adjoint solve may use independent tolerances, maximum steps, and
preconditioner. For a Hermitian operator and symmetric preconditioner, the
primal preconditioner is normally reusable. Gradient accuracy is limited by
both primal and transpose residuals.

## Comparison with GMRES

PCG has short recurrences, one stored search direction, and no growing
orthogonalization basis. It is therefore substantially cheaper in memory than
GMRES. The price is strict structure: applying PCG to a nonsymmetric or
indefinite operator is mathematically invalid. GMRES is the correct general
fallback. MINRES, not currently implemented, is the usual short-recurrence
choice for Hermitian indefinite systems {cite}`paige1975`.

## Normal equations

CG may be applied to $A^HAx=A^Hb$, but the condition number is squared:
$\kappa(A^HA)=\kappa(A)^2$. This can amplify roundoff and slow convergence.
Prefer a formulation that preserves the original operator or use an
appropriate least-squares method when available.

## Preconditioning

A valid PCG preconditioner must preserve positive definiteness. Common choices
include Jacobi for positive diagonal systems, block Jacobi with HPD blocks,
line solves for SPD anisotropic diffusion, and symmetric multigrid cycles. An
arbitrary FGMRES preconditioner is not automatically valid for PCG.

## API summary

- {func}`solvax.pcg.pcg`
- {func}`solvax.pcg.pcg_linear_solve`
- {func}`solvax.pcg.status_name`
- {class}`solvax.pcg.PCGSolution`
- {class}`solvax.pcg.PCGDiagnostics`

Runnable counterpart: `examples/17_pcg.py`.
