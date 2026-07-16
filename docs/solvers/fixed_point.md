# Fixed-point acceleration

Partitioned multiphysics algorithms often define an expensive map

$$
x_{k+1}=G(x_k)
$$

and seek a fixed point $x^*=G(x^*)$. Define the residual

$$
r(x)=G(x)-x.
$$

SOLVAX provides safeguarded vector Aitken relaxation, a bounded-history
Anderson mixing primitive, a complete Aitken fixed-point loop, and matrix-free
FGMRES for affine maps.

## Affine fixed points with FGMRES

For an affine map $G(x)=Lx+c$, the fixed-point equation is the linear system

$$
(I-L)\delta=G(x_0)-x_0,\qquad x^*=x_0+\delta.
$$

`affine_fixed_point_gmres` applies this operator through map evaluations, so no
matrix or Jacobian is assembled. Array and PyTree states, custom inner
products, and right preconditioners use the same contracts as `gmres`.

```python
solution = sx.affine_fixed_point_gmres(
    mapping,
    x0,
    restart=20,
    rtol=1e-8,
)
```

The affine contract is deliberate. For a genuinely nonlinear map, use a
globalized nonlinear primal solver and `root_solve` for implicit derivatives.

## Relaxed iteration

Instead of accepting $G(x_k)$ directly, use

$$
x_{k+1}=x_k+\omega_k r_k.
$$

$\omega_k<1$ under-relaxes unstable coupling; $\omega_k>1$ extrapolates when
the map is safely contractive.

## Vector Aitken relaxation

With $\Delta r_k=r_k-r_{k-1}$, the scalar update used by SOLVAX is

$$
\omega_k=-\omega_{k-1}
\frac{r_{k-1}^H\Delta r_k}{\Delta r_k^H\Delta r_k}.
$$

Only the real part relevant to the scalar relaxation is retained. The result is
kept finite and clipped to `[min_relaxation, max_relaxation]`.

```python
omega = sx.aitken_relaxation(
    previous_residual,
    residual,
    previous_relaxation=1.0,
    min_relaxation=0.05,
    max_relaxation=2.0,
)
```

Use this primitive when the application owns a coupled iteration with physical
stopping gates, logging, rollback, or subsystem failures.

## Complete Aitken solve

```python
solution = sx.aitken_fixed_point(
    mapping,
    x0,
    rtol=1e-8,
    atol=0.0,
    max_steps=100,
    min_relaxation=0.05,
    max_relaxation=2.0,
)
```

The result contains `x`, the true final `residual_norm`, `iterations`,
`converged`, and the final `relaxation`. The loop is implemented with
`jax.lax.while_loop` and supports arrays, `jit`, and `vmap`.

The stopping threshold is

$$
\lVert r(x_k)\rVert_2\le
\max(\mathtt{atol},\mathtt{rtol}\max(\lVert x_0\rVert_2,1)).
$$

## Anderson mixing

Given a history $(x_i,r_i)$ for $i=1,\ldots,h$, Anderson mixing chooses affine
weights whose residual combination is small. SOLVAX solves the regularized
Gram system

$$
(R^HR+\lambda I)w=\mathbf{1},\qquad
\alpha=\frac{w}{\mathbf{1}^T w},
$$

then combines mapped points:

$$
x_{new}=(1-\beta)G(x_h)+
\beta\sum_{i=1}^{h}\alpha_iG(x_i).
$$

Here rows of `residuals` form the history matrix $R$, `regularization`
controls $\lambda$ after internal scale normalization, and `damping` is
$\beta$.

```python
x_next = sx.anderson_mixing(
    iterates,       # shape (history, ...)
    residuals,      # G(iterates) - iterates, same shape
    regularization=1e-8,
    damping=1.0,
    condition_limit=1e6,
)
```

When several mapped quantities must use identical affine coefficients, compute
the weights once from the residual-bearing state and apply them along each
history axis. The trailing shapes may differ:

```python
weights = sx.anderson_weights(residuals, condition_limit=1e6)
mixed_velocity = jnp.tensordot(weights, mapped_velocity, axes=(0, 0))
mixed_flux = jnp.tensordot(weights, mapped_flux, axes=(0, 0))
```

`condition_limit` optionally filters residual-history singular directions
before solving the affine-weight system. This prevents nearly dependent map
histories from squaring an already large condition number in the Gram solve.

Map evaluation and stopping intentionally remain outside this primitive so an
application can keep a bounded ring buffer and its own nonlinear control
policy. Anderson acceleration originated as a mixing method for integral
equations and is closely related to multisecant quasi-Newton updates
{cite}`anderson1965,walker2011`.

## Aitken versus Anderson versus Newton

| Method | Stored history | Derivative information | Typical role |
|---|---:|---|---|
| fixed relaxation | none | none | robust baseline |
| Aitken | one residual | scalar secant estimate | one dominant slow coupling mode |
| Anderson | several residuals | multisecant history | several coupled slow modes |
| Newton | Jacobian or JVP solves | local derivative | fast local convergence near a regular root |

Newton can converge quadratically near a regular solution but requires linear
solves and globalization. Aitken and Anderson reuse only map evaluations but do
not make a fundamentally noncontractive map globally convergent.

## Safeguards and failure modes

- A vanishing Aitken denominator retains the previous relaxation.
- Anderson adds scale-aware regularization and falls back to the newest mapped
  point if weights or their affine normalization become degenerate.
- `condition_limit` bounds the retained residual-history condition number;
  use it when a longer history produces spikes from nearly dependent columns.
- Large Aitken upper bounds can destabilize nonlinear maps; choose physical
  bounds rather than accepting the permissive default blindly.
- Nearly dependent Anderson histories require stronger regularization or a
  shorter history.
- Always recompute the true residual after an accelerated candidate.

## Differentiation

Differentiating through iteration-count branching gives the derivative of the
executed algorithm. For the derivative of the converged equation, define
$f(x)=G(x)-x$ and wrap the primal solver with
{func}`solvax.implicit.root_solve`.

## API summary

- {func}`solvax.fixed_point.aitken_relaxation`
- {func}`solvax.fixed_point.anderson_weights`
- {func}`solvax.fixed_point.anderson_mixing`
- {func}`solvax.fixed_point.aitken_fixed_point`
- {func}`solvax.fixed_point.affine_fixed_point_gmres`
- {class}`solvax.fixed_point.FixedPointSolution`
