# Linear operator models

Operator containers make structure composable without forcing assembly. Every
SOLVAX operator is callable and exposes a `shape`; structured operators also
provide a closed-form `.T` action.

## Matrix-free operator

```python
operator = sx.MatrixFreeOperator(
    apply=lambda v: apply_physics(v),
    shape=(n, n),
    transpose_apply=lambda w: apply_adjoint_physics(w),  # optional
)
```

If `transpose_matvec` is omitted, the transpose is obtained through
`jax.linear_transpose`. Supply an explicit adjoint when it is cheaper, clearer,
or uses a specialized discretization.

The algebraic-transpose identity is

$$
\sum_i(Ax)_iy_i=\sum_jx_j(A^Ty)_j.
$$

Test it numerically for every custom operator. For complex reverse-mode
programs, distinguish this algebraic transpose from a Hermitian adjoint and
validate the JAX cotangent convention used by the complete objective.

## Sum operator

For a structured principal part plus matrix-free corrections,

$$
A=A_0+A_1+\cdots+A_p,
$$

use:

```python
full = sx.SumOperator((structured_core, matrix_free_tail))
```

This is the central preconditioning pattern: solve `full` with FGMRES while
using an exact inverse of `structured_core` as the preconditioner.

## Kronecker operator

`KroneckerOperator(A, B)` represents $A\otimes B$. If
$A\in\mathbb{R}^{p\times q}$, $B\in\mathbb{R}^{r\times s}$, and
$X\in\mathbb{R}^{s\times q}$, then

$$
(A\otimes B)\operatorname{vec}(X)
=\operatorname{vec}(BXA^T).
$$

The reshape identity avoids constructing the full $pr\times qs$ matrix and
costs the two smaller multiplies {cite}`vanloan1993`.

```python
operator = sx.KroneckerOperator(A, B)
y = operator(x)
small_reference = jnp.kron(A, B) @ x
```

Use `materialize()` only for small tests and diagnostics.

## Block-tridiagonal operator

```python
operator = sx.BlockTridiagonalOperator(lower, diag, upper)
y = operator(x_flat)
lower, diag, upper = operator.to_blocks()
```

The callable expects a flat vector but stores the exact block structure. Its
bands feed {func}`solvax.direct.block_thomas_factor` without conversion. This
lets one object serve as the matrix-free outer operator and the source of a
direct preconditioner.

## Bordered operator

Constraints, Lagrange multipliers, and gauge conditions often produce

$$
K=
\begin{bmatrix}
A&B\\
C&0
\end{bmatrix},
\qquad
K\begin{bmatrix}x\\y\end{bmatrix}
=
\begin{bmatrix}Ax+By\\Cx\end{bmatrix}.
$$

```python
K = sx.BorderedOperator(A, b_columns, c_rows)
```

`b_columns.shape == (n, p)` and `c_rows.shape == (p, n)`. The callable acts on
the concatenated `(n + p,)` vector.

## Schur-projected preconditioner

Given an approximate inverse action $\widetilde A^{-1}$, form the dense small
Schur complement

$$
S=C\widetilde A^{-1}B.
$$

For residual $(r_x,r_y)$, the projected application is

$$
y=S^{-1}(C\widetilde A^{-1}r_x-r_y),
$$

$$
x=\widetilde A^{-1}(r_x-By).
$$

```python
precond = sx.schur_projected_precond(a_inv, b_columns, c_rows)
solution = sx.gmres(K, rhs, precond=precond)
```

It is exact when `a_inv` is $A^{-1}$ and the dense Schur complement is solved
exactly. With an approximate inverse, it is a constraint-aware preconditioner
for saddle-point systems {cite}`benzi2005`.

## Model selection

| Operator | Use when | Avoid when |
|---|---|---|
| MatrixFree | only the action is cheap/available | an exact reusable structure is being hidden |
| Sum | principal part plus perturbations | operands have incompatible shapes |
| Kronecker | separable tensor-product action | ordering does not match the vectorization identity |
| BlockTridiagonal | adjacent dense blocks | coupling reaches nonadjacent blocks |
| Bordered | few global constraints | the constraint block is large enough to need its own sparse model |

## API summary

- {class}`solvax.operators.MatrixFreeOperator`
- {class}`solvax.operators.SumOperator`
- {class}`solvax.operators.KroneckerOperator`
- {class}`solvax.operators.BlockTridiagonalOperator`
- {class}`solvax.operators.BorderedOperator`
- {func}`solvax.operators.schur_projected_precond`

Runnable counterparts: `examples/11_kronecker.py` and
`examples/12_operators.py`.
