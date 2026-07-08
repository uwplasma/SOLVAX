# solvax

**Differentiable structured linear solvers, preconditioners and matrix-free
methods in JAX.**

`solvax` is the solver layer that kinetic and PDE codes keep re-implementing,
factored out once: structured direct solves, preconditioned and recycled
Krylov methods, physics-agnostic preconditioners, mixed-precision refinement,
and implicit differentiation — all jit/vmap/grad-transparent on CPU and GPU.

## Quickstart

```python
import solvax as sx

# Block-tridiagonal system: L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k
x = sx.block_thomas(lower, diag, upper, rhs)

# Reuse one elimination across right-hand sides
factors = sx.block_thomas_factor(lower, diag, upper)
x1 = sx.block_thomas_solve(factors, rhs1)
x2 = sx.block_thomas_solve(factors, rhs2)
```

## The block-tridiagonal kernel

For blocks $L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k$, a Schur-complement
sweep from the last block down,

$$\Delta_{N-1} = D_{N-1}, \qquad
\Delta_k = D_k - U_k \Delta_{k+1}^{-1} L_{k+1}, \qquad
\sigma_k = b_k - U_k \Delta_{k+1}^{-1} \sigma_{k+1},$$

followed by substitution upward from block 0, solves the system exactly with
one dense LU and one matrix product per block {cite}`golub2013,demmel1995`.
When the right-hand side vanishes above block $K$ and only the lowest $K$
solution blocks are needed — the typical situation for spectral kinetic
equations, where sources and velocity moments involve only the first few
modes — storage above $K$ can be discarded on the fly, giving memory
$O(K m^2)$ independent of $N$ {cite}`escoto2025`.

```{toctree}
:maxdepth: 1

api
```

## References

```{bibliography}
```
