# SOLVAX

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21651844.svg)](https://doi.org/10.5281/zenodo.21651844)

[![tests](https://github.com/uwplasma/SOLVAX/actions/workflows/tests.yml/badge.svg)](https://github.com/uwplasma/SOLVAX/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/uwplasma/SOLVAX/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/SOLVAX)
[![PyPI](https://img.shields.io/pypi/v/solvax)](https://pypi.org/project/solvax/)
[![docs](https://readthedocs.org/projects/solvax/badge/?version=latest)](https://solvax.readthedocs.io)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Differentiable structured linear solvers, preconditioners and matrix-free methods in JAX.**

`solvax` provides the solver infrastructure that kinetic and PDE codes keep
re-implementing: structured direct solves (batched dense LU, block-tridiagonal
Schur elimination with truncated storage), preconditioned and recycled Krylov
methods, physics-agnostic preconditioners (coarse-operator LU, semicoarsened
geometric and p-multigrid, Kronecker approximations, symmetric additive and
line smoothers),
mixed-precision iterative refinement, and implicit differentiation of every solve —
traceable under `jit`, `vmap` and `grad`, on CPU and GPU.

Three documented exceptions, because "transparent to every transform" would
not be true: the exact-window reverse rule is a `custom_vjp`, so `jax.jacfwd`
and `jax.jvp` raise on it rather than falling back to the taped path (pass the
full window, or differentiate the untruncated entry point, when forward mode is
what you need); `eigenpair_reverse` supports reverse mode but orchestrates
application-supplied eigensolvers on the host, so their compiled kernels sit
inside rather than around that dispatcher; and `solvax.native` host primitives
refuse direct tracing while still composing as external primals with
`eigenpair_reverse`.

It complements general JAX solver libraries with block-structured direct
elimination, coarse-operator and multigrid preconditioning, and Krylov
subspace recycling for parameter continuation. SOLVAX operators are native
JAX pytrees; no external operator abstraction is required.

## Install

```bash
pip install solvax
```

## Quickstart

```python
import jax
import jax.numpy as jnp
import solvax as sx

# Solve a block-tridiagonal system L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k
x = sx.block_thomas(lower, diag, upper, rhs)

# Matrix-free PCG on arrays or arbitrary JAX pytrees
solution = sx.pcg(matvec, rhs, precond=preconditioner, rtol=1e-10)
assert solution.converged

# Solve an expensive affine coupling map without assembling its Jacobian
coupled = sx.affine_fixed_point_gmres(coupling_sweep, initial_state)

# Run a relaxed nonlinear coupling sweep with explicit stopping diagnostics
fixed = sx.fixed_point_iteration(
    coupling_sweep, initial_state, relaxation=0.8, atol=1e-8
)

# Periodic Poisson solve; reuse the symbol when the timestep owns the FFT
grid = 2.0 * jnp.pi * jnp.arange(16) / 16
rho = jnp.sin(grid[:, None]) * jnp.sin(grid[None, :])
dx = dy = 2.0 * jnp.pi / 16
rho_hat = jnp.fft.fftn(rho)
phi = sx.solve_periodic_poisson(rho, spacing=(dx, dy))
poisson_symbol = sx.periodic_poisson_eigenvalues(rho.shape, (dx, dy))
phi_hat = sx.solve_periodic_poisson_spectral(rho_hat, eigenvalues=poisson_symbol)

# Same diagnostics, but gradients use an implicit primal/transpose solve
implicit_solution = sx.pcg_linear_solve(matvec, rhs, precond=preconditioner)

# Reuse one elimination across many right-hand sides
factors = sx.block_thomas_factor(lower, diag, upper)
x1 = sx.block_thomas_solve(factors, rhs1)
x2 = sx.block_thomas_solve(factors, rhs2)

# Generate each block once when reusable factors are needed without a stored
# diagonal band. `row(j)` returns the triple (L_j, D_j, U_j) of row j; the
# parameterized form `block_fn(params, j)` used further down takes the
# parameters as its first argument.
generated_factors = sx.block_thomas_factor_fn(row, n_blocks=N)

# Same factors, a third of the state: keep the Schur LU only and rebuild the
# off-diagonal blocks from `row` during each substitution sweep.
lean_factors = sx.block_thomas_factor_fn(row, n_blocks=N, store_offdiagonals=False)
x_lean = sx.block_thomas_solve(lean_factors, rhs)
adjoint = sx.block_thomas_solve(lean_factors, rhs, transpose=True)

# One generated solve with O(sqrt(N) m^2) factor storage and exact JVP/VJP.
x = sx.block_thomas_checkpointed_fn(row, N, rhs)

# Memory-truncated mode: rhs nonzero only in the lowest K blocks and only the
# lowest K solution blocks needed -> O(K m^2) memory, independent of N.
x_low = sx.block_thomas_truncated(lower, diag, upper, rhs[:3], keep_lowest=3)
```

Differentiate a generated selected-head solve with respect to the compact
parameters that build its rows, at retained state independent of the block
count. Ask for the accuracy you want and get a window that provably delivers
it:

```python
# Smallest window whose relative gradient error is provably below rtol.
cert = sx.certified_adjoint_window(block_fn, N, keep_lowest=3, params=p,
                                   rhs_low=rhs[:3], cotangent=ct, rtol=1e-8)
print(cert.window, cert.certified_relative_error)  # proven, not estimated

def objective(params):
    x_low = sx.block_thomas_truncated_fn(
        block_fn, N, rhs[:3], keep_lowest=3,
        params=params, adjoint_window=cert,
    )
    return loss(x_low)

grad = jax.grad(objective)(p)
```

The certificate is possible because the exact-window rule leaves exactly one
approximation to bound — the omitted rows — while every retained row cotangent
is an exact block of the full solve. It costs two localization sweeps, one
generator Jacobian per row, and a differentiated solve, so it is a setup
computation rather than something to run each optimizer step.

Two cheaper options remain for when that is too much.
`localization_crossover_window` reads the chain's transfer norms and returns a
starting window with `certified=False`, and `check_localized_gradient` widens a
window to see whether the gradient moves — evidence rather than a bound, but it
needs no Jacobians.

For a batch of chains that localize at different rows — a collisionality scan,
say — `adjoint_window` is static and would impose the worst chain's width on
all of them. `plan_chain_windows` cuts the batch into a few static buckets
instead, and `chain_window=` gives each chain its own window inside one:

```python
plan = sx.plan_chain_windows(windows, keep_lowest=3, max_buckets=4)
print(plan.reduction)  # measured: 51% fewer retained rows on a 24-chain nu scan
for window, chains in plan.buckets:
    grads = jax.vmap(lambda c: chain_gradient(c, adjoint_window=window))(chains)
```

Everything is differentiable (`jax.grad` through the solve) and batchable
(`jax.vmap` over stacked systems).

## Differentiate a certified eigenmode without taping its solver

Applications often need a mode shape as well as its eigenvalue, but recording
hundreds of Krylov or timestepper iterations makes reverse-mode optimization
slow and memory hungry. `eigenpair_reverse` leaves mode finding to the
application and differentiates only the converged simple eigenpair:

```python
initial_mode = jnp.ones(2, dtype=jnp.complex128)
parameters = jnp.asarray(0.0)

def build(parameters):
    matrix = jnp.asarray(
        [[0.5 + 0.3 * parameters + 0.2j, 0.0],
         [parameters, -1.0 + 3.0j]],
        dtype=jnp.complex128,
    )
    return lambda vector: matrix @ vector

def primal_solver(parameters, _apply, _start):
    # A real application injects its residual-certified matrix-free solver.
    eigenvalue = 0.5 + 0.3 * parameters + 0.2j
    eigenvector = jnp.asarray(
        [1.0, parameters / (eigenvalue + 1.0 - 3.0j)],
        dtype=jnp.complex128,
    )
    return eigenvalue, eigenvector

def left_solver(_parameters, _apply, _start, _eigenvalue):
    return jnp.asarray([1.0, 0.0], dtype=jnp.complex128)

def objective(parameters):
    eigenvalue, eigenvector = sx.eigenpair_reverse(
        parameters,
        build,
        initial_mode,
        primal_solver=primal_solver,
        left_solver=left_solver,
    )
    normalized = eigenvector / jnp.linalg.norm(eigenvector)
    return jnp.real(eigenvalue) + 0.1 * jnp.real(normalized[1])

gradient = jax.grad(objective)(parameters)
```

For `leftᴴ right = 1`, the eigenvalue rule is
`dλ = leftᴴ (dA) right`; eigenvector observables use one bordered
reduced-resolvent solve. This is independent of state layout and primal
algorithm, avoids differentiating the eigensolver iteration, and rejects
nearly defective eigenpairs through an explicit condition-number gate. That
gate reads the pair you supply, so it is bounded by your own solver's accuracy:
near an exceptional point the solver degrades first, and the vectors it returns
can make the condition number look acceptable when it is not. Treat a value
within an order of magnitude of the limit as unconfirmed.

## What's in the box

| Module | Contents |
|---|---|
| `solvax.operators` | Matrix-free, sum, Kronecker, block-tridiagonal and bordered (constraint-row) operator containers with closed-form transposes |
| `solvax.precond` | Jacobi/block-Jacobi, coarse-operator LU, Galerkin-deflation coarse correction, symmetric additive and alternating-direction line composition, V-/W-/F-cycle multigrid over explicit or semicoarsened rediscretized hierarchies, nearest-Kronecker, mixed-precision wrappers |
| `solvax.transfer` | Separable per-axis restriction/prolongation (full weighting, linear, injection) with periodic, dirichlet and reflective closures, exact variational adjointness, and semicoarsening plans |
| `solvax.smoothers` | Point/block Jacobi, batched tridiagonal line and exact banded plane relaxation, upwind-ordered sweeps for streaming operators, and a measured smoothing factor |
| `solvax.direct` | Block-tridiagonal Schur elimination (block Thomas): full, factor/solve split, selected-head (truncated-storage) mode, exact-window localized adjoint, per-row localization profile and window advisor |
| `solvax.banded` | Non-pivoted banded LU with row equilibration + static pivoting; periodic variant via the Woodbury capacitance trick |
| `solvax.tridiagonal` | Batched scalar tridiagonal solve (reproducible Thomas / fused cuSPARSE backend) and periodic (cyclic) systems via a Sherman--Morrison correction |
| `solvax.elliptic` | N-D fully periodic Poisson inversion and periodic-by-bounded Fourier--Helmholtz solves, with reusable spectral symbols for timestep loops |
| `solvax.krylov` | Flexible restarted GMRES (CGS2 + Givens) over arrays, scalars and arbitrary pytrees with optional custom inner products, and GCROT Krylov subspace recycling with FIFO or harmonic-Ritz (GCRO-DR) deflated restarting |
| `solvax.propagator` | Residual-certified RK4 and nested exponential-Arnoldi eigenmode extraction without materializing the operator |
| `solvax.eigen` | Solver-independent implicit reverse derivatives for externally certified eigenpairs, including eigenvector observables and exceptional-point guards |
| `solvax.pcg` | Matrix-free pytree PCG with preconditioning, fixed-shape residual history, and explicit convergence/breakdown status |
| `solvax.fixed_point` | Plain and Aitken fixed-point loops, bounded-memory (condition-filtered) Anderson, and matrix-free affine fixed-point FGMRES |
| `solvax.implicit` | Matrix-free `newton_krylov` (JFNK) plus implicit-function-theorem `linear_solve` and `root_solve` — gradients cost one extra (transposed) solve |
| `solvax.autodiff` | Bounded-memory chunked forward/reverse Jacobians (`chunked_jacfwd`/`jacrev`/`jacobian`) with automatic chunk sizing |
| `solvax.refine` | Mixed-precision iterative refinement (float32 factor, float64 residuals) |
| `solvax.native` / `native_eigen` | Host-side SuperLU and sparse shift-invert eigenpairs; eager primals compose with implicit eigenpair AD |

Complex-valued GMRES/GCROT, tridiagonal solves, and fixed-point acceleration
use Hermitian inner products and real-valued safeguards. Remaining roadmap:
multi-leaf pytree GCROT operands (GCROT takes arrays of any rank; GMRES is
pytree-native) and expanded GPU batched-LU benchmarks.

```python
# Preconditioned, recycled Krylov across a parameter scan:
sol = sx.gcrot(matvec, b, precond=coarse_inverse, m=50, k=10)
sol2 = sx.gcrot(matvec2, b2, precond=coarse_inverse, recycle=sol.recycle)

# Matrix-free Newton-Krylov (JFNK): Jacobian-vector products via jax.linearize,
# each correction solved by FGMRES over an array or structured pytree state.
root = sx.newton_krylov(residual_fn, x0, precond=approx_inverse, rtol=1e-8)

# Weakly contractive affine coupling map G(x) = L x + c, solved as (I - L) x = c:
fixed = sx.affine_fixed_point_gmres(coupling_map, x0, restart=20)

# Periodic (cyclic) scalar tridiagonal line, corners in sub[0] and sup[-1]:
x_line = sx.cyclic_tridiagonal_solve(sub, dia, sup, line_rhs)

# Differentiable solve wrapping any solver:
x = sx.linear_solve(matvec, b, solver=lambda mv, rhs: sx.gmres(mv, rhs).x)
```

## License

MIT. Developed by the [UW Plasma group](https://github.com/uwplasma).
