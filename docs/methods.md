# Methods and architecture

This page is the complete reference for what `solvax` implements: for every
capability, the method and its governing equations, the source it is drawn
from, the inputs and outputs, and the use case it targets. The rendered
per-function signatures live in the {doc}`api`; this page is the connective
narrative and the mathematics.

## Design principles

`solvax` is a thin layer of *structured* solvers on top of JAX and
[lineax](https://github.com/patrick-kidger/lineax). Four conventions run
through every module:

- **`matvec` callables.** A linear operator is anything callable ``v -> A v``
  on flat arrays. Krylov methods, preconditioners and implicit-diff wrappers
  all speak this protocol, and the containers in {mod}`solvax.operators` are
  themselves callable, so they drop in wherever a `matvec` is expected.
- **Transform transparency.** Unless a function lives in {mod}`solvax.native`,
  it is pure JAX: `jax.jit`, `jax.vmap` and `jax.grad` compose through it. Loop
  state has static shapes (zero-padded Krylov bases, fixed-size recycle pairs)
  so even early-exiting iterations stay jit-able.
- **Factor / solve split.** Every direct method exposes a `*_factor` step that
  does the expensive elimination once and a `*_solve` step that is cheap to
  repeat across right-hand sides — and, where relevant, across the *transposed*
  system, which is exactly what an adjoint needs.
- **Bring your own solver.** Implicit differentiation ({mod}`solvax.implicit`)
  treats the forward solver as a black box and attaches the
  implicit-function-theorem derivative, so the choice of solver and the choice
  of gradient are orthogonal.

The remainder of this page walks the capabilities in dependency order.

## Linear operators — {mod}`solvax.operators`

Physics-agnostic containers wrapping the *action* of a structured map, each
with a closed-form transpose (`.T`) so the single invariant
$\langle A x, y\rangle = \langle x, A^\top y\rangle$ holds without a materialized
matrix.

- **`MatrixFreeOperator`** wraps an arbitrary linear callable; its transpose
  falls back on `jax.linear_transpose` when no explicit adjoint is supplied.
- **`SumOperator`** applies $(A_1 + \dots + A_p)v = \sum_i A_i v$; the transpose
  distributes over the sum. The idiom is a structured principal part plus
  matrix-free perturbations.
- **`KroneckerOperator`** applies $A \otimes B$ through the reshape identity
  $(A\otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^\top)$, costing
  $O(pqs+prs)$ instead of the $O(pqrs)$ of the assembled product {cite}`vanloan1993`.
- **`BlockTridiagonalOperator`** stores dense bands $(L_k, D_k, U_k)$ in the
  {mod}`solvax.direct` layout and applies row $k$ as $L_k x_{k-1} + D_k x_k +
  U_k x_{k+1}$ with batched einsums; `to_blocks()` feeds the block-Thomas
  factorizer directly.
- **`BorderedOperator`** is the saddle-point structure
  $K = \begin{bmatrix} A & B \\ C & 0 \end{bmatrix}$, acting on $[x, y]$ as
  $[Ax + By,\ Cx]$. `schur_projected_precond` turns an approximate $A^{-1}$
  into a preconditioner for the whole system via the dense Schur complement
  $S = C A^{-1} B$,

  $$y = S^{-1}(C A^{-1} r_x - r_y), \qquad x = A^{-1}(r_x - B y),$$

  which is exactly $K^{-1}$ when $A^{-1}$ is exact {cite}`benzi2005`.

*Use case:* express a matrix-free operator once and reuse it as `matvec=` and
its structured part as the preconditioner.

## Structured direct solvers — {mod}`solvax.direct`

For a block-tridiagonal system $L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k$, the
Schur-complement (block Thomas) sweep from the last block down,

$$\Delta_{N-1} = D_{N-1}, \quad
\Delta_k = D_k - U_k \Delta_{k+1}^{-1} L_{k+1}, \quad
\sigma_k = b_k - U_k \Delta_{k+1}^{-1} \sigma_{k+1},$$

then substitution upward, $x_0 = \Delta_0^{-1}\sigma_0$,
$x_k = \Delta_k^{-1}(\sigma_k - L_k x_{k-1})$, solves the system with one dense
LU and one matrix product per block — never an explicit inverse — at
$O(N m^3)$ cost {cite}`golub2013`.

- **`block_thomas` / `block_thomas_factor` / `block_thomas_solve`** are the
  full solve and the factor/solve split. `block_thomas_solve(..., transpose=True)`
  reuses the *same* factors for $A^\top x = b$, because for a fixed elimination
  order the Schur complements of $A^\top$ are exactly $\Delta_k^\top$ — one
  elimination serves the forward and the adjoint solve.
- **`block_thomas_truncated` / `block_thomas_truncated_fn`** exploit a common
  kinetic structure: when the right-hand side vanishes for $k \ge K$ and only
  the lowest $K$ solution blocks are wanted (velocity moments touching only the
  first few spectral modes), the upward substitution stops at $K$ and the
  downward sweep stores nothing above it, so peak memory is $O(K m^2)$,
  *independent of $N$* {cite}`escoto2025`. The `_fn` variant assembles blocks
  on the fly from compact physics coefficients so the full band arrays are
  never materialized.

*Stability:* block LU without pivoting is backward-stable for
block-diagonally-dominant systems {cite}`demmel1995`; each dense block is
factored with partial pivoting, and {mod}`solvax.refine` provides the
iterative-refinement fallback in weakly dominant regimes.

*Use case:* the natural direct solver — and preconditioner — for 1-D
transport, spectral kinetic equations, and any operator with a
{class}`~solvax.operators.BlockTridiagonalOperator` principal part.

## Banded and periodic-banded LU — {mod}`solvax.banded`

Non-pivoted (Doolittle) LU of a banded matrix in `scipy`-layout storage,
carried out column by column with a `jax.lax.scan` so shapes stay static.
Because XLA handles row pivoting poorly, two safeguards substitute for it: row
equilibration (scale each row by $1/\max|\text{row}|$) and static pivoting
(clamp any pivot below a floor and count the clamps, so callers can detect
near-singularity). LU without pivoting is backward-stable for diagonally
dominant systems {cite}`golub2013,demmel1995`.

Periodic (circulant-banded) systems $A = B + UV^\top$ — a banded core plus
wrap-around corners as a low-rank update — are solved with the
Sherman-Morrison-Woodbury identity

$$(B + UV^\top)^{-1} = B^{-1} - B^{-1}U(I + V^\top B^{-1} U)^{-1} V^\top B^{-1},$$

where the tiny capacitance matrix $I + V^\top B^{-1}U$ is LU-factored once, so
each periodic solve costs one banded solve plus $O(\text{bw})$ dense work.

*Use case:* advection-dominated 1-D operators and their periodic variants;
`banded_matvec` applies $A$ without densifying, and the factors compose into
line solves for {func}`~solvax.precond.line_smoother`.

## Batched tridiagonal solve — {mod}`solvax.tridiagonal`

The scalar tridiagonal case $lower_j x_{j-1} + diag_j x_j + upper_j x_{j+1} =
rhs_j$ is common and structured enough to earn a dedicated, hardware-aware fast
path. The system lives on the **leading** axis; every trailing axis of `rhs`
(columns, stacked fields, batch dimensions) is solved at once — the layout that
maps a stack of independent systems onto the vendor batched kernel without an
outer `vmap`. Two backends:

- **Thomas** (`method="thomas"`) — two `jax.lax.scan` sweeps

  $$c'_0 = \tfrac{upper_0}{diag_0},\quad
  c'_j = \frac{upper_j}{diag_j - lower_j c'_{j-1}},\quad
  d'_j = \frac{rhs_j - lower_j d'_{j-1}}{diag_j - lower_j c'_{j-1}},$$

  back-substituted as $x_{n-1} = d'_{n-1}$, $x_j = d'_j - c'_j x_{j+1}$
  {cite}`thomas1949,golub2013`. Fixed arithmetic makes it **bitwise
  reproducible** — the CPU path.
- **Fused** (`method="lax"`) — XLA's batched
  `jax.lax.linalg.tridiagonal_solve` (cuSPARSE `gtsv2` on CUDA). On a GPU the
  $n$ sequential Thomas steps serialize into $n$ latency-bound kernel launches,
  so the single fused kernel is far faster there.

`method="auto"` (default) selects Thomas when the code lowers for CPU (bit
parity, honouring a CPU pin even on an accelerator host) and the fused kernel
otherwise, via `jax.lax.platform_dependent`; systems with fewer than three rows
always use Thomas (cuSPARSE requires $n \ge 3$). The Thomas kernel is ported
verbatim from the parity-proven vmec_jax radial preconditioner.

*Use case:* 1-D radial/field-line preconditioners and the per-axis line solves
of {func}`~solvax.precond.line_smoother`, where the same tridiagonal is applied
across many spectral columns each iteration.

## Krylov methods — {mod}`solvax.krylov`

Right-preconditioned **flexible GMRES** builds the Arnoldi relation
$A Z_m = V_{m+1}\bar H_m$ with $Z_m = [M_1^{-1}v_1, \dots, M_m^{-1}v_m]$; storing
the preconditioned vectors explicitly lets the preconditioner change from step
to step (flexible mode). The correction $x \mathrel{+}= Z_m y$ minimizes
$\|\beta e_1 - \bar H_m y\|$, solved incrementally with Givens rotations so the
residual is available every inner step. Orthogonalization is classical
Gram-Schmidt applied twice (CGS2), which cuts accelerator inner-product latency
while keeping $O(\varepsilon)$ loss of orthogonality {cite}`saad2003`.

**`gcrot`** adds GCROT($m$, $k$)-style subspace recycling: an outer pair
$(C, U)$ with $AU = C$, $C^H C = I$ deflates the operator, and the pair can
be carried across a slowly-varying sequence of solves (parameter continuation)
— pass `solution.recycle` of one solve as `recycle=` of the next. On warm start
$AU$ is recomputed for the current operator and re-orthonormalized by thin QR,
so a stale pair is always consistent {cite}`parks2006,morgan2002`.

Real and complex systems share the same implementation. Complex Arnoldi and
recycling use Hermitian projections, while the incremental least-squares solve
uses scaled unitary complex Givens rotations in the LAPACK ``xLARTG``
convention. The convergence test always uses the true residual norm recomputed
at restart boundaries and after the final cycle.

*Use case:* matrix-free solves where a preconditioner clusters the spectrum
(`gmres`), and parameter scans where consecutive systems share eigenmodes
(`gcrot`).

## Preconditioned conjugate gradients — {mod}`solvax.pcg`

For a Hermitian positive-definite matrix-free operator $A$ and a positive-
definite preconditioner $M$, PCG applies conjugate gradients to the
preconditioned residual without materializing either matrix. With
$r_0=b-Ax_0$, $z_0=M^{-1}r_0$, and $p_0=z_0$, each step is

$$
\alpha_k = \frac{r_k^\ast z_k}{p_k^\ast A p_k},\qquad
x_{k+1}=x_k+\alpha_kp_k,\qquad
r_{k+1}=r_k-\alpha_kAp_k,
$$

$$
z_{k+1}=M^{-1}r_{k+1},\qquad
\beta_k=\frac{r_{k+1}^\ast z_{k+1}}{r_k^\ast z_k},\qquad
p_{k+1}=z_{k+1}+\beta_kp_k.
$$

{func}`~solvax.pcg.pcg` accepts an array or an arbitrary JAX pytree; `matvec`,
`precond`, `b`, and `x0` share that structure. Dot products sum over every leaf,
so coupled field blocks can remain named pytrees rather than being flattened by
application code. Real and complex Hermitian systems are supported, and integer
right-hand sides are promoted to floating point.

The returned {class}`~solvax.pcg.PCGSolution` contains the solution, absolute
and relative residual norms, iteration count, convergence flag, integer status,
and a residual history of shape `max_steps + 1`. Entries after termination
repeat the final norm: this fixed shape is deliberate, keeping the whole result
compatible with `jit` and `vmap`. {func}`~solvax.pcg.status_name` maps a
materialized status to one of `converged`, `max_iterations`,
`non_positive_curvature`, `nonfinite`, or `preconditioner_breakdown`.

Stopping uses $\|r_k\|\le\max(\texttt{atol},\texttt{rtol}\|b\|)$ and positive-
curvature checks are scale-free. A non-positive $p^\ast A p$ reports that the
operator is not positive definite in the explored subspace; a non-positive
$r^\ast M^{-1}r$ reports an invalid or broken preconditioner. Neither condition
is silently converted into convergence.

For gradients, {func}`~solvax.pcg.pcg_linear_solve` preserves the forward
{class}`~solvax.pcg.PCGSolution` diagnostics while registering an implicit VJP
with independent primal and transpose tolerances, iteration limits, and
preconditioners. It differentiates the converged linear system rather than the
iteration count and does not require a second primal solve to recover
diagnostics.

*Use case:* symmetric elliptic operators, normal equations used with care, and
multi-field SPD blocks where a structured line, multigrid, or application-
provided preconditioner removes anisotropy.

## Preconditioners — {mod}`solvax.precond`

Every builder returns a callable $M^{-1}$ suitable for `precond=`. A
preconditioner only has to *cluster the spectrum* of $A M^{-1}$, so an
$O(1)$-accurate inverse of the dominant physics beats an exact inverse of the
wrong terms; flexible GMRES tolerates inexact, step-dependent application.

- **`jacobi` / `block_jacobi`** — (block-)diagonal scaling, $M = \mathrm{diag}(A)$
  or the block diagonal with batched LU {cite}`saad2003`.
- **`coarse_operator`** — the central physics pattern: precondition a hard
  operator with an *exact* solve of a simplified one (physics-coarsened or
  coupling-dropped), so $A A_s^{-1} = I + (A - A_s)A_s^{-1}$ and convergence is
  set by how much physics $A_s$ captures. This is the production "preconditioner
  matrix" (`Pmat`) idiom.
- **`line_smoother`** — alternating-direction block Jacobi:
  $x \leftarrow x + \omega_i M_i^{-1}(b - Ax)$ solving the strongly coupled
  direction exactly, the standard cure for anisotropy {cite}`trottenberg2001`.
- **`p_multigrid`** — a V-cycle over caller-supplied levels (pre-smooth,
  restrict, recurse, prolong, post-smooth); physics-agnostic, covering h- and
  p-/spectral coarsening alike {cite}`trottenberg2001`.
- **`mixed_precision`** — run any preconditioner in low precision (see
  {mod}`solvax.refine`).
- **`kronecker_nkp` / `nearest_kronecker`** — invert $A \otimes B$ with two
  small solves, with the factors extracted from a dense matrix by the
  Van Loan-Pitsianis rearrangement (the nearest Kronecker product is the
  leading singular triplet of a permuted matrix) {cite}`vanloan1993`.

## Fixed-point acceleration — {mod}`solvax.fixed_point`

For a contractive map $G(x)$, vector Aitken acceleration updates a safeguarded
scalar relaxation from successive residuals $r_k = G(x_k)-x_k$. This targets
expensive partitioned multiphysics iterations where each map evaluation is a
converged subsystem solve and unaccelerated fixed-point convergence is stiff.

- **`aitken_fixed_point(mapping, x0, ...)`** reports the final iterate, true
  fixed-point residual, iteration count, convergence flag, and relaxation.
- **`anderson_mixing(iterates, residuals, ...)`** provides a JIT-compatible,
  regularized affine history update for application loops with expensive maps
  and application-specific stopping policies.
- **`aitken_relaxation(previous_residual, residual, ...)`** exposes the same
  safeguarded scalar update for applications that own their coupled loop.
- Relaxation bounds make denominator breakdown and noncontractive transients
  finite and explicit rather than silently producing NaNs.
- The loop is `jit`/`vmap` compatible. For accepted gradients, combine the
  primal solver with {func}`~solvax.implicit.root_solve` so differentiation is
  implicit rather than through iteration-count branching.

*Use case:* accelerate coupled field/fluid, radiation/material, or other
partitioned steady solves without embedding application physics in SOLVAX.

## Implicit differentiation — {mod}`solvax.implicit`

For a parameterized solve $A(\theta)x = b(\theta)$, the implicit function
theorem gives the VJP from a single *transposed* solve,

$$A^\top \lambda = \bar x, \qquad \bar b = \lambda, \qquad \bar A = -\lambda x^\top,$$

and for a root $x^\ast$ of $f(x, \theta) = 0$,
$\mathrm{d}x^\ast/\mathrm{d}\theta = -(\partial f/\partial x)^{-1}(\partial f/\partial \theta)$.
The forward solver is a black box — it may iterate to any tolerance, restart,
precondition — and the adjoint costs exactly one extra solve regardless of the
iteration count {cite}`blondel2022,skene2026`.

- **`linear_solve(matvec, b, solver, ...)`** wraps `jax.lax.custom_linear_solve`.
- **`root_solve(f, x0, solver, ...)`** wraps `jax.lax.custom_root`.

*Use case:* differentiate through a converged equilibrium/steady state built
from any of the solvers above, keeping $O(1)$ memory in the forward iteration
count.

## Memory-chunked autodiff — {mod}`solvax.autodiff`

`jax.jacfwd` / `jax.jacrev` evaluate all $n$ (or $m$) directional derivatives in
one `vmap`, replicating the intermediate program state across the full Jacobian
width. Chunking splits the basis into blocks of `chunk_size`, vmaps each block
and walks the blocks with `jax.lax.map`, so

$$\text{memory} \sim m_0 + m_1\,\texttt{chunk\_size}, \qquad
\text{time} \sim t_0 + t_1\,\frac{n}{\texttt{chunk\_size}},$$

a knob between the fast/hungry `chunk_size = n` (plain `jacfwd`/`jacrev`) and
the lean/slow `chunk_size = 1`. The chunked Jacobian is numerically identical to
the JAX builder — the same JVP/VJP is evaluated for every basis vector, only the
batching changes.

- **`chunked_jacfwd` / `chunked_jacrev`** — column-chunked forward mode (tall
  Jacobians, small input) and row-chunked reverse mode (wide Jacobians, small
  output). Output shape follows the JAX convention `output_shape + input_shape`.
- **`chunked_jacobian(..., mode="fwd"|"rev"|"auto")`** — dispatcher; `"auto"`
  picks the mode with fewer basis vectors.
- **`auto_chunk_size`** — the `chunk_size="auto"` policy: the largest chunk that
  fits an explicit or device-reported memory budget, else a
  $\lceil\sqrt{\dim}\rceil$ heuristic that balances peak memory against the
  number of chunks.
- **`chunk_map`** — the underlying `vmap`/`lax.map` selector, useful on its own.

This is the reusable analogue of DESC's `jac_chunk_size` optimization-memory
option {cite}`panici2023`, shared here across kinetic and equilibrium codes.

*Use case:* optimization Jacobians of a large parameter vector, and
materializing matrix-free operators, that would otherwise exceed accelerator
memory.

## Mixed-precision refinement — {mod}`solvax.refine`

Given an approximate solver $M \approx A^{-1}$ carried out in *low* precision,
iterative refinement recovers high-precision accuracy by defect correction,

$$r_i = b - A x_i,\qquad d_i = M r_i,\qquad x_{i+1} = x_i + d_i,$$

with the residual accumulated in high precision. Each sweep contracts the error
by roughly $u_f\,\kappa(A)$ and converges to working-precision accuracy when
$\kappa(A)u_f < 1$ {cite}`carson2018`. `as_low_precision` wraps a solver to run
in a lower dtype internally — the standard way to exploit fast float32/float16
hardware while keeping a float64 result.

## Native host bridge — {mod}`solvax.native`

A non-differentiable escape hatch to SciPy's SuperLU for general sparse systems
outside the structured solvers. It runs on the host CPU, entirely outside the
JAX trace, so it must **not** be called under `jit`, `vmap` or `grad` (a guard
raises a clear error otherwise). Factor once with `SpluFactorization`, solve
many; `splu_solve` is the one-shot wrapper.
