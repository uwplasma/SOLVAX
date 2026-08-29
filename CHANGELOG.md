# Changelog

## Unreleased

### Fixed-work implicit loops

- Added opt-in fixed-length scan control to GMRES and Newton--Krylov, with
  converged slots masked and the existing early-exit defaults unchanged.
- Made happy breakdown and zero-residual norm/Givens paths carry finite zero
  cotangents, so fixed-work solves compose with reverse mode inside an outer
  `lax.scan`.
- Pinned float64 agreement with the early-exit solve to `1e-14` relative and
  checked scan-embedded gradients against dense or analytic references.

## 0.18.0 - 2026-08-28

### Globalized nonlinear continuation

- Added JIT-able pseudo-transient JFNK with a user metric, shifted
  preconditioner, hard admissibility/backtracking gates, switched-evolution
  pseudo-time updates, and safeguarded Eisenstat--Walker forcing.
- Shared the public forcing primitive with adaptive ordinary Newton--Krylov.
- Added fixed-size true-residual histories and explicit accepted/rejected,
  nonlinear-evaluation, linear-failure, and Krylov-work diagnostics.
- Added host-orchestrated adaptive branch continuation with complete stage
  records and square pseudo-arclength bordered residual/corrector helpers.
- Rebalanced hosted CI into exhaustive current-stack shards and focused
  compatibility/backend lanes, retaining the 95% branch-coverage gate while
  reducing the complete hosted evidence cycle below ten minutes.

## 0.17.0 - 2026-08-22

### Implicit affine fixed-point derivatives

- Upgraded `affine_fixed_point_gmres` in place to use
  `jax.lax.custom_linear_solve`. JVPs and VJPs now require one tangent or
  transposed FGMRES solve and never record the primal Krylov iterations.
- Added independent transpose preconditioner, tolerance, and restart controls
  while retaining the existing matrix-free affine-map API, PyTree support,
  diagnostics, and result type.
- Verified analytical parameter gradients, explicit implicit-solve structure,
  and reverse temporary memory independent of the maximum restart count. A
  representative 8,192-unknown primal remains within 1.3% of the previous warm
  runtime.

## 0.16.0 - 2026-08-22

### Memory-bounded recurrence adjoints

- Added `checkpointed_fori_loop`, an exact finite-recurrence JVP/VJP with a
  two-level `O(N / C + C)` retained-state bound and a square-root default.
  It accepts array or pytree state and keeps loop bounds and checkpoint width
  explicit static controls.
- Verified primal, JVP, and VJP equality to the uncheckpointed recurrence and
  compiled reverse temporary memory below half of a full recurrence tape.
- Made periodic-Poisson spacing traceable so compiled geometry derivatives can
  use the reusable Fourier symbol while eager invalid spacing still fails
  clearly.
- Documented the derivative policy for raw iterative primals, implicit solves,
  checkpointed recurrences, localized rules, and reverse-only APIs.

## 0.15.0 - 2026-08-22

### Fully periodic Poisson inversion

- Added an N-dimensional Fourier Poisson solve with explicit zero-mode
  projection, selectable solution mean, real/complex support, and separate
  reusable-symbol and spectral-coefficient entry points for timestep loops.
- Verified manufactured modes, nullspace handling, shape contracts, JIT, and
  reverse-mode differentiation. The README and elliptic guide document both
  the direct and transform-reuse APIs.

## 0.14.0 - 2026-08-21

### Stationary fixed-point iteration

- Added `fixed_point_iteration`, a JAX-native relaxed iteration with one map
  evaluation per update, optional application-defined residual norm, explicit
  convergence diagnostics, and either tolerance-based stopping or a static
  step count for differentiable unrolled algorithms.
- Documented the selection boundary among plain iteration, Aitken, Anderson,
  affine-map FGMRES, and implicit differentiation in the README and solver
  guide.
- Verified the release on Python 3.10 and 3.12, current/minimum/optional JAX
  stacks, Linux and macOS, with 98.16% combined coverage.

## 0.13.0 - 2026-08-04

### Reusable block-Thomas factors without the off-diagonal bands

- `block_thomas_factor_fn` gains `store_offdiagonals=False`, which returns a
  `GeneratedBlockTridiagFactors` holding only the Schur LU factors and their
  pivots. `block_thomas_solve` accepts it exactly like `BlockTridiagFactors`
  and rebuilds `L_k` and `U_k` from the same `block_fn`, one block at a time,
  during each substitution sweep. Both the primal and the `transpose=True`
  solve are supported and stay exact.
- Reusable factors nominally hold three `(N, m, m)` arrays; only the Schur LU
  is irrecoverable. Dropping the other two cuts retained state to a third, and
  to a sixth against float64 bands when `factor_dtype=jnp.float32` puts the LU
  in single precision — the two options compose. Measured at `N=48`, `m=24` as
  the summed byte count of the factor pytree's leaves: 668160 bytes of stored
  bands against 225792 and 115200, or 0.3379 and 0.1724 of the full-band state,
  the excess over 1/3 and 1/6 being the pivot indices, which both policies keep.
  On the scale where that decides something: a family of
  chains whose three float64 bands come to 53.3 GB retains 17.8 GB as Schur LU
  alone, and 8.9 GB with a float32 LU — the difference between fitting on a
  24 GB device and not.
- The cost is generator evaluations, not arithmetic: `block_fn` is called once
  per index to factor and twice per index per solve. Triangular-solve and
  matrix-product counts are unchanged, and the elimination still runs exactly
  once, so this remains a factor-once/solve-many path. It is not a second
  `block_thomas_checkpointed_fn`, which re-eliminates on every application and
  is therefore unusable as a preconditioner.
- Both sweeps thread the right-hand side through the scan *carry* and take only
  the row index and that row's Schur factors as scan inputs. This is required,
  not cosmetic: on every JAX release before 0.10.0 a `lax.scan` cannot be
  transposed if a value linear in the differentiated input arrives as a scan
  input — the transpose rule classifies every input of a plainly traced scan as
  a residual and then asserts that no residual is an undefined primal, which a
  two-line cumulative-sum scan is enough to trip. A linear carry has always
  been transposable. Reverse mode is unaffected either way, because partial
  evaluation sets the classification honestly there, so the symptom is confined
  to `jax.linear_transpose`. XLA fuses the carry's per-step read and write into
  an in-place slice update, so no sweep does work proportional to the block
  count per step, and the solve holds one right-hand-side-shaped buffer instead
  of three. `block_thomas_solve`'s unrolled sweeps exist for the same
  underlying reason; the comment there now says so.
- `jax.grad` and `jax.linear_transpose` therefore behave as they do on stored
  bands. Tests compare gradients against the stored-band path, primal and
  transposed; state the adjoint property directly as
  `<A^-T u, v> == <u, A^-1 v>` so it is pinned by arithmetic rather than by
  JAX's transposition machinery; assert structurally that no
  right-hand-side-shaped floating input reappears in either scan; and compare
  residuals rather than solution vectors on a deliberately near-singular pinned
  chain, where a solution comparison would measure conditioning instead of the
  code.

## 0.12.0 - 2026-08-02

### Fixed

- The Thomas pivot guard was a fixed `1e-12`, neither scale- nor
  dtype-invariant: enormous beside a float32 system of order one, negligible
  beside a float64 system scaled to `1e12`. It is now `sqrt(eps)` of the
  working dtype times the coefficient scale, reduced along the solve axis only
  so a sharded batch stays collective-free -- and so each system in a batch is
  guarded against its own coefficients rather than the largest anywhere.
- The Fourier--Helmholtz solver hard-cast its geometry to `float64`, silently
  upgrading an x64-disabled program, or meaning nothing at all with x64 off. It
  now infers the working precision from the caller and rejects complex
  geometry explicitly.
- `nystrom_preconditioner`'s null-direction test compared against exact zero.
  On a rank-deficient operator those eigenvalues emerge from an SVD as rounding
  noise, so the result depended on the linear-algebra backend: the same zero
  operator gave the identity under one JAX release and arbitrary O(1) values
  under another. The test is now relative, with a floor at the construction's
  own resolution.

### Added

- `certified_adjoint_window` returns the smallest window whose relative
  gradient error is *provably* within a requested tolerance. This is the
  certificate `localization_crossover_window` has always declined to give, and
  it is available because the exact-window rule leaves exactly one
  approximation to bound: the source cotangent and every retained row cotangent
  are exact blocks of the full solve, so the tail identity holds with equality
  and each of its factors has a computable envelope. Making the statement
  *relative* needs a lower bound on the gradient, which norms of the summands
  cannot supply -- they cannot see cancellation -- so one windowed gradient
  supplies it through the reverse triangle inequality. Conservative by two and a
  half to seven orders of magnitude depending on the family, and a chain that
  does not localize gets the full window rather than a plausible guess.
- `plan_chain_windows` for batches whose chains localize at different rows.
  `adjoint_window` is static, so one `vmap` runs the whole batch at the widest
  window any chain needs; on a collisionality scan the criterion gives up on the
  least collisional chain and drags everything to the full window. Padding to
  the maximum does not fix that -- it is the problem -- so the planner cuts the
  batch into a few static buckets by dynamic programming over the distinct
  window values. Measured on a 24-chain scan: four buckets remove 51% of the
  retained rows against a per-chain ideal of 59%.
- `chain_window=` on `block_thomas_truncated_fn`, a traced per-chain window
  under a static bound. It changes which rows contribute, not the retained
  state, and is bit-identical to the uniform path when the two agree.
- Forward-mode differentiation through the windowed rule now raises an error
  that names what to do instead. JAX's own message is true and useless -- "can't
  apply forward-mode autodiff (jvp) to a custom_vjp function" -- and a code
  whose gradients are reverse but whose audits are forward needs to be told
  which entry point supports both.
- `nystrom_preconditioner` reports `spectrum_span`, the smallest retained
  eigenvalue over the largest, so a caller can see whether the sketch reached
  into the decaying tail or sat on a flat plateau. `nystrom_preconditioner_adaptive`
  grows the rank until that estimate clears a target and returns the rank used.
- `RECYCLE_DRIFT_ADVISORY`, a calibrated threshold for when to drop a recycle
  pair, with the measurements behind it in the docstring. The calibration says
  something worth knowing: reuse keeps paying up to a drift of about 0.6, far
  further than a cautious guess would suggest, and the penalty for keeping a
  stale pair is a few percent rather than a factor.
- A scope-and-scaling suite covering tridiagonal behaviour across eight decades
  of scaling, the complex banded path at sizes up to 512 with several
  bandwidths, dtype promotion at the public entry points, and the forward-mode
  message.

### Fixed

- `nystrom_preconditioner` returned `NaN` for a zero or rank-deficient
  operator. The stabilizing shift is proportional to the sketch norm, so an
  operator with no range got no shift, the core Cholesky was singular, and the
  triangular solve divided by it -- before `mu` was ever used, which is why a
  positive `mu` did not help either. The shift now has a floor, and the
  eigenvalue scaling takes the null-space limit (leave the direction alone)
  instead of evaluating 0/0.
- The banded static-pivot clamp took its sign from the real part alone, so a
  small complex pivot was rotated onto the real axis: `1e-18j` became a purely
  real `1e-12`. It now clamps the magnitude and keeps the direction, which
  reduces to the previous behaviour for real pivots.
- `gcrot`'s recycle-drift diagnostic averaged per-column projected residual
  norms and called that the mean sine of the principal angles. It is not: the
  principal angles are a property of the two subspaces, while that average
  changes under a unitary remixing of the same subspace (measured: 1e-4 on
  random 4-dimensional subspaces of R^40). It is now the mean of the singular
  values of the projected residual, which are the sines themselves -- and which
  do not lose half their digits for nearly equal subspaces, the regime the
  diagnostic exists for.

### Added

- A `types` CI job running mypy. The package advertises `Typing :: Typed` and
  ships `py.typed`, so downstream type-checkers trust these annotations;
  nothing verified them until now. Nine modules with pre-existing findings are
  listed in `[tool.mypy]` so the gate holds the line rather than blocking on
  debt, and everything else -- including `direct.py` -- is checked.

- Add `shard_batch`, an explicit `jax.shard_map` wrapper for independent
  numerical batches. It fixes the input batch axis and one output batch axis
  to a named mesh axis, with validation for ranks and mesh names.
- Add an eight-device test that checks the per-device batch size, output
  sharding, numerical result, and zero-collective HLO.

### Differentiable externally solved eigenpairs

- `eigenpair_reverse` differentiates a caller-certified right/left eigenpair
  implicitly, including eigenvector observables, an application-supplied
  transposed reduced-resolvent solve, and an exceptional-point condition guard.
- Propagator helpers provide RK4 stability estimation, continuous-operator
  residual stopping, and multi-candidate extraction without prescribing an
  application eigensolver.
- A bounded sparse bridge -- `sparse_operator_matrix`, `sparse_eigenpairs` and
  `SpluFactorization` -- converts each JAX column block straight to CSR instead
  of materializing a dense n-by-n host matrix, and lets one shifted LU serve
  both the right and the conjugate-transpose Arnoldi so implicit reverse mode
  does not refactor the same operator. It is eager and CPU-factorized: JAX
  supplies the operator actions and parameter derivatives, the native primal is
  an external certified solve, and it refuses to be traced rather than
  pretending otherwise.
- The exceptional-point condition gate measures what it can see, not the true
  condition number. On a deliberately near-defective pair at `eps = 1e-18` the
  true condition number was 5e8 while the gate measured 4.5e7, and the
  derivative came out 0.955 against an analytic 0.5. It guards the obvious
  degeneracy and is not a certificate of well-conditioning.

## 0.11.2 - 2026-07-28

### An archivable release

- The library code is **identical to 0.11.1**. There is no behaviour change, no
  bug fix and no new feature; every number produced under `0.11.1` is
  reproduced bit-for-bit here.
- `0.11.1` could not be archived. The Zenodo integration created a record and
  then failed to fetch the source archive from GitHub -- `Record '21656172' has
  no file 'uwplasma/SOLVAX-v0.11.1.zip'` -- during a window when Zenodo was
  rate-limiting this network. Because the record exists, every retry of the
  release webhook now returns `409 Conflict`, so that version cannot be
  archived under its own DOI.
- A citable archive matters here: the release is what a paper's availability
  statement points at, and pointing at a version with no archive would be a
  claim that is not true. `0.11.2` exists to be archivable.
- If you are already on `0.11.1` there is no reason to upgrade beyond citing a
  DOI that resolves.

## 0.11.1 - 2026-07-28

### A version number that names one tree

- Five commits landed after `0.11.0` was tagged while `__version__` still read
  `0.11.0`, so the same version string described two different trees: the one
  published to the index and the one measurements were then run against. That
  is a small thing until someone tries to reproduce a number, at which point it
  is the whole problem. This release gives the current tree its own version.
- The only library change since `0.11.0` is in `solvax.native`, which now
  detects JAX tracers without reaching for `jax.core.Tracer`. `jax.core` has
  been shrinking toward private status across releases; the guard degrades to
  the tracer protocol if it moves, rather than raising `AttributeError` from
  inside a check whose job is to produce a clear error. No solver path and no
  numerical result changes.
- The rest is test and CI work: coverage of the degenerate tridiagonal paths
  (`n = 1`, `n = 2`), explicit tests for the JAX seams the library depends on,
  and a CI fix -- three suites were being collected twice, which pushed the
  macOS job into its timeout.

## 0.11.0 - 2026-07-28

### The residual diagnostic works on the exact-window path

- `block_thomas_truncated_fn_with_residual` gains `params` and
  `adjoint_window`. Supplying `params` selects the exact-window reverse rule
  for the solution instead of taping the elimination; the residual comes back
  from the *same* forward sweep, so asking for both no longer costs two.
- Before this, a caller who wanted the Schur-system residual and a bounded
  reverse pass had to run the elimination twice. Measured on a stellarator
  drift-kinetic chain (`N_xi = 128`, `m = 81`) that cost 1.7x on a
  forward-only solve --- the case where the residual is usually the only thing
  wanted. It is now 18.0 -> 18.5 ms at unchanged memory.
- At full window the solution and the residual are bitwise identical to the
  taped path. The gradient goes from 90.6 to 50.6 ms and 41.6 to 33.2 MiB; at
  the advised window, 9.4 MiB and flat in `N_xi`.
- The residual is returned through `stop_gradient`. It is a diagnostic, and
  the reverse rule ignores its cotangent --- which is sound only because that
  cotangent is structurally zero. A test pins `d(residual)/dp == 0`.

### Known limitation, now documented

- The exact-window rule is a `custom_vjp`, so JAX cannot push **forward-mode**
  autodiff through it: `jax.jacfwd` and `jax.jvp` raise rather than falling
  back. Integrating it into a production solver made this concrete --- a code
  whose gradients are reverse but whose derivative audits are forward must
  keep an unwindowed path, and applying the window is a per-call-site choice
  rather than a global switch. This is stated in the docstrings and in the
  release notes; it is not new behaviour, only newly written down.

## 0.10.1 - 2026-07-28

Corrections to 0.10.0. Nothing in the exact-window algorithm changed; what
changed is that the documented ways of calling it now work.

### `LocalizationWindow` can be passed where the docstring says it can

- `LocalizationWindow` defines `__index__` so the advisor's result can be handed
  straight back to the solver as `adjoint_window`. Both public entry points
  compared it against a bound *before* coercing it, so the documented call
  raised `TypeError`. Coercion now happens first, through `operator.index`, at
  `block_thomas_truncated` and `block_thomas_truncated_fn` alike. A float is
  still rejected, now with a message that says why.
- The only test covering this checked `int(advice) == advice.window` rather than
  the call the method exists for. There are now tests that pass the record
  through both entry points, under `jit`, and through a gradient.

### Documentation that runs

- The README's gradient example differentiated a closed-over solution, so it
  returned zeros. It now builds the objective inside the function being
  differentiated, and shows `check_localized_gradient` alongside it.
- The README reused `lower`/`diag`/`upper` for both block and scalar
  tridiagonal shapes, and `block_fn` for both the plain generator `row(j)` and
  the parameterized `block_fn(params, j)`. Both are disambiguated.
- `examples/25_localized_adjoint_window.py` passed the advisor's record where an
  integer was wanted, named a file that does not exist in its `Run:` line, and
  described the advised window more strongly than the advisor warrants.
- `block_thomas_truncated`'s argument documentation still described the
  superseded leading-window re-solve that 0.10.0 replaced.
- The 0.10.0 release notes named a field `profile` that is called
  `primal_profile`, and said `check_localized_gradient` compares against the
  full-window gradient; it compares against a *wider* window.

### The gate that would have caught all of it

- `tests/test_documentation_runs.py` executes every README python block and
  every example in a subprocess against the tree under test, checks that the
  documented `LocalizationWindow` fields exist, checks that no documented
  gradient is identically zero, and checks that every `Run:` line names a file
  that exists. CI runs it. Linting alone passed all six defects above.

## 0.10.0 - 2026-07-27

### Uniform `adjoint_window` semantics across both entry points

- `block_thomas_truncated` (stored bands) now routes its bands through the same
  generated construction as `block_thomas_truncated_fn`, so both entry points
  produce **bitwise-identical** finite-window gradients and `adjoint_window`
  has one meaning across the API. Previously the array path closed the window
  with a leading-principal subsystem, whose retained states are not blocks of
  the full solution; that closure is kept privately and exercised only as an
  ablation in the test suite.
- This is a behaviour change at finite window for the stored-band path. Full
  window is unaffected: it was exact before and is exact now.

### The window advisor is explicitly an estimate

- `suggest_adjoint_window` is renamed `localization_crossover_window` and now
  returns a `LocalizationWindow` dataclass carrying the per-row transfer
  profile, the crossover row, and `certified=False`. The name and the field
  both say what the old name did not: the value is a diagnostic read off the
  chain's transfer norms, not a window certified against a tolerance.
- `suggest_adjoint_window` remains as a deprecated alias returning the integer
  window, and will be removed in 0.12.0.
- `check_localized_gradient` is added for confirming a chosen window against
  the full-window gradient on a problem small enough to afford it.

### Optional chunking backend

- The `adv-jax-math` chunking backend is optional and selected at runtime;
  `available_backends()` reports what is installed. SOLVAX pins no JAX version
  and no Python version on its account, and the default install does not
  require the extra. CI runs a dedicated job with the extra installed and
  asserts the backend is actually reachable before exercising it.

### Notes

- 0.9.1 was published without a changelog entry, so two code states could both
  report `0.9.1` while differing in their exact-window semantics and public
  API. This release restores the correspondence between version and behaviour;
  anything depending on the finite-window gradient should pin `>=0.10.0`.

## 0.9.0 - 2026-07-25

### Exact-window localized adjoint (breaking behaviour change)

- The generated-block parameter VJP behind `block_thomas_truncated_fn(...,
  params=, adjoint_window=)` is now an **exact-window** construction. The
  previous release reconstructed the retained primal and adjoint states from
  the leading principal submatrix; those states are not blocks of the full
  solution, so the gradient carried an interface-induced error *inside* the
  retained window in addition to the omitted tail. **The API is unchanged, so
  code written against 0.8.x silently gets the stronger guarantee.**
- What is now exact at every window: the selected forward blocks, the source
  cotangent `bar b = P_K lambda` (including `w = 0`), every retained row
  cotangent `j < W`, any parameter whose generator derivative vanishes above
  `W`, and the full-window gradient. The only approximation is the omission of
  operator rows `j >= W`.
- Measured against the superseded closure, the advantage grows with the window
  (2x at `w = 0`, 24x at `w = 4`) because the closure's interface error decays
  more slowly than the true tail. The old path is retained privately as
  `_leading_principal_params_bar` for ablation only.
- `_block_thomas_selected_fn_state` decouples the source support from the
  retained length so the reverse rule can request its one-block halo. Both
  sweeps still visit all `N` rows: the method is memory-localized, not
  work-localized.
- Complex systems: the convention is established by explicit tests against
  ordinary reverse mode rather than inferred from the real case.

### Graded localization and a-priori window selection

- `localization_profile_fn` returns the exact per-row factors
  `rho_k = ||Delta_k^{-1} L_k||` from the Schur recursion the forward solve
  already sweeps, so the diagnostic costs only the norm estimates.
- `suggest_adjoint_window` turns the profile into a window: retained rows must
  reach past the row where `rho_k` first falls below one. If the chain never
  localizes within its length the full window is returned.
- A single uniform rate is the wrong model for chains whose coupling and
  diagonal scale differently with the row index; fitting one to the leading
  rows reports the non-localized head and underestimates the window's value.


- Added `solvax.transfer`: separable restriction and prolongation operators for
  structured grids, built as one small matrix per coarsened axis and applied as
  per-axis contractions, so an N-D transfer never materializes an N-D operator
  and unmasked axes are not contracted at all. Full-weighting and injection
  restriction, linear and injection prolongation, and periodic, dirichlet and
  reflective closures; the default pair satisfies `R = 2^-d P^T` exactly over
  the `d` coarsened axes. `coarsening_plan` schedules a semicoarsening
  hierarchy, stopping each axis as its parity or minimum size runs out.

- Added `solvax.smoothers`: relaxation sweeps in the multigrid
  `smooth(matvec, x, b)` protocol — damped point and block Jacobi, batched
  tridiagonal line relaxation (periodic or not), exact banded plane relaxation
  over two axes, upwind-ordered sweeps whose ordering follows a caller-supplied
  advection field, and their multiplicative composition. `smoothing_factor`
  measures Brandt's high-frequency error-reduction rate by power iteration on
  the Fourier-projected error propagation operator. Measured against local
  Fourier analysis in the test suite: damped Jacobi at `2/3`, line relaxation
  on an anisotropic operator at `weak / (strong + weak)`, and an upwind sweep
  beating every wind-agnostic ordering by a factor that grows with the mesh
  Peclet number (over 200x at Pe = 100).

- Extended the multigrid cycle. `multigrid` takes an explicit level list and
  adds V-, W- and F-cycles, a general per-level cycle index, independent pre-
  and post-smoothing counts, and a configurable damped-Jacobi weight;
  `p_multigrid` keeps its signature and defaults and forwards the new options.
  `semicoarsening_hierarchy` assembles the levels for a structured grid from
  the transfers plus a caller-supplied *rediscretization* of the operator on
  each coarse grid, rather than a Galerkin triple product. Measured
  h-independence on an advection-dominated anisotropic operator: 0.0064 to
  0.0072 residual reduction per cycle from 32x16 to 256x16, with
  multigrid-preconditioned GMRES converging in 2 iterations at every size
  while unpreconditioned GMRES does not converge at all.

- `gmres` and `gcrot` now iterate arrays of any rank in their own layout
  instead of requiring a flat vector, so a multidimensional state never has to
  be raveled and unraveled around each operator application. Flat `(n,)`
  arrays keep the existing matrix path, and GCROT's recycle pair — a basis
  rather than a state — stays in flat `(n, k)` columns.

- Added `recycle_strategy="harmonic"` to `gcrot`: harmonic-Ritz deflated
  restarting (GCRO-DR) over the space augmented by the current recycle pair,
  alongside the existing one-direction-per-cycle FIFO update. The eigenproblem
  is small and dense and the new pair is reconstructed from the Arnoldi
  relation, so deflated restarting costs no extra operator applications. On a
  spectrum with six eigenvalues decades below the bulk it converges in 42
  matvecs against 90 for FIFO, where unpreconditioned GMRES does not converge.

- Added `recycled_linear_solve`: the same one-transposed-solve implicit
  adjoint as `linear_solve`, but the forward solve's Krylov recycle space is
  handed to the adjoint solve rather than discarded. Measured on a spectrum
  with eight small outliers: 11 adjoint inner iterations with reuse against 47
  without, at identical gradients.

- Added `block_thomas_checkpointed_fn`, an exact generated full-system solve
  that recomputes each radial segment during substitution. Its default
  square-root checkpoint spacing reduces factor storage from `O(N m^2)` to
  `O(sqrt(N) m^2)` while preserving JIT, JVP, and VJP through an implicit
  primal/transpose solve.
- Made automatic Jacobian chunking square-root bounded on every device.
  Device capacity alone no longer silently widens a batch; callers can still
  request the existing largest-fitting policy with `max_memory_bytes`.

## 0.8.7 - 2026-07-19

- Added the GPU measurement records (`benchmarks/results/gpu/`, 2x RTX A4000)
  and their docs columns: collective counts identical to the emulated-mesh
  schedule over NCCL; ideal weak scaling for single-reduction PCG (and a
  recorded non-scaling finding for the fused lax tridiagonal layout under
  sharding); the bounded adjoint flat on GPU; the physics-scale
  generated-block inversion whose N=4096, m=195 gradient runs in 33.7 MiB
  where the naive tape would exceed the 16 GB card; and the per-platform
  framing of the mixed-precision backward cost. New kinetic-inversion
  benchmark docs page.

- Added the baseline comparison benchmark (`benchmarks/benchmark_baselines.py`,
  `pip install solvax[bench]`): head-to-head against `jax.scipy`, `lineax`, and
  `scipy.sparse` on the problem suite at identical tolerance with no
  preconditioning, performance-profile ratios, and work-precision series
  including a solution-plus-gradient series through the implicit adjoint.
  SOLVAX matches the SciPy reference iteration counts exactly and runs within
  ~20% of the fastest JAX baseline at these sizes.

- `block_thomas_truncated_fn` gained a `params`/`adjoint_window` path with a
  structure-preserving custom VJP for generated blocks: the right-hand-side
  gradient is an exactly generated truncated solve of the transpose, and the
  `params` gradient pulls windowed band cotangents back through `block_fn`'s
  own derivative — forward and reverse both run at memory independent of the
  block count, with no band arrays materialized in either direction. Measured
  flat from N=32 to N=1024 (~30x less reverse scratch than the taped generated
  path at N=1024); full-window gradients are bitwise identical to the
  array-band bounded adjoint.

- Added the transport-inversion application benchmark
  (`benchmarks/benchmark_kinetic_inversion.py`): damped Newton recovers a
  collisionality profile exactly (quadratic convergence to 1.6e-14) from
  truncated low-moment observations of a spectral kinetic ladder, with the
  gradient and Hessian both flowing through
  `block_thomas_truncated(adjoint_window=w)` and validated against finite
  differences. The extended-profile misfit Hessian spectrum documents that
  the quadratic profile coefficient is unidentifiable from truncated moments,
  and the memory record separates the solve-tape savings of the bounded
  adjoint from the band-array floor of the array-band API.

- Added `nystrom_preconditioner`: a rank-`ell` randomized Nystrom
  preconditioner for SPD systems `(A + mu I) x = b`, built from `ell` operator
  applications with an explicit PRNG key — deterministic, jit-able, and
  differentiable through the sketch and eigenfactors. Bounds the
  preconditioned condition number by a small constant in expectation when the
  rank exceeds about twice the mu-effective dimension (Frangella, Tropp &
  Udell 2023); the scalable coarse correction when no grid hierarchy exists.

- Warm-started :func:`gcrot` reports `recycle_drift`, the mean principal-angle
  sine between the incoming recycle image space and its re-established span
  under the current operator — zero for an unchanged operator, growing
  linearly with the operator step, so optimization loops can monitor whether
  their step size keeps recycling effective. Joint primal+adjoint recycling
  along a continuation trajectory is demonstrated and measured in
  `benchmarks/benchmark_recycling.py`.

- Added the one-command reproduction driver (`python -m benchmarks.reproduce`):
  regenerates every committed measurement record after writing a
  hardware/software manifest and validating the timer against a known
  reference interval; `--quick` is CI-smoked. Tagged releases carry Zenodo
  metadata (`.zenodo.json`) so archives include the records.

- Benchmarks are now part of the documentation: a new Benchmarks section
  renders the committed measurement records (`benchmarks/results/*.json`) —
  bounded-memory adjoint scaling, mixed-precision adjoint accuracy and cost,
  communication accounting, and the problem-suite sweeps — each with its exact
  reproduce command and methodology notes, alongside a test-taxonomy page.

- Added the research-grade benchmark problem suite (`benchmarks/problems.py`):
  convection-diffusion (Peclet sweep), indefinite Helmholtz (wavenumber sweep),
  anisotropic diffusion (ratio sweep), Poisson, and the kinetic
  block-tridiagonal operator, each dense-verifiable; plus the sweep driver
  (`benchmarks/benchmark_sweeps.py`) recording iterations-to-tolerance,
  convergence, achieved residual, and warm wall time against the
  `jax.scipy.sparse.linalg` baselines at identical tolerance. CI smoke-runs the
  dense verification mode.

- Added a sharding and communication test suite on an eight-device emulated CPU
  mesh (`tests/test_sharding.py`), pinning sharding preservation through pytree
  Krylov solves and collective-operation counts of compiled primal and adjoint
  solves, plus `benchmarks/benchmark_collectives.py` and a sharding guide. The
  measured invariant: reverse-mode solves cost exactly one extra solve's worth
  of collectives, and sharded batched tridiagonal solves are collective-free in
  both directions.

- `mixed_precision_block_thomas` gained an opt-in `implicit_adjoint` custom VJP:
  the adjoint system is solved by the same working-precision refinement reusing
  the transposed low-precision factors — zero additional factorizations, no
  differentiation through the factorization, and the gradient inherits the
  refined forward error rather than the factorization precision.

- `block_thomas_truncated` gained an opt-in `adjoint_window` argument selecting a
  structure-preserving custom VJP: the right-hand-side gradient is the exact
  transposed truncated solve and the band gradients come from a leading
  `(keep_lowest + adjoint_window)`-block re-solve, so the *differentiated* solve
  runs at memory independent of the block count (versus the linear-in-`N` tape of
  plain reverse mode). Band-gradient error decays geometrically in the window.

## 0.8.6 - 2026-07-17

- `tridiagonal_solve` and `cyclic_tridiagonal_solve` accept complex operands:
  real bands with a complex right-hand side solve the real and imaginary parts
  independently (keeping real band storage and the fused accelerator kernel),
  while genuinely complex bands use the portable Thomas path. The fused
  primitive is wrapped in an implicit linear solve, so the `"lax"` backend is
  now forward- and reverse-differentiable.

## 0.8.5 - 2026-07-16

- Added `additive_tridiagonal_line_preconditioner` for differentiable additive
  line inverses over nonperiodic array axes and an optional cyclic final axis.
- `schur_projected_precond` accepts an optional border-border block
  `d_block`, generalizing the projected Schur preconditioner from the
  saddle-point case `[[A, B], [C, 0]]` to a general bordered matrix
  `[[A, B], [C, D]]` with `S = C A^{-1} B - D`.

## 0.8.4 - 2026-07-15

- Extended `linear_solve` with an independent `transpose_solver` and optional
  `has_aux` diagnostics while preserving implicit JVP and VJP behavior.
- Exposed safeguarded Anderson weights for reuse across differently shaped
  coupled-state histories.

## 0.8.3 - 2026-07-14

- Added `additive_preconditioner`, a positive weighted combination of
  inverse actions for symmetry-preserving additive line, block, and Schwarz
  preconditioning on arrays or arbitrary PyTrees.

## 0.8.2 - 2026-07-14

- Added `galerkin_deflation`, a balanced symmetry-preserving Galerkin coarse
  correction for fixed SPD preconditioners used with PCG.

## 0.8.1 - 2026-07-13

- Added `solvax.elliptic`: a spectral Fourier--Helmholtz elliptic solve for
  separable Helmholtz-type problems on a periodic axis and a bounded axis
  (`build_fourier_helmholtz_operator`, `solve_fourier_helmholtz`,
  `FourierHelmholtzOperator`). Fourier-transforms the periodic axis and solves
  the remaining per-mode tridiagonal system in the bounded axis; `jit`/`grad`/
  `vmap` transparent. This is the `lap phi = rhs` inversion used by reduced
  drift-plane / vorticity models.

## 0.8.0 - 2026-07-13

- Extended FGMRES beyond flat arrays: `gmres` now solves scalar, array, and
  arbitrary matching-pytree operands through a leaf-wise Arnoldi basis (no
  `ravel_pytree`, preserving leaf-level sharding), and accepts an optional
  `inner_product` callback for weighted or mesh-wide (distributed) products.
  The optimized flat-array and GCROT paths are unchanged.
- Added `newton_krylov`, a matrix-free Jacobian-free Newton-Krylov (JFNK) root
  solver. Jacobian-vector products come from `jax.linearize`; each correction
  is solved by SOLVAX FGMRES. It supports array or pytree states, right
  preconditioning, custom inner products, an independent nonlinear norm, and
  reports separate nonlinear and linear convergence flags.
- Added `affine_fixed_point_gmres`, which solves an affine fixed-point map
  `G(x)=Lx+c` as the matrix-free system `(I-L)x=c`, and gave `anderson_mixing`
  optional spectral condition filtering of ill-conditioned histories.
- Added a batched, differentiable cyclic-tridiagonal solve that retains the
  hardware-aware Thomas/cuSPARSE backend through an exact rank-one
  (Sherman-Morrison) correction.
- `lu_solve_banded` now promotes a real right-hand side against complex factors
  instead of silently truncating the imaginary part.

## 0.7.0 - 2026-07-12

- Added opt-in single-reduction PCG for sharded systems. Its algebraically
  equivalent recurrence lets XLA batch per-iteration scalar products into one
  tuple all-reduce while retaining residual diagnostics and implicit gradients.

## 0.6.1 - 2026-07-11

- Mark the distributed package as PEP 561 typed so strict downstream type
  checking analyzes SOLVAX's annotated public API.

## 0.6.0 — 2026-07-11

- Added complex-valued GMRES/GCROT with scaled unitary Givens rotations and
  Hermitian Arnoldi/recycle projections.
- Added complex fixed-point acceleration with real Aitken safeguards and the
  Hermitian Anderson residual Gram matrix.
- Restored block-Thomas linear-transpose compatibility on current JAX while
  preserving reusable factors and warm-solve performance.
- Made Jacobi preconditioners explicit PyTrees so mixed-precision wrappers cast
  stored factor state as well as runtime vectors.
- Added supported-minimum/current JAX CI rows, manual draft-PR validation, GPU
  compatibility evidence, and a complex implicit-gradient example.

## 0.5.1 — 2026-07-11

- Added `pcg_linear_solve`, which retains fixed-shape primal diagnostics while
  applying an implicit VJP with independently controlled transpose solves.

## 0.5.0 — 2026-07-11

- Added matrix-free preconditioned conjugate gradients on arbitrary JAX pytrees.
- Added fixed-shape residual histories and explicit convergence, iteration-limit,
  non-positive-curvature, nonfinite, and preconditioner-breakdown statuses.
- Added real/complex, x32/x64, JIT, scale-invariance, preconditioning, and
  implicit-gradient tests plus a cold/warm benchmark fixture.

## 0.4.0

- Added safeguarded Aitken and bounded-memory Anderson fixed-point acceleration.
