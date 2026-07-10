"""Host-side SuperLU bridge for general sparse systems.

`solvax.native` is a non-differentiable escape hatch to SciPy's SuperLU for
sparse systems that fall outside the structured solvers. It runs on the host
CPU, entirely outside the JAX trace — so it must NOT be called under jit, vmap,
or grad (a guard raises a clear error if you try). Factor once, solve many.

Requires SciPy: `pip install solvax[native]`.
Expected runtime: about a second on a laptop CPU.
"""

import numpy as np
import scipy.sparse as sp

import solvax as sx

n = 500
rng = np.random.default_rng(0)
A = sp.random(n, n, density=0.01, format="csr", random_state=rng) + 5.0 * sp.eye(n)
b = rng.standard_normal(n)

# Factor once, reuse across right-hand sides.
lu = sx.SpluFactorization(A)
x1 = lu.solve(b)
x2 = lu.solve(2.0 * b)
print(f"reused factorization residual: {np.linalg.norm(A @ np.asarray(x1) - b):.2e}")
print(f"linearity check (x2 == 2 x1) : {np.allclose(np.asarray(x2), 2.0 * np.asarray(x1))}")

# One-shot convenience wrapper.
x = sx.splu_solve(A, b)
print(f"one-shot solve residual      : {np.linalg.norm(A @ np.asarray(x) - b):.2e}")
