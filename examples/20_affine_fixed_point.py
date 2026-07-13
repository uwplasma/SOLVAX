"""Matrix-free FGMRES for a weakly contractive affine coupling map.

Partitioned multiphysics couplings often produce an affine fixed-point map
``G(x) = L x + c`` whose spectral radius is just below one, so fixed-point
relaxation and even Anderson mixing converge slowly. `affine_fixed_point_gmres`
solves the equivalent linear system ``(I - L) x = c`` with matrix-free FGMRES,
evaluating only the map: no matrix or Jacobian is assembled. It returns a
`KrylovSolution` with the recomputed true residual.

Expected runtime: under a second on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)

n = 200
rng = np.random.default_rng(0)

# Random L scaled to spectral radius 0.98: contractive but only weakly.
raw = rng.standard_normal((n, n))
spectral_radius = np.max(np.abs(np.linalg.eigvals(raw)))
matrix = jnp.asarray(0.98 * raw / spectral_radius)
offset = jnp.asarray(rng.standard_normal(n))


def coupling_map(x):
    return matrix @ x + offset


solution = sx.affine_fixed_point_gmres(
    coupling_map,
    jnp.zeros(n),
    restart=60,
    rtol=1e-12,
)

reference = np.linalg.solve(np.eye(n) - np.asarray(matrix), np.asarray(offset))
matches = np.allclose(np.asarray(solution.x), reference, atol=1e-8)
print("converged:", bool(solution.converged))
print("GMRES iterations:", int(solution.iterations))
print("residual norm:", float(solution.residual_norm))
print("matches dense (I - L) solve:", bool(matches))

# Contrast: 200 plain fixed-point sweeps barely move at this spectral radius.
picard = jnp.zeros(n)
for _ in range(200):
    picard = coupling_map(picard)
picard_residual = float(jnp.linalg.norm(coupling_map(picard) - picard))
print("picard residual after 200 sweeps:", picard_residual)
