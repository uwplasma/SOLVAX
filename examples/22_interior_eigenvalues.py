"""Interior eigenvalues of a matrix-free operator, and their gradients.

Many stability problems ask for an eigenvalue that is *interior*: small in
magnitude compared with the spectral radius, because the spectrum is dominated
by fast oscillatory modes that carry no information about stability. Plain
Arnoldi cannot find those — its Ritz values approximate the periphery, so it
returns the fastest modes no matter how large the subspace grows.

`harmonic_krylov_schur` combines Krylov-Schur restarting with harmonic Ritz
extraction about a target, which reaches interior eigenvalues without the large
linear solves a shift-and-invert transformation would need. `eigenvalue` wraps
it with an analytic derivative, so a growth rate can be optimized directly.

Expected runtime: under a minute on a laptop CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

import solvax as sx

jax.config.update("jax_enable_x64", True)


def operator_with_spectrum(eigenvalues, seed=0):
    """Dense operator with a prescribed spectrum, for checking against truth."""
    rng = np.random.default_rng(seed)
    n = len(eigenvalues)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return jnp.asarray(q @ np.diag(eigenvalues) @ q.conj().T)


# --- 1. An interior eigenvalue hidden behind fast oscillatory modes. ---------
# One weakly unstable mode at 0.5 + 0.2i; everything else is stable but
# oscillates ~100x faster. This is the structure of a linearized kinetic or
# fluid operator, where |Im| >> |Re| across the spectrum.
n = 300
rng = np.random.default_rng(0)
target = 0.5 + 0.2j
bulk = (rng.standard_normal(n - 1) * 0.3 - 1.0) + 1j * rng.standard_normal(n - 1) * 60
spectrum = np.concatenate([[target], bulk])
a = operator_with_spectrum(spectrum)
v0 = jnp.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))

print(f"spectral radius {np.abs(spectrum).max():.1f}, wanted |lambda| {abs(target):.2f}")
print(f"ratio {np.abs(spectrum).max() / abs(target):.0f} — the wanted mode is deep inside\n")

solution = sx.harmonic_krylov_schur(
    lambda v: a @ v, v0, sigma=target, k=1, m=32, tol=1e-9,
    max_restarts=250, which="target",
)
found = complex(solution.eigenvalues[0])
print(f"harmonic Krylov-Schur : {found.real:+.10f}{found.imag:+.10f}i")
print(f"exact                 : {target.real:+.10f}{target.imag:+.10f}i")
print(f"residual {float(solution.residuals[0]):.2e}   restarts {solution.restarts}"
      f"   matvecs {solution.matvecs}   orthogonality {solution.orthogonality:.1e}\n")

# What plain Rayleigh-Ritz returns on a comparable subspace, for contrast: the
# peripheral modes, off by two orders of magnitude.
peripheral = sx.harmonic_krylov_schur(
    lambda v: a @ v, v0, sigma=0.0, k=1, m=32, tol=1e-9,
    max_restarts=5, which="largest_real",
)
print(f"5 restarts chasing largest-real instead: {complex(peripheral.eigenvalues[0]):.4f}")
print("   — the periphery, which is why the target matters\n")


# --- 2. Structured state: the operator never forms a matrix. -----------------
# A caller with a multidimensional state passes it through unchanged; only the
# matrix-vector product is required.
shape = (8, 4)
diagonal = jnp.asarray(
    np.concatenate(
        [
            [0.4 + 0.1j],
            (rng.standard_normal(31) - 2.0)
            + 1j * rng.standard_normal(31) * 20,
        ]
    )
).reshape(shape)


def structured(state):
    """A diagonal plus a shift — assembled the way a discretized PDE would be."""
    return diagonal * state + 0.05 * jnp.roll(state, 1, axis=0)


structured_solution = sx.harmonic_krylov_schur(
    structured, jnp.ones(shape, dtype=complex), sigma=0.4 + 0.1j, k=1, m=16,
    tol=1e-10, max_restarts=100, which="target",
)
print(f"structured operator: lambda = {complex(structured_solution.eigenvalues[0]):.8f}")
print(f"   eigenvector shape {structured_solution.eigenvectors.shape} — state shape preserved\n")

# A recycled block keeps competing branches distinct. Supplying a shifted
# inverse here makes the *search space* rational; the reported residuals remain
# those of the original diagonal operator.
cluster = jnp.asarray(
    [0.50 + 0.20j, 0.46 + 0.18j, *(-1.0 + 1j * np.arange(2, 32))]
)
cluster_start = jnp.ones(cluster.shape, dtype=complex)
cluster_shift = 0.48 + 0.20j
cluster_solution = sx.block_harmonic_krylov(
    lambda v: cluster * v,
    cluster_start,
    subspace_apply=lambda v: v / (cluster - cluster_shift),
    sigma=cluster_shift,
    k=2,
    m=10,
    block_size=3,
    tol=1e-10,
    max_restarts=5,
)
print(f"block candidates: {np.asarray(cluster_solution.eigenvalues)}")
print(f"   original-operator residuals {np.asarray(cluster_solution.residuals)}\n")


# --- 3. Differentiating the growth rate. ------------------------------------
# The derivative uses  dlambda = (w^H dA v) / (w^H v)  rather than
# differentiating the restarted iteration, so its cost is one extra solve for
# the left eigenvector plus one JVP of the operator — independent of how many
# restarts convergence needed.
small = 120
base = operator_with_spectrum(
    np.concatenate([[target], (rng.standard_normal(small - 1) * 0.3 - 1.0)
                    + 1j * rng.standard_normal(small - 1) * 5])
)
perturbation = jnp.asarray(
    rng.standard_normal((small, small)) + 1j * rng.standard_normal((small, small))
) * 0.01
start = jnp.asarray(rng.standard_normal(small) + 1j * rng.standard_normal(small))
options = dict(sigma=target, m=24, tol=1e-11, max_restarts=200, which="target")


def growth_rate(theta):
    """Real part of the eigenvalue — a stability objective."""
    return jnp.real(
        sx.eigenvalue(theta, lambda p: (lambda x: (base + p * perturbation) @ x),
                      start, **options)
    )


analytic = float(jax.grad(growth_rate)(0.0))
step = 1e-5
difference = float((growth_rate(step) - growth_rate(-step)) / (2 * step))
print(f"d(growth rate)/dtheta  analytic    {analytic:+.12e}")
print(f"                       finite diff {difference:+.12e}")
print(f"                       agreement   {abs(analytic - difference) / abs(difference):.2e}")

# The paired API also returns the differentiable right eigenvector. Its phase is
# arbitrary, so downstream objectives should use invariant combinations such
# as normalized quadratic or quartic mode weights.
pair_value, pair_vector = sx.eigenpair(
    jnp.asarray(0.0),
    lambda p: (lambda x: (base + p * perturbation) @ x),
    start,
    **options,
)
participation = jnp.sum(jnp.abs(pair_vector) ** 4) / jnp.sum(
    jnp.abs(pair_vector) ** 2
) ** 2
print(f"phase-invariant mode participation {float(participation):.8e}")
print(f"paired eigenvalue {complex(pair_value):+.8f}")
