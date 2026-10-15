"""Block-Thomas Schur sweep: dense U@X versus Kronecker streaming apply.

Off-diagonal Legendre blocks of the DKES/PAS operator are a*S + b*M with
S = diag(w) (iota Dt (x) I + I (x) Dz) and M diagonal. The dense sweep pays
2 m^3 per gemm; the structured one 2 m^2 (Nt + Nz).
"""
import sys, time
import jax, jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve
jax.config.update("jax_enable_x64", True)

def spectral_d(n):
    k = jnp.fft.fftfreq(n, 1.0 / n) * 1j
    k = k.at[n // 2].set(0) if n % 2 == 0 else k
    e = jnp.eye(n)
    return jnp.real(jnp.fft.ifft(k[:, None] * jnp.fft.fft(e, axis=0), axis=0))

def problem(nt, nz, nl, key):
    m = nt * nz
    k1, k2, k3 = jax.random.split(key, 3)
    dt, dz = spectral_d(nt), spectral_d(nz)
    w = 1.0 + 0.1 * jax.random.normal(k1, (m,))
    mir = 0.2 * jax.random.normal(k2, (m,))
    iota = 0.87
    s = w[:, None] * (iota * jnp.kron(dt, jnp.eye(nz)) + jnp.kron(jnp.eye(nt), dz))
    l = jnp.arange(nl, dtype=jnp.float64)
    a_lo, b_lo = l / (2 * l - 1 + 1e-30), l * (l - 1) / (2 * l - 1 + 1e-30)
    a_up, b_up = (l + 1) / (2 * l + 3), -(l + 1) * (l + 2) / (2 * l + 3)
    diag = l[:, None] * (l[:, None] + 1) * 1e-2 + 1.0 + 0.1 * jax.random.normal(k3, (nl, m))
    return dict(dt=dt, dz=dz, w=w, mir=mir, iota=iota, s=s, diag=diag,
                a_lo=a_lo, b_lo=b_lo, a_up=a_up, b_up=b_up)

def s_apply(p, x):  # S @ x for x (m, r) without forming S
    y = x.reshape(p["dt"].shape[0], p["dz"].shape[0], -1)
    y = p["iota"] * jnp.einsum("ab,bzr->azr", p["dt"], y) + jnp.einsum("zy,ayr->azr", p["dz"], y)
    return p["w"][:, None] * y.reshape(x.shape)

def sweep(p, structured):
    s, mir, diag = p["s"], p["mir"], p["diag"]
    def off(a, b, x):
        return a * (s_apply(p, x) if structured else s @ x) + b * mir[:, None] * x
    def blk_lower(k):
        return p["a_lo"][k] * s + p["b_lo"][k] * jnp.diag(mir)
    def step(carry, k):
        lu = carry
        x = lu_solve(lu, blk_lower(k + 1))
        dk = jnp.diag(diag[k]) - off(p["a_up"][k], p["b_up"][k], x)
        new = lu_factor(dk)
        return new, new[0][0, 0]
    nl = diag.shape[0]
    last = lu_factor(jnp.diag(diag[-1]))
    lu, tr = jax.lax.scan(step, last, jnp.arange(nl - 1)[::-1])
    return lu[0], tr

def main():
    nt, nz, nl = (int(a) for a in sys.argv[1:4])
    reps = 5
    p = problem(nt, nz, nl, jax.random.PRNGKey(0))
    out = {}
    for name, st in (("dense", False), ("kron", True)):
        f = jax.jit(lambda pp, st=st: sweep(pp, st))
        t0 = time.perf_counter(); r = jax.block_until_ready(f(p)); cold = time.perf_counter() - t0
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(f(p)); ts.append(time.perf_counter() - t0)
        out[name] = r
        print(f"{jax.devices()[0].platform} m={nt*nz} nl={nl} {name}: cold {cold:.3f} s, warm min {min(ts):.4f} s median {sorted(ts)[reps//2]:.4f} s")
    err = float(jnp.max(jnp.abs(out["dense"][0] - out["kron"][0])) / jnp.max(jnp.abs(out["dense"][0])))
    print(f"max rel diff of final Schur LU: {err:.2e}")
    assert err < 1e-11

if __name__ == "__main__":
    main()
