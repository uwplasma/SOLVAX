# Special functions API

```{eval-rst}
.. automodule:: solvax.special
   :members:
   :member-order: bysource
```

## Conventions and accuracy

`ellipk(m)` and `ellipe(m)` take the parameter `m = k**2`, as SciPy's `ellipk` and `ellipe` do, not the modulus
`k`. `plasma_dispersion` is the Landau-continued function of Fried and Conte (1961), analytic in the whole plane;
below the real axis it differs from the Cauchy integral by `2 i sqrt(pi) exp(-zeta**2)`.

| Function | Method | Measured float64 error | Reference |
|---|---|---|---|
| `carlson_rf`, `carlson_rd` | duplication, 16 fixed steps, fifth-order series (Carlson 1995) | 1e-14 relative | SciPy `elliprf`, `elliprd` |
| `ellipk`, `ellipe` | `R_F(0, 1-m, 1)`, `K - m R_D(0, 1-m, 1)/3` | 6e-15 relative, m in [-100, 1 - 1e-12] | arithmetic-geometric mean |
| `bessel_j0`, `bessel_j1` | Cephes rational forms (Moshier, BSD-3 via SciPy): in `x**2` up to 5, Hankel form with rational P, Q above | 3.3e-16 absolute, `abs(x) <= 60`, 1e6, 1e8, 1e15; 0 at `inf` | 40-digit mpmath |
| `faddeeva` | Weideman (1994) N = 40 below `abs(z) = 8`, continued fraction above, reflection below the axis | 1e-13 relative in modulus, `abs(Re z), abs(Im z) <= 30` | SciPy `wofz`, mpmath |

Derivatives are closed forms: `dK/dm = (K - R_D/3) / (2 (1 - m))`, `dE/dm = -R_D / 6`, `J0' = -J1`,
`J1' = J0 - J1/x`, `w' = -2 z w + 2 i / sqrt(pi)`; second derivatives follow from the same rules. In float32
expect a few 1e-6 relative.

Limits: `m > 1` gives NaN and complex `m` raises `TypeError`; `dE/dm` at `m = 1` (true value `-inf`) is finite
and not meaningful. On the real axis `Re w = exp(-x**2)` and `Im Z = sqrt(pi) exp(-x**2)` hold to 1e-14
relative; just off the axis (`0 < abs(Im z) <= 1e-2`) the error of `Re w` is 6e-17 absolute, so where
`Re w` is itself that small (`abs(Re z) > 4`) it has few correct digits. Derivatives of `w` are NaN where
`w` overflows below the axis.

`bessel_j0` and `bessel_j1` have the call signature of the hand-written functions in GKX's `core.velocity`
(accurate to 4.9e-9 absolute). Under `jit` on 1e6 points in [0, 100] (Apple M-series CPU, 2 threads,
load about 10) `bessel_j0` takes 3.4-3.9 ms against 1.7 ms for GKX's on JAX 0.12.1 and 1.3-1.7 ms against
1.1-1.8 ms on JAX 0.6.2; `jax.scipy.special.bessel_jn` takes 45-57 ms.
