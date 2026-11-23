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
| `bessel_j0`, `bessel_j1` | series below 2, Miller recurrence below 25, Hankel expansion above | 1.3e-15 absolute, `abs(x) <= 1e3` | 30-digit mpmath |
| `faddeeva` | Weideman (1994) N = 40 below `abs(z) = 8`, continued fraction above, reflection below the axis | 8e-15 relative, `abs(Re z), abs(Im z) <= 30` | SciPy `wofz` |

Derivatives are closed forms: `dK/dm = (K - R_D/3) / (2 (1 - m))`, `dE/dm = -R_D / 6`, `J0' = -J1`,
`J1' = J0 - J1/x`, `w' = -2 z w + 2 i / sqrt(pi)`. The identity `Z' = -2 (1 + zeta Z)` holds to 1.3e-12 relative
at `abs(zeta) = 64`, where `1 + zeta Z` cancels to 1e-4. In float32 expect about 1e-6 relative.

JAX itself: `jax.scipy.special.wofz` is absent in 0.4.38 and 0.6.2 and in 0.12.1 errs by 3.1e-13 relative (it is
Weideman's N = 32 form); `jax.scipy.special.bessel_jn` is accurate to 5e-16 below `x = 20` but errs by 4e-6 at
`x = 40` and by 19 at `x = 1e3`. `bessel_j0` and `bessel_j1` take the same argument as the hand-written
functions of GKX's `core.velocity`; those are accurate to 4.9e-9 absolute, which is the largest difference between
the two.
