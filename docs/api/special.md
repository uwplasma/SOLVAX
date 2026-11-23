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
| `bessel_j0`, `bessel_j1` | series below 2, Miller recurrence below 25, Hankel expansion above | 4.4e-16 absolute, `abs(x) <= 1e3` | SciPy `j0`, `j1`, `jv` |
| `faddeeva` | Weideman (1994) N = 40 below `abs(z) = 8`, continued fraction above, reflection below the axis | 8e-15 relative, `abs(Re z), abs(Im z) <= 30` | SciPy `wofz` |

Derivatives are closed forms: `dK/dm = (K - R_D/3) / (2 (1 - m))`, `dE/dm = -R_D / 6`, `J0' = -J1`,
`J1' = J0 - J1/x`, `w' = -2 z w + 2 i / sqrt(pi)`. In float32 expect about 1e-6 relative.

`bessel_j0` and `bessel_j1` take the same argument as the hand-written functions of GKX's
`core.velocity`, which are accurate to 4.9e-9 absolute.
