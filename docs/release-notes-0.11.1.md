# SOLVAX 0.11.1

A provenance release. `0.11.0` was tagged and published, and then five commits
landed while `__version__` still read `0.11.0` -- so one version string named two
different trees: the one on the index, and the one measurements were run
against. Anyone reproducing a number would have had no way to tell which. This
release gives the current tree its own version.

## The one library change

`solvax.native` detects JAX tracers without naming `jax.core.Tracer`. That
attribute is the documented check, but `jax.core` has been shrinking toward
private status across JAX releases and already resolves into `jax._src`. The
guard now uses it when it exists and falls back to the tracer protocol when it
does not, so a JAX upgrade that moves it produces the intended clear error
rather than an `AttributeError` raised from inside the guard.

No solver path changes. No numerical result changes. Every record produced
under `0.11.0` reproduces bit-identically under `0.11.1`.

## Tests and CI

- The degenerate tridiagonal paths (`n = 1`, `n = 2`) are covered directly
  rather than only through the general case.
- The JAX behaviours the library leans on -- tracer identity, `custom_vjp`
  being reverse-only, `scan` carry structure -- are pinned by explicit tests, so
  an upgrade that changes one of them fails here instead of somewhere subtler.
- Three suites were listed as explicit CI steps while `pytest -n auto` already
  collected them, running each twice; the macOS job hit its 15-minute timeout as
  a result. The duplication is gone, and the example-execution suite runs on one
  operating system rather than two.

## Upgrading

Nothing to do. `pip install --upgrade solvax`.
