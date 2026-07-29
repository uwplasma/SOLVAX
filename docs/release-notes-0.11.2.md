# SOLVAX 0.11.2

**The library code is identical to 0.11.1.** No behaviour change, no bug fix, no
new feature. Every number produced under `0.11.1` is reproduced bit-for-bit
here. This release exists for one reason: `0.11.1` cannot be archived.

## What happened

The Zenodo integration created a record for `v0.11.1` and then failed to fetch
the source archive from GitHub:

```
Record '21656172' has no file 'uwplasma/SOLVAX-v0.11.1.zip'
```

The failure was transient — Zenodo was rate-limiting requests from this network
at the time, returning `403 Access to this resource has been restricted due to
unusual traffic` and then gateway timeouts. But the record was already created,
so every subsequent retry of the release webhook returns `409 Conflict` and the
version can never be archived under its own DOI.

## Why it is worth a version number

A release is what a citation resolves to. SOLVAX is referenced from a
manuscript whose availability statement names the release that produced its
measurements, and naming a version with no archive would be a claim that is not
true. Rather than cite a DOI that does not resolve, or a different version from
the one measured, `0.11.2` gives the same code an archivable release.

## Upgrading

Nothing to do. If you are on `0.11.1` the only difference you will observe is
that `solvax.__version__` reports `0.11.2` and that a DOI for it exists.

```bash
pip install --upgrade solvax
```
