"""Lightweight, physics-agnostic linear-operator containers.

Thin equinox modules wrapping the *action* of a linear map, interoperable
with the plain-callable convention used throughout solvax: everything in
``solvax.krylov`` takes a ``matvec`` callable, and every operator here is
itself callable (``op(v) == op.matvec(v)``), so operators can be passed
directly as ``matvec=`` or ``precond=`` arguments. The single invariant all
operators maintain is adjoint consistency,

    <A x, y> = <x, A^T y>    for all x, y,

with ``A^T`` produced by the ``.T`` property.

Structure exploited per container:

- ``MatrixFreeOperator`` wraps an arbitrary linear callable; its transpose
  falls back on :func:`jax.linear_transpose` when no hand-written
  transpose is supplied.
- ``SumOperator`` applies ``(A_1 + ... + A_p) v = sum_i A_i v``; the
  transpose distributes over the sum.
- ``KroneckerOperator`` applies ``(A (x) B) v`` through the reshape
  identity ``(A (x) B) vec(X) = vec(B X A^T)`` (equivalently, with the
  row-major flattening used by ``jnp.reshape``, ``A X B^T``), so the
  ``(pr) x (qs)`` product is never formed: cost is two small matrix
  products instead of one huge one.
- ``BlockTridiagonalOperator`` stores dense bands ``(L_k, D_k, U_k)`` in
  the layout of ``solvax.direct`` and applies all blocks with batched
  einsums; ``to_blocks()`` feeds ``solvax.direct.block_thomas_factor``
  directly, pairing the operator with its natural preconditioner.
- ``BorderedOperator`` is the saddle-point (KKT-like) structure

      K = [[A, B],
           [C, 0]],       K [x, y] = [A x + B y, C x],

  arising when constraint or source rows border a physics block ``A``.
  :func:`schur_projected_precond` turns an approximate inverse of ``A``
  *alone* into a preconditioner for the full bordered system via the
  (small, dense) Schur complement ``S = C A^{-1} B``:

      y = S^{-1} (C A^{-1} r_x - r_y),    x = A^{-1} (r_x - B y),

  which is exactly ``K^{-1}`` when ``A^{-1}`` is exact — so a
  preconditioner built for the physics block preconditions the
  constrained system.

References
----------
- M. Benzi, G. H. Golub & J. Liesen, "Numerical solution of saddle point
  problems", Acta Numerica 14, 1 (2005) — block preconditioners and Schur
  complement methods for bordered/KKT systems.
- C. F. Van Loan, "The ubiquitous Kronecker product", J. Comput. Appl.
  Math. 123, 85 (2000) — the vec/reshape identity.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
  (2003), chapter 9 — preconditioned Krylov methods with matrix-free
  operators.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve


def _shape_of(op: Any) -> tuple[int, int]:
    """Return the ``(n_out, n_in)`` shape of an operator or dense matrix."""
    shape = getattr(op, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(
            "operator has no 2-D `shape`; wrap plain callables in "
            "MatrixFreeOperator(apply, shape=...) first"
        )
    return tuple(shape)


def _apply(op: Any, v: jax.Array) -> jax.Array:
    """Apply an operator, plain callable, or dense matrix to a vector."""
    return op(v) if callable(op) else op @ v


def _transpose(op: Any) -> Any:
    """Transpose an operator or dense matrix (both expose ``.T``)."""
    if not hasattr(op, "T"):
        raise ValueError(
            "operator has no `.T`; wrap plain callables in "
            "MatrixFreeOperator(apply, shape=...) first"
        )
    return op.T


def _materialize(op: Any) -> jax.Array:
    """Densify an operator (via ``materialize``) or pass a matrix through."""
    return op.materialize() if hasattr(op, "materialize") else jnp.asarray(op)


class _LinearOperator(eqx.Module):
    """Shared behaviour: operators are callables with a generic densifier."""

    def __call__(self, v: jax.Array) -> jax.Array:
        """Alias for :meth:`matvec`, so operators drop in as plain callables."""
        return self.matvec(v)

    def materialize(self) -> jax.Array:
        """Assemble the dense matrix by applying the operator to identity columns.

        Costs ``n_in`` operator applications (one vmapped batch) and
        ``O(n_out * n_in)`` memory — intended for small sizes, testing, and
        building coarse/direct preconditioners, not production solves.

        Returns:
            Dense matrix of shape ``self.shape``.
        """
        _, n_in = self.shape
        return jax.vmap(self.matvec, in_axes=1, out_axes=1)(jnp.eye(n_in))


class MatrixFreeOperator(_LinearOperator):
    """A linear operator defined only by its action ``v -> A v``.

    Attributes:
        apply: linear callable computing ``A @ v`` on flat ``(n_in,)``
            arrays; must be pure JAX (traceable).
        transpose_apply: optional callable computing ``A^T @ w``. When
            omitted, ``.T`` falls back on :func:`jax.linear_transpose`.
        shape: static ``(n_out, n_in)`` extent of the map (keyword-only).
    """

    apply: Callable
    transpose_apply: Callable | None = None
    shape: tuple[int, int] = eqx.field(static=True, kw_only=True)

    def matvec(self, v: jax.Array) -> jax.Array:
        """Apply the operator: ``A @ v``."""
        return self.apply(v)

    @property
    def T(self) -> MatrixFreeOperator:  # noqa: N802
        """The transposed operator ``A^T``.

        When ``transpose_apply`` was supplied it is used directly (and the
        forward ``apply`` becomes the transpose of the transpose, so
        ``op.T.T`` is free). Otherwise the transpose is derived with
        :func:`jax.linear_transpose`: each application then costs roughly
        one forward evaluation of ``apply`` (its jaxpr is transposed rule
        by rule), plus a re-trace per call outside ``jit`` — supply an
        explicit ``transpose_apply`` when the adjoint is hot.

        Returns:
            A :class:`MatrixFreeOperator` with shape ``(n_in, n_out)``.
        """
        if self.transpose_apply is not None:
            return MatrixFreeOperator(
                self.transpose_apply, self.apply, shape=(self.shape[1], self.shape[0])
            )
        apply = self.apply
        n_in = self.shape[1]

        def transposed(w: jax.Array) -> jax.Array:
            primal = jax.ShapeDtypeStruct((n_in,), w.dtype)
            (out,) = jax.linear_transpose(apply, primal)(w)
            return out

        return MatrixFreeOperator(transposed, apply, shape=(self.shape[1], self.shape[0]))


class SumOperator(_LinearOperator):
    """A sum of linear operators, applied term by term: ``(sum_i A_i) v``.

    Typical use: a structured principal part (e.g. a
    :class:`BlockTridiagonalOperator` that a direct solve can precondition)
    plus matrix-free perturbations or coupling terms.

    Attributes:
        terms: tuple of operators, plain linear callables, or dense
            matrices, all with the same ``(n_out, n_in)`` extent.
    """

    terms: tuple = eqx.field(converter=tuple)

    @property
    def shape(self) -> tuple[int, int]:
        """The common ``(n_out, n_in)`` shape, read off the first shaped term."""
        for term in self.terms:
            shape = getattr(term, "shape", None)
            if shape is not None and len(shape) == 2:
                return tuple(shape)
        raise ValueError(
            "no term exposes a 2-D `shape`; include at least one shaped "
            "operator or wrap a callable in MatrixFreeOperator"
        )

    def matvec(self, v: jax.Array) -> jax.Array:
        """Apply every term and sum: ``sum_i A_i v``."""
        results = [_apply(term, v) for term in self.terms]
        return jax.tree_util.tree_reduce(lambda x, y: x + y, results)

    @property
    def T(self) -> SumOperator:  # noqa: N802
        """The transpose, distributed over the terms: ``(sum A_i)^T = sum A_i^T``.

        Plain callables (no ``.T`` of their own) are wrapped in
        :class:`MatrixFreeOperator` with this operator's shape and
        transposed via :func:`jax.linear_transpose` — see
        :attr:`MatrixFreeOperator.T` for the cost.
        """
        shape = self.shape

        def transpose_term(term: Any) -> Any:
            if hasattr(term, "T"):
                return term.T
            return MatrixFreeOperator(term, shape=shape).T

        return SumOperator(tuple(transpose_term(term) for term in self.terms))


class KroneckerOperator(_LinearOperator):
    """The Kronecker product ``A (x) B`` applied without forming it.

    Uses the reshape identity ``(A (x) B) vec(X) = vec(B X A^T)``; with the
    row-major flattening of ``jnp.reshape`` this reads, for ``A`` of shape
    ``(p, q)`` and ``B`` of shape ``(r, s)``,

        (A (x) B) v = (A X B^T).reshape(-1),    X = v.reshape(q, s),

    so one matvec costs ``O(pqs + prs)`` instead of the ``O(pqrs)`` of the
    assembled ``(pr) x (qs)`` matrix.

    Attributes:
        a: left factor — an operator (anything with ``matvec``-style
            ``__call__``, ``shape``, ``.T``) or a dense matrix. Wrap plain
            callables in :class:`MatrixFreeOperator`.
        b: right factor, same protocol as ``a``.
    """

    a: Any
    b: Any

    @property
    def shape(self) -> tuple[int, int]:
        """``(p * r, q * s)`` for ``A`` of shape ``(p, q)``, ``B`` of ``(r, s)``."""
        (p, q), (r, s) = _shape_of(self.a), _shape_of(self.b)
        return (p * r, q * s)

    def matvec(self, v: jax.Array) -> jax.Array:
        """Apply ``(A (x) B) v`` via two small products on the reshaped vector."""
        (_, q), (_, s) = _shape_of(self.a), _shape_of(self.b)
        x = v.reshape(q, s)
        ax = jax.vmap(lambda col: _apply(self.a, col), in_axes=1, out_axes=1)(x)
        axbt = jax.vmap(lambda row: _apply(self.b, row))(ax)
        return axbt.reshape(-1)

    @property
    def T(self) -> KroneckerOperator:  # noqa: N802
        """``(A (x) B)^T = A^T (x) B^T``."""
        return KroneckerOperator(_transpose(self.a), _transpose(self.b))

    def materialize(self) -> jax.Array:
        """Assemble the dense product with :func:`jnp.kron` — small sizes only.

        Returns:
            Dense matrix of shape ``(p * r, q * s)``.
        """
        return jnp.kron(_materialize(self.a), _materialize(self.b))


class BlockTridiagonalOperator(_LinearOperator):
    """Block-tridiagonal operator over dense per-block bands.

    Row ``k`` of the action is ``L_k x_{k-1} + D_k x_k + U_k x_{k+1}``,
    computed for all blocks at once with batched einsums over shifted
    slices (no Python loop over blocks). The band layout matches
    ``solvax.direct``: ``lower[0]`` and ``upper[-1]`` are carried but
    ignored, so :meth:`to_blocks` feeds
    :func:`solvax.direct.block_thomas_factor` unchanged — the natural
    direct preconditioner for this operator.

    Attributes:
        lower: sub-diagonal blocks ``L_k``, shape ``(n_blocks, m, m)``;
            ``lower[0]`` is ignored.
        diag: diagonal blocks ``D_k``, shape ``(n_blocks, m, m)``.
        upper: super-diagonal blocks ``U_k``, shape ``(n_blocks, m, m)``;
            ``upper[-1]`` is ignored.
    """

    lower: jax.Array
    diag: jax.Array
    upper: jax.Array

    def __check_init__(self):
        if not (
            self.diag.ndim == 3
            and self.diag.shape[1] == self.diag.shape[2]
            and self.lower.shape == self.diag.shape
            and self.upper.shape == self.diag.shape
        ):
            raise ValueError(
                "lower, diag, upper must share shape (n_blocks, m, m); got "
                f"{self.lower.shape}, {self.diag.shape}, {self.upper.shape}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_blocks * m, n_blocks * m)``."""
        n_blocks, m, _ = self.diag.shape
        return (n_blocks * m, n_blocks * m)

    def matvec(self, v: jax.Array) -> jax.Array:
        """Apply the operator to a flat ``(n_blocks * m,)`` vector."""
        n_blocks, m, _ = self.diag.shape
        x = v.reshape(n_blocks, m)
        y = jnp.einsum("kij,kj->ki", self.diag, x)
        y = y.at[1:].add(jnp.einsum("kij,kj->ki", self.lower[1:], x[:-1]))
        y = y.at[:-1].add(jnp.einsum("kij,kj->ki", self.upper[:-1], x[1:]))
        return y.reshape(-1)

    @property
    def T(self) -> BlockTridiagonalOperator:  # noqa: N802
        """The transpose: bands swap and every block transposes.

        ``(A^T)_{k,k-1} = U_{k-1}^T`` and ``(A^T)_{k,k+1} = L_{k+1}^T``,
        implemented as a roll of the transposed bands (the wrapped-around
        ``lower[0]`` / ``upper[-1]`` entries land in the ignored slots).
        """
        swap = lambda blocks: jnp.swapaxes(blocks, -1, -2)  # noqa: E731
        return BlockTridiagonalOperator(
            lower=jnp.roll(swap(self.upper), 1, axis=0),
            diag=swap(self.diag),
            upper=jnp.roll(swap(self.lower), -1, axis=0),
        )

    def to_blocks(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(lower, diag, upper)`` for :func:`solvax.direct.block_thomas_factor`."""
        return (self.lower, self.diag, self.upper)

    def materialize(self) -> jax.Array:
        """Assemble the dense ``(n_blocks m) x (n_blocks m)`` matrix — small sizes only.

        Returns:
            Dense matrix with the three block bands scattered in place.
        """
        n_blocks, m, _ = self.diag.shape
        r = jnp.arange(n_blocks)[:, None] * m + jnp.arange(m)  # (n_blocks, m)
        dense = jnp.zeros((n_blocks * m, n_blocks * m), self.diag.dtype)
        dense = dense.at[r[:, :, None], r[:, None, :]].set(self.diag)
        dense = dense.at[r[1:, :, None], r[:-1, None, :]].set(self.lower[1:])
        dense = dense.at[r[:-1, :, None], r[1:, None, :]].set(self.upper[:-1])
        return dense


class BorderedOperator(_LinearOperator):
    """The bordered (KKT-like) operator ``[[A, B], [C, 0]]``.

    Acts on the concatenated vector ``[x, y]`` as
    ``[A x + B y, C x]`` — the structure of a physics block ``A``
    augmented with constraint rows ``C`` and source/coupling columns
    ``B`` (Benzi, Golub & Liesen 2005). Pair with
    :func:`schur_projected_precond` to recycle a preconditioner for ``A``
    on the full constrained system.

    Attributes:
        a: the ``(n, n)`` principal block — an operator or dense matrix
            (wrap plain callables in :class:`MatrixFreeOperator`).
        b_cols: border columns ``B``, shape ``(n, p)``.
        c_rows: border rows ``C``, shape ``(q, n)``.
    """

    a: Any
    b_cols: jax.Array
    c_rows: jax.Array

    def __check_init__(self):
        n_out, n_in = _shape_of(self.a)
        if self.b_cols.ndim != 2 or self.b_cols.shape[0] != n_out:
            raise ValueError(f"b_cols must have shape ({n_out}, p); got {self.b_cols.shape}")
        if self.c_rows.ndim != 2 or self.c_rows.shape[1] != n_in:
            raise ValueError(f"c_rows must have shape (q, {n_in}); got {self.c_rows.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_out + q, n_in + p)``."""
        n_out, n_in = _shape_of(self.a)
        return (n_out + self.c_rows.shape[0], n_in + self.b_cols.shape[1])

    def matvec(self, v: jax.Array) -> jax.Array:
        """Apply to the concatenated vector: ``[A x + B y, C x]``."""
        n_in = _shape_of(self.a)[1]
        x, y = v[:n_in], v[n_in:]
        return jnp.concatenate([_apply(self.a, x) + self.b_cols @ y, self.c_rows @ x])

    @property
    def T(self) -> BorderedOperator:  # noqa: N802
        """``[[A, B], [C, 0]]^T = [[A^T, C^T], [B^T, 0]]``.

        The border rows become the transposed columns and vice versa.
        """
        return BorderedOperator(_transpose(self.a), self.c_rows.T, self.b_cols.T)

    def materialize(self) -> jax.Array:
        """Assemble the dense bordered matrix — small sizes only.

        Returns:
            Dense matrix ``[[A, B], [C, 0]]``.
        """
        a = _materialize(self.a)
        zero = jnp.zeros((self.c_rows.shape[0], self.b_cols.shape[1]), a.dtype)
        return jnp.block([[a, self.b_cols], [self.c_rows, zero]])


def schur_projected_precond(
    a_inv: Callable, b_cols: jax.Array, c_rows: jax.Array
) -> Callable:
    """Preconditioner for a bordered system from an approximate inverse of ``A``.

    Given ``a_inv ~ A^{-1}`` for the principal block alone, forms the small
    dense Schur complement ``S = C A^{-1} B`` once (``p`` applications of
    ``a_inv`` to the columns of ``B``, then an LU factorization of the
    ``q x p`` result, which must be square) and returns the projection

        y = S^{-1} (C a_inv(r_x) - r_y),    x = a_inv(r_x - B y),

    i.e. the exact inverse of ``[[A, B], [C, 0]]`` with ``A^{-1}``
    replaced by ``a_inv`` throughout (Benzi, Golub & Liesen, Acta
    Numerica 2005, section 5). With ``a_inv`` exact, the preconditioned
    operator is the identity and GMRES converges in one iteration; with an
    approximate ``a_inv``, the border is still eliminated exactly through
    the projected Schur system, so a preconditioner built for the physics
    block ``A`` preconditions the full constrained system. Each
    application costs two calls to ``a_inv`` plus one small triangular
    solve.

    Args:
        a_inv: callable ``r -> A^{-1} r`` (approximate is fine) on flat
            ``(n,)`` arrays; must be linear and pure JAX.
        b_cols: border columns ``B``, shape ``(n, p)``.
        c_rows: border rows ``C``, shape ``(p, n)`` — the Schur complement
            must be square.

    Returns:
        A callable ``[r_x, r_y] -> [x, y]`` on concatenated ``(n + p,)``
        vectors, suitable as ``precond=`` for :func:`solvax.krylov.gmres`
        on the matching :class:`BorderedOperator`.
    """
    ainv_b = jax.vmap(a_inv, in_axes=1, out_axes=1)(b_cols)
    schur = c_rows @ ainv_b
    if schur.shape[0] != schur.shape[1]:
        raise ValueError(f"Schur complement must be square; got shape {schur.shape}")
    schur_lu = lu_factor(schur)
    n = c_rows.shape[1]

    def precond(r: jax.Array) -> jax.Array:
        r_x, r_y = r[:n], r[n:]
        y = lu_solve(schur_lu, c_rows @ a_inv(r_x) - r_y)
        x = a_inv(r_x - b_cols @ y)
        return jnp.concatenate([x, y])

    return precond
