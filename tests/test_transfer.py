"""Tests for solvax.transfer: grid transfers, their adjointness and accuracy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvax import (
    coarse_axis_size,
    coarse_grid_shape,
    coarsenable,
    coarsening_plan,
    grid_transfer,
    prolongation_matrix,
    restriction_matrix,
)

jax.config.update("jax_enable_x64", True)

# (boundary, fine length) triples that satisfy each closure's parity rule.
CLOSURES = (("periodic", 16), ("dirichlet", 15), ("reflective", 16))


@pytest.mark.parametrize(("boundary", "n_fine"), CLOSURES)
def test_full_weighting_is_half_the_prolongation_transpose(boundary, n_fine):
    """The variational relation R = c P^T, with c = 1/2 per coarsened axis."""
    restrict = restriction_matrix(n_fine, boundary=boundary)
    prolong = prolongation_matrix(n_fine, boundary=boundary)
    n_coarse = coarse_axis_size(n_fine, boundary=boundary)

    assert restrict.shape == (n_coarse, n_fine)
    assert prolong.shape == (n_fine, n_coarse)
    np.testing.assert_allclose(restrict, 0.5 * prolong.T, atol=1e-15)


@pytest.mark.parametrize(("boundary", "n_fine"), CLOSURES)
def test_injection_pair_is_exactly_adjoint(boundary, n_fine):
    restrict = restriction_matrix(n_fine, kind="injection", boundary=boundary)
    prolong = prolongation_matrix(n_fine, kind="injection", boundary=boundary)
    np.testing.assert_allclose(restrict, prolong.T, atol=1e-15)
    # Injection samples one fine point per coarse point and nothing else.
    assert int(jnp.count_nonzero(restrict)) == restrict.shape[0]


@pytest.mark.parametrize(("boundary", "n_fine"), CLOSURES)
def test_restriction_preserves_constants(boundary, n_fine):
    """Rows sum to one, so a constant residual restricts to the same constant."""
    for kind in ("full_weighting", "injection"):
        restrict = restriction_matrix(n_fine, kind=kind, boundary=boundary)
        np.testing.assert_allclose(restrict.sum(axis=1), 1.0, atol=1e-15)


@pytest.mark.parametrize(("boundary", "n_fine"), (("periodic", 16), ("reflective", 16)))
def test_linear_prolongation_preserves_constants(boundary, n_fine):
    prolong = prolongation_matrix(n_fine, boundary=boundary)
    n_coarse = coarse_axis_size(n_fine, boundary=boundary)
    np.testing.assert_allclose(prolong @ jnp.ones(n_coarse), 1.0, atol=1e-15)


def test_dirichlet_prolongation_interpolates_against_the_zero_boundary():
    """A constant is not a Dirichlet function: the end points see the zero
    boundary value and correctly interpolate halfway down to it."""
    n_fine = 15
    prolong = prolongation_matrix(n_fine, boundary="dirichlet")
    interpolated = prolong @ jnp.ones(coarse_axis_size(n_fine, boundary="dirichlet"))
    np.testing.assert_allclose(interpolated[1:-1], 1.0, atol=1e-15)
    np.testing.assert_allclose(interpolated[jnp.array([0, -1])], 0.5, atol=1e-15)


def test_periodic_stencils_wrap_around():
    """The periodic full-weighting stencil reaches across the seam."""
    restrict = restriction_matrix(8, boundary="periodic")
    np.testing.assert_allclose(restrict[0], [0.5, 0.25, 0, 0, 0, 0, 0, 0.25], atol=1e-15)
    prolong = prolongation_matrix(8, boundary="periodic")
    np.testing.assert_allclose(prolong[:, 0], [1.0, 0.5, 0, 0, 0, 0, 0, 0.5], atol=1e-15)
    # Without the wrap the last fine point would receive nothing at all.
    np.testing.assert_allclose(prolong.sum(axis=1), 1.0, atol=1e-15)


def test_dirichlet_prolongation_is_exact_on_piecewise_linear_functions():
    """Interior unknowns with zero outside: interpolation of a coarse-grid
    piecewise-linear function that respects the boundary condition is exact."""
    n_fine = 31
    n_coarse = coarse_axis_size(n_fine, boundary="dirichlet")
    fine_h = 1.0 / (n_fine + 1)
    fine_x = fine_h * jnp.arange(1, n_fine + 1)
    coarse_x = 2.0 * fine_h * jnp.arange(1, n_coarse + 1)
    tent = lambda x: jnp.minimum(x, 1.0 - x)  # zero at both boundaries  # noqa: E731
    prolong = prolongation_matrix(n_fine, boundary="dirichlet")
    np.testing.assert_allclose(prolong @ tent(coarse_x), tent(fine_x), atol=1e-15)


def test_periodic_prolongation_is_second_order_on_smooth_modes():
    """Linear interpolation of a resolved mode converges at O(h^2)."""
    errors = []
    for n_fine in (32, 64, 128):
        n_coarse = coarse_axis_size(n_fine, boundary="periodic")
        fine = jnp.cos(2.0 * jnp.pi * jnp.arange(n_fine) / n_fine)
        coarse = jnp.cos(2.0 * jnp.pi * jnp.arange(n_coarse) / n_coarse)
        interpolated = prolongation_matrix(n_fine, boundary="periodic") @ coarse
        errors.append(float(jnp.max(jnp.abs(interpolated - fine))))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.5 <= ratio <= 4.5 for ratio in ratios), ratios


def test_reflective_transfers_are_cell_centered_with_a_mirror_closure():
    restrict = restriction_matrix(8, boundary="reflective")
    # Interior rows carry the cell-centered full weighting [1/8, 3/8, 3/8, 1/8];
    # the boundary row folds the mirrored weight back onto the first cell.
    np.testing.assert_allclose(restrict[1, 1:5], [0.125, 0.375, 0.375, 0.125], atol=1e-15)
    np.testing.assert_allclose(restrict[0, :3], [0.5, 0.375, 0.125], atol=1e-15)
    prolong = prolongation_matrix(8, boundary="reflective")
    np.testing.assert_allclose(prolong[0], [1.0, 0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(prolong[1], [0.75, 0.25, 0.0, 0.0], atol=1e-15)


def test_grid_transfer_is_adjoint_in_n_dimensions():
    """<R x, y> = 2^-d <x, P y> over the d coarsened axes, in N-D."""
    shape = (8, 5, 15, 3)
    coarsen = (True, False, True, False)
    boundary = ("periodic", "periodic", "dirichlet", "reflective")
    restrict, prolong = grid_transfer(shape, coarsen, boundary=boundary)
    coarse = coarse_grid_shape(shape, coarsen, boundary=boundary)
    assert coarse == (4, 5, 7, 3)

    rng = np.random.default_rng(0)
    fine = jnp.asarray(rng.standard_normal(shape))
    coarse_vector = jnp.asarray(rng.standard_normal(coarse))
    left = jnp.vdot(restrict(fine), coarse_vector)
    right = 0.25 * jnp.vdot(fine, prolong(coarse_vector))
    assert float(left) == pytest.approx(float(right), rel=1e-12)


def test_grid_transfer_leaves_masked_axes_untouched():
    """An unmasked axis is never contracted: no axis mask means the identity."""
    shape = (6, 4)
    rng = np.random.default_rng(1)
    fine = jnp.asarray(rng.standard_normal(shape))

    identity_restrict, identity_prolong = grid_transfer(shape, (False, False))
    np.testing.assert_array_equal(identity_restrict(fine), fine)
    np.testing.assert_array_equal(identity_prolong(fine), fine)

    # Coarsening only axis 0 acts column by column, exactly as the 1-D matrix.
    restrict, _ = grid_transfer(shape, (True, False))
    expected = restriction_matrix(shape[0]) @ fine
    np.testing.assert_allclose(restrict(fine), expected, atol=1e-15)


def test_grid_transfer_carries_trailing_field_axes():
    shape = (8, 4)
    restrict, prolong = grid_transfer(shape, (True, True))
    rng = np.random.default_rng(2)
    fields = jnp.asarray(rng.standard_normal((*shape, 3)))
    transferred = restrict(fields)
    assert transferred.shape == (4, 2, 3)
    for field in range(3):
        np.testing.assert_allclose(
            transferred[..., field], restrict(fields[..., field]), atol=1e-15
        )
    assert prolong(transferred).shape == (*shape, 3)


def test_grid_transfer_is_a_pytree_and_jit_transparent():
    shape = (8, 8)
    restrict, prolong = grid_transfer(shape, (True, True))
    fine = jnp.ones(shape)
    compiled = jax.jit(lambda value: restrict(value))
    np.testing.assert_allclose(compiled(fine), restrict(fine), atol=1e-15)

    # The transfers are equinox modules, so they may cross a jit boundary as
    # arguments rather than only as closures.
    applied = jax.jit(lambda operator, value: operator(value))(restrict, fine)
    np.testing.assert_allclose(applied, restrict(fine), atol=1e-15)
    assert jax.tree.leaves(restrict)

    # Differentiation flows through the contraction.
    objective = lambda value: jnp.sum(prolong(restrict(value)) ** 2)  # noqa: E731
    gradient = jax.grad(objective)(fine)
    assert gradient.shape == shape
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_transfers_respect_the_requested_dtype():
    restrict = restriction_matrix(8, dtype=jnp.float32)
    assert restrict.dtype == jnp.float32
    prolong = prolongation_matrix(8, dtype=jnp.float32)
    assert prolong.dtype == jnp.float32


def test_coarse_axis_size_and_coarsenable():
    assert coarse_axis_size(16) == 8
    assert coarse_axis_size(15, boundary="dirichlet") == 7
    assert coarse_axis_size(16, boundary="reflective") == 8
    assert coarsenable(16, min_size=8)
    assert not coarsenable(16, min_size=9)
    assert not coarsenable(15)  # odd length, periodic closure
    assert coarsenable(15, boundary="dirichlet", min_size=7)
    assert not coarsenable(16, boundary="dirichlet")


def test_coarsening_plan_stops_axes_independently():
    plan = coarsening_plan(
        (64, 6, 33), (True, True, True), levels=4,
        boundary=("periodic", "periodic", "dirichlet"), min_size=4,
    )
    assert plan.shapes == ((64, 6, 33), (32, 6, 16), (16, 6, 16), (8, 6, 16), (4, 6, 16))
    # Axis 1 stops at 6 -> 3 < min_size; axis 2 becomes even and cannot halve
    # again under the dirichlet closure; axis 0 keeps going alone.
    assert plan.masks == (
        (True, False, True),
        (True, False, False),
        (True, False, False),
        (True, False, False),
    )
    assert len(plan.shapes) == len(plan.masks) + 1


def test_coarsening_plan_truncates_when_nothing_can_coarsen():
    plan = coarsening_plan((8, 8), (False, False), levels=3)
    assert plan.shapes == ((8, 8),)
    assert plan.masks == ()

    exhausted = coarsening_plan((8,), (True,), levels=5, min_size=4)
    assert exhausted.shapes == ((8,), (4,))


def test_semicoarsening_keeps_the_masked_axis_at_full_resolution():
    plan = coarsening_plan((32, 32), (True, False), levels=3)
    assert [shape[1] for shape in plan.shapes] == [32, 32, 32, 32]
    assert [shape[0] for shape in plan.shapes] == [32, 16, 8, 4]


def test_transfer_validation():
    with pytest.raises(ValueError, match="unknown boundary"):
        coarse_axis_size(8, boundary="neumann")
    with pytest.raises(ValueError, match="even axis length"):
        coarse_axis_size(7)
    with pytest.raises(ValueError, match="odd axis length"):
        coarse_axis_size(8, boundary="dirichlet")
    with pytest.raises(ValueError, match="unknown restriction"):
        restriction_matrix(8, kind="linear")
    with pytest.raises(ValueError, match="unknown prolongation"):
        prolongation_matrix(8, kind="full_weighting")
    with pytest.raises(ValueError, match="one entry per axis"):
        coarse_grid_shape((8, 8), (True,))
    with pytest.raises(ValueError, match="one entry per axis"):
        grid_transfer((8, 8), (True,))
    with pytest.raises(ValueError, match="one entry per axis"):
        grid_transfer((8, 8), (True, True), boundary=("periodic",))
    with pytest.raises(ValueError, match="one entry per axis"):
        coarsening_plan((8, 8), (True,), levels=1)
    with pytest.raises(ValueError, match="levels must be"):
        coarsening_plan((8,), (True,), levels=-1)
    restrict, _ = grid_transfer((8, 8), (True, True))
    with pytest.raises(ValueError, match="leading axes"):
        restrict(jnp.ones((4, 4)))
    with pytest.raises(ValueError, match="leading axes"):
        restrict(jnp.ones(8))
