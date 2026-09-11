"""Dense-oracle and exact integer-kernel checks for fixed-cell recovery."""

import itertools

import numpy as np
import pytest

from ssa.cell_summary_minimax import (
    cell_summary_minimax, decode_cell_summary, summarize_cells,
)


@pytest.mark.parametrize("weights,ids,selected,cells", [
    ([], [], [], 3),
    ([.75, .25], [0, 0], [], 1),
    ([.75, .25], [0, 0], [0], 1),
    ([.2, .2, .2, .2], [1, 1, 1, 1], [], 3),
    ([9., -2., 0., 4., -3.], [0, 0, 0, 0, 0], [], 1),
    ([9., -2., 0., 4., -3.], [0, 1, 0, 1, 0], [3, 0], 4),
    ([9., -2., 0., 4., -3.], [0, 1, 0, 1, 0], [4, 0, 3, 1, 2], 4),
])
def test_sharp_witness_and_decoder(weights, ids, selected, cells):
    p = np.array(weights)
    profile = cell_summary_minimax(weights, ids, selected, cells=cells)
    h = profile["dual_witness"]
    assert h.dtype == np.int8
    assert np.all(np.abs(h) <= 1)
    np.testing.assert_array_equal(h[selected], 0)
    # Integer additions certify feasibility exactly, without a numerical rank test.
    for cell in range(cells):
        assert sum(int(h[i]) for i, label in enumerate(ids) if label == cell) == 0
    assert profile["kernel_exact"] and profile["selected_witness_exact"]
    assert float(p @ h) == pytest.approx(profile["radius"], abs=1e-14)
    assert profile["radius"] <= profile["mean_radius"] + 1e-14
    for sign in [-1, 1]:
        v = sign * h
        sums = summarize_cells(v, ids, cells=cells)
        np.testing.assert_array_equal(sums, 0)
        decoded = decode_cell_summary(profile, sums, v[selected])
        assert decoded == 0
        assert abs(decoded - p @ v) == pytest.approx(profile["radius"], abs=1e-14)


def test_minimax_by_exhaustive_small_cube_and_median_grid():
    p = np.array([.03, .4, -.2, .21, .06])
    ids = np.array([0, 0, 0, 1, 1])
    selected = [4]
    profile = cell_summary_minimax(p, ids, selected)
    errors = []
    for signs in itertools.product([-1., 1.], repeat=len(p)):
        v = np.array(signs)
        out = decode_cell_summary(profile, summarize_cells(v, ids), v[selected])
        errors.append(abs(out - p @ v))
    assert max(errors) == pytest.approx(profile["radius"])
    for c0, c1 in itertools.product(np.linspace(-1, 1, 31), repeat=2):
        residual = p - np.array([c0, c1])[ids]
        residual[selected] = 0
        assert np.abs(residual).sum() + 1e-14 >= profile["radius"]


def test_deterministic_ties_and_median_not_normalized():
    profile = cell_summary_minimax([1., 1., 1., 1., 1.], [0] * 5)
    np.testing.assert_array_equal(profile["dual_witness"], [-1, -1, 0, 1, 1])
    # Normalized target does not make the lifted median coefficients normalized.
    profile = cell_summary_minimax([.8, .1, .1], [0, 0, 0])
    assert profile["coefficients"].sum() * 3 == pytest.approx(.3)
    assert profile["radius"] == pytest.approx(.7)
    assert profile["mean_radius"] > profile["radius"]


def test_scalar_scaling_and_vector_output_bounds_and_signed_identity():
    rng = np.random.default_rng(42)
    p = rng.normal(size=31)
    ids = rng.integers(0, 7, size=31)
    selected = np.array([18, 2, 30])
    profile = cell_summary_minimax(p, ids, selected, cells=9)
    for dimension in [None, 1, 5]:
        shape = (31,) if dimension is None else (31, dimension)
        values = rng.uniform(-3., 3., size=shape)
        out = decode_cell_summary(profile, summarize_cells(values, ids, cells=9), values[selected])
        error = out - p @ values
        np.testing.assert_allclose(error, -profile["residual"] @ values, atol=2e-14)
        assert np.max(np.abs(error)) <= 3 * profile["radius"] + 1e-13
        B = np.max(np.abs(values)) if dimension is None else np.max(np.linalg.norm(values, axis=1))
        assert np.linalg.norm(error) <= B * profile["radius"] + 1e-13
    values = 7.5 * profile["dual_witness"]
    assert abs(p @ values) == pytest.approx(7.5 * profile["radius"])


def test_selected_correction_uses_full_sums_and_preserves_order():
    p = np.array([.2, .6, .15, .05])
    v = np.array([[1., 3.], [-2., 5.], [7., 0.], [2., -4.]])
    ids = [0, 1, 0, 1]
    selected = [3, 0]
    profile = cell_summary_minimax(p, ids, selected)
    np.testing.assert_array_equal(profile["selected"], selected)
    # One unread key per cell is recovered exactly from the full sum.
    assert profile["radius"] == 0
    np.testing.assert_allclose(decode_cell_summary(profile, summarize_cells(v, ids), v[selected]), p @ v)
    all_selected = [2, 0, 3, 1]
    full = cell_summary_minimax(p, ids, all_selected)
    np.testing.assert_allclose(decode_cell_summary(full, summarize_cells(v, ids), v[all_selected]), p @ v)


def test_mean_control_and_nonmutation():
    p = np.array([.8, .1, .1, .5, .01])
    ids = np.array([0, 0, 0, 1, 1])
    selected = np.array([4])
    values = np.arange(15.).reshape(5, 3)
    originals = [x.copy() for x in [p, ids, selected, values]]
    profile = cell_summary_minimax(p, ids, selected, cells=3)
    sums = summarize_cells(values, ids, cells=3)
    sums_before = sums.copy()
    means = profile["mean_coefficients"].copy()
    out = decode_cell_summary(profile, sums, values[selected], coefficients=means)
    np.testing.assert_allclose(out - p @ values, -profile["mean_residual"] @ values)
    for original, actual in zip(originals, [p, ids, selected, values]):
        np.testing.assert_array_equal(original, actual)
    np.testing.assert_array_equal(sums, sums_before)
    np.testing.assert_array_equal(means, profile["mean_coefficients"])
    profile["weights"][0] = 99
    profile["cell_ids"][0] = 2
    profile["selected"][0] = 0
    for original, actual in zip(originals, [p, ids, selected, values]):
        np.testing.assert_array_equal(original, actual)


@pytest.mark.parametrize("p,ids,selected,cells", [
    ([1], [0, 1], [], None), ([np.nan], [0], [], None),
    ([[1]], [0], [], None), ([1], [-1], [], None),
    ([1], [0.], [], None), ([1], [False], [], None),
    ([1], [0], [1], None), ([1], [0], [-1], None),
    ([1], [0], [0, 0], None), ([1], [0], [0.], None),
    ([1], [0], [], 0), ([1], [0], [], True),
    ([], [], [], -1), ([], [], [], 1.5),
])
def test_invalid_profiles(p, ids, selected, cells):
    with pytest.raises(ValueError):
        cell_summary_minimax(p, ids, selected, cells=cells)


def test_invalid_summaries_and_decode_shapes():
    profile = cell_summary_minimax([.8, .2], [0, 0], [1])
    with pytest.raises(ValueError):
        summarize_cells([np.nan, 1.], [0, 0])
    with pytest.raises(ValueError):
        summarize_cells([1.], [0, 0])
    for sums, exact, coefficients in [([3.], [], None), ([3., 4.], [1.], None),
                                      ([3.], [np.inf], None), ([3.], [1.], [1., 2.]),
                                      ([[3., 4.]], [1.], None)]:
        with pytest.raises(ValueError):
            decode_cell_summary(profile, sums, exact, coefficients=coefficients)


def test_empty_zero_cells_scalar_and_vector():
    profile = cell_summary_minimax([], [])
    assert profile["radius"] == 0
    assert decode_cell_summary(profile, [], []) == 0
    np.testing.assert_array_equal(
        decode_cell_summary(profile, np.empty((0, 4)), np.empty((0, 4))), np.zeros(4))


def test_even_median_retains_equal_subnormal_weights():
    smallest = np.nextafter(0., 1.)
    for weight in [smallest, -smallest]:
        profile = cell_summary_minimax([weight, weight], [0, 0])
        assert profile["coefficients"][0] == weight
        assert profile["radius"] == 0
