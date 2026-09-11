"""Dense-oracle checks, not a formal IEEE rounding proof."""
import numpy as np
import pytest

from ssa.partial_coordinate_attention import CoordinateBounds
from ssa.score_tail_certificate import exact_score_histogram, tail_profile_from_item_caps


@pytest.mark.parametrize("mode", ["block_interval", "global_abs"])
@pytest.mark.parametrize("prefix", [0, 1, 7, 8, 9, 31])
def test_dense_oracle_causal_prefix_refinement(mode, prefix):
    rng = np.random.default_rng(15)
    K = rng.normal(size=(40, 9))
    q = rng.normal(size=9)
    index = CoordinateBounds(K[:prefix], block=8, mode=mode)
    state = index.start(q, beta=1 / 3)
    exact = (K[:prefix] * ((1 / 3) * q)).sum(axis=1)
    for coords in [[], [3, 0], [5], [8, 6, 4], [1, 2, 7]]:
        old_lo, old_hi = state.lower.copy(), state.upper.copy()
        state.refine_coordinates(coords)
        assert np.all(state.lower <= exact + 1e-13)
        assert np.all(state.upper >= exact - 1e-13)
        assert np.all(state.lower >= old_lo - 1e-13)
        assert np.all(state.upper <= old_hi + 1e-13)
    np.testing.assert_array_equal(state.lower, exact)
    np.testing.assert_array_equal(state.upper, exact)
    assert state.coordinate_scalars_read == prefix * 9
    assert not state.opened.any()


def test_future_mutation_and_partial_boundary_are_excluded():
    K = np.arange(48, dtype=float).reshape(12, 4)
    snapshot = CoordinateBounds(K[:5], block=4)
    state = snapshot.start([1, -1, 2, -2])
    lo, hi = state.lower.copy(), state.upper.copy()
    K[:] = 1e20
    state.refine_coordinates([])
    np.testing.assert_array_equal(state.lower, lo)
    np.testing.assert_array_equal(state.upper, hi)
    np.testing.assert_array_equal(snapshot.minimum[1], [16, 17, 18, 19])
    assert not snapshot.K.flags.writeable


@pytest.mark.parametrize("mode", ["block_interval", "global_abs"])
def test_extreme_member_not_hidden_by_harmless_centroid(mode):
    K = np.array([[0, 1e6], [0, -1e6], [1, 0], [1, 0]], float)
    state = CoordinateBounds(K, block=2, mode=mode).start([1, 1])
    state.refine_coordinates([0])
    assert state.partial[0] == 0
    assert state.upper[0] >= 1e6
    assert not state.topk_certificate([2])["certified"]
    state.open_keys([0])
    # Its sibling could still attain the same extreme cap until refined.
    assert not state.topk_certificate([0])["certified"]
    state.refine_coordinates([1])
    assert state.topk_certificate([0])["certified"]


def test_ledger_opening_refinement_and_duplicates():
    K = np.arange(30, dtype=float).reshape(6, 5)
    state = CoordinateBounds(K).start(np.ones(5))
    state.refine_coordinates([0, 0, 3])
    assert state.coordinate_scalars_read == 12
    logits = state.open_keys([4, 1, 4])
    np.testing.assert_array_equal(logits, K[[1, 4]].sum(axis=1))
    assert state.coordinate_scalars_read == 18
    state.open_keys([1])
    state.refine_coordinates([3, 0])
    assert state.coordinate_scalars_read == 18
    state.refine_coordinates([1, 2, 4])
    assert state.coordinate_scalars_read == 30
    assert state.read_mask.all()
    assert state.opened.sum() == 2
    work = state.work()
    assert work["fully_scored_keys"] == 6
    assert work["index_build_key_scalars"] == 30
    assert work["index_archive_bytes"] == 240
    assert work["reference_refresh_coordinate_slots"] >= 30


def test_ties_and_zero_query_do_not_claim_strict_topk():
    state = CoordinateBounds(np.ones((5, 4))).start(np.zeros(4))
    np.testing.assert_array_equal(state.coordinate_order, np.arange(4))
    state.refine_coordinates(np.arange(4))
    assert state.topk_certificate([0])["strict_margin"] == 0
    assert not state.topk_certificate([0])["certified"]
    assert state.topk_certificate([])["vacuous"]
    assert state.topk_certificate(np.arange(5))["vacuous"]


def test_abs_query_order_has_stable_coordinate_ties():
    state = CoordinateBounds(np.zeros((3, 4))).start([-2, 2, -3, 0])
    np.testing.assert_array_equal(state.coordinate_order, [2, 0, 1, 3])


@pytest.mark.parametrize("mode", ["block_interval", "global_abs"])
def test_tail_caps_dominate_dense_mass_and_full_refinement_histogram(mode):
    rng = np.random.default_rng(67)
    K, q = rng.normal(size=(39, 8)), rng.normal(size=8)
    state = CoordinateBounds(K, block=7, mode=mode).start(q, beta=0.3)
    scores = (K * (0.3 * q)).sum(axis=1)
    selected = [1, 5, 12]
    state.open_keys(selected)
    unread = ~state.opened
    truth = exact_score_histogram(scores[unread]).log_mass_upper
    old = np.inf
    for coords in [[], [0, 1], [2, 3], [4, 5], [6, 7]]:
        state.refine_coordinates(coords)
        tail = tail_profile_from_item_caps(state.upper[unread], levels=16)
        assert tail.log_mass_upper >= truth - 1e-12
        retained = min(old, tail.log_mass_upper)
        assert retained <= old
        old = retained
    histogram = exact_score_histogram(state.upper[unread])
    assert histogram.log_mass_upper == truth


@pytest.mark.parametrize("K", [np.ones(3), np.ones((2, 0)), [[np.nan]], [[np.inf]]])
def test_invalid_keys(K):
    with pytest.raises(ValueError):
        CoordinateBounds(K)


@pytest.mark.parametrize("block", [0, -1, 1.2, True])
def test_invalid_block(block):
    with pytest.raises(ValueError):
        CoordinateBounds([[1]], block=block)


@pytest.mark.parametrize("q,beta", [([1, 2], 1), ([np.nan], 1), ([1], 0),
                                   ([1], -1), ([1], np.inf)])
def test_invalid_query(q, beta):
    with pytest.raises(ValueError):
        CoordinateBounds([[1]]).start(q, beta=beta)


@pytest.mark.parametrize("indices", [[-1], [3], [0.1], [[0]], [True]])
def test_invalid_read_indices(indices):
    state = CoordinateBounds(np.ones((3, 3))).start(np.ones(3))
    with pytest.raises(ValueError):
        state.refine_coordinates(indices)
    with pytest.raises(ValueError):
        state.open_keys(indices)


def test_overflow_rejected_not_certified():
    with pytest.raises(ValueError):
        CoordinateBounds([[1e308]]).start([1e308])
    with pytest.raises(ValueError):
        CoordinateBounds([[1]]).start([1e308], beta=1e308)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        CoordinateBounds([[1]], mode="centroid_only")
