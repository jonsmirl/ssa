"""Dense-oracle tests for deterministic score-tail attention certificates."""
import math

import numpy as np
import pytest
import torch

from ssa.core import dense_read, softmax
from ssa.score_tail_certificate import (
    RoutingMetricSelection,
    ScoreTailCertifiedAttention,
    best_log_mass_upper,
    certificate_margin,
    certificate_softplus_loss,
    exact_score_histogram,
    mass_share_upper_from_logs,
    tail_profile_from_bands,
    tail_profile_from_item_caps,
)


def _check_dense(index, q, beta, result, prefix):
    dense, p, _ = dense_read(q, index.K[:prefix], index.V[:prefix], beta)
    selected = result.indices
    sparse = softmax(beta * (index.K[selected] @ q))
    np.testing.assert_allclose(result.output, sparse @ index.V[selected], atol=2e-12, rtol=2e-12)
    omitted = np.ones(prefix, dtype=bool)
    omitted[selected] = False
    actual_mass = float(p[omitted].sum())
    assert actual_mass <= result.mass_upper + 3e-12
    assert np.linalg.norm(dense - result.output) <= result.output_error_upper + 3e-11
    actual_kl = float(np.sum(sparse * np.log(sparse / p[selected])))
    assert actual_kl <= result.kl_upper + 3e-11


def test_disjoint_tail_bound_dominates_actual_mass_and_one_level_is_max_bound():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=127)
    cap_slack = rng.uniform(0, 0.4, size=len(scores))
    caps = scores + cap_slack
    for levels in (1, 2, 9, 64):
        profile = tail_profile_from_item_caps(caps, levels=levels)
        assert np.exp(scores).sum() <= np.exp(profile.log_mass_upper) * (1 + 2e-15)
    one = tail_profile_from_item_caps(caps, levels=1)
    assert np.exp(one.log_mass_upper) == pytest.approx(len(caps) * np.exp(caps.max()))


def test_exact_histogram_equals_true_mass():
    scores = np.array([2.0, -1.0, 2.0, 0.5, -1.0])
    profile = exact_score_histogram(scores)
    assert np.exp(profile.log_mass_upper) == pytest.approx(np.exp(scores).sum())


def test_empty_bands_and_repeated_thresholds_are_supported():
    caps = np.array([-2.0, -2.0, 1.0])
    profile = tail_profile_from_item_caps(
        caps, upper_edges=np.array([-2.0, -2.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(profile.counts, [2, 0, 0, 1, 0])
    empty = tail_profile_from_item_caps([], upper_edges=[0.0, 0.0, 1.0])
    assert empty.source_items == 0 and empty.log_mass_upper == -np.inf


def test_adding_thresholds_never_worsens_the_same_cap_profile():
    caps = np.array([-3.0, -1.2, -0.2, 0.7, 1.0])
    counts = np.array([5, 3, 7, 2, 1])
    coarse = tail_profile_from_item_caps(caps, counts, upper_edges=[-1.0, 1.0])
    refined = tail_profile_from_item_caps(caps, counts,
                                          upper_edges=[-3.0, -2.0, -1.0, 0.0, 1.0])
    assert refined.log_mass_upper <= coarse.log_mass_upper
    assert best_log_mass_upper(coarse.log_mass_upper, refined.log_mass_upper) == refined.log_mass_upper
    lowered = tail_profile_from_item_caps(caps, counts // 2,
                                          upper_edges=[-3.0, -2.0, -1.0, 0.0, 1.0])
    assert lowered.log_mass_upper <= refined.log_mass_upper


def test_lowering_certified_disjoint_band_counts_tightens_bound():
    coarse = tail_profile_from_bands([-2.0, 0.0, 3.0], [9, 8, 7])
    tighter = tail_profile_from_bands([-2.0, 0.0, 3.0], [9, 5, 2])
    assert tighter.log_mass_upper < coarse.log_mass_upper


def test_minimum_of_existing_admissible_caps_stays_admissible():
    true_log_mass = math.log(7.0)
    kept = best_log_mass_upper(
        true_log_mass + 4.0,  # radius
        true_log_mass + 2.0,  # Bennett/covariance/peeled
        true_log_mass + 3.0,  # parent
        true_log_mass + 1.0,  # child sum
        true_log_mass + 0.5)  # tail profile
    assert kept >= true_log_mass


def test_margin_is_exact_stopping_test_and_softplus_boundary_is_log_two():
    eta = 0.1
    for log_a, log_z in ((0.0, 4.0), (4.0, 0.0), (math.log(1), math.log(9))):
        share = mass_share_upper_from_logs(log_a, log_z)
        margin = certificate_margin(log_a, log_z, eta)
        assert (share <= eta) == (margin <= 2e-15)
        loss = certificate_softplus_loss(log_a, log_z, eta)
        # softplus(M) is never zero; log(2), not zero, is its exact hard boundary.
        assert (loss <= math.log(2) + 2e-15) == (margin <= 2e-15)


@pytest.mark.parametrize("prefix", [1, 17, 32, 53, 96])
@pytest.mark.parametrize("levels", [1, 4, 16])
def test_reader_is_sound_for_arbitrary_causal_prefixes(prefix, levels):
    rng = np.random.default_rng(8)
    K, V = rng.normal(size=(96, 7)), rng.normal(size=(96, 4))
    q = rng.normal(size=7)
    index = ScoreTailCertifiedAttention(K, V, 16)
    route = index.route_by_block_mean(q, 2, prefix=prefix)
    result = index.read(q, beta=1.7, routing=route, tail_levels=levels,
                        prefix=prefix, max_blocks=max(1, min(3, math.ceil(prefix / 16))))
    _check_dense(index, q, 1.7, result, prefix)


def test_prefix_does_not_observe_future_keys_or_values():
    rng = np.random.default_rng(19)
    K, V = rng.normal(size=(80, 5)), rng.normal(size=(80, 3))
    q, prefix = rng.normal(size=5), 45
    a_index = ScoreTailCertifiedAttention(K, V, 8)
    a_route = a_index.route_by_block_mean(q, 2, prefix=prefix)
    a = a_index.read(q, routing=a_route, prefix=prefix, max_blocks=3)
    K[prefix:], V[prefix:] = 1e9, -1e9
    b_index = ScoreTailCertifiedAttention(K, V, 8)
    b_route = b_index.route_by_block_mean(q, 2, prefix=prefix)
    b = b_index.read(q, routing=b_route, prefix=prefix, max_blocks=3)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_allclose(a.output, b.output)
    assert a.mass_upper == b.mass_upper


def test_equal_and_concentrated_logits_and_full_refinement():
    rng = np.random.default_rng(4)
    V = rng.normal(size=(64, 3))
    for K, q, beta in ((np.zeros((64, 4)), np.ones(4), 1.0),
                       (np.r_[np.full((8, 1), 10.0), np.zeros((56, 1))], [1.0], 3.0)):
        index = ScoreTailCertifiedAttention(K, V, 8)
        route = index.route_by_block_mean(q, 1)
        full = index.read(q, beta=beta, routing=route, tail_levels=8, mass_tol=0)
        assert full.certified and full.keys_scored == len(K)
        _check_dense(index, q, beta, full, len(K))


def test_deterministic_ties_choose_larger_block_index_first():
    index = ScoreTailCertifiedAttention(np.zeros((32, 2)), np.ones((32, 1)), 8)
    route = index.route_by_block_mean([1, 0], 2)
    np.testing.assert_array_equal(route.block_indices, [3, 2])
    adapted = RoutingMetricSelection.from_ccc_row(2, [7, 3, 0], certified=True)
    np.testing.assert_array_equal(adapted.block_indices, [7, 3])
    assert adapted.top_blocks_certified and adapted.metric_name == "ccc_routing_metric"
    read = index.read_ccc_row([1, 0], 2, [3, 2, 0], ccc_certified=True,
                              max_blocks=2, mass_tol=0.1)
    assert read.routing_top_blocks_certified
    assert read.attention_score_bound == "direct_block_mean_plus_residual_radius"


def test_ccc_adapter_accepts_tensor_contract():
    adapted = RoutingMetricSelection.from_ccc_row(
        torch.tensor(2, dtype=torch.int32),
        torch.tensor([5, 1, 0], dtype=torch.int32), certified=True)
    np.testing.assert_array_equal(adapted.block_indices, [5, 1])


def test_centroid_hidden_extreme_is_not_miscertified():
    # Block zero has harmless mean but one extreme key. Mean routing misses it;
    # the separate residual-radius bridge keeps the attention certificate sound.
    K = np.zeros((32, 2))
    K[0, 0], K[1:8, 0] = 14.0, -2.0
    K[8:16, 0] = 1.0
    V = np.eye(32)
    index = ScoreTailCertifiedAttention(K, V, 8)
    route = index.route_by_block_mean([1, 0], 1)
    assert 0 not in route.block_indices
    limited = index.read([1, 0], routing=route, tail_levels=8, mass_tol=0.1, max_blocks=1)
    assert not limited.certified
    _check_dense(index, [1, 0], 1.0, limited, 32)
    full = index.read([1, 0], routing=route, tail_levels=8, mass_tol=0.1)
    assert full.certified and 0 in (full.indices // 8)
    _check_dense(index, [1, 0], 1.0, full, 32)


def test_value_output_certificate_can_stop_before_mass_certificate():
    index = ScoreTailCertifiedAttention(np.zeros((64, 2)), np.ones((64, 3)), 8)
    route = index.route_by_block_mean([0, 0], 1)
    output_only = index.read([0, 0], routing=route, mass_tol=None, error_tol=0)
    assert output_only.certified and output_only.keys_scored == 8
    assert output_only.output_error_upper == 0 and output_only.mass_upper > 0.8
    _check_dense(index, np.zeros(2), 1.0, output_only, 64)
