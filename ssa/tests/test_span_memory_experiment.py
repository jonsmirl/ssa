"""Protocol tests for the CPU association-memory diagnostic.

These test leakage, causal reads, and accounting, not model quality or a
deterministic compressed-attention certificate.
"""

import json

import numpy as np
import pytest

from ssa.span_memory_experiment import (
    centroid_attention,
    reconstruct_attention,
    route_blocks,
    run_case,
)


def sample(seed=7, n=96):
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(n, 4))
    queries = rng.normal(size=(n, 4))
    values = np.tanh(keys @ rng.normal(size=(4, 3)))
    values += 0.05 * rng.normal(size=values.shape)
    return queries, keys, values


def experiment(queries, keys, values, **kwargs):
    config = dict(budget=256, seed=3, train_stop=48, calibration=16,
                  batch_size=8, rate=0.5, query_positions=[55, 71, 95],
                  block=8, top_blocks=2)
    config.update(kwargs)
    return run_case(queries, keys, values, **config)


def dense(q, keys, values):
    scores = keys @ q / np.sqrt(keys.shape[1])
    weights = np.exp(scores - scores.max())
    return weights @ values / weights.sum()


def test_route_is_causal_unique_and_ties_are_deterministic():
    keys = np.zeros((40, 4))
    query = np.ones(4)
    ids = route_blocks(keys, query, 19, block=8, top_blocks=1)
    # Flat complete blocks select the first block; partial boundary is kept.
    np.testing.assert_array_equal(ids, np.r_[np.arange(8), np.arange(16, 19)])
    assert len(np.unique(ids)) == len(ids)
    assert (ids < 19).all()
    poisoned = keys.copy()
    poisoned[19:] = 1e12
    np.testing.assert_array_equal(
        ids, route_blocks(poisoned, query, 19, block=8, top_blocks=1))


@pytest.mark.parametrize("prefix", [1, 7, 8, 9, 16, 19])
def test_route_full_budget_recovers_every_visible_key(prefix):
    queries, keys, _ = sample(n=40)
    ids = route_blocks(keys, queries[prefix - 1], prefix,
                       block=8, top_blocks=100)
    np.testing.assert_array_equal(np.sort(ids), np.arange(prefix))


def test_oracle_reconstruction_uses_predictions_only_for_unselected_values():
    queries, keys, values = sample(n=12)
    predicted = np.full_like(values, -2.0)
    ids = np.array([0, 4, 11])
    expected_values = predicted.copy()
    expected_values[ids] = values[ids]
    actual = reconstruct_attention(queries[-1], keys, values, ids, predicted)
    np.testing.assert_allclose(actual, dense(queries[-1], keys, expected_values),
                               rtol=1e-12, atol=1e-12)
    altered = predicted.copy()
    altered[ids] = 1e8
    np.testing.assert_allclose(
        reconstruct_attention(queries[-1], keys, values, ids, altered), actual,
        rtol=1e-12, atol=1e-12)


def test_full_selection_recovers_dense_for_both_attention_helpers():
    queries, keys, values = sample(n=12)
    centers = keys[[0, 6]]
    assignment = ((keys[:, None] - centers[None]) ** 2).sum(-1).argmin(-1)
    counts = np.bincount(assignment, minlength=2)
    sums = np.zeros((2, values.shape[1]))
    np.add.at(sums, assignment, values)
    ids = np.arange(len(keys))
    expected = dense(queries[-1], keys, values)
    np.testing.assert_allclose(
        reconstruct_attention(queries[-1], keys, values, ids,
                              np.full_like(values, 1e5)), expected,
        rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        centroid_attention(queries[-1], keys, values, ids, centers, counts, sums),
        expected, rtol=1e-12, atol=1e-12)


def test_centroid_subtracts_exact_selected_count_and_value_sum():
    # Equal logits make one-cell approximation exact only if replacements do
    # not double-count the selected item in the streaming summary.
    keys = np.zeros((4, 2))
    values = np.array([[100., -3.], [1., 4.], [2., 5.], [3., 6.]])
    actual = centroid_attention(np.array([1., 2.]), keys, values, np.array([0]),
                                np.zeros((1, 2)), np.array([4]),
                                values.sum(0, keepdims=True))
    np.testing.assert_allclose(actual, values.mean(0), rtol=1e-12, atol=1e-12)


def test_frozen_fit_does_not_read_suffix_values():
    queries, keys, values = sample()
    original = experiment(queries, keys, values)
    changed = values.copy()
    changed[48:] = 1000 + changed[48:]
    poisoned = experiment(queries, keys, changed)
    assert original["models"].keys() == poisoned["models"].keys()
    for name in original["models"]:
        assert (original["models"][name]["frozen_fit_sha256"] ==
                poisoned["models"][name]["frozen_fit_sha256"]), name


def test_later_tokens_cannot_change_earlier_frozen_or_online_outputs():
    queries, keys, values = sample()
    original = experiment(queries, keys, values)
    changed_queries, changed_keys, changed_values = (
        array.copy() for array in (queries, keys, values))
    for array in (changed_queries, changed_keys, changed_values):
        array[56:] += 100
    poisoned = experiment(changed_queries, changed_keys, changed_values)
    for name, model in original["models"].items():
        for mode in ("frozen_attention", "online_attention",
                     "online_centroid_attention"):
            if mode not in model:
                continue
            first = model[mode][0]
            other = poisoned["models"][name][mode][0]
            assert first["position"] == other["position"] == 55
            np.testing.assert_allclose(first["output"], other["output"],
                                       rtol=1e-12, atol=1e-12,
                                       err_msg=f"future leakage: {name}/{mode}")


def test_report_matches_state_budget_read_budget_and_feature_families():
    result = experiment(*sample())
    models = result["models"]
    assert set(models) == {"linear_joint", "linear_sequential", "tanh_joint",
                           "tanh_sequential", "cells"}
    for model in models.values():
        assert 0 < model["state_scalars"] <= 256
        assert model["feature_dimension"] > 0
        assert len(model["frozen_fit_sha256"]) == 64
        assert model["heldout_value"]["relative_frobenius_error"] >= 0
        assert model["heldout_value"]["mse"] >= 0
        for mode in ("frozen_attention", "online_attention"):
            assert [row["position"] for row in model[mode]] == [55, 71, 95]
            assert all(row["selected_count"] <= 24 for row in model[mode])
    for family in ("linear", "tanh"):
        joint, sequential = models[family + "_joint"], models[family + "_sequential"]
        assert joint["feature_dimension"] == sequential["feature_dimension"]
        assert joint["state_scalars"] == sequential["state_scalars"]
    reference = models["cells"]["online_attention"]
    for model in models.values():
        assert ([row["selected_count"] for row in model["online_attention"]] ==
                [row["selected_count"] for row in reference])
    json.dumps(result, allow_nan=False)


def test_interference_keeps_old_associations_and_new_batch_metrics_separate():
    from ssa.span_memory import joint_update

    queries, keys, values = sample()
    report = experiment(queries, keys, values)
    rows = report["models"]["linear_joint"]["interference"]
    assert [row["fit_start"] for row in rows] == [8, 16, 24, 32, 40]
    for row in rows:
        assert 0 <= row["fraction_old_associations_worsened"] <= 1
        assert row["anchor_count"] == row["fit_start"]
        assert row["new_batch_residual_fro_after"] <= row["new_batch_residual_fro_before"] + 1e-10
        assert row["new_batch_irreducible_residual_fro"] <= row["new_batch_residual_fro_after"] + 1e-10
    # Independently reconstruct the second update; the old rows are not the
    # current fitting batch, and mean row norms are not Frobenius norms.
    normalized = ((keys - keys[:16].mean(0)) /
                  np.maximum(keys[:16].std(0), 1e-6))
    features = np.column_stack((np.ones(len(keys)), normalized))
    weights, _ = joint_update(np.zeros((5, 3)), features[:8], values[:8], rate=.5)
    old_before = features[:8] @ weights - values[:8]
    new_before = features[8:16] @ weights - values[8:16]
    updated, _ = joint_update(weights, features[8:16], values[8:16], rate=.5)
    old_after = features[:8] @ updated - values[:8]
    new_after = features[8:16] @ updated - values[8:16]
    row = rows[0]
    assert row["old_mean_l2_before"] == pytest.approx(np.linalg.norm(old_before, axis=1).mean())
    assert row["old_mean_l2_after"] == pytest.approx(np.linalg.norm(old_after, axis=1).mean())
    assert row["new_batch_residual_fro_before"] == pytest.approx(np.linalg.norm(new_before))
    assert row["new_batch_residual_fro_after"] == pytest.approx(np.linalg.norm(new_after))


def test_zero_values_are_finite_and_relative_error_is_explicitly_undefined():
    queries, keys, values = sample()
    values[:] = 0
    report = experiment(queries, keys, values)
    for model in report["models"].values():
        assert model["heldout_value"]["relative_frobenius_error"] is None
        assert model["heldout_value"]["mse"] == 0
        for mode in ("frozen_attention", "online_attention"):
            assert all(row["l2_error"] == 0 for row in model[mode])
    json.dumps(report, allow_nan=False)


def test_extra_measurements_do_not_change_the_persistent_stream():
    data = sample()
    single = experiment(*data, query_positions=[95])
    # Non-batch-aligned observations previously flushed partial updates into
    # persistent state, changing the later joint-memory output.
    multiple = experiment(*data, query_positions=[52, 69, 95])
    for name, model in single["models"].items():
        other = multiple["models"][name]
        assert model["frozen_fit_sha256"] == other["frozen_fit_sha256"]
        for mode in ("frozen_attention", "online_attention",
                     "online_centroid_attention"):
            if mode in model:
                np.testing.assert_allclose(model[mode][-1]["output"],
                                           other[mode][-1]["output"],
                                           rtol=1e-12, atol=1e-12,
                                           err_msg=f"measurement changes state: {name}/{mode}")
        assert model["online_value_rows_admitted"] == 48
        assert other["online_value_rows_admitted"] == 48
        if name != "cells":
            assert model["online_update_rows_processed"] == 48
            # Prefixes53 and70 require temporary batches of5 and6 rows; these
            # rows are read again by subsequent committed updates.
            assert other["online_update_rows_processed"] == 48 + 5 + 6


def test_attention_helpers_do_not_read_unselected_archive_values():
    queries, keys, values = sample(n=12)
    ids = np.array([0, 4, 11])
    centers = keys[[0, 6]]
    assignment = ((keys[:, None] - centers[None]) ** 2).sum(-1).argmin(-1)
    counts = np.bincount(assignment, minlength=2)
    sums = np.zeros((2, values.shape[1]))
    np.add.at(sums, assignment, values)
    predicted = np.full_like(values, 0.25)
    poisoned = np.full_like(values, np.nan)
    poisoned[ids] = values[ids]
    # Summaries/predictions were built before archive values became unavailable.
    # Only the selected exact entries may be consumed at this query.
    for reader in (
        lambda archive: reconstruct_attention(queries[-1], keys, archive, ids, predicted),
        lambda archive: centroid_attention(queries[-1], keys, archive, ids,
                                           centers, counts, sums),
    ):
        np.testing.assert_array_equal(reader(poisoned), reader(values))
        bad_selected = poisoned.copy()
        bad_selected[ids[0]] = np.nan
        with pytest.raises(ValueError):
            reader(bad_selected)


def test_sparse_read_is_finite_when_all_selected_dense_weights_underflow():
    queries, keys, values = sample()
    # Block0 has a unique enormous logit but an unattractive mean. It remains
    # unopened, and dense-normalized weights at every selected key become zero.
    keys[0, 0] = 100
    keys[1:8, 0] = -1000
    queries[:] = 0
    queries[:, 0] = 1e5
    report = experiment(queries, keys, values)
    assert all(route["actual_selected_mass"] == 0 for route in report["routes"])
    assert all(0 not in route["indices"] for route in report["routes"])
    for row, route in zip(report["sparse_attention"], report["routes"]):
        ids = np.asarray(route["indices"])
        np.testing.assert_allclose(row["output"],
                                   dense(queries[row["position"]], keys[ids], values[ids]),
                                   rtol=1e-12, atol=1e-12)
    json.dumps(report, allow_nan=False)
