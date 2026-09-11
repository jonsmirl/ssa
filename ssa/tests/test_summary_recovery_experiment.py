"""Protocol checks: chronological validation, causal reads and charged ridge state."""

import json

import numpy as np
import pytest

from ssa.summary_recovery_experiment import (
    audit_cases, feature_spec, features, run_case, run_synthetic, select_penalty,
)


@pytest.fixture
def fixture():
    rng = np.random.default_rng(284)
    Q = rng.normal(size=(96, 4))
    K = rng.normal(size=(96, 4))
    V = np.tanh(K @ rng.normal(size=(4, 3))) + .1*rng.normal(size=(96, 3))
    return Q, K, V


def _run(fixture, **overrides):
    kwargs = dict(budget=256, calibration=16, fit_stop=32, train_stop=48,
                  batch_size=8, query_positions=[55, 71, 95],
                  penalties=(.01, 1., 100.), include_existing=False,
                  block=8, top_blocks=1)
    kwargs.update(overrides)
    return run_case(*fixture, **kwargs)


def _query_outputs(report, index):
    return [report["sparse_attention"][index]["output"],
            report["oracle_selected_only"]["rows"][index]["output"],
            report["global_sum_median"]["rows"][index]["output"],
            report["cells"]["rows"][index]["median_output"],
            report["cells"]["rows"][index]["mean_output"]] + [
        model[protocol][index]["output"]
        for model in report["models"].values()
        for protocol in ("frozen_attention", "online_attention",
                         "matched_joint_attention", "matched_sequential_attention")]


def test_future_queries_keys_values_cannot_change_earlier_outputs(fixture):
    before = _run(fixture)
    poisoned = tuple(x.copy() for x in fixture)
    rng = np.random.default_rng(45)
    for array in poisoned:
        array[56:] = 20*rng.normal(size=array[56:].shape)
    after = _run(poisoned)
    assert before["routes"][0] == after["routes"][0]
    np.testing.assert_allclose(_query_outputs(before, 0), _query_outputs(after, 0), rtol=0, atol=0)
    assert before["cells"]["rows"][0] == after["cells"]["rows"][0]
    for kind in before["models"]:
        assert before["models"][kind]["validation"] == after["models"][kind]["validation"]
        assert before["models"][kind]["frozen_fit_sha256"] == after["models"][kind]["frozen_fit_sha256"]


def test_suffix_values_cannot_choose_penalty_or_change_frozen_fit(fixture):
    before = _run(fixture)
    poisoned = tuple(x.copy() for x in fixture)
    poisoned[2][48:] = np.random.default_rng(55).normal(size=(48, 3))*100
    after = _run(poisoned)
    for kind in before["models"]:
        left, right = before["models"][kind], after["models"][kind]
        assert left["penalty"] == right["penalty"]
        assert left["validation"] == right["validation"]
        assert left["frozen_fit_sha256"] == right["frozen_fit_sha256"]
        assert left["training_value"] == right["training_value"]
        assert left["frozen_heldout_value"] != right["frozen_heldout_value"]


def test_penalty_selection_matches_independent_chronological_validation(fixture):
    _, K, V = fixture
    report = _run(fixture)
    for kind, model in report["models"].items():
        spec, r, _ = feature_spec(K[:16], 256, 3, 0, kind)
        X = features(K, spec)
        expected = []
        for penalty in (.01, 1., 100.):
            W = np.linalg.solve(X[:32].T @ X[:32]+penalty*np.eye(r), X[:32].T @ V[:32])
            expected.append(float(np.mean((X[32:48]@W-V[32:48])**2)))
        validation = model["validation"]
        np.testing.assert_allclose(validation["validation_mse"], expected, atol=1e-13)
        assert validation["selected_index"] == int(np.argmin(expected))
        assert model["penalty"] == (.01, 1., 100.)[int(np.argmin(expected))]
        assert validation["fit_rows"] == 32 and validation["validation_rows"] == 16
        assert validation["fit_value_rows_across_candidates"] == 96
        assert validation["validation_value_rows_across_candidates"] == 48


def test_tied_validation_prefers_first_supplied_penalty():
    penalty, report = select_penalty(np.ones((12, 2)), np.zeros((12, 3)), 4, 8, (100., .01, 1.))
    assert penalty == 100.
    assert report["selected_index"] == 0


def test_query_schedule_invariance_and_partial_copy_work(fixture):
    one = _run(fixture, query_positions=[95])
    many = _run(fixture, query_positions=[52, 69, 95])
    np.testing.assert_allclose(_query_outputs(one, -1), _query_outputs(many, -1), rtol=0, atol=0)
    assert one["routes"][-1] == many["routes"][-1]
    assert one["cells"]["rows"][-1] == many["cells"]["rows"][-1]
    for kind in one["models"]:
        left, right = one["models"][kind], many["models"][kind]
        assert left["last_online_state_sha256"] == right["last_online_state_sha256"]
        # Persistent batch grid reads 48 rows once. Discarded partial copies
        # separately read rows [48,53) and [64,70): five plus six operations.
        assert left["online_rows_processed_including_query_copies"] == 48
        assert right["online_rows_processed_including_query_copies"] == 59
        assert left["work"]["fit_value_rows"] == 96
        assert right["work"]["fit_value_rows"] == 107
        assert left["work"]["distinct_admitted_value_rows"] == right["work"]["distinct_admitted_value_rows"] == 96
        assert right["work"]["temporary_query_state_copy_scalars"] > 0


def test_all_models_and_matched_controls_fit_same_storage_cap(fixture):
    report = _run(fixture)
    assert 0 < report["cells"]["state_scalars"] <= 256
    assert report["cells"]["state_scalars"] == report["cells"]["count"]*(4+3+1)
    for model in report["models"].values():
        r = model["feature_dimension"]
        feature_state = 2*4+(4+1)*(r-1)
        persistent = r*r+2*r*3+2
        assert model["feature_state_scalars"] == feature_state
        assert model["statistics_accounting"]["persistent_scalars"] == persistent
        assert model["state_scalars"] == feature_state+persistent <= 256
        assert model["matched_feature_control_state_scalars"] == feature_state+r*3 <= 256
        assert model["state_bytes_float64"] == 8*model["state_scalars"]
        for protocol in ("frozen_attention", "online_attention",
                         "matched_joint_attention", "matched_sequential_attention"):
            assert [row["selected_count"] for row in model[protocol]] == [
                len(row["indices"]) for row in report["routes"]]
    assert report["shared_work"]["exact_selected_value_reads"] == sum(
        len(row["indices"]) for row in report["routes"])
    assert report["shared_work"]["oracle_key_logits_and_dense_truth_value_rows"] == 56+72+96


def test_full_selection_recovers_dense_and_zero_cell_radius(fixture):
    report = _run(fixture, top_blocks=100)
    for i, route in enumerate(report["routes"]):
        assert sorted(route["indices"]) == list(range(route["position"]+1))
        assert route["omitted_mass"] == 0
        cell = report["cells"]["rows"][i]
        assert cell["radius"] == 0
        assert cell["mean_coefficient_radius"] == 0
        assert cell["dual_target"] == 0
        assert cell["median_l2_error"] < 1e-14
        assert report["sparse_attention"][i]["l2_error"] < 1e-14
        for model in report["models"].values():
            for protocol in ("frozen_attention", "online_attention",
                             "matched_joint_attention", "matched_sequential_attention"):
                assert model[protocol][i]["l2_error"] < 1e-14
            assert model["online_attention"][i]["reader_gain"] == 0


def test_exact_kernel_duality_and_vector_bound_checks(fixture):
    report = _run(fixture)
    for row in report["cells"]["rows"]:
        assert row["kernel_max_abs_integer"] == 0
        assert row["selected_witness_max_abs_integer"] == 0
        assert row["primal_dual_gap"] < 1e-14
        assert row["signed_identity_deficit"] < 1e-14
        assert row["output_bound_deficit"] < 1e-14
        assert row["radius"] <= row["mean_coefficient_radius"]+1e-14
        assert row["radius"] <= row["omitted_mass_without_summary"]+1e-14
        assert row["median_l2_error"] <= row["row_norm_scaled_output_bound"]+1e-14


def test_reader_gain_and_attention_match_independent_dense_prefix_ridge(fixture):
    Q, K, V = fixture
    report = _run(fixture)
    for kind, model in report["models"].items():
        spec, r, _ = feature_spec(K[:16], 256, 3, 0, kind)
        X = features(K, spec)
        for index, route in enumerate(report["routes"]):
            pos = route["position"]
            prefix = pos+1
            logits = K[:prefix]@Q[pos]/2
            p = np.exp(logits-logits.max())
            p /= p.sum()
            ids = route["indices"]
            unread = np.ones(prefix, dtype=bool)
            unread[ids] = False
            a = p[unread]@X[:prefix][unread]
            for protocol, admitted in (("frozen_attention", 48), ("online_attention", prefix)):
                design = X[:admitted]
                inverse_read = np.linalg.solve(design.T@design+model["penalty"]*np.eye(r), a)
                expected_gain = np.linalg.norm(inverse_read@design.T)
                W = np.linalg.solve(design.T@design+model["penalty"]*np.eye(r), design.T@V[:admitted])
                out = p[ids]@V[ids]+a@W
                row = model[protocol][index]
                assert row["reader_gain"] == pytest.approx(expected_gain, rel=2e-10, abs=1e-13)
                assert row["reader_gain"] <= row["universal_reader_gain_bound"]+1e-12
                assert row["signed_identity_deficit"] < 1e-13
                np.testing.assert_allclose(row["output"], out, rtol=1e-10, atol=1e-12)


def test_zero_tail_oracle_control_has_no_hidden_tail_values(fixture):
    Q, K, V = fixture
    report = _run(fixture)
    for index, route in enumerate(report["routes"]):
        pos, ids = route["position"], route["indices"]
        logits = K[:pos+1]@Q[pos]/2
        p = np.exp(logits-logits.max())
        p /= p.sum()
        np.testing.assert_allclose(report["oracle_selected_only"]["rows"][index]["output"], p[ids]@V[ids])
        assert report["cells"]["rows"][index]["radius"] <= report["global_sum_median"]["rows"][index]["radius"]+1e-14


def test_numerical_audit_rejects_bad_witness(fixture):
    report = _run(fixture)
    assert audit_cases([report])["passed"]
    report["cells"]["rows"][0]["kernel_max_abs_integer"] = 1
    assert not audit_cases([report])["passed"]


def test_predefined_synthetic_controls():
    report = run_synthetic()
    assert set(report) == {"realizable_linear", "independent_values", "concentrated_attention"}
    assert audit_cases(list(report.values()))["passed"]
    # Fixed witness of useful fitting, not a universal guarantee of improvement.
    fit = report["realizable_linear"]["models"]["linear_projection"]
    assert fit["summary"]["online_attention"] < .001
    assert report["concentrated_attention"]["cells"]["mean_radius"] > .5


def test_deterministic_json_finite_and_input_nonmutation(fixture):
    original = tuple(x.copy() for x in fixture)
    first, second = _run(fixture), _run(fixture)
    for report in (first, second):
        assert report.pop("seconds_including_controls_and_oracles") >= 0
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(second, allow_nan=False, sort_keys=True)
    for before, after in zip(original, fixture):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("overrides", [
    {"query_positions": [47]}, {"query_positions": [96]},
    {"query_positions": [71, 55]}, {"query_positions": [55, 55]},
    {"query_positions": []}, {"fit_stop": 48}, {"calibration": 33},
    {"penalties": (0., 1.)}, {"penalties": (np.nan,)},
])
def test_invalid_protocol_inputs(fixture, overrides):
    with pytest.raises(ValueError):
        _run(fixture, **overrides)
