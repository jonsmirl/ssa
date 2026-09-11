"""Dense-oracle protocol checks for coordinate-to-tail certified reads."""

import math

import numpy as np
import pytest

from ssa.partial_coordinate_experiment import evaluate_case, run_read, synthetic


@pytest.fixture
def arrays():
    rng = np.random.default_rng(918)
    return rng.normal(size=(53, 6)), rng.normal(size=(53, 3)), rng.normal(size=6)


def _run(arrays, **kwargs):
    options = dict(prefix=43, coordinates=2, seed_values=7, max_values=11,
                   batch=3, block=8, levels=16, eta=.1)
    options.update(kwargs)
    return run_read(*arrays, **options)


def _dense_check(arrays, row):
    K, V, q = arrays
    n, ids = row["prefix"], np.array(row["indices"])
    scores = K[:n]@q/math.sqrt(K.shape[1])
    p = np.exp(scores-scores.max())
    p /= p.sum()
    restricted = np.exp(scores[ids]-scores[ids].max())
    restricted /= restricted.sum()
    sparse = restricted@V[ids]
    lifted = np.zeros(n)
    lifted[ids] = restricted
    other = np.ones(n, dtype=bool)
    other[ids] = False
    delta = float(p[other].sum())
    np.testing.assert_allclose(row["output"], sparse, rtol=2e-13, atol=2e-13)
    assert row["actual_omitted_mass"] == pytest.approx(delta, abs=2e-14)
    assert .5*np.abs(lifted-p).sum() == pytest.approx(delta, abs=2e-14)
    assert delta <= row["omitted_mass_upper"]+2e-13
    assert row["actual_supported_kl"] == pytest.approx(
        np.sum(restricted*np.log(restricted/p[ids])), abs=2e-13)
    assert row["actual_supported_kl"] <= row["kl_upper"]+2e-13
    assert np.linalg.norm(sparse-p@V[:n]) <= row["output_error_upper"]+2e-12
    assert all(value <= 2e-12 for value in row["audit"].values())
    if row["certified"]:
        assert delta <= row["eta"]+2e-13
        assert row["omitted_mass_upper"] <= row["eta"]+2e-13
    for spec in row["top_set"].values():
        if spec["certified"]:
            assert sorted(spec["selected_top_ids"]) == sorted(spec["oracle_top_ids"])


@pytest.mark.parametrize("prefix", [1, 7, 8, 17, 43, 53])
@pytest.mark.parametrize("coordinates", [0, 2, 6])
@pytest.mark.parametrize("mode", ["global_abs", "block_interval"])
def test_dense_oracle_for_arbitrary_prefix_and_coordinate_counts(arrays, prefix, coordinates, mode):
    row = _run(arrays, prefix=prefix, coordinates=coordinates, mode=mode,
               max_values=min(prefix, 11))
    _dense_check(arrays, row)
    assert len(set(row["indices"])) == len(row["indices"])
    assert all(0 <= i < prefix for i in row["indices"])
    assert set(range((prefix//8)*8, prefix)) <= set(row["indices"])


def test_future_keys_and_values_cannot_change_prefix_read(arrays):
    before = _run(arrays)
    K, V, q = [a.copy() for a in arrays]
    K[43:] = np.nan
    V[43:] = np.inf
    after = _run((K, V, q))
    for key in before:
        if key != "seconds_including_build_and_oracle":
            assert before[key] == after[key]


def test_values_cannot_change_route_mass_certificate_or_work(arrays):
    before = _run(arrays)
    K, V, q = [a.copy() for a in arrays]
    V[:] = np.random.default_rng(21).normal(size=V.shape)*100
    after = _run((K, V, q))
    for key in ("indices", "trace", "work", "omitted_mass_upper",
                "actual_omitted_mass", "top_set", "stop_reason"):
        assert before[key] == after[key]
    assert before["output"] != after["output"]


@pytest.mark.parametrize("coordinates", [0, 2, 6])
def test_fixed_coordinate_and_value_budget_has_no_duplicate_key_reads(arrays, coordinates):
    row = _run(arrays, coordinates=coordinates, eta=.001)
    n, d = row["prefix"], arrays[0].shape[1]
    k = len(row["indices"])
    assert k == 11 and row["stop_reason"] == "value_budget"
    work = row["work"]
    assert work["coordinate_scalars_read"] == n*coordinates+k*(d-coordinates)
    assert work["coordinate_scalars_read"] <= n*d
    assert work["value_rows_read"] == k
    assert work["value_scalars_read"] == k*arrays[1].shape[1]
    assert work["opened_keys"] == k
    assert work["fully_scored_keys"] == (n if coordinates == d else k)
    assert work["index_build_key_scalars"] == n*d
    assert work["index_value_bound_rows"] == n
    assert row["oracle_work"]["truth_value_rows"] == n


def test_progressive_schedule_retains_best_mass_bound_and_charges_unique_reads(arrays):
    row = _run(arrays, coordinates=0, coordinate_schedule=(2, 4, 6), eta=.001)
    trace = row["trace"]
    assert [t["coordinates"] for t in trace[:4]] == [0, 2, 4, 6]
    assert all(a["omitted_mass_upper"] >= b["omitted_mass_upper"]-2e-14
               for a, b in zip(trace, trace[1:]))
    assert row["work"]["coordinate_scalars_read"] == row["prefix"]*6
    assert row["work"]["value_rows_read"] == 11
    _dense_check(arrays, row)


def test_full_coordinate_unquantized_cap_is_actual_residual_mass(arrays):
    row = _run(arrays, coordinates=6)
    assert row["unquantized_item_cap_mass_upper"] == pytest.approx(
        row["actual_omitted_mass"], abs=2e-14)
    assert row["one_threshold_mass_upper"]+2e-14 >= row["omitted_mass_upper"]
    assert row["omitted_mass_upper"]+2e-14 >= row["unquantized_item_cap_mass_upper"]
    assert row["work"]["fully_scored_keys"] == row["prefix"]
    assert row["work"]["value_rows_read"] < row["prefix"]


def test_full_value_read_equals_dense(arrays):
    row = _run(arrays, seed_values=43, max_values=43, coordinates=0)
    _dense_check(arrays, row)
    assert row["indices"] == list(range(43))
    assert row["actual_omitted_mass"] == row["omitted_mass_upper"] == 0
    assert row["output_error_upper"] == row["kl_upper"] == 0
    assert row["actual_output_l2"] < 2e-13
    assert row["certified"]


def test_zero_query_uniform_mass_and_deterministic_ties(arrays):
    K, V, _ = arrays
    row = _run((K, V, np.zeros(6)), coordinates=0, max_values=7, seed_values=7)
    assert row["indices"] == [0, 1, 2, 3, 40, 41, 42]
    assert row["actual_omitted_mass"] == pytest.approx(36/43)
    assert row["unquantized_item_cap_mass_upper"] == pytest.approx(36/43)
    assert not row["top_set"]["1"]["certified"]
    assert not row["top_set"]["16"]["certified"]
    _dense_check((K, V, np.zeros(6)), row)


def test_harmless_centroid_extreme_unread_key_prevents_false_certificate():
    K = np.zeros((32, 2))
    K[17, 1] = 100
    K[18:24, 1] = -100/6
    V = np.arange(32.)[:, None]
    q = np.array([1., .5])
    np.testing.assert_allclose(K[16:24].mean(axis=0), 0, atol=1e-14)
    row = run_read(K, V, q, coordinates=1, max_values=4, seed_values=4,
                   batch=2, block=8, eta=.1)
    assert row["indices"] == [0, 1, 2, 3]
    assert row["actual_omitted_mass"] > .999999
    assert row["omitted_mass_upper"] > .999999
    assert not row["certified"]
    assert not row["top_set"]["1"]["certified"]
    _dense_check((K, V, q), row)


def test_single_mass_level_is_residual_maximum(arrays):
    row = _run(arrays, levels=1)
    assert row["omitted_mass_upper"] == pytest.approx(row["one_threshold_mass_upper"])
    _dense_check(arrays, row)


def test_exact_top_one_does_not_imply_small_omitted_attention_mass():
    K = np.column_stack((np.linspace(.01, 0., 32), np.zeros(32)))
    V = np.arange(32.)[:, None]
    q = np.array([1., 0.])
    row = run_read(K, V, q, coordinates=2, max_values=4, seed_values=4,
                   block=8, eta=.1)
    assert row["top_set"]["1"]["certified"]
    assert row["top_set"]["1"]["recall"] == 1
    assert not row["certified"]
    assert row["actual_omitted_mass"] > .8
    _dense_check((K, V, q), row)


def test_concentrated_exact_mass_can_stop_after_one_value():
    K = np.zeros((32, 2))
    K[0, 0] = 50
    V = np.arange(32.)[:, None]
    q = np.array([1., 0.])
    row = run_read(K, V, q, coordinates=1, max_values=4, seed_values=1,
                   block=8, eta=.01)
    assert row["indices"] == [0]
    assert row["certified"]
    assert row["stop_reason"] == "certificate"
    assert row["work"]["value_rows_read"] == 1
    assert row["work"]["coordinate_scalars_read"] == 33
    _dense_check((K, V, q), row)


@pytest.mark.parametrize("kind", ["concentrated", "random", "adversarial"])
def test_synthetic_comparison_runs_all_modes_with_identical_queries(kind):
    data = synthetic(kind, n=192, d=6, queries=2)
    report = evaluate_case(*data, coordinate_counts=(0, 2, 6), block=8)
    expected = [int(pos)+1 for pos in data[3]]
    for family in ("fixed_budget", "stopping", "progressive_10percent_value_cap"):
        for experiment in report[family].values():
            assert [row["prefix"] for row in experiment["rows"]] == expected
            assert experiment["summary"]["queries"] == 2
    assert report["audit"]["violations_above_1e_9"] == 0


@pytest.mark.parametrize("overrides", [
    {"prefix": 0}, {"prefix": 54}, {"coordinates": -1}, {"coordinates": 7},
    {"coordinate_schedule": (2, 4)}, {"coordinate_schedule": (4, 3)},
    {"max_values": 2}, {"eta": 0}, {"eta": 1}, {"batch": 0},
])
def test_invalid_protocol_settings_rejected(arrays, overrides):
    with pytest.raises(ValueError):
        _run(arrays, **overrides)
