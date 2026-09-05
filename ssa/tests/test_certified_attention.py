"""Dense-oracle checks for the summary-only attention error certificates."""
import numpy as np
import pytest

from ssa.certified_attention import CertifiedBlockAttention
from ssa.core import dense_read, softmax


def check_against_dense(index, q, beta, result, prefix=None):
    end = len(index.K) if prefix is None else prefix
    dense, p, _ = dense_read(q, index.K[:end], index.V[:end], beta)
    ids = result.indices
    assert len(np.unique(ids)) == len(ids) == result.keys_scored
    assert np.all((0 <= ids) & (ids < end))
    kept = softmax(beta * (index.K[ids] @ q))
    np.testing.assert_allclose(result.output, kept @ index.V[ids], atol=2e-12, rtol=2e-12)
    missing = np.ones(end, dtype=bool)
    missing[ids] = False
    dropped = p[missing].sum()
    assert dropped <= result.mass_upper + 2e-12
    assert np.linalg.norm(result.output - dense) <= result.output_error_upper + 2e-11
    sparse = np.zeros(end)
    sparse[ids] = kept
    tv = 0.5 * np.abs(sparse - p).sum()
    np.testing.assert_allclose(tv, dropped, atol=2e-12)
    kl = float(np.sum(kept * np.log(kept / p[ids])))
    assert kl <= result.kl_upper + 2e-11
    np.testing.assert_allclose(kl, -np.log(p[ids].sum()), atol=2e-11)


@pytest.mark.parametrize("beta", [0.0, 0.2, 4.0, 30.0])
@pytest.mark.parametrize("block", [1, 7, 32])
def test_random_partitions_caps_and_causal_prefixes(beta, block):
    rng = np.random.default_rng(123)
    K, V = rng.normal(size=(95, 9)), rng.normal(size=(95, 5))
    index = CertifiedBlockAttention(K, V, block)
    q = rng.normal(size=9)
    for prefix in (1, 53, 95):
        for cap in (1, 2, 5, 100):
            result = index.read(q, beta, prefix=prefix, max_blocks=cap,
                                mass_tol=1e-4, error_tol=1e-3)
            check_against_dense(index, q, beta, result, prefix)
            assert result.blocks_opened <= cap
            assert result.certified == (result.mass_upper <= 1e-4
                                        and result.output_error_upper <= 1e-3)


def test_high_margin_opens_one_block_with_small_certified_error():
    K = np.zeros((1024, 3))
    K[:16, 0] = 10
    V = np.random.default_rng(4).normal(size=(1024, 4))
    index = CertifiedBlockAttention(K, V, 16)
    result = index.read([1, 0, 0], beta=3, mass_tol=1e-8, error_tol=1e-8)
    assert result.certified
    assert result.keys_scored == 16
    assert result.bounds_evaluated == 64
    check_against_dense(index, [1, 0, 0], 3, result)


def test_flat_logits_require_mass_and_kl_even_with_exact_argmax():
    index = CertifiedBlockAttention(np.zeros((64, 2)), np.eye(64), 8)
    capped = index.read([1, 0], max_blocks=1, mass_tol=0.01)
    assert not capped.certified
    assert capped.mass_upper == pytest.approx(7 / 8)
    assert capped.kl_upper == pytest.approx(np.log(8))  # nested-uniform-support theorem
    check_against_dense(index, [1, 0], 1, capped)
    full = index.read([1, 0], mass_tol=0.01)
    assert full.certified and full.keys_scored == 64
    assert full.certificate_checks == 4
    assert full.mass_upper == full.kl_upper == full.output_error_upper == 0


def test_equal_values_certify_output_despite_large_omitted_mass():
    index = CertifiedBlockAttention(np.zeros((128, 3)), np.ones((128, 5)), 8)
    output_only = index.read([0, 0, 0], error_tol=0)
    assert output_only.certified and output_only.keys_scored == 8
    assert output_only.output_error_upper == 0
    assert output_only.mass_upper > 0.9
    both = index.read([0, 0, 0], error_tol=0, mass_tol=0.01)
    assert both.certified and both.keys_scored == 128


def test_small_mass_distinctive_value_still_requires_escalation():
    K, V = np.array([[0.0], [-20.0]]), np.array([[0.0], [1e12]])
    index = CertifiedBlockAttention(K, V, 1)
    capped = index.read([1], mass_tol=1e-6, error_tol=1, max_blocks=1)
    assert capped.mass_upper < 1e-6
    assert capped.output_error_upper > 1000 and not capped.certified
    check_against_dense(index, [1], 1, capped)
    assert index.read([1], mass_tol=1e-6, error_tol=1).keys_scored == 2


def test_weighted_value_bound_ignores_redundant_high_mass_block():
    # Omitted high-mass values match the kept output. Only a tiny tail differs.
    index = CertifiedBlockAttention([[0], [0], [-20]], [[0], [0], [1]], 1)
    result = index.read([1], initial_blocks=[0], max_blocks=1, error_tol=1e-8)
    assert result.mass_upper > 0.49
    assert result.certified and result.output_error_upper < 2e-9
    check_against_dense(index, [1], 1, result)


def test_prefix_has_no_future_dependence_and_initial_blocks_are_deduplicated():
    rng = np.random.default_rng(12)
    K, V = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
    q, prefix = rng.normal(size=3), 19
    a = CertifiedBlockAttention(K, V, 8).read(q, prefix=prefix,
                                            initial_blocks=[0, 0], max_blocks=2)
    K[prefix:], V[prefix:] = 1e6, -1e6
    b = CertifiedBlockAttention(K, V, 8).read(q, prefix=prefix,
                                            initial_blocks=[0], max_blocks=2)
    np.testing.assert_array_equal(a.output, b.output)
    np.testing.assert_array_equal(a.indices, b.indices)
    assert a.mass_upper == b.mass_upper
    assert a.output_error_upper == b.output_error_upper
    assert a.keys_scored == 11  # one full seed block plus three visible boundary keys


def test_log_space_extremes_and_zero_tolerance_do_not_false_certify():
    index = CertifiedBlockAttention([[10000], [-10000]], [[0], [1]], 1)
    limited = index.read([1], mass_tol=0, max_blocks=1)
    assert not limited.certified
    assert 0 < limited.mass_upper < 1e-300
    assert 0 < limited.kl_upper < 1e-300
    assert 0 < limited.output_error_upper < 1e-300
    assert index.read([1], mass_tol=0).keys_scored == 2
    # Large common logit offsets must not overflow or change the decision.
    rng = np.random.default_rng(0)
    K, V = rng.normal(size=(32, 1)), rng.normal(size=(32, 2))
    a = CertifiedBlockAttention(K, V, 4).read([1], max_blocks=2)
    b = CertifiedBlockAttention(K + 10000, V, 4).read([1], max_blocks=2)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_allclose(a.output, b.output, atol=1e-11)
    assert a.mass_upper == pytest.approx(b.mass_upper)


def test_snapshot_does_not_alias_callers_arrays():
    K, V = np.arange(8.0)[:, None], np.ones((8, 2))
    index = CertifiedBlockAttention(K, V, 2)
    K[:] = 0
    V[:] = 5
    np.testing.assert_array_equal(index.V, np.ones((8, 2)))
    assert index.K[-1, 0] == 7
    with pytest.raises(ValueError):
        index.V[0] = 0


@pytest.mark.parametrize("kwargs", [
    {"beta": -1}, {"mass_tol": -1}, {"mass_tol": 2}, {"error_tol": np.nan},
    {"prefix": 0}, {"prefix": 9}, {"prefix": 1.5}, {"max_blocks": 0},
    {"initial_blocks": [4]}, {"initial_blocks": [-1]},
    {"prefix": 7, "initial_blocks": [0], "max_blocks": 1},
])
def test_invalid_requests_are_rejected(kwargs):
    index = CertifiedBlockAttention(np.zeros((8, 2)), np.zeros((8, 1)), 2)
    with pytest.raises(ValueError):
        index.read([1, 0], **kwargs)


@pytest.mark.parametrize("K,V,block", [
    ([], [], 1), ([[0]], [[0], [1]], 1), ([[np.nan]], [[0]], 1),
    ([[0]], [[np.inf]], 1), ([[0]], [[0]], 0), ([[0]], [[0]], 1.5),
])
def test_invalid_snapshots_are_rejected(K, V, block):
    with pytest.raises(ValueError):
        CertifiedBlockAttention(K, V, block)


def test_overflowed_value_summary_cannot_produce_a_certificate():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="summary arithmetic"):
            CertifiedBlockAttention([[0], [0]], [[1e308], [1e308]], 2)
