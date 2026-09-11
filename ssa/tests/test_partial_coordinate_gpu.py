"""Dense-oracle CUDA checks; these test, not prove, floating-point guards."""
import math

import pytest
import torch

pytest.importorskip("triton")
from ssa.partial_coordinate_gpu import CoordinateGPUIndex

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _data(n=193, heads=2, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(312)
    return (torch.randn(n, 64, generator=g, device="cuda", dtype=dtype),
            torch.randn(n, 8, generator=g, device="cuda", dtype=dtype),
            torch.randn(heads, 64, generator=g, device="cuda", dtype=dtype))


def _audit(K, V, q, result):
    scores = q.double() @ K.double().T / 8
    probability = scores.softmax(-1)
    dense = probability @ V.double()
    assert torch.all(result["lower"].double() <= scores + 1e-10)
    assert torch.all(result["upper"].double() >= scores - 1e-10)
    omitted = probability.masked_fill(result["selected_mask"], 0).sum(-1)
    assert torch.all(omitted <= result["mass_upper"] + 1e-10)
    error = (result["output"].double() - dense).norm(dim=-1)
    assert torch.all(error <= result["output_error_upper"] + 1e-8)
    for h, counts in enumerate(result["counts"]):
        kept = result["selected_mask"][h]
        exact_kl = scores[h].logsumexp(0) - scores[h, kept].logsumexp(0)
        assert exact_kl <= result["kl_upper"][h] + 1e-10
        assert counts["selected_keys"] == int(kept.sum())


@pytest.mark.parametrize("r", [0, 8, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_intervals_mass_output_and_dense_fallback(r, dtype):
    K, V, q = _data(dtype=dtype)
    ix = CoordinateGPUIndex(K, V)
    out = ix.read(q, coordinates=r, eta=.1, seed=16)
    _audit(K, V, q, out)
    assert out["certified"].all()
    assert torch.all(out["mass_upper"] <= .1)
    for c in out["counts"]:
        assert c["key_coordinate_scalars"] == len(K) * r + c["selected_keys"] * (64-r)


@pytest.mark.parametrize("n", [1, 63, 64, 65, 191])
def test_causal_prefix_and_boundary(n):
    K, V, q = _data(n=192)
    K[n:] = torch.nan
    V[n:] = torch.nan
    ix = CoordinateGPUIndex(K[:n], V[:n])
    out = ix.read(q, coordinates=32, max_values=min(n, 64), seed=16)
    _audit(K[:n], V[:n], q, out)
    assert out["selected_mask"][:, n//64*64:].all()


def test_equal_logits_deterministic_ties_and_value_independence():
    K, V, q = _data(n=256)
    q.zero_()
    ix = CoordinateGPUIndex(K, V)
    a = ix.read(q, coordinates=8, max_values=16, seed=16)
    b = CoordinateGPUIndex(K, V * 13).read(q, coordinates=8, max_values=16, seed=16)
    assert a["selected_mask"][:, :16].all()
    assert not a["selected_mask"][:, 16:].any()
    assert torch.equal(a["selected_mask"], b["selected_mask"])
    _audit(K, V, q, a)
    assert not a["certified"].any()


def test_full_coordinates_do_not_fetch_all_values():
    K, V, q = _data(n=256)
    K.zero_(); K[0, 0] = 100
    q.zero_(); q[:, 0] = 8
    out = CoordinateGPUIndex(K, V).read(q, coordinates=64, seed=4)
    _audit(K, V, q, out)
    assert out["certified"].all()
    for c in out["counts"]:
        assert c["fully_scored_keys"] == 256
        assert c["selected_keys"] == 4
        assert c["value_scalars"] == 32


def test_centroid_harmless_extreme_key_is_bounded():
    K, V, q = _data(n=256)
    K.zero_(); K[:, 63] = -100/63
    K[::64, 63] = 100
    q.zero_(); q[:, 0] = 2; q[:, 63] = 1
    assert K[:64].mean(0).abs().max() < 1e-5
    out = CoordinateGPUIndex(K, V).read(q, coordinates=1, seed=8, max_values=8)
    _audit(K, V, q, out)
    assert not out["certified"].any()


def test_snapshot_copies_inputs_and_validates():
    K, V, q = _data(n=64)
    ix = CoordinateGPUIndex(K, V)
    K.zero_(); V.zero_()
    assert ix.K.abs().sum() > 0
    assert ix.V.abs().sum() > 0
    with pytest.raises(ValueError):
        ix.read(q, coordinates=65)
    with pytest.raises(ValueError):
        ix.read(q, eta=0)
    with pytest.raises(ValueError):
        ix.read(q, coordinates=1.5)
    with pytest.raises(ValueError, match="float16, bfloat16 or float32"):
        ix.read(q.double())


def test_forced_full_read_matches_dense_and_zero_tail():
    K, V, q = _data(n=128)
    out = CoordinateGPUIndex(K, V).read(q, coordinates=32, seed=128)
    _audit(K, V, q, out)
    assert out["selected_mask"].all()
    assert torch.equal(out["mass_upper"], torch.zeros_like(out["mass_upper"]))
    dense = (q.double() @ K.double().T / math.sqrt(64)).softmax(-1) @ V.double()
    torch.testing.assert_close(out["output"].double(), dense, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("r", [0, 8, 32, 64])
@pytest.mark.parametrize("eta", [.1, .01])
def test_cached_tail_order_preserves_policy_and_certificates(r, eta):
    K, V, q = _data(n=513)
    ix = CoordinateGPUIndex(K, V)
    baseline = ix.read(q, coordinates=r, eta=eta, seed=16, reuse_tail_order=False)
    cached = ix.read(q, coordinates=r, eta=eta, seed=16, reuse_tail_order=True)
    _audit(K, V, q, baseline)
    _audit(K, V, q, cached)
    assert torch.equal(baseline["selected_mask"], cached["selected_mask"])
    assert torch.equal(baseline["output"], cached["output"])
    torch.testing.assert_close(baseline["mass_upper"], cached["mass_upper"], atol=1e-12, rtol=1e-12)
    for old, new, old_trace, new_trace in zip(baseline["counts"], cached["counts"], baseline["trace"], cached["trace"]):
        assert [x["selected"] for x in old_trace] == [x["selected"] for x in new_trace]
        assert new["key_coordinate_scalars"] == old["key_coordinate_scalars"]
        assert new["sort_items"] <= old["sort_items"]
        assert new["tail_reduction_slots"] == len(K)
        assert new["tail_preprocessing_slots"] > 0
        assert old["tail_preprocessing_slots"] == 0
        assert new["tail_reduction_slots"] + new["tail_preprocessing_slots"] < old["tail_reduction_slots"]


def test_cached_tail_order_preserves_ties_with_fixed_budget():
    K, V, q = _data(n=256)
    q.zero_()
    ix = CoordinateGPUIndex(K, V)
    old = ix.read(q, coordinates=8, seed=16, max_values=63, reuse_tail_order=False)
    new = ix.read(q, coordinates=8, seed=16, max_values=63, reuse_tail_order=True)
    assert torch.equal(old["selected_mask"], new["selected_mask"])
    assert new["selected_mask"][:, :63].all()
    assert not new["selected_mask"][:, 63:].any()


@pytest.mark.parametrize("common_score", [1e8, -1e8, 1e9, -1e9])
def test_value_softmax_remains_normalized_at_large_common_logits(common_score):
    K, V, q = _data(n=128)
    K.zero_(); K[:, 0] = common_score
    V.fill_(1)
    q.zero_(); q[:, 0] = 8
    out = CoordinateGPUIndex(K, V).read(q, coordinates=32, seed=128)
    _audit(K, V, q, out)
    torch.testing.assert_close(out["output"], torch.ones_like(out["output"]), atol=1e-6, rtol=1e-6)
