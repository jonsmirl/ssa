"""Dense-oracle and CUDA-graph checks for a device-only fixed proposal."""
import inspect

import pytest
import torch

pytest.importorskip("triton")
from ssa.device_coordinate_attention import DeviceCoordinateIndex, DeviceCoordinatePlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _data(n=257, heads=2, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(491)
    return (torch.randn(n, 64, device="cuda", generator=g, dtype=dtype),
            torch.randn(n, 8, device="cuda", generator=g, dtype=dtype),
            torch.randn(heads, 64, device="cuda", generator=g, dtype=dtype))


def _audit(K, V, q, result, eta=.1):
    score = q.double() @ K.double().T / 8
    p = score.softmax(1)
    dense = p @ V.double()
    proposed_omitted = p.masked_fill(result["proposed_selected_mask"], 0).sum(1)
    omitted = p.masked_fill(result["selected_mask"], 0).sum(1)
    assert torch.all(proposed_omitted <= result["proposed_mass_upper"] + 1e-10)
    assert torch.all(omitted <= result["mass_upper"] + 1e-10)
    assert torch.all(result["lower"].double() <= score + 1e-10)
    assert torch.all(result["upper"].double() >= score - 1e-10)
    error = (result["output"].double() - dense).norm(dim=1)
    assert torch.all(error <= result["output_error_upper"] + 1e-8)
    assert result["numerically_valid"].all()
    assert result["certified"].all()
    assert torch.all(result["mass_upper"] <= eta)
    exact_kl = score.logsumexp(1) - score.masked_fill(~result["selected_mask"], -torch.inf).logsumexp(1)
    assert torch.all(exact_kl <= result["kl_upper"] + 1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("r", [0, 8, 32, 64])
def test_fixed_proposal_mass_and_fallback(r, dtype):
    K, V, q = _data(dtype=dtype)
    plan = DeviceCoordinateIndex(K, V).prepare(q, coordinates=r, fraction=.25)
    result = plan.run()
    _audit(K, V, q, result)
    c = result["counts"]
    assert torch.equal(c["proposal_key_coordinate_scalars"], torch.full_like(c["key_coordinate_scalars"], len(K)*r+plan.count*(64-r)))
    assert torch.equal(c["fallback_key_coordinate_scalars"], (~result["accepted"]).long()*len(K)*64)
    assert torch.equal(c["value_scalars"], torch.where(result["accepted"], plan.count, len(K))*V.shape[1])


@pytest.mark.parametrize("n", [1, 63, 64, 65, 127])
def test_causal_prefix_snapshot_and_partial_boundary(n):
    K, V, q = _data(n=128)
    K[n:] = torch.nan; V[n:] = torch.nan
    plan = DeviceCoordinateIndex(K[:n], V[:n]).prepare(q, coordinates=32, fraction=.1)
    result = plan.run()
    _audit(K[:n], V[:n], q, result)
    assert result["proposed_selected_mask"][:, n//64*64:].all()


def test_mixed_accept_reject_and_no_wasted_proposal_values():
    K, V, q = _data(n=256)
    K.zero_(); K[0, 0] = 80
    q.zero_(); q[0, 0] = 8
    result = DeviceCoordinateIndex(K, V).prepare(q, coordinates=32, fraction=.125).run()
    _audit(K, V, q, result)
    assert result["accepted"].tolist() == [True, False]
    assert result["counts"]["value_scalars"].tolist() == [32*8, 256*8]
    assert result["counts"]["fallback_key_coordinate_scalars"].tolist() == [0, 256*64]
    assert result["selected_mask"][1].all()
    assert int(result["selected_mask"][0].sum()) == 32


def test_equal_logits_deterministic_ties():
    K, V, q = _data(n=256)
    q.zero_()
    result = DeviceCoordinateIndex(K, V).prepare(q, fraction=.125).run()
    assert result["proposed_selected_mask"][:, :32].all()
    assert not result["proposed_selected_mask"][:, 32:].any()
    assert not result["accepted"].any()
    _audit(K, V, q, result)


@pytest.mark.parametrize("common_score", [1e8, -1e8, 1e9, -1e9])
@pytest.mark.parametrize("fraction", [.125, 1.])
def test_large_common_scores_preserve_sparse_and_fallback_normalization(common_score, fraction):
    K, V, q = _data(n=128)
    K.zero_(); K[:, 0] = common_score
    V.fill_(1)
    q.zero_(); q[:, 0] = 8
    result = DeviceCoordinateIndex(K, V).prepare(q, coordinates=32, fraction=fraction).run()
    _audit(K, V, q, result)
    torch.testing.assert_close(result["output"], torch.ones_like(result["output"]), atol=1e-6, rtol=1e-6)


def test_full_fallback_matches_dense():
    K, V, q = _data(n=1024)
    result = DeviceCoordinateIndex(K, V).prepare(q, coordinates=8, fraction=.01).run()
    assert not result["accepted"].any()
    _audit(K, V, q, result)
    dense = (q.double() @ K.double().T / 8).softmax(1) @ V.double()
    torch.testing.assert_close(result["output"].double(), dense, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_graph_replay_changed_queries_changes_gpu_decision(dtype):
    K, V, q = _data(n=256, dtype=dtype)
    K.zero_(); K[0, 0] = 80
    q.zero_(); q[0, 0] = 8
    plan = DeviceCoordinateIndex(K, V).prepare(q, coordinates=32, fraction=.125)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            plan.run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = plan.run()
    graph.replay()
    assert captured["accepted"].tolist() == [True, False]
    _audit(K, V, q, captured)
    q.zero_(); q[1, 0] = 8
    graph.replay()
    assert captured["accepted"].tolist() == [False, True]
    _audit(K, V, q, captured)


def test_prepare_validates_without_silent_query_downcast():
    K, V, q = _data(n=64)
    ix = DeviceCoordinateIndex(K, V)
    with pytest.raises(ValueError):
        ix.prepare(q.double())
    with pytest.raises(ValueError):
        ix.prepare(q, fraction=0)
    with pytest.raises(ValueError):
        ix.prepare(q, coordinates=65)


def test_run_source_has_no_scalar_extraction_or_dynamic_selection():
    source = inspect.getsource(DeviceCoordinatePlan.run)
    for forbidden in (".item(", ".tolist(", ".cpu(", ".nonzero(", ".synchronize("):
        assert forbidden not in source
