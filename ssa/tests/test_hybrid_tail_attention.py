import torch
import pytest
from torch.nn import functional as F

from ssa.hybrid_tail_attention import hybrid_tail_attention


def data():
    torch.manual_seed(41)
    return (torch.randn(1, 4, 23, 6), torch.randn(1, 2, 23, 6), torch.randn(1, 2, 23, 5))


def test_complete_selection_matches_dense_gqa_at_partial_prefix():
    q, k, v = data()
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    for tail in (False, True):
        result = hybrid_tail_attention(q, k, v, block=4, top_blocks=8, cells=3,
                                       query_chunk=5, use_tail=tail)
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)


def test_future_keys_values_and_queries_cannot_change_past_outputs():
    q, k, v = data()
    before = hybrid_tail_attention(q, k, v, block=4, top_blocks=1, cells=3, query_chunk=5)
    q[:, :, 13:] = 100
    k[:, :, 13:] = -50
    v[:, :, 13:] = 12
    after = hybrid_tail_attention(q, k, v, block=4, top_blocks=1, cells=3, query_chunk=7)
    torch.testing.assert_close(before[:, :, :13], after[:, :, :13], rtol=2e-5, atol=2e-6)


def test_tail_gain_has_finite_nonzero_gradient_through_real_attention_output():
    q, k, v = data()
    q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
    gains = torch.zeros(4, requires_grad=True)
    output = hybrid_tail_attention(q, k, v, block=4, top_blocks=1, cells=3,
                                   query_chunk=5, log_gain=gains)
    grad = torch.autograd.grad(output.square().sum(), (q, k, v, gains))
    for g in grad:
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0


def test_gqa_matches_explicit_key_value_head_repetition():
    q, k, v = data()
    gqa = hybrid_tail_attention(q, k, v, block=4, top_blocks=1, cells=3)
    expanded = hybrid_tail_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                                     block=4, top_blocks=1, cells=3)
    torch.testing.assert_close(gqa, expanded)


def test_supplied_tree_blocks_obey_causal_cut_and_duplicate_exclusion():
    q, k, v = data()
    routes = torch.tensor([0, 0, 1, 2, 3, 4, 5, 100, -1]).expand(1, 4, 23, -1)
    result = hybrid_tail_attention(q, k, v, block=4, cells=3, routing_blocks=routes)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)


def test_existing_tree_adapter_matches_flat_routes_and_has_no_future_leak(monkeypatch):
    pytest.importorskip("faiss")
    from ssa import cascade_router
    from ssa.tail_tree_router import tail_tree_routes
    monkeypatch.setattr(cascade_router, "DEV", "cpu")
    q, k, v = data()
    routes = tail_tree_routes(q, k, block=4, top_blocks=2)
    result = hybrid_tail_attention(q, k, v, block=4, cells=3, routing_blocks=routes)
    flat = hybrid_tail_attention(q, k, v, block=4, cells=3, top_blocks=2)
    torch.testing.assert_close(result, flat, rtol=2e-5, atol=2e-6)
    q[:, :, 13:] = 1e4
    k[:, :, 13:] = -1e5
    after = tail_tree_routes(q, k, block=4, top_blocks=2)
    torch.testing.assert_close(routes[:, :, :13], after[:, :, :13])


def test_checkpoint_recomputation_matches_tail_gain_gradient():
    from torch.utils.checkpoint import checkpoint
    q, k, v = data()
    gain = torch.zeros(4, requires_grad=True)
    def attention(q, k, v, gain):
        return hybrid_tail_attention(q, k, v, block=4, cells=3, top_blocks=1, log_gain=gain)
    direct = attention(q, k, v, gain)
    recomputed = checkpoint(attention, q, k, v, gain, use_reentrant=False)
    torch.testing.assert_close(direct, recomputed)
    torch.testing.assert_close(torch.autograd.grad(direct.square().sum(), gain)[0],
                               torch.autograd.grad(recomputed.square().sum(), gain)[0])
