import pytest
import torch
from ssa.tail_tree_router import batched_tail_tree_routes
from ssa.hybrid_tail_attention import hybrid_tail_attention


@pytest.mark.parametrize("n", [3, 17, 65, 129, 259])
def test_batched_tree_exhaustive_beam_matches_causal_flat(n):
    torch.manual_seed(n)
    q, k, v = torch.randn(1, 4, n, 5), torch.randn(1, 2, n, 5), torch.randn(1, 2, n, 6)
    routes, stats = batched_tail_tree_routes(q, k, block=4, beam=128, query_chunk=13, return_stats=True)
    if n > 4:
        actual = hybrid_tail_attention(q, k, v, block=4, cells=3, routing_blocks=routes)
        expected = hybrid_tail_attention(q, k, v, block=4, cells=3, top_blocks=2)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    assert stats["node_bound_slots"] > 0
    assert ((routes < torch.arange(n)[None, None, :, None] // 4) | (routes == -1)).all()


def test_fixed_beam_future_isolation_and_chunk_invariance():
    torch.manual_seed(47)
    q, k = torch.randn(2, 4, 259, 5), torch.randn(2, 2, 259, 5)
    before = batched_tail_tree_routes(q, k, block=4, beam=4, query_chunk=17)
    replay = batched_tail_tree_routes(q, k, block=4, beam=4, query_chunk=31)
    assert torch.equal(before, replay)
    q[:, :, 147:] = 1e6; k[:, :, 147:] = -1e6
    after = batched_tail_tree_routes(q, k, block=4, beam=4, query_chunk=23)
    assert torch.equal(before[:, :, :147], after[:, :, :147])


def test_batched_tree_equal_scores_deterministic():
    q, k = torch.zeros(1, 4, 99, 5), torch.zeros(1, 2, 99, 5)
    a = batched_tail_tree_routes(q, k, block=4, beam=4)
    b = batched_tail_tree_routes(q, k, block=4, beam=4)
    assert torch.equal(a, b)
    assert not ((a[..., 0] == a[..., 1]) & (a[..., 0] >= 0)).any()
