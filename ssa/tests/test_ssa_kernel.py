"""GPU-gated correctness for the block-sparse SSA kernel (skipped on CPU / without FlexAttention)."""
import pytest
import torch

cuda = torch.cuda.is_available()


@pytest.mark.skipif(not cuda, reason="SSA kernel needs CUDA + FlexAttention")
def test_full_budget_matches_dense():
    """With every causal block selected (top_c = #blocks), block-sparse SSA == dense causal attention."""
    from ssa.ssa_kernel import ssa_flex, dense, BLOCK
    torch.manual_seed(0)
    n = 4 * BLOCK
    q = torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    out_dense = dense(q, k, v)
    out_ssa = ssa_flex(q, k, v, top_c=n // BLOCK, local=n // BLOCK)   # select everything
    assert torch.allclose(out_ssa, out_dense, atol=2e-2)


@pytest.mark.skipif(not cuda, reason="SSA kernel needs CUDA")
def test_routing_is_sparse_and_causal():
    """A tight budget selects a strict subset, and never a future block."""
    from ssa.ssa_kernel import block_route, BLOCK
    torch.manual_seed(0)
    n = 8 * BLOCK
    q = torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    _, _, sel = block_route(q, k, top_c=2, local=1)
    nb = n // BLOCK
    qi = torch.arange(nb, device="cuda")
    future = qi[None, None, :, None] < qi[None, None, None, :]        # kv block > query block
    assert not (sel & future).any()                                  # no future blocks
    assert sel.float().mean().item() < 1.0                           # strictly sparse
