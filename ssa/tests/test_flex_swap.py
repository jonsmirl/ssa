"""The flex kernel path inside the real-model swap (ssa_kernel.block_route_budget + gemma_ssa impl='flex').

CPU tests pin the budget router's causal/counting invariants; GPU-gated tests check the fused kernel path
reproduces dense sdpa / the analytic path and never leaks the future (including non-block-multiple n)."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
import pytest

cuda = torch.cuda.is_available()


# ---- CPU: block_route_budget invariants ----------------------------------------------------------

def test_budget_router_causal_and_counts():
    from ssa.ssa_kernel import block_route_budget
    torch.manual_seed(0)
    n, block = 16 * 128, 128
    nb = n // block
    q = torch.randn(1, 1, n, 64); k = torch.randn(1, 1, n, 64)
    kv_num, kv_idx, sel = block_route_budget(q, k, block, budget_frac=0.25, local=1)
    qi = torch.arange(nb)
    assert not (sel & (qi[None, None, :, None] < qi[None, None, None, :])).any()   # no future block
    assert (kv_num >= 1).all()                                                     # own block kept
    for i in range(nb):
        ids = kv_idx[0, 0, i, :int(kv_num[0, 0, i])]
        assert (ids <= i).all() and i in ids.tolist()


def test_budget_router_monotone_in_budget():
    from ssa.ssa_kernel import block_route_budget
    torch.manual_seed(1)
    n, block = 32 * 128, 128
    q = torch.randn(1, 1, n, 64); k = torch.randn(1, 1, n, 64)
    counts = []
    for frac in (1.0, 0.5, 0.25, 0.1):
        kv_num, _, _ = block_route_budget(q, k, block, budget_frac=frac, local=0)
        counts.append(int(kv_num.sum()))
    assert counts[0] > counts[1] > counts[2] >= counts[3], counts


# ---- GPU: the fused kernel path --------------------------------------------------------------------

def _mod(groups=1, scaling=1.0):
    return SimpleNamespace(num_key_value_groups=groups, is_sliding=False, scaling=scaling, layer_idx=0)


@pytest.mark.skipif(not cuda, reason="flex kernel needs CUDA")
def test_flex_full_budget_matches_dense_sdpa():
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import ssa_attention_forward
    torch.manual_seed(0)
    n, d = 512, 64
    q = torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    G.CFG = G.SSAConfig(block=128, budget_frac=1.0, impl="flex")
    out, _ = ssa_attention_forward(_mod(groups=1), q, k, v, scaling=1.0 / (d ** 0.5))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    assert torch.allclose(out, ref, atol=2e-2), (out - ref).abs().max().item()


@pytest.mark.skipif(not cuda, reason="flex kernel needs CUDA")
def test_flex_matches_analytic_at_full_budget():
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import ssa_attention_forward
    torch.manual_seed(0)
    n, d = 512, 64
    q = torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    G.CFG = G.SSAConfig(block=128, budget_frac=1.0, impl="analytic")
    ref, _ = ssa_attention_forward(_mod(), q, k, v, scaling=1.0 / (d ** 0.5))
    G.CFG = G.SSAConfig(block=128, budget_frac=1.0, impl="flex")
    out, _ = ssa_attention_forward(_mod(), q, k, v, scaling=1.0 / (d ** 0.5))
    assert torch.allclose(out, ref, atol=2e-2), (out - ref).abs().max().item()


@pytest.mark.skipif(not cuda, reason="flex kernel needs CUDA")
def test_flex_no_future_leak():
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import ssa_attention_forward
    torch.manual_seed(0)
    n, d, cut = 512, 64, 256
    q = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    G.CFG = G.SSAConfig(block=128, budget_frac=0.5, impl="flex")
    out1, _ = ssa_attention_forward(_mod(), q, k.clone(), v.clone(), scaling=1.0 / (d ** 0.5))
    k2, v2 = k.clone(), v.clone()
    k2[:, :, cut + 1:] += 5.0 * torch.randn_like(k2[:, :, cut + 1:])
    v2[:, :, cut + 1:] += 5.0 * torch.randn_like(v2[:, :, cut + 1:])
    out2, _ = ssa_attention_forward(_mod(), q, k2, v2, scaling=1.0 / (d ** 0.5))
    assert torch.allclose(out1[:, :cut + 1], out2[:, :cut + 1], atol=2e-2)


@pytest.mark.skipif(not cuda, reason="flex kernel needs CUDA")
def test_flex_pad_nonmultiple_length():
    """n not a multiple of block: padding + (kv < n) mask_mod must still match dense on the real tokens."""
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import ssa_attention_forward
    torch.manual_seed(0)
    n, d = 300, 64
    q = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    G.CFG = G.SSAConfig(block=128, budget_frac=1.0, impl="flex")
    out, _ = ssa_attention_forward(_mod(), q, k, v, scaling=1.0 / (d ** 0.5))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    assert out.shape == (1, n, 1, d)
    assert torch.allclose(out, ref, atol=2e-2), (out - ref).abs().max().item()
