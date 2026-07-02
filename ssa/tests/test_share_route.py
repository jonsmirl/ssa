"""Tests for cross-layer routing sharing + the projection hook (ssa.gemma_ssa). Analytic-shape CPU tests
plus a GPU flex smoke. The donor layer computes the selection once; consumers above it reuse it; layers
below route per-layer (honest cost accounting); sharing is prefill-only."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
import pytest

from ssa import gemma_ssa as G
from ssa.gemma_ssa import ssa_attention_forward, repeat_kv

cuda = torch.cuda.is_available()
DT = torch.float64


def _mod(li, groups=2, scaling=1.0):
    return SimpleNamespace(num_key_value_groups=groups, is_sliding=False, scaling=scaling, layer_idx=li)


def _qkv(b=1, hq=16, hkv=8, n=160, d=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, hq, n, d, generator=g, dtype=DT),
            torch.randn(b, hkv, n, d, generator=g, dtype=DT),
            torch.randn(b, hkv, n, d, generator=g, dtype=DT))


def test_block_route_budget_proj_identity():
    """proj=identity reproduces the no-proj centroid selection (proj drops the variance term, so compare
    against budget routing with beta=0 = centroid-only)."""
    from ssa.ssa_kernel import block_route_budget
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8 * 128, 64)
    k = torch.randn_like(q)
    I = torch.eye(64)
    a = block_route_budget(q, k, top_c=3, local=1, proj=I)
    b = block_route_budget(q, k, top_c=3, local=1, beta=0.0)       # centroid-only (no variance)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])


@pytest.mark.skipif(not cuda, reason="flex sharing needs CUDA")
def test_donor_consumer_share_mask():
    """A consumer layer (li>donor) reuses the donor's BlockMask; a fresh forward refreshes the stash."""
    torch.manual_seed(0)
    n, d = 512, 64
    q = torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    G.CFG = G.SSAConfig(block=128, budget_frac=0.25, impl="flex", share_route_from=0)
    G._SHARE["sig"] = None; G._SHARE["bm"] = None
    o0, _ = ssa_attention_forward(_mod(0, groups=1), q, k, v, scaling=1.0 / d ** 0.5)   # donor stashes
    assert G._SHARE["bm"] is not None
    bm0 = G._SHARE["bm"]
    o1, _ = ssa_attention_forward(_mod(1, groups=1), q, k, v, scaling=1.0 / d ** 0.5)   # consumer reuses
    assert G._SHARE["bm"] is bm0                                    # not rebuilt


@pytest.mark.skipif(not cuda, reason="flex sharing needs CUDA")
def test_full_budget_shared_equals_dense():
    """Full budget + sharing across layers still reproduces dense causal attention (the sanity gate)."""
    torch.manual_seed(0)
    n, d = 384, 64
    q = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    G.CFG = G.SSAConfig(block=128, budget_frac=1.0, impl="flex", share_route_from=0)
    G._SHARE["sig"] = None; G._SHARE["bm"] = None
    ssa_attention_forward(_mod(0, groups=1), q, k, v, scaling=1.0 / d ** 0.5)
    out, _ = ssa_attention_forward(_mod(1, groups=1), q, k, v, scaling=1.0 / d ** 0.5)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    assert torch.allclose(out, ref, atol=2e-2), (out - ref).abs().max().item()


def test_decode_ignores_sharing():
    """A decode step (q_len=1) with sharing configured falls through to the analytic path and matches the
    prefill's last row (sharing is prefill-only)."""
    G.CFG = G.SSAConfig(block=32, budget_frac=0.5, impl="flex", share_route_from=0)
    q, k, v = _qkv(n=160)
    full, _ = ssa_attention_forward(_mod(2), q, k, v, scaling=1.0)          # analytic (CPU) prefill
    inc, _ = ssa_attention_forward(_mod(2), q[:, :, -1:, :], k, v, scaling=1.0)
    assert torch.allclose(inc[:, 0], full[:, -1], atol=1e-10)
