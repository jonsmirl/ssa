"""
Unit tests for the Gemma-4 SSA attention swap (ssa/gemma_ssa.py), on synthetic tensors with
Gemma-4's exact full-layer shapes (hq=16, hkv=8, head_dim=512, GQA group 2). No model load —
these validate the routing/selection math so the GPU window is spent on real forwards, not debugging.

The load-bearing one is `test_full_budget_equals_dense` (the kappa=100% sanity gate): with the budget
covering every visible block, SSA must reproduce dense attention bit-for-bit.
"""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
import pytest

from ssa import gemma_ssa as G
from ssa.gemma_ssa import ssa_attention_forward, repeat_kv

torch.manual_seed(0)
DT = torch.float64  # high precision so the dense-equality gate is exact


def _module(groups=2, is_sliding=False, scaling=1.0):
    return SimpleNamespace(num_key_value_groups=groups, is_sliding=is_sliding, scaling=scaling)


def _qkv(b=1, hq=16, hkv=8, n=160, d=512):
    q = torch.randn(b, hq, n, d, dtype=DT)
    k = torch.randn(b, hkv, n, d, dtype=DT)
    v = torch.randn(b, hkv, n, d, dtype=DT)
    return q, k, v


def dense_ref(q, k, v, scaling, groups):
    """Plain causal softmax attention with GQA repeat. Returns (b,n,hq,d)."""
    k = repeat_kv(k, groups); v = repeat_kv(v, groups)
    n = q.shape[2]
    s = torch.matmul(q, k.transpose(-1, -2)) * scaling
    causal = torch.arange(n)[None, :] <= torch.arange(n)[:, None]
    s = s.masked_fill(~causal[None, None], float("-inf"))
    w = torch.softmax(s, dim=-1, dtype=torch.float32).to(v.dtype)  # match production attention
    return torch.matmul(w, v).transpose(1, 2).contiguous()


def test_full_budget_equals_dense():
    """kappa=100%: budget covers every visible block -> SSA == dense (the sanity gate)."""
    G.CFG = G.SSAConfig(block=32, budget_frac=1.0)
    q, k, v = _qkv()
    out, _ = ssa_attention_forward(_module(), q, k, v, scaling=1.0)
    ref = dense_ref(q, k, v, 1.0, groups=2)
    assert out.shape == ref.shape == (1, 160, 16, 512)
    assert torch.allclose(out, ref, atol=1e-10), (out - ref).abs().max().item()


def test_top_c_covering_all_blocks_equals_dense():
    """top_c >= #blocks is also the dense path."""
    G.CFG = G.SSAConfig(block=32, top_c=999)
    q, k, v = _qkv(n=128)
    out, _ = ssa_attention_forward(_module(), q, k, v, scaling=1.0)
    ref = dense_ref(q, k, v, 1.0, groups=2)
    assert torch.allclose(out, ref, atol=1e-10)


def test_causal_no_future_leak():
    """Output at position i must not change when future keys/values are perturbed."""
    G.CFG = G.SSAConfig(block=32, budget_frac=0.5)
    q, k, v = _qkv(n=128)
    out1, _ = ssa_attention_forward(_module(), q, k.clone(), v.clone(), scaling=1.0)
    cut = 64
    k2, v2 = k.clone(), v.clone()
    k2[:, :, cut + 1:] += 5.0 * torch.randn_like(k2[:, :, cut + 1:])
    v2[:, :, cut + 1:] += 5.0 * torch.randn_like(v2[:, :, cut + 1:])
    out2, _ = ssa_attention_forward(_module(), q, k2, v2, scaling=1.0)
    assert torch.allclose(out1[:, :cut + 1], out2[:, :cut + 1], atol=1e-10)


def test_gqa_shapes_and_grouping():
    """16 q-heads / 8 kv-heads, group 2: output is (b,n,16,512) and uses repeat_kv consistently."""
    G.CFG = G.SSAConfig(block=32, budget_frac=1.0)
    q, k, v = _qkv(b=2, hq=16, hkv=8, n=96, d=512)
    out, _ = ssa_attention_forward(_module(groups=2), q, k, v, scaling=1.0)
    assert out.shape == (2, 96, 16, 512)
    assert torch.allclose(out, dense_ref(q, k, v, 1.0, 2), atol=1e-10)


def test_incremental_decode_matches_prefill():
    """KV-cache incremental decoding (q_len=1, kv_len=N) must reproduce the prefill's last-position
    output. This is the bug that silently corrupted the sparse-budget sweep numbers: the mask was
    indexed by query length instead of key length, so generation degraded under sparsity."""
    for frac in (0.5, 0.25):
        G.CFG = G.SSAConfig(block=32, budget_frac=frac)
        q, k, v = _qkv(n=160)
        full, _ = ssa_attention_forward(_module(), q, k, v, scaling=1.0)
        inc, _ = ssa_attention_forward(_module(), q[:, :, -1:, :], k, v, scaling=1.0)
        assert inc.shape == (1, 1, 16, 512)
        assert torch.allclose(inc[:, 0], full[:, -1], atol=1e-10), \
            (frac, (inc[:, 0] - full[:, -1]).abs().max().item())


def test_routing_finds_the_needle():
    """At a tight budget (top_c=1, no local pull to the needle's block), cumulant routing must
    select the block holding a key strongly aligned with the probe query, so the SSA output at the
    probe matches dense (both dominated by that key)."""
    G.CFG = G.SSAConfig(block=32, top_c=1, local_w=0, beta=2.0)
    b, hq, d = 1, 1, 512
    n = 128                                   # 4 blocks of 32; probe in block 3, needle in block 1
    q, k, v = _qkv(b=b, hq=hq, hkv=hq, n=n, d=d)
    u = F.normalize(torch.randn(d, dtype=DT), dim=0)
    probe = n - 1
    q[0, 0, probe] = u                        # probe query points along u
    needle = 40                               # inside block 1
    k[0, 0, needle] = 12.0 * u                # huge-logit key aligned with the probe
    v[0, 0, needle] = torch.arange(d, dtype=DT) / d  # a distinctive value to detect
    out, _ = ssa_attention_forward(_module(groups=1), q, k, v, scaling=1.0)
    ref = dense_ref(q, k, v, 1.0, 1)
    # probe output tracks the dense probe output — routing found block 1 (small residual differs
    # because SSA and dense attend different distractor sets; the needle dominates both).
    assert torch.allclose(out[0, probe], ref[0, probe], atol=3e-3), \
        (out[0, probe] - ref[0, probe]).abs().max().item()
    # and it is essentially the needle's value (the needle was selected and dominates the softmax)
    assert torch.allclose(out[0, probe], v[0, 0, needle], atol=3e-3)


def test_sliding_layer_falls_back():
    """A sliding layer (is_sliding=True) must defer to the installed fallback, untouched by SSA."""
    G.CFG = G.SSAConfig(block=32, budget_frac=0.1, route_full_only=True)
    sentinel = object()
    G._FALLBACK = lambda module, q, k, v, mask=None, scaling=None, dropout=0.0, **kw: (sentinel, None)
    q, k, v = _qkv(n=64)
    out, _ = ssa_attention_forward(_module(is_sliding=True), q, k, v, scaling=1.0)
    assert out is sentinel
    G._FALLBACK = None


def test_lower_budget_attends_fewer_keys():
    """Sanity: shrinking the budget strictly reduces the number of attended key positions."""
    q, k, v = _qkv(b=1, hq=1, hkv=1, n=256, d=512)
    counts = []
    for frac in (1.0, 0.5, 0.25, 0.1):
        G.CFG = G.SSAConfig(block=32, budget_frac=frac, local_w=0)
        m = G._selection_mask(q, repeat_kv(k, 1), G.CFG) if frac < 1.0 else None
        if m is None:
            n = q.shape[2]
            counts.append(n * (n + 1) // 2)   # dense causal count
        else:
            counts.append(int((m[0, 0] == 0.0).sum().item()))
    assert counts[0] > counts[1] > counts[2] > counts[3], counts
