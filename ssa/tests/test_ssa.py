"""Fast CPU assertions for the SSA reproduction layer (no training)."""
import torch
import torch.nn.functional as F
from ssa.ssa_demo import ssa_attention, batched_kmeans


def _qkv(n=48, d=16, h=2, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, h, n, d, generator=g) for _ in range(3))


def test_full_budget_recovers_dense():
    """With every cluster selected (top_c = n_clusters) and a full local window, SSA = dense causal."""
    q, k, v = _qkv()
    n = q.size(2)
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ssa = ssa_attention(q, k, v, n_clusters=4, top_c=4, local_w=n)   # selects all keys
    assert torch.allclose(ssa, dense, atol=1e-5)


def test_attended_fraction_below_one():
    """A tight budget attends a strict fraction of the causal region."""
    q, k, v = _qkv()
    _, frac = ssa_attention(q, k, v, n_clusters=8, top_c=2, local_w=4, return_frac=True)
    assert 0.0 < frac < 1.0


def test_fraction_falls_with_context():
    """Fixed top_c, clusters ∝ √n ⟹ attended fraction shrinks as n grows (the O(n·k) signature)."""
    import math
    fracs = []
    for n in (128, 512, 2048):
        g = torch.Generator().manual_seed(1)
        q = torch.randn(1, 2, n, 16, generator=g); k = torch.randn(1, 2, n, 16, generator=g)
        C = max(4, int(round(1.5 * math.sqrt(n))))
        _, fr = ssa_attention(q, k, k, n_clusters=C, top_c=2, local_w=8, return_frac=True)
        fracs.append(fr)
    assert fracs[0] > fracs[1] > fracs[2]


def test_kmeans_assignments_valid():
    _, k, _ = _qkv()
    a = batched_kmeans(k, C=5)
    assert a.shape == (2, 2, 48) and int(a.min()) >= 0 and int(a.max()) < 5
