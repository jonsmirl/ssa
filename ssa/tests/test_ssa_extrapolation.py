"""Fast CPU assertions for the RoPE extrapolation model (no training)."""
import torch
from ssa.ssa_extrapolation import build_rope, apply_rope, RoPEModel


def test_rope_preserves_norm():
    """RoPE is a rotation per coordinate-pair, so it preserves each token's norm."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 64)
    cos, sin = build_rope(16, 64, "cpu")
    y = apply_rope(x, cos[None, None], sin[None, None])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_rope_model_forward_shape():
    m = RoPEModel(50, d=64, n_layer=2, n_head=4)
    out = m(torch.randint(0, 50, (2, 48)))
    assert out.shape == (2, 48, 50)


def test_rope_runs_at_arbitrary_length():
    """No learned positional cap — the model accepts a length it was never sized for (extrapolation)."""
    m = RoPEModel(50, d=64, n_layer=2, n_head=4)
    out = m(torch.randint(0, 50, (1, 500)))
    assert out.shape == (1, 500, 50)
