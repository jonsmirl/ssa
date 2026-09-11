"""CPU checks for the long-context driver's loss accounting and input gates."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ssa.fullscale_tail_runner import score_window, validate_protocol


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 8)

    def forward(self, ids, use_cache=False):
        hidden = self.embedding(ids)
        for _ in range(2):
            query = hidden[:, None]
            hidden = hidden + F.scaled_dot_product_attention(query, query, query, is_causal=True)[:, 0]
        return SimpleNamespace(last_hidden_state=hidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = nn.Linear(8, 13)
        self.config = SimpleNamespace(num_hidden_layers=2)


@pytest.fixture
def cpu_model(monkeypatch):
    # The production runner is CUDA-only; mock timing, not the attention/math.
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    torch.manual_seed(10)
    return TinyModel().eval()


@pytest.mark.parametrize("length,chunk", [(17, 1), (17, 6), (5, 256)])
def test_chunked_projection_matches_full_shifted_loss(cpu_model, length, chunk):
    ids = torch.arange(2 * length).reshape(2, length) % 13
    row, last = score_window(cpu_model, ids, "dense", 0., {}, projection_chunk=chunk)
    logits = cpu_model.lm_head(cpu_model.model(ids).last_hidden_state)
    target = ids[:, 1:]
    expected = F.cross_entropy(logits[:, :-1].flatten(0, 1), target.flatten(), reduction="sum")
    assert row["tokens"] == ids.numel()
    assert row["targets"] == target.numel()
    assert row["nll_sum"] == pytest.approx(float(expected.detach()), rel=2e-7)
    assert row["top1_correct"] == int((logits[:, :-1].argmax(-1) == target).sum())
    torch.testing.assert_close(last, logits[:, -1])


def test_probe_projects_last_token_without_targets(cpu_model):
    ids = torch.tensor([[2, 4, 7]])
    row, last = score_window(cpu_model, ids, "dense", 0., {}, last_only=True)
    assert row["targets"] == 0 and row["ce_nats"] is None
    assert row["nll_sum"] == 0 and last.shape == (1, 13)


def test_layer_gate_restores_hook_on_failure(cpu_model):
    original = F.scaled_dot_product_attention
    cpu_model.config.num_hidden_layers = 3
    with pytest.raises(AssertionError, match="every transformer layer"):
        score_window(cpu_model, torch.tensor([[2, 4, 7]]), "dense", 0., {})
    assert F.scaled_dot_product_attention is original


@pytest.mark.parametrize("kwargs", [
    {"contexts": ""}, {"contexts": "512,"}, {"contexts": "0"},
    {"contexts": "512,512"}, {"probe_contexts": "-1"},
    {"probe_contexts": "8192,8192"}, {"window_limit": -1},
    {"max_tail_share": float("nan")}, {"max_tail_share": float("inf")},
    {"max_tail_share": -0.1}, {"max_tail_share": 1.1},
])
def test_protocol_rejects_invalid_runs(kwargs):
    options = dict(contexts="512,8192", probe_contexts="8192,131072", window_limit=0, max_tail_share=None)
    options.update(kwargs)
    with pytest.raises(ValueError):
        validate_protocol(**options)


def test_protocol_accepts_empty_probe_set_and_share_endpoints():
    for share in (None, 0., 0.25, 1.):
        validate_protocol("512,8192", "", 1, share)


def test_score_window_rejects_empty_input_or_projection_chunk(cpu_model):
    with pytest.raises(ValueError, match="projection_chunk"):
        score_window(cpu_model, torch.tensor([[1, 2]]), "dense", 0., {}, projection_chunk=0)
    with pytest.raises(ValueError, match="nonempty"):
        score_window(cpu_model, torch.empty(1, 0, dtype=torch.long), "dense", 0., {})
