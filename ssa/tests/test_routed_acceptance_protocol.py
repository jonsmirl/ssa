"""CPU protocol tests; mocked hidden states do not validate Qwen quality."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F

from ssa import routed_acceptance_demo as demo
from ssa.endpoint_acceptance import accept_logits


@pytest.fixture
def protocol(monkeypatch):
    # None of these tests initialize CUDA or load a model/checkpoint.
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1234)
    reference = torch.tensor([[[3., 1., 0.], [3., 1., 0.], [3., 3., 0.],
                               [3., 1., 0.], [0., 1., 3.], [9., 0., 0.]]])
    candidate = torch.tensor([[[4., 1., 0.], [1., 4., 0.], [4., 1., 0.],
                               [4., 4., 0.], [0., 2., 4.], [0., 9., 0.]]])
    hidden = {"sparse": reference, "tail": candidate}
    paired = Mock(return_value=(hidden, {"sparse": .2, "tail": .3}))
    monkeypatch.setattr(demo, "paired_hidden", paired)
    model = SimpleNamespace(lm_head=Mock(side_effect=lambda x: x))
    return model, paired, reference, candidate


@pytest.mark.parametrize("chunk", [1, 2, 9])
def test_paired_window_charges_two_projections_per_chunk_and_scores_shifted_targets(protocol, chunk):
    model, paired, reference, candidate = protocol
    ids = torch.tensor([[2, 0, 1, 2, 0, 2]])
    row = demo.paired_window(model, ids, None, {}, projection_chunk=chunk)
    paired.assert_called_once_with(model, ids, None, {})
    expected_calls = 2 * ((ids.shape[1] - 2) // chunk + 1)
    assert row["projection_calls"] == model.lm_head.call_count == expected_calls
    assert row["full_model_forwards"] == 2
    assert row["targets"] == 5 and row["tokens"] == 6
    assert row["accepted_rows"] == 2
    assert row["reference_ties"] == row["candidate_ties"] == 1
    assert row["changed_top1_proposals"] == 1
    assert row["preservation_violations"] == 0
    assert row["forward_seconds"] == {"sparse": .2, "tail": .3}
    assert row["peak_allocated_gb"] == 1234 / 1e9
    reference, candidate = reference[:, :-1], candidate[:, :-1]
    accepted, _ = accept_logits(reference, candidate)
    for name, logits in (("reference", reference), ("candidate", candidate), ("accepted", accepted)):
        expected = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="sum")
        assert row["modes"][name]["nll_sum"] == pytest.approx(float(expected))
        assert row["modes"][name]["correct"] == int((logits.argmax(-1) == ids[:, 1:]).sum())


def test_acceptance_is_label_independent_at_fixed_computed_endpoints(protocol, monkeypatch):
    model, _, _, _ = protocol
    decisions = []

    def observe(reference, candidate):
        output, info = accept_logits(reference, candidate)
        decisions.append((output.clone(), info["accepted"].clone()))
        return output, info

    monkeypatch.setattr(demo, "accept_logits", observe)
    # Changing ids also changes real model inputs. Here hidden states are held
    # fixed deliberately to isolate label use in the endpoint decision itself.
    first = demo.paired_window(model, torch.tensor([[2, 0, 0, 0, 0, 0]]), None, {}, 9)
    second = demo.paired_window(model, torch.tensor([[2, 1, 1, 1, 1, 1]]), None, {}, 9)
    assert len(decisions) == 2
    assert all(torch.equal(a, b) for a, b in zip(decisions[0], decisions[1]))
    assert first["accepted_rows"] == second["accepted_rows"]
    assert first["modes"]["accepted"]["nll_sum"] != second["modes"]["accepted"]["nll_sum"]
