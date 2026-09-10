"""Algebra checks for the certificate-margin training objective."""
import math

import torch

from ssa.score_tail_training import hard_profile_margin_torch


def test_torch_training_margin_matches_hard_share_and_softplus_boundary():
    blocks = torch.tensor([
        [[2.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[-1.0, 0.0], [-1.0, 0.0]],
    ])
    q = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([0])
    margin, log_a, log_z = hard_profile_margin_torch(
        blocks, q, target, temperature=1.0, eta=0.1)
    share = torch.exp(log_a) / (torch.exp(log_z) + torch.exp(log_a))
    assert bool(share <= 0.1) == bool(margin <= 0)
    loss = torch.nn.functional.softplus(margin)
    assert bool(loss <= math.log(2)) == bool(margin <= 0)


def test_training_margin_backpropagates_through_supplied_hard_summaries():
    blocks = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(2), requires_grad=True)
    q = torch.randn(2, 5, generator=torch.Generator().manual_seed(3))
    target = torch.tensor([0, 2])
    margin, _, _ = hard_profile_margin_torch(blocks, q, target)
    torch.nn.functional.softplus(margin).mean().backward()
    assert blocks.grad is not None and torch.isfinite(blocks.grad).all()
    assert float(blocks.grad.abs().sum()) > 0
