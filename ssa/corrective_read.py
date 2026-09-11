"""Fixed-state corrective routing with exact selected reads and no predicted tail.

The caller routes the first query unchanged using ``TokenBallTree`` and then
may route ``CorrectiveRouter.propose`` against the same causal snapshot. The
proposal changes routing only: every selected attention logit uses the original
query. Tree beam search is approximate, not an attention certificate. Dense
teachers, training losses, and archive scans do not belong to this controller.

Only gathered K/V are inspected by the read helpers. Callers must use a prefix
snapshot (including when constructing the tree) to enforce causality and charge
all selected reads, routing work, and any external training teacher separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .trainable_repair import TokenBallTree


@dataclass
class SelectedSummary:
    """Exact statistics of one selected set; no claim about omitted mass."""

    key_mean: torch.Tensor
    value_mean: torch.Tensor
    entropy: torch.Tensor
    log_partition: torch.Tensor
    count: torch.Tensor


def summarize_selected(q, K, V, indices, beta=1.0):
    """Read a distinct selected set from a shared archive, allowing -1 padding.

    Shapes are q[B,dk], K[n,dk], V[n,dv], indices[B,k]. Repeated valid ids
    are rejected rather than double counted; repetitions of padding are legal.
    Empty rows yield zero means/entropy and log-partition -infinity. This is
    ordinary floating-point selected softmax, not an interval certificate.
    """
    if (q.ndim != 2 or K.ndim != 2 or V.ndim != 2 or indices.ndim != 2
            or K.shape[0] != V.shape[0] or q.shape[1] != K.shape[1]
            or len(q) != len(indices) or K.shape[1] < 1 or V.shape[1] < 1):
        raise ValueError("query, archive, and selected-index shapes must agree")
    if indices.dtype != torch.long:
        raise ValueError("selected indices must have dtype torch.long")
    if not q.is_floating_point() or K.dtype != q.dtype or V.dtype != q.dtype:
        raise ValueError("query, keys, and values must share a floating dtype")
    if any(t.device != q.device for t in (K, V, indices)):
        raise ValueError("query, archive, and indices must share a device")
    if not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not torch.isfinite(q).all():
        raise ValueError("attention queries must be finite")
    valid = indices >= 0
    if (indices < -1).any() or (indices >= len(K)).any():
        raise ValueError("selected ids must be -1 or inside the archive snapshot")
    ordered = indices.sort(dim=1).values
    if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any():
        raise ValueError("selected ids must be distinct within each row")

    # Do not read archive row zero for padding: even that row may be poisoned.
    gathered_k = K[indices[valid]]
    gathered_v = V[indices[valid]]
    if not torch.isfinite(gathered_k).all() or not torch.isfinite(gathered_v).all():
        raise ValueError("selected keys and values must be finite")
    keys = q.new_zeros((*indices.shape, K.shape[1]))
    values = q.new_zeros((*indices.shape, V.shape[1]))
    keys[valid] = gathered_k
    values[valid] = gathered_v
    count = valid.sum(-1)
    if indices.shape[1] == 0:
        return SelectedSummary(q.new_zeros(q.shape), q.new_zeros((len(q), V.shape[1])),
                               q.new_zeros(len(q)), q.new_full((len(q),), -torch.inf), count)
    scores = beta * (keys * q[:, None, :]).sum(-1)
    if not torch.isfinite(scores[valid]).all():
        raise ValueError("selected logits overflowed; use a safer floating dtype")
    scores = scores.masked_fill(~valid, -torch.inf)
    any_valid = valid.any(-1)
    safe_scores = torch.where(any_valid[:, None], scores, torch.zeros_like(scores))
    safe_z = torch.logsumexp(safe_scores, dim=-1)
    weights = torch.exp(safe_scores - safe_z[:, None]).masked_fill(~valid, 0)
    # Invalid scores are never multiplied by zero (0 * -inf would be NaN).
    entropy = -(weights * (safe_scores - safe_z[:, None]).masked_fill(~valid, 0)).sum(-1)
    return SelectedSummary((weights[..., None] * keys).sum(1),
                           (weights[..., None] * values).sum(1), entropy,
                           safe_z.masked_fill(~any_valid, -torch.inf), count)


def exact_union_read(q, K, V, indices, beta=1.0):
    """Exact selected-softmax output at the original query, with no tail term.

    The caller constructs a disjoint union (e.g. tree routing with exclusion).
    Duplicate valid ids are errors, not additional attention mass.
    """
    return summarize_selected(q, K, V, indices, beta).value_mean


class CorrectiveRouter(nn.Module):
    """Bounded GRU state proposes a residual routing query from observed reads.

    ``propose`` returns (routing_query, state). State is caller-owned, reset by
    default, and has exactly ``state_dim`` scalars per query. Carrying it across
    tokens requires explicit causal ordering. The routing movement has norm at
    most ``max_delta * ||q||``; this is not an attention/output error bound.

    ``feedback=False`` supplies zero summary channels as a query-only control.
    It must be trained separately for a matched control experiment. In either
    mode, selected values are never predicted or added to the final output.
    """

    def __init__(self, dk, dv, state_dim=64, max_delta=1.0):
        super().__init__()
        if any(not isinstance(d, int) or d < 1 for d in (dk, dv, state_dim)):
            raise ValueError("dimensions must be positive integers")
        if not math.isfinite(max_delta) or max_delta < 0:
            raise ValueError("max_delta must be finite and nonnegative")
        self.dk, self.dv, self.state_dim, self.max_delta = dk, dv, state_dim, max_delta
        self.cell = nn.GRUCell(2 * dk + dv + 3, state_dim)
        self.query_head = nn.Linear(state_dim, dk)

    def propose(self, q, summary, feedback=True, state=None):
        if q.ndim != 2 or q.shape[1] != self.dk or not torch.isfinite(q).all():
            raise ValueError("query must be a finite batch of matching vectors")
        if (summary.key_mean.shape != q.shape
                or summary.value_mean.shape != (len(q), self.dv)
                or any(x.shape != (len(q),) for x in
                       (summary.entropy, summary.log_partition, summary.count))):
            raise ValueError("selected-summary shapes must match the query")
        if state is None:
            state = q.new_zeros(len(q), self.state_dim)
        if (state.shape != (len(q), self.state_dim) or not torch.isfinite(state).all()
                or (state.abs() > 1).any()):
            raise ValueError("state must be finite, correctly shaped, and within [-1, 1]")
        if feedback:
            log_z = torch.where(summary.count > 0, summary.log_partition,
                                torch.zeros_like(summary.log_partition))
            features = torch.cat((F.normalize(summary.key_mean, dim=-1),
                                  F.normalize(summary.value_mean, dim=-1),
                                  torch.tanh(summary.entropy)[:, None],
                                  torch.tanh(log_z / 32)[:, None],
                                  torch.tanh(torch.log1p(summary.count.to(q.dtype)) / 8)[:, None]), dim=-1)
            if not torch.isfinite(features).all():
                raise ValueError("observed summary features must be finite")
        else:
            features = q.new_zeros((len(q), self.dk + self.dv + 3))
        h = self.cell(torch.cat((F.normalize(q, dim=-1), features), dim=-1), state)
        direction = torch.tanh(self.query_head(h))
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1)
        routed = q + self.max_delta * q.norm(dim=-1, keepdim=True) * direction
        return routed, h

    def forward(self, q, summary, feedback=True, state=None):
        return self.propose(q, summary, feedback=feedback, state=state)
