"""Trainable sparse attention repair with separate routing and attention queries.

The archive is immutable for one call. TokenBallTree implements a geometric,
fixed-beam reference search over individual tokens. Its center/radius priority
uses the same containment inequality as cascade_router.CausalTree; beam pruning
is approximate and is never an attention certificate. Build a separate snapshot
for each causal prefix: future vectors cannot affect its partition or summaries.

RecurrentRepairAttention consumes only gathered keys/values at inference. A GRU
steers subsequent queries using the accumulated attention output. Optional dense
teacher routing logits are emitted only during training. The fixed attention query
defines all exact logits in the selected union, independently of the router.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TokenBallTree:
    """Immutable binary median-split tree, one original token per leaf.

    Construction costs O(n d log n), storage O(n d). A beam B tests at most
    O(B log n) nodes per query; it need not find the exact nearest token. Exclusion
    applies at leaves. Padded/empty leaf slots have id -1 and never contribute.
    This reference uses float arithmetic, without an interval certificate.
    """

    def __init__(self, keys, *, prefix=None):
        if keys.ndim != 2 or not torch.isfinite(keys).all():
            raise ValueError("routing keys must be a finite matrix")
        end = len(keys) if prefix is None else prefix
        if not isinstance(end, int) or not 1 <= end <= len(keys):
            raise ValueError("prefix must select a nonempty snapshot")
        self.keys = keys[:end].detach().clone()
        x = self.keys.double().cpu().numpy()
        self.n, self.d = x.shape
        size = 1 << (self.n - 1).bit_length()
        centers = np.zeros((2 * size, self.d))
        radii = np.zeros(2 * size)
        leaves = np.full(2 * size, -1, dtype=np.int64)
        valid = np.zeros(2 * size, dtype=bool)

        def build(node, ids):
            if not len(ids):
                return
            valid[node] = True
            centers[node] = x[ids].mean(0)
            radii[node] = np.linalg.norm(x[ids] - centers[node], axis=1).max()
            if node >= size:
                leaves[node] = ids[0]
                return
            axis = np.argmax(x[ids].var(0))
            order = np.lexsort((ids, x[ids, axis]))
            ordered = ids[order]
            cut = (len(ids) + 1) // 2
            build(2 * node, ordered[:cut])
            build(2 * node + 1, ordered[cut:])

        build(1, np.arange(self.n))
        self.size = size
        self.centers = torch.as_tensor(centers, device=keys.device, dtype=keys.dtype)
        self.radii = torch.as_tensor(radii, device=keys.device, dtype=keys.dtype)
        self.leaves = torch.as_tensor(leaves, device=keys.device)
        self.valid = torch.as_tensor(valid, device=keys.device)

    @torch.no_grad()
    def route(self, q, budget, *, beam=32, excluded=None):
        if q.ndim != 2 or q.shape[1] != self.d or not torch.isfinite(q).all():
            raise ValueError("routing query must be a finite batch of matching vectors")
        if budget < 1 or beam < budget:
            raise ValueError("positive budget must not exceed beam")
        b = len(q)
        nodes = torch.ones(b, 1, dtype=torch.long, device=q.device)
        evaluations = 0
        while int(nodes[0, 0]) < self.size:
            nodes = torch.stack((2 * nodes, 2 * nodes + 1), dim=-1).flatten(1)
            score = (self.centers[nodes] * q[:, None]).sum(-1)
            score += self.radii[nodes] * q.norm(dim=-1, keepdim=True)
            score.masked_fill_(~self.valid[nodes], -torch.inf)
            if int(nodes[0, 0]) >= self.size and excluded is not None:
                ids = self.leaves[nodes]
                repeated = (ids[..., None] == excluded[:, None, :]).any(-1)
                score.masked_fill_(repeated, -torch.inf)
            evaluations += nodes.shape[1]
            # Stable node order makes ties deterministic across replay.
            rank = torch.argsort(score, dim=1, descending=True, stable=True)[:, :beam]
            nodes = nodes.gather(1, rank)
        ids = self.leaves[nodes]
        score = (self.keys[ids.clamp_min(0)] * q[:, None]).sum(-1)
        score.masked_fill_(ids < 0, -torch.inf)
        if excluded is not None:
            score.masked_fill_((ids[..., None] == excluded[:, None, :]).any(-1), -torch.inf)
        rank = torch.argsort(score, dim=1, descending=True, stable=True)[:, :budget]
        result = ids.gather(1, rank)
        result = result.masked_fill(~torch.isfinite(score.gather(1, rank)), -1)
        return result, evaluations


@dataclass
class RepairOutput:
    output: torch.Tensor
    indices: torch.Tensor
    outputs_by_round: list[torch.Tensor]
    routing_queries: list[torch.Tensor]
    route_logits: list[torch.Tensor]
    state: torch.Tensor
    node_evaluations: int
    log_partition: torch.Tensor


class RecurrentRepairAttention(nn.Module):
    """Differentiable selected attention + bounded GRU routing state.

    Accepts q[B,d], K[B,n,d], V[B,n,dv] and one shared tree snapshot. `state`
    can be carried into the next token/query; the attention accumulator and
    selected ids always reset when q changes. `mode=static` bypasses the GRU
    for a matched disjoint retry control. No oracle is read by the controller.
    """

    def __init__(self, routing_dim, value_dim, state_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.cell = nn.GRUCell(value_dim + routing_dim, state_dim)
        self.query_head = nn.Linear(state_dim, routing_dim)

    def forward(self, q, K, V, tree, routing_query, *, rounds=2, budget=4,
                beam=32, beta=1.0, state=None, mode="learned", teacher_logits=False):
        if mode not in ("learned", "static") or rounds < 1:
            raise ValueError("invalid mode or round count")
        if K.shape[:2] != V.shape[:2] or K.shape[1] != tree.n or K.shape[0] != len(q):
            raise ValueError("archive shapes must agree with the prefix snapshot")
        b = len(q)
        h = q.new_zeros(b, self.state_dim) if state is None else state
        rq = routing_query
        log_z = q.new_full((b,), -torch.inf)
        out = V.new_zeros(b, V.shape[-1])
        selected, outputs, queries, route_logits = [], [], [], []
        evaluations = 0
        for r in range(rounds):
            queries.append(rq)
            excluded = torch.cat(selected, dim=1) if selected else None
            if teacher_logits and r:
                logits = rq @ tree.keys.T
                if excluded is not None:
                    # Scatter only valid ids. Masking all candidates is legal
                    # for inference, but callers must omit exhausted teacher rows.
                    mask = torch.zeros_like(logits, dtype=torch.long)
                    mask.scatter_add_(1, excluded.clamp_min(0), (excluded >= 0).long())
                    logits = logits.masked_fill(mask > 0, -torch.inf)
                route_logits.append(logits)
            ids, cost = tree.route(rq, budget, beam=beam, excluded=excluded)
            evaluations += cost
            valid = ids >= 0
            gathered_k = K.gather(1, ids.clamp_min(0)[..., None].expand(-1, -1, K.shape[-1]))
            gathered_v = V.gather(1, ids.clamp_min(0)[..., None].expand(-1, -1, V.shape[-1]))
            scores = beta * (gathered_k * q[:, None]).sum(-1)
            scores = scores.masked_fill(~valid, -torch.inf)
            any_valid = valid.any(-1)
            # Avoid undefined gradients from logsumexp(-inf, ..., -inf).
            finite_batch_z = torch.logsumexp(torch.where(any_valid[:, None], scores,
                                                         torch.zeros_like(scores)), dim=-1)
            batch_z = finite_batch_z.masked_fill(~any_valid, -torch.inf)
            safe_z = torch.where(any_valid, batch_z, torch.zeros_like(batch_z))
            weights = torch.exp(scores - safe_z[:, None])
            batch_out = (weights[..., None] * gathered_v).sum(1)
            joined = torch.logaddexp(log_z, batch_z)
            safe_joined = torch.where(torch.isfinite(joined), joined, torch.zeros_like(joined))
            out = torch.exp(log_z - safe_joined)[:, None] * out + torch.exp(batch_z - safe_joined)[:, None] * batch_out
            log_z = joined
            selected.append(ids)
            outputs.append(out)
            if mode == "learned":
                h = self.cell(torch.cat((out, routing_query), dim=-1), h)
                rq = F.normalize(self.query_head(h), dim=-1)
        return RepairOutput(out, torch.cat(selected, 1), outputs, queries,
                            route_logits, h, evaluations, log_z)


class PositiveTailState:
    """Fixed-size positive-feature summary for the omitted attention tail.

    Given nonnegative feature maps, keep sum(phi(k)) and sum(phi(k) v).
    Subtract the selected approximate contributions and replace them by exact
    exponentials. This costs O(m(dv+1)) state and O(k m) per read. It estimates
    the tail; neither positivity nor this decomposition certifies kernel error.
    A single feature (m=1) is a constant-logit tail baseline.
    """

    def __init__(self, features, values):
        if features.ndim != 2 or values.ndim != 2 or len(features) != len(values):
            raise ValueError("features and values must be aligned matrices")
        if not torch.isfinite(features).all() or (features < 0).any() or not torch.isfinite(values).all():
            raise ValueError("features must be finite nonnegative; values must be finite")
        self.features = features
        self.values = values
        self.z = features.sum(0)
        self.numerator = features.T @ values

    def read(self, q_features, selected, selected_logits, *, log_scale=0.0):
        """Exact selected logits + approximate residual at the same scale.

        `log_scale` supplies a common shift for exact and approximate kernel
        weights. Candidate features may encode exp(s-log_scale); it is the
        caller's obligation to use the same scale for the exact logits.
        """
        if len(torch.unique(selected)) != len(selected):
            raise ValueError("selected ids must be distinct")
        remainder_z = self.z - self.features[selected].sum(0)
        remainder_n = self.numerator - self.features[selected].T @ self.values[selected]
        z_tail = (q_features @ remainder_z).clamp_min(0)
        n_tail = q_features @ remainder_n
        weights = torch.exp(selected_logits - log_scale)
        z_exact = weights.sum()
        n_exact = weights @ self.values[selected]
        return (n_exact + n_tail) / (z_exact + z_tail), z_tail / (z_exact + z_tail)


class ClusterTailState:
    """Incremental fixed-size cell counts/value sums, with no archive references.

    Centers are fixed before appending. Each append assigns keys to their nearest
    center. A query replaces the approximation for its selected keys by their
    exact weights. Only the caller's already-selected K/V are read at query time.
    State storage is cells*(dv+1) scalars plus fixed centers. Reset at sequence
    boundaries; append only the visible prefix. No future-dependent fitting.
    """

    def __init__(self, centers, value_dim):
        if centers.ndim != 2 or not torch.isfinite(centers).all() or value_dim < 1:
            raise ValueError("centers must be a finite matrix; value dimension positive")
        self.centers = centers.detach().clone()
        self.counts = centers.new_zeros(len(centers))
        self.sums = centers.new_zeros(len(centers), value_dim)
        self.tokens = 0

    def append(self, keys, values):
        if (keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values)
                or keys.shape[1] != self.centers.shape[1] or values.shape[1] != self.sums.shape[1]
                or not torch.isfinite(keys).all() or not torch.isfinite(values).all()):
            raise ValueError("append needs finite, aligned key/value matrices")
        ids = torch.cdist(keys, self.centers).argmin(-1)
        self.counts = self.counts.scatter_add(0, ids, self.counts.new_ones(len(ids)))
        self.sums = self.sums.index_add(0, ids, values)
        self.tokens += len(keys)

    def read(self, q, selected_keys, selected_values, selected_logits, *, beta=1.0,
             log_gain=0.0, corrector=None):
        if not len(selected_keys) or len(selected_keys) != len(selected_values) or len(selected_keys) > self.tokens:
            raise ValueError("need a nonempty selected subset of the appended prefix")
        # Caller must provide each selected key once, all from this snapshot.
        ids = torch.cdist(selected_keys, self.centers).argmin(-1)
        selected_counts = torch.zeros_like(self.counts).scatter_add(0, ids, self.counts.new_ones(len(ids)))
        counts = self.counts - selected_counts
        if (counts < 0).any():
            raise ValueError("selected cell counts exceed the appended prefix")
        sums = self.sums - torch.zeros_like(self.sums).index_add(0, ids, selected_values)
        log_z = torch.logsumexp(selected_logits, 0)
        sparse = selected_logits.softmax(0) @ selected_values
        if corrector is not None:
            return corrector(q[None], sparse[None], log_z[None], counts[None], sums[None])[0][0]
        log_mass = beta * (self.centers @ q) + log_gain + counts.clamp_min(1).log()
        log_mass = log_mass.masked_fill(counts == 0, -torch.inf)
        weights = torch.cat((log_z[None], log_mass)).softmax(0)
        means = sums / counts.clamp_min(1)[:, None]
        return weights[0] * sparse + weights[1:] @ means
