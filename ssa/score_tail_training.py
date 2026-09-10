"""Controlled certificate-margin training experiment.

The certificate objective uses the existing mean-plus-residual-radius block cap
at its finest profile (one cap per unopened block):

``M_eta = log(A_tail) - log(Z_S) - log(eta/(1-eta))`` and
``L_cert = softplus(M_eta)``.

Training is a smooth, almost-everywhere surrogate because block radii use a max
and the selected target block is fixed by the generated example. Evaluation
discards training summaries and rebuilds the hard 16-level profile through
:class:`ScoreTailCertifiedAttention`. Nothing here proves optimization or
held-out generalization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .prune_regularizer import DEV, non_target_variance_penalty
from .score_tail_certificate import ScoreTailCertifiedAttention


def hard_profile_margin_torch(blocks, q, target, *, temperature=0.05, eta=0.1):
    """Finest-profile hard margin from exact target mass and admissible block caps."""
    if not 0 < eta < 1:
        raise ValueError("eta must lie strictly between zero and one")
    batch, clusters, members = len(q), blocks.shape[0], blocks.shape[1]
    means = blocks.mean(1)
    radii = (blocks - means[:, None]).norm(dim=-1).amax(1)
    upper = (q @ means.T + q.norm(dim=-1, keepdim=True) * radii[None]) / temperature
    target_blocks = blocks[target]
    target_logits = torch.einsum("bd,bmd->bm", q, target_blocks) / temperature
    log_z = torch.logsumexp(target_logits, dim=1)
    mask = torch.arange(clusters, device=blocks.device)[None, :] != target[:, None]
    log_terms = math.log(members) + upper
    log_a = torch.logsumexp(log_terms.masked_fill(~mask, float("-inf")), dim=1)
    margin = log_a - log_z - math.log(eta / (1 - eta))
    return margin, log_a, log_z


def train(objective, *, n_clusters=24, cluster_size=16, d=28, steps=600,
          batch_size=128, noise=0.15, lr=3e-3, temperature=0.05,
          variance_weight=16.0, certificate_weight=0.05, eta=0.1, seed=0):
    if objective not in ("baseline", "variance", "certificate", "hybrid"):
        raise ValueError("unknown objective")
    torch.manual_seed(seed)
    n = n_clusters * cluster_size
    keys = torch.nn.Parameter(torch.randn(n, d, device=DEV))
    cluster_id = torch.arange(n, device=DEV) // cluster_size
    optimizer = torch.optim.AdamW([keys], lr=lr)
    generator = torch.Generator(device=DEV).manual_seed(seed)
    final = {}
    for _ in range(steps):
        normalized = F.normalize(keys, dim=-1)
        index = torch.randint(0, n, (batch_size,), generator=generator, device=DEV)
        q = F.normalize(
            normalized[index]
            + noise * torch.randn(batch_size, d, generator=generator, device=DEV), dim=-1)
        logits = q @ normalized.T / temperature
        retrieval = F.cross_entropy(logits, index)
        target = cluster_id[index]
        blocks = normalized.view(n_clusters, cluster_size, d)
        variance = non_target_variance_penalty(blocks, q, target)
        margin, _, _ = hard_profile_margin_torch(
            blocks, q, target, temperature=temperature, eta=eta)
        cert = F.softplus(margin).mean()
        loss = retrieval
        if objective in ("variance", "hybrid"):
            loss = loss + variance_weight * variance
        if objective in ("certificate", "hybrid"):
            loss = loss + certificate_weight * cert
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final = {"total": float(loss.detach()), "retrieval": float(retrieval.detach()),
                 "variance": float(variance.detach()),
                 "certificate_softplus": float(cert.detach()),
                 "mean_margin": float(margin.detach().mean()),
                 "hard_margin_nonpositive_fraction": float((margin <= 0).float().mean())}
    return F.normalize(keys, dim=-1).detach().cpu().numpy(), final


def evaluate(keys, *, cluster_size=16, noise=0.15, trials=256, eta=0.1,
             tail_levels=16, route_fraction=0.10, max_block_fraction=None, seed=1):
    rng = np.random.default_rng(seed)
    values = np.tanh(keys[:, :8])
    reader = ScoreTailCertifiedAttention(keys, values, cluster_size)
    rows = []
    for _ in range(trials):
        source = int(rng.integers(len(keys)))
        q = keys[source] + noise * rng.standard_normal(keys.shape[1])
        q /= np.linalg.norm(q) + 1e-12
        dense_top = int((keys @ q).argmax())
        visible_blocks = math.ceil(len(keys) / cluster_size)
        route = reader.route_by_block_mean(q, max(1, math.ceil(route_fraction * visible_blocks)))
        max_blocks = (None if max_block_fraction is None else
                      max(len(route.block_indices), math.ceil(max_block_fraction * visible_blocks)))
        result = reader.read(q, beta=1 / 0.05, routing=route,
                             tail_levels=tail_levels, mass_tol=eta, max_blocks=max_blocks)
        selected = set(result.indices.tolist())
        logits = (keys @ q) / 0.05
        top = logits.max()
        weights = np.exp(logits - top); weights /= weights.sum()
        missing = np.ones(len(keys), dtype=bool); missing[result.indices] = False
        actual = float(weights[missing].sum())
        rows.append((source in selected, dense_top in selected, result.certified,
                     result.keys_scored, result.blocks_opened, result.bounds_evaluated,
                     result.mass_upper, actual, result.certificate_margin))
    a = np.asarray(rows, dtype=np.float64)
    finite_margin = a[np.isfinite(a[:, 8]), 8]
    return {
        "source_retrieval_accuracy": float(a[:, 0].mean()),
        "dense_argmax_retrieval_accuracy": float(a[:, 1].mean()),
        "certified_fraction": float(a[:, 2].mean()),
        "mean_keys_scored": float(a[:, 3].mean()),
        "mean_blocks_opened": float(a[:, 4].mean()),
        "mean_bound_evaluations": float(a[:, 5].mean()),
        "mean_certified_omitted_mass": float(a[:, 6].mean()),
        "mean_actual_omitted_mass": float(a[:, 7].mean()),
        "maximum_soundness_deficit": float(np.maximum(a[:, 7] - a[:, 6], 0).max()),
        "full_read_fraction": float((a[:, 3] == len(keys)).mean()),
        "mean_finite_final_margin": (float(finite_margin.mean()) if len(finite_margin) else None),
        "max_block_fraction": max_block_fraction,
        "tail_levels": tail_levels,
    }


def run(out, *, steps=600, seed=0):
    result = {
        "method": "certificate-margin training comparison",
        "command": f"python -m ssa.score_tail_training --steps {steps} --out {out}",
        "device": DEV,
        "hardware": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "hardware": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "parameters": {"steps": steps, "seed": seed, "variance_weight": 16.0,
                       "certificate_weight": 0.05, "eta": 0.1, "tail_levels": 16},
        "scope": "controlled learned-key task; hard held-out summaries rebuilt after training",
        "objectives": {},
        "fences": [
            "softplus training loss is not a certificate",
            "hard evaluation recomputes partitions, radii, counts, and margins",
            "optimization and held-out generalization are not proved",
        ],
    }
    for objective in ("baseline", "variance", "certificate", "hybrid"):
        keys, training = train(objective, steps=steps, seed=seed)
        result["objectives"][objective] = {
            "final_training": training,
            "hard_evaluation_to_certificate": evaluate(keys, seed=seed + 1),
            "hard_evaluation_at_25pct_block_cap": evaluate(
                keys, max_block_fraction=0.25, seed=seed + 1),
        }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/score_tail_training.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args.out, steps=args.steps, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
