"""Test whether fixed-state retries can repair a fixed-budget sparse read.

This experiment separates three questions:

1. Does merely repeating/widening a read add anything beyond the same total
   one-shot budget?  (No: exact streaming accumulation makes them identical.)
2. Can a bounded state help when an opened value contains information about
   where to read next?  (Yes, on a controlled pointer/clue task.)
3. Can raw downstream CE train a controller through exact hard top-k misses?
   (Not through the selection index; a routing surrogate supplies that path.)

Run::

    python -m ssa.recurrent_repair_experiment \
      --cache /tmp/ssa_bennett_qwen_8192.npz \
      --out runs/recurrent_repair.json

The Qwen portion replays measured Q/K geometry but does not train a model.  It
uses block-mean routing as a deliberately simple reference, not the user's
router checkpoint.  All attention diagnostics are dense-oracle measurements,
not deterministic score-tail certificates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .recurrent_repair import fixed_query_repair, route_repair_rounds


def _hard_recall(model, candidates, device):
    x = torch.eye(candidates, device=device)
    return float((model(x).argmax(-1) == torch.arange(candidates, device=device)).float().mean())


def _train_controller(mode, *, candidates=32, steps=300, batch=256, seed=0, device="cpu"):
    """Train a clue->route-query map and evaluate it with exact hard argmax.

    The routed-only downstream prediction is the fixed value at the selected
    address.  Exact argmax severs the derivative to the query.  ``route_ce`` is
    explicit target-address supervision; ``straight_through`` has hard forward
    choices but uses soft routing derivatives and is therefore also a surrogate.
    """
    torch.manual_seed(seed)
    model = torch.nn.Linear(candidates, candidates, bias=False, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)
    eye = torch.eye(candidates, device=device)
    initial = _hard_recall(model, candidates, device)
    zero_grad_steps = 0
    final_loss = math.nan
    for _ in range(steps):
        target = torch.randint(candidates, (batch,), device=device)
        clue = eye[target]
        scores = model(clue)
        hard_id = scores.argmax(-1)
        if mode == "raw_ce":
            # Only the selected archive value reaches the downstream head.
            prediction_logits = 8.0 * eye[hard_id]
            loss = F.cross_entropy(prediction_logits, target) + 0.0 * scores.sum()
        elif mode == "route_ce":
            loss = F.cross_entropy(scores, target)
        elif mode == "straight_through":
            soft = torch.softmax(scores, dim=-1)
            hard = F.one_hot(hard_id, candidates).to(soft.dtype)
            weights = hard + soft - soft.detach()
            loss = F.cross_entropy(8.0 * weights, target)
        else:
            raise ValueError(mode)
        opt.zero_grad(); loss.backward()
        norm = math.sqrt(sum(float((p.grad * p.grad).sum()) for p in model.parameters()))
        zero_grad_steps += norm == 0.0
        opt.step()
        final_loss = float(loss.detach())
    return {
        "initial_hard_top1_recall": initial,
        "final_hard_top1_recall": _hard_recall(model, candidates, device),
        "zero_controller_gradient_steps": zero_grad_steps,
        "steps": steps,
        "final_training_loss": final_loss,
        "hard_evaluation": True,
    }


def _pointer_demo():
    n, d = 64, 8
    routing = np.zeros((n, d)); routing[0, 0] = 4; routing[47, 5] = 4
    K = np.zeros((n, d)); K[47, 7] = 14
    V = np.zeros((n, d)); V[0, 5] = 1; V[47, 2] = 1
    q_attention = np.eye(d)[7]

    def use_opened_clue(state, sparse_output, _round):
        state = sparse_output
        return state, state

    read = route_repair_rounds(
        q_attention, K, V, routing, np.eye(d)[0], use_opened_clue,
        rounds=2, budget=1)
    return {
        "context_keys": n,
        "controller_state_scalars": d,
        "per_round_budget": 1,
        "selected_ids": [r.indices.tolist() for r in read.rounds],
        "retained_mass_by_round": [r.actual_retained_mass for r in read.rounds],
        "output_error_by_round": [r.actual_output_error for r in read.rounds],
        "interpretation": "the first opened value is an explicit address clue for the second read",
    }


def _block_order(q, K, prefix, block):
    full, partial = divmod(prefix, block)
    means = []
    for b in range(full):
        means.append(K[b * block:(b + 1) * block].mean(axis=0))
    if partial:
        means.append(K[full * block:prefix].mean(axis=0))
    means = np.asarray(means)
    ids = np.arange(len(means))
    return np.lexsort((-ids, -(means @ q)))


def _ids_for_blocks(block_ids, prefix, block):
    return np.concatenate([
        np.arange(int(b) * block, min((int(b) + 1) * block, prefix)) for b in block_ids
    ])


def _qwen_replay(cache, *, queries=32, block=64, per_round_fraction=0.025,
                 rounds=4, seed=0):
    path = Path(cache)
    if not path.exists():
        return {"available": False, "path": str(path)}
    z = np.load(path)
    K = z["K"].astype(np.float64)
    Q = z["Q"].astype(np.float64)
    rng = np.random.default_rng(seed)
    eligible = np.arange(max(block, len(K) // 2), len(K))
    positions = np.sort(rng.choice(eligible, min(queries, len(eligible)), replace=False))
    beta = 1 / math.sqrt(K.shape[1])
    V = np.tanh(K[:, :min(8, K.shape[1])])  # deterministic proxy; mass uses only real Q/K
    mass_rows, error_rows, key_rows, repeat_rows, equality = [], [], [], [], []
    oracle_mass = []
    for pos in positions:
        prefix = int(pos) + 1
        q = Q[pos]
        order = _block_order(q, K, prefix, block)
        per_round = max(1, math.ceil(per_round_fraction * len(order)))
        batches = []
        for r in range(rounds):
            chunk = order[r * per_round:(r + 1) * per_round]
            if len(chunk):
                batches.append(_ids_for_blocks(chunk, prefix, block))
        read = fixed_query_repair(q, K, V, batches, beta=beta, prefix=prefix)
        one_ids = np.concatenate(batches)
        one = fixed_query_repair(q, K, V, [one_ids], beta=beta, prefix=prefix)
        repeated = fixed_query_repair(q, K, V, [batches[0]] * len(batches), beta=beta, prefix=prefix)
        mass_rows.append([r.actual_retained_mass for r in read.rounds])
        error_rows.append([r.actual_output_error for r in read.rounds])
        key_rows.append([r.cumulative_keys for r in read.rounds])
        repeat_rows.append(repeated.rounds[-1].actual_retained_mass)
        equality.append(float(np.max(np.abs(read.output - one.output))))

        logits = beta * (K[:prefix] @ q)
        weights = np.exp(logits - logits.max()); weights /= weights.sum()
        k = len(one_ids)
        oracle_mass.append(float(np.partition(weights, len(weights) - k)[-k:].sum()))
    mass_rows = np.asarray(mass_rows)
    error_rows = np.asarray(error_rows)
    key_rows = np.asarray(key_rows)
    return {
        "available": True,
        "path": str(path),
        "geometry": "Qwen2.5-0.5B layer 18 KV head 0 post-RoPE Q/K",
        "router": "contiguous block means; static query; disjoint exclusion",
        "queries": len(positions),
        "block": block,
        "rounds": rounds,
        "per_round_block_fraction": per_round_fraction,
        "mean_keys_scored_by_round": key_rows.mean(axis=0).tolist(),
        "mean_actual_retained_mass_by_round": mass_rows.mean(axis=0).tolist(),
        "median_actual_retained_mass_by_round": np.median(mass_rows, axis=0).tolist(),
        "mean_proxy_output_error_by_round": error_rows.mean(axis=0).tolist(),
        "mean_repeated_same_read_final_mass": float(np.mean(repeat_rows)),
        "mean_exact_top_key_oracle_mass_at_equal_key_budget": float(np.mean(oracle_mass)),
        "max_staged_vs_one_shot_output_difference": float(np.max(equality)),
        "certificate": None,
        "fence": "dense-oracle measurement; this replay does not contain the user's trained router",
    }


def run(out, cache, *, seed=0, steps=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    result = {
        "experiment": "bounded-state recurrent repair for fixed-budget sparse attention",
        "command": f"python -m ssa.recurrent_repair_experiment --cache {cache} --out {out}",
        "device": device,
        "seed": seed,
        "pointer_demo": _pointer_demo(),
        "hard_router_training": {
            mode: _train_controller(mode, seed=seed, steps=steps, device=device)
            for mode in ("raw_ce", "route_ce", "straight_through")
        },
        "qwen_static_replay": _qwen_replay(cache, seed=seed),
        "claims": {
            "proved_by_construction": [
                "streaming state equals softmax attention on the selected union",
                "fixed-query staged disjoint reads equal a one-shot read of the same union",
            ],
            "measured_not_proved": [
                "opened clues can improve a later route on the controlled distribution",
                "the training modes reach the reported hard-routing recall",
            ],
            "not_claimed": [
                "raw CE discovers useful retry queries in a pretrained transformer",
                "fixed state reconstructs arbitrary unseen KV content",
                "the user's router has the block-mean replay's behavior",
                "conditional subquadraticity without a fixed number of rounds and fixed per-round budget",
            ],
        },
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    path = Path(out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/ssa_bennett_qwen_8192.npz")
    p.add_argument("--out", default="runs/recurrent_repair.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()
    result = run(args.out, args.cache, seed=args.seed, steps=args.steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
