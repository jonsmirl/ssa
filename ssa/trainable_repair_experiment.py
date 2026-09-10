"""Train/evaluate a GRU controller on fresh continuous-address archive tasks.

No source/target address or answer label is fed to the controller. The first
value contains a rotated target address (an explicit task assumption). Archives
and label assignments are sampled afresh; evaluation uses independently generated
key banks, including longer contexts. All reported reads use hard tree routing.
Dense target-address supervision is optional and confined to training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from .trainable_repair import RecurrentRepairAttention, TokenBallTree


def bank(n, d, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g, device=device), dim=-1)


def batch(tree, rotation, batch_size, generator, classes=16, no_clue=False):
    keys = tree.keys
    n, d = keys.shape
    dev = keys.device
    source = torch.randint(n, (batch_size,), generator=generator, device=dev)
    offset = torch.randint(1, n, (batch_size,), generator=generator, device=dev)
    target = (source + offset) % n
    row = torch.arange(batch_size, device=dev)
    K = keys[None].expand(batch_size, -1, -1).clone()
    q = keys[source]
    # Original attention is sharply concentrated on the answer, although its
    # routing address is available only by reading the clue. No target id leaks
    # into the controller's inputs; K[target] is scored only when opened.
    K[row, target] = q * 1.6
    ptr = F.normalize(torch.randn(batch_size, n, d, generator=generator, device=dev), dim=-1)
    ptr[row, source] = keys[target] @ rotation
    if no_clue:
        ptr[row, source] = 0
    labels = torch.randint(classes, (batch_size, n), generator=generator, device=dev)
    answers = labels[row, target]
    payload = F.one_hot(labels, classes).to(keys.dtype)
    payload[row, source] = 0
    ptr[row, target] = 0
    V = torch.cat((ptr, payload), dim=-1)
    return q, K, V, target, answers


def train(objective, *, seed, device, steps, n=256, d=24, budget=4, beam=32):
    torch.manual_seed(seed)
    rotation = torch.linalg.qr(torch.randn(d, d, device=device)).Q
    tree = TokenBallTree(bank(n, d, seed + 100, device))
    model = RecurrentRepairAttention(d, d + 16, state_dim=64).to(device)
    # Same initialization/data for every objective. This head is the state-only
    # control, with no additional archive reads after round one.
    state_head = nn.Linear(64, 16).to(device)
    params = list(model.parameters()) + list(state_head.parameters())
    opt = torch.optim.AdamW(params, lr=0.002, weight_decay=0.001)
    g = torch.Generator(device=device).manual_seed(seed + 200)
    history = []
    for step in range(steps):
        q, K, V, target, answer = batch(tree, rotation, 64, g)
        read = model(q, K, V, tree, q, budget=budget, rounds=2, beam=beam,
                     beta=20, teacher_logits=objective != "raw_ce")
        ce = F.cross_entropy(12 * read.output[:, d:], answer)
        loss = ce + 0 * sum(p.sum() for p in model.parameters())
        teacher_on = objective == "route_ce" or (objective == "warmup_then_ce" and step < steps // 2)
        if teacher_on:
            # Targets already found on round one require no residual routing.
            missing = ~(read.indices[:, :budget] == target[:, None]).any(-1)
            if missing.any():
                route_ce = F.cross_entropy(12 * read.route_logits[0][missing], target[missing])
                loss = loss + route_ce
        # A first-round-only independent state control observes exactly the same
        # sparse output and query. It may change the answer without another read.
        h0 = q.new_zeros(len(q), 64)
        state = model.cell(torch.cat((read.outputs_by_round[0].detach(), q), -1), h0)
        state_ce = F.cross_entropy(state_head(state.detach()), answer)
        loss = loss + state_ce
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            hit = (read.indices == target[:, None]).any(-1).float().mean()
            row = {"step": step, "ce": float(ce.detach()), "target_recall": float(hit),
                   "teacher_on": teacher_on}
            history.append(row)
            print(objective, seed, row, flush=True)
    return model.eval(), state_head.eval(), rotation, history


@torch.no_grad()
def evaluate(model, state_head, rotation, *, n, seed, device, queries=512,
             budget=4, beam=32, no_clue=False):
    d = rotation.shape[0]
    tree = TokenBallTree(bank(n, d, seed + 10000, device))
    g = torch.Generator(device=device).manual_seed(seed + 20000)
    records = {m: [] for m in ("one_shot", "static_retry", "learned_retry", "state_only")}
    max_union_diff = 0.0
    for start in range(0, queries, 32):
        q, K, V, target, answer = batch(tree, rotation, min(32, queries - start), g, no_clue=no_clue)
        dense_logits = 20 * (q[:, None] * K).sum(-1)
        p = dense_logits.softmax(-1)
        dense_out = (p[..., None] * V).sum(1)
        for mode in records:
            first_only = mode == "state_only"
            learned = mode == "learned_retry"
            started = time.perf_counter()
            r = model(q, K, V, tree, q, rounds=2 if mode.endswith("retry") else 1,
                      budget=budget if mode != "one_shot" else 2 * budget,
                      beam=beam, beta=20, mode="learned" if learned else "static")
            logits = 12 * r.output[:, d:]
            if first_only:
                h = model.cell(torch.cat((r.output, q), -1), q.new_zeros(len(q), 64))
                logits = state_head(h)
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.perf_counter() - started
            selected = r.indices
            found = (selected == target[:, None]).any(-1)
            first_found = (selected[:, :budget] == target[:, None]).any(-1)
            mass = (p.gather(1, selected.clamp_min(0)) * (selected >= 0)).sum(-1)
            error = (r.output - dense_out).norm(dim=-1)
            ce = F.cross_entropy(logits, answer, reduction="none")
            hit = logits.argmax(-1) == answer
            for j in range(len(q)):
                records[mode].append({
                    "target_recall": float(found[j]), "answer_accuracy": float(hit[j]),
                    "ce": float(ce[j]), "retained_mass": float(mass[j]),
                    "output_error": float(error[j]), "keys_scored": int((selected[j] >= 0).sum()),
                    "node_evaluations": r.node_evaluations, "initial_miss": not bool(first_found[j]),
                    "milliseconds_per_query": elapsed * 1000 / len(q),
                })
            # Independent dense oracle over exactly selected ids, not full n.
            ss = dense_logits.gather(1, selected.clamp_min(0)).masked_fill(selected < 0, -torch.inf)
            vv = V.gather(1, selected.clamp_min(0)[..., None].expand(-1, -1, V.shape[-1]))
            check = (ss.softmax(-1)[..., None] * vv).sum(1)
            max_union_diff = max(max_union_diff, float((check - r.output).abs().max()))
    result = {}
    for mode, rows in records.items():
        result[mode] = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        misses = [row for row in rows if row["initial_miss"]]
        result[mode]["initial_miss_count"] = len(misses)
        result[mode]["recovery_given_initial_miss"] = (
            sum(row["target_recall"] for row in misses) / len(misses) if misses else None)
    return {"n": n, "queries": queries, "no_clue": no_clue, "modes": result,
            "maximum_union_output_difference": max_union_diff,
            "controller_state_scalars": model.state_dim,
            "accumulator_scalars": d + 16 + 2,
            "retained_id_capacity": 2 * budget}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/trainable_repair")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--lengths", default="256,1024,4096")
    p.add_argument("--queries", type=int, default=512)
    args = p.parse_args()
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    report = {"config": vars(args), "device": device,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("trainable_repair.py", "trainable_repair_experiment.py")},
              "scope": "controlled rotated-address task; fresh banks and random labels; hard token-tree reads",
              "training_teacher": "target address supervised only for route_ce; no teacher at evaluation",
              "runs": []}
    for seed in map(int, args.seeds.split(",")):
        for objective in ("raw_ce", "route_ce", "warmup_then_ce"):
            model, head, rotation, history = train(objective, seed=seed, device=device, steps=args.steps)
            torch.save({"model": model.state_dict(), "state_head": head.state_dict(),
                        "rotation": rotation, "config": {"routing_dim": 24, "value_dim": 40, "state_dim": 64}},
                       root / f"{objective}_{seed}.pt")
            row = {"seed": seed, "objective": objective, "history": history, "evaluation": []}
            for n in map(int, args.lengths.split(",")):
                evaluation = evaluate(model, head, rotation, n=n, seed=seed + 700, device=device,
                                      queries=args.queries)
                row["evaluation"].append(evaluation)
                print(objective, seed, n, evaluation["modes"]["learned_retry"], flush=True)
            row["no_clue"] = evaluate(model, head, rotation, n=1024, seed=seed + 700,
                                      device=device, queries=args.queries, no_clue=True)
            report["runs"].append(row)
            (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("saved", root / "results.json", flush=True)


if __name__ == "__main__":
    main()
