"""Fixed-size tail state with exact selected-key replacement on real Qwen Q/K/V.

This tests an output correction at UNCHANGED selected-key budget. Coarse key
cells hold counts and value sums. A small network predicts a per-key exponential
weight in each cell from q and the observed sparse read. Selected approximate
weights are subtracted, then their exact weights are inserted. No tail values
are opened at query time. This is an approximation, not a mass certificate.

Cell centers are fitted on the first 512 keys only. The network is trained on
queries before position 4096; all final queries lie at/after 4096. Prefix states
contain only keys preceding or at that query, including on held-out positions.
The cumulative-state tensor is a batch-training convenience: the serving state
is one counts/value-sums table, O(cells*(value_dim+1)), independent of context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def prepare(cache, cells=32, seed=0, device="cpu", block=64, top_blocks=2):
    z = np.load(cache)
    Q, K, V = [torch.tensor(z[key], device=device, dtype=torch.float32) for key in ("Q", "K", "V")]
    n, d = K.shape
    if n < 8192:
        raise ValueError("the prescribed train/test split requires at least 8192 positions")
    torch.manual_seed(seed)
    centers = K[:512][torch.randperm(512, device=device)[:cells]].clone()
    for _ in range(15):
        assignment = torch.cdist(K[:512], centers).argmin(-1)
        for c in range(cells):
            if (assignment == c).any():
                centers[c] = K[:512][assignment == c].mean(0)
    assignment = torch.cdist(K, centers).argmin(-1)
    features = F.one_hot(assignment, cells).float()
    counts = features.cumsum(0)
    sums = (features[..., None] * V[:, None]).cumsum(0)
    means = K[:n // block * block].reshape(-1, block, d).mean(1)
    # Fixed train/validation/test positions. All modes share the exact same ids.
    positions = torch.arange(512, 8192, 16, device=device)
    result = {k: [] for k in ("q", "sparse", "dense", "log_z", "tail_counts", "tail_sums",
                              "true_log_tail", "keys", "mass", "positions", "true_cell_logmass")}
    for pos in positions.tolist():
        q = Q[pos]
        full = (pos + 1) // block
        route = (means[:full] @ q).argsort(descending=True, stable=True)[:top_blocks]
        ids = (route[:, None] * block + torch.arange(block, device=device)).flatten()
        if (pos + 1) % block:
            ids = torch.cat((ids, torch.arange(full * block, pos + 1, device=device)))
        logits = K[:pos + 1] @ q / d**0.5
        kept_logits = logits[ids]
        log_z = kept_logits.logsumexp(0)
        sparse = kept_logits.softmax(0) @ V[ids]
        dense = logits.softmax(0) @ V[:pos + 1]
        tail_count = counts[pos] - features[ids].sum(0)
        tail_sum = sums[pos] - features[ids].T @ V[ids]
        omitted = torch.ones(pos + 1, device=device, dtype=torch.bool); omitted[ids] = False
        tail_logits = logits.masked_fill(~omitted, -torch.inf)
        true_cell_mass = torch.stack([tail_logits.masked_fill(assignment[:pos + 1] != c,
                                                              -torch.inf).logsumexp(0) for c in range(cells)])
        vals = (q, sparse, dense, log_z, tail_count, tail_sum, tail_logits.logsumexp(0),
                torch.tensor(len(ids), device=device), logits.softmax(0)[ids].sum(),
                torch.tensor(pos, device=device), true_cell_mass)
        for key, val in zip(result, vals):
            result[key].append(val)
    result = {k: torch.stack(v) for k, v in result.items()}
    return result, centers


class TailCorrector(nn.Module):
    """Predict coarse residual mass; reconstruct its value via fixed-size state."""

    def __init__(self, centers, value_dim):
        super().__init__()
        self.register_buffer("centers", centers)
        d = centers.shape[1]
        self.net = nn.Sequential(nn.Linear(d + value_dim + 1, 128), nn.SiLU(),
                                 nn.Linear(128, len(centers)))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, sparse, log_z, counts, sums, *, learned=True, log_gain=0.0):
        logits = q @ self.centers.T / q.shape[-1]**0.5
        logits = logits + log_gain
        if learned:
            x = torch.cat((q / 4, sparse, log_z[:, None] / 20), -1)
            logits = logits + self.net(x)
        log_count = counts.clamp_min(1).log()
        log_mass = (logits + log_count).masked_fill(counts <= 0, -torch.inf)
        # This softmax joins the exact selected partition to positive tail cells.
        weights = torch.cat((log_z[:, None], log_mass), -1).softmax(-1)
        tail_means = sums / counts.clamp_min(1)[..., None]
        output = weights[:, :1] * sparse + (weights[:, 1:, None] * tail_means).sum(1)
        return output, log_mass, weights[:, 1:].sum(-1)


def subset(data, ids):
    return {key: val[ids] for key, val in data.items()}


def run_model(model, data, learned=True, log_gain=0.0):
    return model(data["q"], data["sparse"], data["log_z"], data["tail_counts"],
                 data["tail_sums"], learned=learned, log_gain=log_gain)


@torch.no_grad()
def evaluate(model, data, log_gain=0.0):
    base = (data["sparse"] - data["dense"]).norm(dim=-1)
    outputs = {"sparse": data["sparse"], "centroid_tail": run_model(model, data, False)[0],
               "learned_tail": run_model(model, data)[0],
               "calibrated_centroid_tail": run_model(model, data, False, log_gain)[0],
               "tail_mean_only": data["tail_sums"].sum(1) / data["tail_counts"].sum(1)[:, None]}
    # An oracle assigns exact omitted mass to each coarse cell but still uses
    # its uniform value mean: it exposes approximation loss inside those cells.
    weights = torch.cat((data["log_z"][:, None], data["true_cell_logmass"]), -1).softmax(-1)
    means = data["tail_sums"] / data["tail_counts"].clamp_min(1)[..., None]
    outputs["oracle_cell_mass"] = weights[:, :1] * data["sparse"] + (weights[:, 1:, None] * means).sum(1)
    result = {}
    for name, out in outputs.items():
        error = (out - data["dense"]).norm(dim=-1)
        result[name] = {"mean_l2_error": float(error.mean()), "median_l2_error": float(error.median()),
                        "mean_squared_error": float((out - data["dense"]).square().mean()),
                        "fraction_better_than_sparse": float((error < base).float().mean()),
                        "p95_l2_error": float(torch.quantile(error, 0.95))}
    result["queries"] = len(base)
    result["mean_keys_scored"] = float(data["keys"].float().mean())
    result["mean_actual_selected_mass"] = float(data["mass"].mean())
    result["learned_tail_mass_mae"] = float((run_model(model, data)[2] - (1 - data["mass"])).abs().mean())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/ssa_qwen_qkv_8192.npz")
    p.add_argument("--out", default="runs/tail_state")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    data, centers = prepare(args.cache, args.cells, args.seed, device)
    train = subset(data, data["positions"] < 3072)
    validation = subset(data, (data["positions"] >= 3072) & (data["positions"] < 4096))
    test = subset(data, data["positions"] >= 4096)
    model = TailCorrector(centers, data["sparse"].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    g = torch.Generator(device=device).manual_seed(args.seed + 20)
    best, best_state, best_step = float("inf"), None, None
    history = []
    for step in range(args.steps):
        b = subset(train, torch.randint(len(train["q"]), (64,), device=device, generator=g))
        out, lm, _ = run_model(model, b)
        target_weights = torch.cat((b["log_z"][:, None], b["true_cell_logmass"]), -1).softmax(-1)
        pred_logits = torch.cat((b["log_z"][:, None], lm), -1)
        # Teacher distribution over kept mass and omitted cells, masked safely
        # for empty cells. This trains positive mass, not only an arbitrary output.
        log_p = pred_logits.log_softmax(-1)
        distill = -(target_weights * log_p.masked_fill(target_weights == 0, 0)).sum(-1).mean()
        loss = F.mse_loss(out, b["dense"]) + 0.05 * distill
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                score = F.mse_loss(run_model(model, validation)[0], validation["dense"]).item()
            if score < best:
                best = score; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_step = step
            history.append({"step": step, "validation_mse": score})
            print(history[-1], flush=True)
    model.load_state_dict(best_state)
    # Choose only on validation: a scalar repairs the systematic exponential
    # mass underestimate without fitting a high-dimensional query network.
    with torch.no_grad():
        gain_sweep = [{"log_gain": gain / 2,
                       "validation_mse": F.mse_loss(run_model(model, validation, False, gain / 2)[0],
                                                     validation["dense"]).item()}
                      for gain in range(-4, 33)]
    log_gain = min(gain_sweep, key=lambda row: row["validation_mse"])["log_gain"]
    report = {"config": vars(args), "device": device, "best_validation_step": best_step,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "fixture_sha256": hashlib.sha256(Path(args.cache).read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train": evaluate(model, train, log_gain), "validation": evaluate(model, validation, log_gain),
              "test": evaluate(model, test, log_gain), "history": history,
              "calibrated_log_gain": log_gain, "validation_gain_sweep": gain_sweep,
              "serving_state_scalars": args.cells * (data["sparse"].shape[-1] + 1),
              "scope": "one Qwen head/document; temporal split; real values; approximate tail, no certificate",
              "work": "state append: nearest of cells centers + one count/value-sum update; query: cells summaries + fixed selected keys",
              "router": "two complete contiguous blocks by mean score plus visible partial block; flat reference"}
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "centers": centers}, root / "model.pt")
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
