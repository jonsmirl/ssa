"""Dense-oracle diagnostic of prototype versus actual-cell-mean tail estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/ssa_qwen_qkv_8192.npz")
    p.add_argument("--out", default="runs/tail_mass_diagnostic.json")
    args = p.parse_args()
    data = np.load(args.cache)
    Q, K, V = (data[x].astype(np.float64) for x in ("Q", "K", "V"))
    centers = K[np.linspace(0, 63, 16).astype(int)]
    assignment = ((K[:, None] - centers[None])**2).sum(-1).argmin(-1)
    gain = json.loads(Path("runs/qwen_tail_final/results.json").read_text())["trained_log_gains"][18][0]
    records = []
    for pos in np.linspace(512, len(K) - 1, 32).astype(int):
        q, k, v = Q[pos], K[:pos + 1], V[:pos + 1]
        full = pos // 64
        means = k[:full * 64].reshape(full, 64, -1).mean(1)
        route = np.argsort(-(means @ q), kind="stable")[:2]
        selected = np.concatenate(((route[:, None] * 64 + np.arange(64)).ravel(), np.arange(full * 64, pos + 1)))
        tail = np.ones(pos + 1, bool); tail[selected] = False
        logits = k @ q / np.sqrt(k.shape[-1]); shift = logits.max()
        w = np.exp(logits - shift)
        zS, zT = w[~tail].sum(), w[tail].sum()
        dense = w @ v / w.sum(); sparse = w[~tail] @ v[~tail] / zS
        counts = np.bincount(assignment[:pos + 1][tail], minlength=16)
        keysum = np.zeros_like(centers); valuesum = np.zeros((16, V.shape[1]))
        np.add.at(keysum, assignment[:pos + 1][tail], k[tail])
        np.add.at(valuesum, assignment[:pos + 1][tail], v[tail])
        actualmean = keysum / np.maximum(counts, 1)[:, None]
        row = {"position": int(pos), "true_tail_share": zT / (zS + zT), "sparse_mse": float(np.mean((sparse-dense)**2)), "modes": {}}
        for name, mu, g in (("prototype_saved", centers, gain), ("jensen", actualmean, 0.)):
            perkey = np.exp(mu @ q / np.sqrt(k.shape[-1]) + g - shift)
            mass = perkey * counts
            numerator = perkey @ valuesum
            output = (zS * sparse + numerator) / (zS + mass.sum())
            coarse = numerator / mass.sum()
            direction, error = coarse - sparse, dense - sparse
            row["modes"][name] = {"mass_ratio_to_true": float(mass.sum()/zT),
                                  "tail_share": float(mass.sum()/(zS+mass.sum())),
                                  "output_mse": float(np.mean((output-dense)**2)),
                                  "direction_alignment": float(error @ direction),
                                  "maximum_improving_share": float(2*(error @ direction)/max(direction @ direction,1e-300))}
        records.append(row)
    result = {"scope": "diagnostic previously inspected Qwen layer18 KV0 fixture, not validation selection", "fixture_sha256": hashlib.sha256(Path(args.cache).read_bytes()).hexdigest(), "queries": records}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for name in ("prototype_saved", "jensen"):
        ratios = np.asarray([r["modes"][name]["mass_ratio_to_true"] for r in records])
        print(name, "mass ratios median/max", np.median(ratios), ratios.max(), "overestimates", (ratios>1+1e-10).sum())


if __name__ == "__main__":
    main()
