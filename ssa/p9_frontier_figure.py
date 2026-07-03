"""
The P9 trained frontier — recall vs load for the four token-mixers (D1), the selection-vs-compression
crossover measured for a TRAINED model. Reads paper/figures/p9_compare.json.
Run: python3 -m ssa.p9_frontier_figure
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"dense": ("#e8806f", "o", "dense attn — selection (κ=n)"),
         "ssa": ("#5aa9e6", "s", "SSA — selection (κ≈dh)"),
         "deltanet": ("#3fbf90", "D", "DeltaNet — compression (state dh)"),
         "linear": ("#c9a0dc", "^", "linear attn — compression, no gate")}


def main():
    d = json.load(open("paper/figures/p9_compare.json"))
    meta = d["meta"]
    dh = meta.get("head_dim", 32)
    d1 = {r["mixer"]: r["recall"] for r in d["rows"] if r["test"] == "D1"}

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    for mixer, (color, mk, lab) in STYLE.items():
        if mixer not in d1:
            continue
        loads = sorted(int(k) for k in d1[mixer])
        rec = [d1[mixer][str(l)] for l in loads]
        ax.plot(loads, rec, color=color, lw=2.4, marker=mk, ms=6, label=lab)
    ax.axvline(dh, color="#888", lw=1.2, ls=(0, (2, 3)))
    ax.text(dh * 1.03, 0.06, f"DeltaNet state dh={dh}", color="#aaa", fontsize=8, rotation=90, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(int(k) for k in next(iter(d1.values()))))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("MQAR load — # key-value pairs m", color="#bdbdbd")
    ax.set_ylabel("trained recall at query positions", color="#bdbdbd")
    ax.text(0.0, 1.045, "TRAINED — micro-LM on MQAR, matched state (P9)",
            transform=ax.transAxes, color="#8a8a8a", fontsize=8.4, fontweight="bold")
    ax.set_title("Selection vs compression, trained: recall vs load", color="white",
                 fontsize=13.5, loc="left", pad=30)
    ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.12, labelcolor="#ddd")
    fig.text(0.012, 0.02, "Trained end-to-end; synthetic MQAR (not natural language). Selection (dense/SSA) is "
             "content-addressed → flat in load; both\ncompression corners degrade past the state dim dh — DeltaNet "
             "more (its erase forgets older queried pairs) than additive linear.",
             color="#6f6f6f", fontsize=6.6)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig("paper/figures/p9_frontier.png", dpi=150, facecolor=fig.get_facecolor())
    print("wrote paper/figures/p9_frontier.png")


if __name__ == "__main__":
    main()
