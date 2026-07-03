"""
The P9 gate/aux result (D2) — read- vs write-salient recall at overload for {ssa, deltanet no-gate, +gate,
+gate+aux}: the TRAINED mirror of P8's write-time-vs-read-time-relevance split. A learned write gate lifts
the compression corner only where a write-time signal exists (write-salient), not where relevance is
read-time-only. Reads paper/figures/p9_compare.json.  Run: python3 -m ssa.p9_gate_figure
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["ssa (selection)", "deltanet no-gate", "deltanet +gate", "deltanet +gate+aux"]
COLORS = {"ssa (selection)": "#5aa9e6", "deltanet no-gate": "#8a8a8a",
          "deltanet +gate": "#3fbf90", "deltanet +gate+aux": "#2f9e78"}
REGIMES = [("write_salient", "write-salient — marker keys (keepable at write time)"),
           ("read_salient", "read-salient — stock MQAR (query unknown at write time)")]


def main():
    d = json.load(open("paper/figures/p9_compare.json"))
    dh = d["meta"].get("head_dim", 16)
    rows = {(r["regime"], r["arm"]): r["recall"] for r in d["rows"] if r["test"] == "D2"}

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    x = np.arange(len(REGIMES)); w = 0.2
    for i, arm in enumerate(ARMS):
        vals = [rows.get((rg, arm), 0.0) for rg, _ in REGIMES]
        ax.bar(x + (i - 1.5) * w, vals, w, label=arm, color=COLORS[arm], edgecolor="#111", linewidth=0.5)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", color="#ccc", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in REGIMES], fontsize=8.6, color="#ddd")
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("trained recall at query positions", color="#bdbdbd")
    ax.text(0.0, 1.055, f"TRAINED — micro-LM on MQAR at overload m=32 > state dh={dh} (P9-D2)",
            transform=ax.transAxes, color="#8a8a8a", fontsize=8.4, fontweight="bold")
    ax.set_title("Does a learned write gate close the compression gap?", color="white",
                 fontsize=13.5, loc="left", pad=30)
    ax.grid(True, axis="y", color="#2a2a2a", lw=0.5)
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.15, labelcolor="#ddd", ncol=1)
    fig.text(0.012, 0.02, "The trained mirror of P8's 0.10-vs-1.00 split. Write-salient (reserved marker keys) is "
             "solved by every arm — the no-gate delta rule too — so the gate adds nothing;\nread-salient leaves "
             "DeltaNet near its capacity floor (~0.6) with or without gate/aux, far below selection (SSA 0.89). "
             "Synthetic MQAR, trained end-to-end.", color="#6f6f6f", fontsize=6.4)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig("paper/figures/p9_gate.png", dpi=150, facecolor=fig.get_facecolor())
    print("wrote paper/figures/p9_gate.png")


if __name__ == "__main__":
    main()
