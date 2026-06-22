"""
Router scaling — MEASURED: flat (n/b)² GEMM router vs the bounded-candidate treecode (beam descend).

Reads paper/figures/treecode_router_measured.json (real wall-clock from treecode_bench, 16 GB GPU) and
draws router time vs context length: each router solid where measured (256K→2M), dashed as its own
power-law extrapolation. Two honest markers the data forces:
  * the flat router's nb² matrix OOMs at ~3.5M (H·nb²·2 bytes > GPU) — its dashed line STOPS there;
  * the treecode keeps running (linear memory) and its dashed line continues; the wall-clock crossover
    where it also gets FASTER is far out (~16M), but the decisive line is the flat OOM, not the crossover.

Run: python3 -m ssa.treecode_router_figure  ->  paper/figures/treecode_router_scaling.png
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("paper/figures/treecode_router_measured.json"))
n = np.array([r["n"] for r in rows], float)
flat = np.array([r["flat_ms"] for r in rows], float)
desc = np.array([r["descend_ms"] for r in rows], float)

pf = np.polyfit(np.log(n), np.log(flat), 1)
pd = np.polyfit(np.log(n), np.log(desc), 1)
flat_law = lambda x: np.exp(pf[1]) * x ** pf[0]
desc_law = lambda x: np.exp(pd[1]) * x ** pd[0]
cross = np.exp((pf[1] - pd[1]) / (pd[0] - pf[0]))          # flat_law == desc_law
OOM = 3.5e6                                                # flat nb² matrix exceeds the 16 GB GPU here

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.4, 6.3))
fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
ax.set_xscale("log"); ax.set_yscale("log")

ax.axvspan(n[-1], 1.6e7, color="white", alpha=0.05)
ax.axvline(OOM, color="#e8806f", ls=":", lw=1.3, alpha=0.8)
ax.text(OOM * 1.05, 270, "flat router OOMs\n(nb² matrix > 16 GB)", color="#e8b0a6", fontsize=8.3, va="top")

# flat: measured solid, extrapolated dashed -> stops at the OOM wall
xf = np.logspace(np.log10(n[-1]), np.log10(OOM), 60)
ax.plot(n, flat, color="#e8806f", lw=2.6, marker="o", ms=5, label=f"flat (n/b)² GEMM — measured  (~n^{pf[0]:.1f})")
ax.plot(xf, flat_law(xf), color="#e8806f", lw=2.0, ls=(0, (5, 3)), label="flat — extrapolated, then OOM")
ax.scatter([OOM], [flat_law(OOM)], marker="x", s=80, color="#e8806f", zorder=6)

# treecode: measured solid, extrapolated dashed -> continues past the wall
xt = np.logspace(np.log10(n[-1]), np.log10(1.6e7), 80)
ax.plot(n, desc, color="#3fbf90", lw=2.6, marker="o", ms=5,
        label=f"treecode beam descend — measured  (~n^{pd[0]:.1f})")
ax.plot(xt, desc_law(xt), color="#3fbf90", lw=2.0, ls=(0, (5, 3)), label="treecode — extrapolated (runs: linear memory)")

ax.annotate("2M: flat still 2.4× faster\n(better constant)", xy=(n[-1], desc[-1]), xytext=(2.9e5, 330),
            color="#bbb", fontsize=8.2, arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
if n[-1] < cross < 1.6e7:
    ax.annotate(f"time crossover ~{cross/1e6:.0f}M\n(but flat is long dead by OOM)",
                xy=(cross, desc_law(cross)), xytext=(5.0e6, 60), color="#9fd9c4", fontsize=8.2,
                arrowprops=dict(arrowstyle="->", color="#9fd9c4", lw=0.9))

ticks = [262144, 524288, 1048576, 2097152, 4194304, 8388608, 1.6e7]
ax.set_xticks(ticks); ax.set_xticklabels(["256K", "512K", "1M", "2M", "4M", "8M", "16M"])
ax.set_xlim(2.3e5, 1.6e7); ax.set_ylim(1, 600)
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Router wall-clock per call (ms, log)", color="#bdbdbd")
ax.text(0.0, 1.10, "ROUTER SCALING — measured (solid) + power-law extrapolation (dashed), one 16 GB GPU",
        transform=ax.transAxes, color="#8a8a8a", fontsize=9, fontweight="bold")
ax.set_title("Flat (n/b)² GEMM router vs the bounded-candidate treecode", color="white",
             fontsize=14, loc="left", pad=24)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="lower right", fontsize=8.2, framealpha=0.12, labelcolor="#dddddd")
fig.tight_layout()
out = "paper/figures/treecode_router_scaling.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"flat ~ n^{pf[0]:.2f},  treecode ~ n^{pd[0]:.2f},  time-crossover ~ {cross/1e6:.1f}M (flat OOMs ~3.5M first)")
