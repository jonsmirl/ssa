"""
One chart: dense O(n²)  vs  Subquadratic's CLAIM  vs  our MEASURED kernel — solid = measured,
dashed = projection.

Common axis = attention compute *relative to dense @4K* (so it is hardware-neutral and dense is the
n² line). Each curve's vertical gap below dense IS its speedup. Sources:
  * dense + our SSA kernel: real wall-clock (paper/figures/kernel_speed_measured.json, one GPU), solid
    where measured (4K->262K), dashed power-law projection beyond.
  * SubQ's claim: their published speedups (7.2x@128K, 52.2x@1M) + the 1,000x@12M headline, turned into
    relative compute = dense / speedup, dashed (their claimed trajectory; the 12M point is the claim).
Caveat (on the figure): speedups are ratios vs each system's OWN dense on different hardware — the
ratio is the comparable quantity, not absolute ms.

Run: python3 -m ssa.unified_scaling_figure  ->  paper/figures/unified_scaling.png
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("paper/figures/kernel_speed_measured.json"))
n_m = np.array([r["n"] for r in rows], float)
dense = np.array([r["dense_ms"] for r in rows], float)
ssa = np.array([r["ssa_ms"] for r in rows], float)
ref = dense[0]                                   # normalize to dense @ 4K
dense_rel, ssa_rel = dense / ref, ssa / ref

fit = n_m >= 16384                               # power laws past the ~8K crossover
pd_ = np.polyfit(np.log(n_m[fit]), np.log(dense_rel[fit]), 1)
ps_ = np.polyfit(np.log(n_m[fit]), np.log(ssa_rel[fit]), 1)
dense_law = lambda n: np.exp(pd_[1]) * n ** pd_[0]
ssa_law = lambda n: np.exp(ps_[1]) * n ** ps_[0]

subq_n = np.array([131072.0, 1048576.0, 12e6])   # 128K, 1M (published), 12M (claim)
subq_sp = np.array([7.2, 52.2, 1000.0])
subq_rel = dense_law(subq_n) / subq_sp           # claimed compute relative to dense
n_ext = np.logspace(np.log10(262144), np.log10(12e6), 120)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.6, 6.5))
fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(262144, 12e6, color="white", alpha=0.05)
ax.text(2.7e6, 1.4, "projection (dashed)\nbeyond measured", color="#777", fontsize=8, ha="center")

# dense O(n^2) — the n^2 line
ax.plot(n_m, dense_rel, color="#e8806f", lw=2.6, marker="o", ms=4.5, label="dense  O(n²) — measured")
ax.plot(n_ext, dense_law(n_ext), color="#e8806f", lw=2.0, ls=(0, (5, 3)), label="dense  O(n²) — projected")
# our measured kernel
ax.plot(n_m, ssa_rel, color="#3fbf90", lw=2.6, marker="o", ms=4.5, label="our SSA kernel — measured (22× @262K)")
ax.plot(n_ext, ssa_law(n_ext), color="#3fbf90", lw=2.0, ls=(0, (5, 3)),
        label=f"our SSA — projected (~{dense_law(12e6) / ssa_law(12e6):.0f}× @12M)")
# SubQ's claim
ax.plot(subq_n, subq_rel, color="#f2c14e", lw=1.8, ls=(0, (4, 3)), zorder=4, label="SubQ — claimed trajectory")
ax.scatter(subq_n[:2], subq_rel[:2], color="#f2c14e", s=55, zorder=5, label="SubQ published (7.2×, 52.2×)")
ax.scatter(subq_n[2:], subq_rel[2:], marker="*", s=300, color="#ffd23f", edgecolor="white", linewidth=0.6,
           zorder=6, label="SubQ claim 1,000× @12M")

ax.annotate("at 12M: SubQ claims ~1,000×;\nour measured-and-projected ~260×",
            xy=(12e6, subq_rel[2]), xytext=(1.1e6, 6.5), color="#e8d6a0", fontsize=8.3,
            arrowprops=dict(arrowstyle="->", color="#e8d6a0", lw=0.9))

ticks = [4096, 16384, 65536, 262144, 1048576, 12e6]
ax.set_xticks(ticks); ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "12M"])
ax.set_xlim(4096, 1.4e7); ax.set_ylim(1, 6e6)
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Attention compute relative to dense @4K  (log; lower = faster)", color="#bdbdbd")
ax.text(0.0, 1.10, "SCALING — dense O(n²) vs SubQ's claim vs our measured kernel (solid=measured, dashed=projected)",
        transform=ax.transAxes, color="#8a8a8a", fontsize=8.4, fontweight="bold")
ax.set_title("Attention compute: the claim, the baseline, and the measurement", color="white",
             fontsize=14, loc="left", pad=24)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.12, labelcolor="#dddddd")
fig.text(0.012, 0.012, "Speedups are ratios vs each system's own dense on different hardware; the ratio is the "
         "comparable quantity, not absolute ms. SubQ's 12M point is the claim, above its own 2-point scaling.",
         color="#6f6f6f", fontsize=6.6)
fig.tight_layout(rect=(0, 0.028, 1, 1))
out = "paper/figures/unified_scaling.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"@12M relative compute: dense={dense_law(12e6):.0f}, ours={ssa_law(12e6):.0f} "
      f"({dense_law(12e6)/ssa_law(12e6):.0f}×), SubQ-claim={subq_rel[2]:.0f} (1000×)")
