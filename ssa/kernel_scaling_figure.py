"""
Measured kernel scaling — our actual wall-clock results (solid) + math extrapolation (dashed).

Reads the real measurement (paper/figures/kernel_speed_measured.json: dense FlashAttention vs the SSA
block-sparse kernel, H=8, d=64, fp16, one GPU) and draws attention wall-clock vs context length:

  * DENSE (red): measured solid, then the O(n²) extrapolation dashed — this is the "n²" line.
  * SSA   (green): measured solid, then the model extrapolation dashed (power law fit past the ~8K
    crossover to the measured points — i.e. the realized scaling of the shipped flat-block-router kernel).

Honesty: the timed kernel (ssa_kernel.py) uses the FLAT block router; the hierarchical/treecode router
(hierarchical_routing.py) is a reference not yet wired into it (RESULTS § "treecode reality check"), so
the dashed SSA line extrapolates the *flat* kernel we actually measured — it bends above pure O(n). The
hierarchical router is the design path that would pull it back down toward O(n); shown dotted, labelled
as a design target (unbuilt at scale), NOT a measured result.

Run:  python3 -m ssa.kernel_scaling_figure   ->  paper/figures/kernel_scaling_measured.png
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("paper/figures/kernel_speed_measured.json"))
n_m = np.array([r["n"] for r in rows], float)
dense_m = np.array([r["dense_ms"] for r in rows], float)
ssa_m = np.array([r["ssa_ms"] for r in rows], float)

# power-law fits on the clean scaling region (past the ~8K crossover)
fit = n_m >= 16384
pd = np.polyfit(np.log(n_m[fit]), np.log(dense_m[fit]), 1)   # dense exponent ~ n^1.9
ps = np.polyfit(np.log(n_m[fit]), np.log(ssa_m[fit]), 1)     # SSA exponent (flat kernel) ~ n^1.4
dense_law = lambda n: np.exp(pd[1]) * n ** pd[0]
ssa_law = lambda n: np.exp(ps[1]) * n ** ps[0]

n_ext = np.logspace(np.log10(262144), np.log10(12e6), 200)
sp_262k = dense_m[-1] / ssa_m[-1]
sp_12m = dense_law(12e6) / ssa_law(12e6)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.6, 6.4))
fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(262144, 12e6, color="white", alpha=0.05)
ax.text(2.4e6, 0.18, "extrapolation\n(beyond measured)", color="#777", fontsize=8.3, ha="center")

# DENSE — the n^2 line
ax.plot(n_m, dense_m, color="#e8806f", lw=2.6, marker="o", ms=5, label="dense FlashAttention — measured")
ax.plot(n_ext, dense_law(n_ext), color="#e8806f", lw=2.0, ls=(0, (5, 3)), label="dense  O(n²) — extrapolated")
# SSA — our actual results
ax.plot(n_m, ssa_m, color="#3fbf90", lw=2.6, marker="o", ms=5, label="SSA kernel — measured (flat block router)")
ax.plot(n_ext, ssa_law(n_ext), color="#3fbf90", lw=2.0, ls=(0, (5, 3)),
        label=f"SSA — extrapolated (measured n^{ps[0]:.1f} scaling)")
# the hierarchical/treecode router is now MEASURED at the router level (treecode_router_scaling.png):
# it is SLOWER than the flat router in range, so it does not lift this green line — no O(n) line drawn.
ax.text(1.6e6, 6.0, "treecode (hierarchical) router measured separately —\n"
        "treecode_router_scaling.png: slower in range, wins only\n"
        "past the flat router's ~3.5M OOM wall",
        color="#8fb8cc", fontsize=7.2, ha="center", va="center")

ax.annotate(f"{sp_262k:.0f}× measured\n@262K", xy=(262144, dense_m[-1]), xytext=(7.0e4, 2.0e3),
            color="#ddd", fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#666", lw=0.8))
ax.text(1.30e7, dense_law(12e6), f"~{sp_12m:.0f}×\n@12M", color="#e8b0a6", fontsize=8.5, va="center")
ax.text(8400, 0.5, "crossover\n~8K", color="#888", fontsize=8, ha="center")

ticks = [4096, 16384, 65536, 262144, 1048576, 12e6]
ax.set_xticks(ticks); ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "12M"])
ax.set_xlim(4096, 1.55e7); ax.set_ylim(0.15, 3e6)
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Attention wall-clock per call (ms, log)", color="#bdbdbd")
ax.text(0.0, 1.10, "SCALING — measured (solid) + math extrapolation (dashed), one GPU",
        transform=ax.transAxes, color="#8a8a8a", fontsize=9, fontweight="bold")
ax.set_title("Attention compute: dense O(n²) vs SSA — our measured kernel",
             color="white", fontsize=14, loc="left", pad=24)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="upper left", fontsize=8.2, framealpha=0.12, labelcolor="#dddddd")
fig.tight_layout()
out = "paper/figures/kernel_scaling_measured.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"dense ~ n^{pd[0]:.2f},  SSA(flat) ~ n^{ps[0]:.2f}")
print(f"measured speedup @262K = {sp_262k:.1f}x ;  extrapolated @12M = {sp_12m:.0f}x (flat kernel)")
