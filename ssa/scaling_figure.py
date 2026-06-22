"""
The honest scaling chart — SubQ's "O(n²) vs O(n)" redrawn from SubQ's OWN two data points.

The marketing chart plots two textbook curves (n² and n) and labels the straight line "SubQ O(n)".
This redraws it as *speedup vs context length* using only SubQ's two published speedups
(7.2×@128K, 52.2×@1M) plus the per-query-block cost model this rig derived:

    speedup(n) = n / (kappa + n/b²)          # dense n² over (attention n·kappa + flat router (n/b)²)

Fitting that to the two points pins kappa (keys attended) and b (block size), hence the FLAT-router
ceiling b². The figure contrasts two router families:
  * FLAT / fixed-block router — saturates toward a hard b² ceiling past ~1M;
  * SSA's HIERARCHICAL (treecode) router — O(n) selection, so it tracks the linear ideal.

Honesty (kept from this rig's own audit, RESULTS § "treecode reality check"):
  * the green hierarchical curve is the router's *complexity* — node-count validated (a) and a
    wall-clock crossover measured at ~1M (b); the shipped kernel still uses the flat block router and
    the naive 2-level regressed at 2M, so 1M→12M is DASHED (design/asymptotic, not a measured result);
  * SubQ's two points sit on the bend where both families coincide (≤1M), so the data alone can't tell
    them apart;
  * SubQ's 1,000×@12M sits above the flat ceiling AND above an O(n) router at the fitted kappa
    (~670×) — i.e. inflated regardless of router, reachable only by a tighter budget (a quality cost).

The necessity behind the ceiling is machine-checked: Substrate Lean `flat_router_work` /
`subquadratic_forces_skip`.

Run:  python3 -m ssa.scaling_figure      ->  paper/figures/scaling_claim_vs_realized.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- fit the per-query-block model n/(kappa + n/b^2) to SubQ's two published points ---
PTS_N = np.array([131072.0, 1048576.0])      # 128K, 1M
PTS_S = np.array([7.2, 52.2])                # SubQ published prefill speedups
A = np.array([[1.0, PTS_N[0]], [1.0, PTS_N[1]]])
kappa, inv_b2 = np.linalg.solve(A, PTS_N / PTS_S)
b2 = 1.0 / inv_b2
b = b2 ** 0.5
CLAIM_N, CLAIM_S = 12e6, 1000.0
SPLIT = 1.05e6                                # solid (supported) | dashed (extrapolation) boundary

n = np.logspace(np.log10(32768), np.log10(12e6), 600)
flat = n / (kappa + n / b2)                                       # flat/fixed-block router: saturates at b^2
hier = n / (kappa + np.log2(np.maximum(n / b, 2)) / b)           # SSA hierarchical (treecode) router: ~O(n)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.4, 6.4))
fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
ax.set_xscale("log"); ax.set_yscale("log")

ax.axvspan(SPLIT, 12e6, color="white", alpha=0.05)
ax.text(3.6e6, 1.22, "1M → 12M: extrapolation\n(no SubQ data; design/asymptotic)",
        color="#777", fontsize=8.3, ha="center", va="bottom")


def split_plot(y, color, label, lw):
    m = n <= SPLIT
    ax.plot(n[m], y[m], color=color, lw=lw, label=label)
    ax.plot(n[~m], y[~m], color=color, lw=lw, ls=(0, (4, 3)))


split_plot(hier, "#3fbf90", "SSA: hierarchical (treecode) router — O(n) selection [design]", 2.7)
split_plot(flat, "#e8806f", "flat / fixed-block router — saturates at b²", 2.7)
ax.axhline(b2, color="#e8806f", ls=":", lw=1.1, alpha=0.7)
ax.text(3.4e4, b2 * 1.06, f"flat-router ceiling  b² ≈ {b2:.0f}×", color="#e8806f", fontsize=9, va="bottom")
ax.text(1.30e7, hier[-1], "≈670×", color="#3fbf90", fontsize=9, va="center", ha="left")

ax.scatter(PTS_N, PTS_S, color="white", s=58, zorder=5, edgecolor="#000", linewidth=0.5,
           label="SubQ published (7.2×@128K, 52.2×@1M) — on the bend, families coincide")
ax.scatter([CLAIM_N], [CLAIM_S], marker="*", s=350, color="#ff5d5d", zorder=6,
           edgecolor="white", linewidth=0.6, label="SubQ claim 1,000×@12M")
ax.annotate("above the flat ceiling AND\nabove an O(n) router (~670×)",
            xy=(CLAIM_N, CLAIM_S), xytext=(1.55e6, 1550), color="#ff9d9d", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#ff9d9d", lw=1.0))

ticks = [32768, 65536, 131072, 262144, 524288, 1048576, 12e6]
ax.set_xticks(ticks); ax.set_xticklabels(["32K", "64K", "128K", "256K", "512K", "1M", "12M"])
ax.set_xlim(32768, 1.55e7); ax.set_ylim(1, 2700)
ax.set_yticks([1, 3, 10, 30, 100, 300, 1000]); ax.set_yticklabels(["1×", "3×", "10×", "30×", "100×", "300×", "1000×"])
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Prefill speedup vs dense  (log)", color="#bdbdbd")
ax.text(0.0, 1.115, "SCALING — an independent reading of SubQ's own two data points",
        transform=ax.transAxes, color="#8a8a8a", fontsize=9, fontweight="bold")
ax.set_title("Attention speedup: a flat router ceilings; SSA's hierarchical router stays O(n)",
             color="white", fontsize=13.5, loc="left", pad=24)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="lower right", fontsize=8.1, framealpha=0.12, labelcolor="#dddddd")
fig.text(0.012, 0.012,
         "Green = the hierarchical router's O(n) selection complexity (node-count validated; wall-clock crossover measured at ~1M, RESULTS §b); "
         "1M→12M dashed = design/asymptotic, not a measured result. Even an O(n) router reaches ~670×, not 1,000×, at the fitted κ≈18k.",
         color="#6f6f6f", fontsize=6.6)
fig.tight_layout(rect=(0, 0.028, 1, 1))

os.makedirs("paper/figures", exist_ok=True)
out = "paper/figures/scaling_claim_vs_realized.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"fit:  kappa={kappa:.0f} keys,  b={b:.1f}  ->  flat-router ceiling b^2={b2:.0f}x")
for nn in (1.048576e6, 12e6):
    print(f"  {nn/1e6:5.2f}M : flat={nn/(kappa+nn/b2):6.0f}x (ceil {b2:.0f}x) | "
          f"hier(SSA)={nn/(kappa+np.log2(nn/b)/b):6.0f}x | claim {1000 if nn>2e6 else '-'}")
