"""
Unified scaling chart — in SubQ's orientation (attention COMPUTE vs context; lower = faster, dense O(n²)
rising at the top). One frame for the whole program:
  * dense O(n²)             — "today's models", the rising curve (measured + projected).
  * our flat-router kernel  — MEASURED (4K→262K) + projected; stays well below dense but its speedup is
                              capped because the argsort BlockMask build (~n^2.12) rises nearly as fast.
  * our IVF-router kernel   — projected (the faiss-GPU IVF router, GPU-measured to 8M, removes the maskbuild)
                              — drops onto the floor.
  * n·κ floor               — the attention-only cost (the lowest any selector can reach).
  * SubQ                    — its claim as compute = dense / reported speedup (published 7.2×,52.2× + 1,000×@12M).
Built from paper/figures/{kernel_speed_measured,cost_profile}.json. Run: python3 -m ssa.unified_scaling_figure
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ks = json.load(open("paper/figures/kernel_speed_measured.json"))
n_m = np.array([r["n"] for r in ks], float)
dense_m = np.array([r["dense_ms"] for r in ks], float)
ssa_m = np.array([r["ssa_ms"] for r in ks], float)
fit = json.load(open("paper/figures/cost_profile.json"))["fit"]
pa = lambda key, x: fit[key][1] * x ** fit[key][0]
block = 128

m = n_m >= 16384
dl = np.polyfit(np.log(n_m[m]), np.log(dense_m[m]), 1)
dense_law = lambda x: np.exp(dl[1]) * x ** dl[0]
flat_kernel = lambda x: pa("attention", x) + pa("router", x) + pa("maskbuild", x)
ivf_kernel = lambda x: pa("attention", x) + pa("router", x) * (x / block) ** -0.5 + pa("maskbuild", x) * (x / block) ** -1.0
floor = lambda x: pa("attention", x)

xe = np.logspace(np.log10(262144), np.log10(12e6), 140)
xall = np.logspace(np.log10(4096), np.log10(12e6), 200)
sub_n = np.array([131072.0, 1048576.0, 12e6])
sub_sp = np.array([7.2, 52.2, 1000.0])
sub_ms = dense_law(sub_n) / sub_sp                 # SubQ's claim as compute = dense / speedup

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.6, 6.5))
fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(262144, 12e6, color="white", alpha=0.05)

# dense O(n²) — the rising "today's models" curve
ax.plot(n_m, dense_m, color="#e8806f", lw=2.7, marker="o", ms=4.5, label="dense O(n²) — measured")
ax.plot(xe, dense_law(xe), color="#e8806f", lw=2.0, ls=(0, (5, 3)))
ax.text(1.31e7, dense_law(12e6), "  dense O(n²)", color="#e8806f", fontsize=9, va="center")
# our flat-router kernel
ax.plot(n_m, ssa_m, color="#f2c14e", lw=2.6, marker="o", ms=4.5, label="our flat-router kernel — measured")
ax.plot(xe, flat_kernel(xe), color="#f2c14e", lw=1.9, ls=(0, (5, 3)))
# our IVF-router kernel — from the MEASURED faiss-GPU IVF router (router_gpu_compare) + attention floor
gr = json.load(open("paper/figures/router_gpu_compare.json"))
ivf_n = np.array([r["n"] for r in gr], float)
ivf_router_meas = np.array([r["ivf_ms"] for r in gr], float)
ivf_kernel_meas = floor(ivf_n) + ivf_router_meas              # attention + measured IVF router (maskbuild ~0)
ax.plot(ivf_n, ivf_kernel_meas, color="#3fbf90", lw=2.5, marker="s", ms=6, zorder=6,
        label="our IVF-router kernel — MEASURED router + attention")
pr = np.polyfit(np.log(ivf_n), np.log(ivf_router_meas), 1)   # extrapolate the measured router to 12M
xv = np.logspace(np.log10(ivf_n[-1]), np.log10(12e6), 30)
ax.plot(xv, floor(xv) + np.exp(pr[1]) * xv ** pr[0], color="#3fbf90", lw=1.9, ls=(0, (4, 3)))
# floor
ax.plot(xall, floor(xall), color="#9a9aff", lw=1.6, ls=(0, (1, 2)), label="n·κ floor (attention-only)")
# SubQ claim (as compute)
ax.scatter(sub_n[:2], sub_ms[:2], color="white", s=55, zorder=5, label="SubQ published (7.2×, 52.2×)")
ax.scatter(sub_n[2:], sub_ms[2:], marker="*", s=320, color="#ff5d5d", edgecolor="white", linewidth=0.6,
           zorder=6, label="SubQ claim 1,000×@12M")

ax.annotate("flat kernel: speedup capped\n(argsort BlockMask ~ n²·¹²)", xy=(12e6, flat_kernel(12e6)),
            xytext=(8e5, 9e3), color="#e8d6a0", fontsize=8, arrowprops=dict(arrowstyle="->", color="#e8d6a0", lw=0.8))
ax.annotate("IVF router (GPU-measured to 8M)\n→ kernel drops onto the floor", xy=(4194304, floor(4194304) + 31.4),
            xytext=(2.4e4, 1.7e3), color="#9fd9c4", fontsize=8, arrowprops=dict(arrowstyle="->", color="#9fd9c4", lw=0.8))

ax.set_xticks([4096, 16384, 65536, 262144, 1048576, 12e6]); ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "12M"])
ax.set_xlim(4096, 1.55e7); ax.set_ylim(0.15, 3e6)
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Attention compute — wall-clock per call (ms, log; lower = faster)", color="#bdbdbd")
ax.text(0.0, 1.15, "UNIFIED SCALING — measured (solid) + projection (dashed), one 16 GB GPU",
        transform=ax.transAxes, color="#8a8a8a", fontsize=8.6, fontweight="bold")
ax.set_title("Attention compute: dense O(n²), our kernels, the floor, and SubQ's claim", color="white",
             fontsize=13.5, loc="left", pad=34)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="upper left", fontsize=7.8, framealpha=0.12, labelcolor="#ddd")
fig.text(0.012, 0.012, "SubQ's claim plotted as compute = (our dense fit) / (its reported speedup); ratios "
         "across hardware. IVF-kernel line projects the GPU-measured IVF router (router_gpu_compare) into the "
         "full kernel; floor = the linear attention term; benign geometry (P1).",
         color="#6f6f6f", fontsize=6.3)
fig.tight_layout(rect=(0, 0.028, 1, 1))
out = "paper/figures/unified_scaling.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"@12M (ms): dense={dense_law(12e6):.0f} flat={flat_kernel(12e6):.0f} ivf={ivf_kernel(12e6):.0f} "
      f"floor={floor(12e6):.0f} ; SubQ-claim={sub_ms[2]:.0f}")


if __name__ == "__main__":
    pass
