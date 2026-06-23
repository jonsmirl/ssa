"""
Unified scaling chart (v2): speedup over dense vs context, telling the whole program in one frame.
  * dense O(n²)                  — the 1× baseline.
  * our flat-router kernel       — MEASURED (4K→262K) + projected; plateaus ~20× because the argsort
                                    BlockMask build (P0: ~n^2.12) grows nearly as fast as dense.
  * our IVF-router kernel        — projected (the faiss-GPU IVF router, GPU-measured to 8M, removes the
                                    maskbuild) — uncaps and climbs toward the floor.
  * routing-free floor           — dense / attention (the n·κ floor); the max speedup if routing were free.
  * SubQ                         — its two published speedups + the 1,000×@12M claim, for reference.
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

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.6, 6.5))
fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(262144, 12e6, color="white", alpha=0.05)

ax.axhline(1.0, color="#888", lw=1.2, ls=(0, (4, 3)), label="dense O(n²) — 1× baseline")
ax.plot(xall, dense_law(xall) / floor(xall), color="#9a9aff", lw=1.6, ls=(0, (1, 2)),
        label="routing-free floor (dense / attention)")
# flat-router kernel: measured solid + projected dashed
ax.plot(n_m, dense_m / ssa_m, color="#f2c14e", lw=2.6, marker="o", ms=4.5,
        label="our flat-router kernel — measured")
ax.plot(xe, dense_law(xe) / flat_kernel(xe), color="#f2c14e", lw=1.9, ls=(0, (5, 3)),
        label="flat kernel — projected (plateaus: maskbuild cap)")
# IVF-router kernel: projected
ax.plot(xe, dense_law(xe) / ivf_kernel(xe), color="#3fbf90", lw=2.4, ls=(0, (4, 3)),
        label="our IVF-router kernel — projected (GPU-validated)")
# SubQ
ax.scatter(sub_n[:2], sub_sp[:2], color="white", s=55, zorder=5, label="SubQ published (7.2×, 52.2×)")
ax.scatter(sub_n[2:], sub_sp[2:], marker="*", s=320, color="#ff5d5d", edgecolor="white", linewidth=0.6,
           zorder=6, label="SubQ claim 1,000×@12M")

ax.annotate("flat kernel plateaus ~20×\n(argsort BlockMask ~ n²·¹²)", xy=(12e6, dense_law(12e6) / flat_kernel(12e6)),
            xytext=(7e5, 2.7), color="#e8d6a0", fontsize=8, arrowprops=dict(arrowstyle="->", color="#e8d6a0", lw=0.8))
ax.annotate("IVF router removes the cap →\nclimbs toward the floor", xy=(4e6, dense_law(4e6) / ivf_kernel(4e6)),
            xytext=(2.4e5, 700), color="#9fd9c4", fontsize=8, arrowprops=dict(arrowstyle="->", color="#9fd9c4", lw=0.8))

ax.set_xticks([4096, 16384, 65536, 262144, 1048576, 12e6]); ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "12M"])
ax.set_xlim(4096, 1.4e7); ax.set_ylim(0.4, 4000)
ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1×", "10×", "100×", "1000×"])
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Prefill speedup over dense  (log)", color="#bdbdbd")
ax.text(0.0, 1.15, "UNIFIED SCALING — measured (solid) + projection (dashed), one 16 GB GPU",
        transform=ax.transAxes, color="#8a8a8a", fontsize=8.6, fontweight="bold")
ax.set_title("The maskbuild cap, the IVF router, and the floor", color="white", fontsize=14, loc="left", pad=34)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="upper left", fontsize=7.8, framealpha=0.12, labelcolor="#ddd")
fig.text(0.012, 0.012, "Floor is the routing-free max at this config's budget; the recall-viable budget (hence "
         "the achievable floor) is geometry-dependent (P1: 0.4% co-trained → 50% adversarial). IVF-kernel line "
         "projects the GPU-measured IVF router (router_gpu_compare) into the full kernel; benign geometry.",
         color="#6f6f6f", fontsize=6.3)
fig.tight_layout(rect=(0, 0.028, 1, 1))
out = "paper/figures/unified_scaling.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"@12M speedup: floor={dense_law(12e6)/floor(12e6):.0f}x flat={dense_law(12e6)/flat_kernel(12e6):.0f}x "
      f"ivf={dense_law(12e6)/ivf_kernel(12e6):.0f}x ; SubQ claim 1000x")


if __name__ == "__main__":
    pass
