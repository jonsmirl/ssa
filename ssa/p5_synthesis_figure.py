"""
P5 — synthesis figure: dense O(n²) vs the measured flat-router SSA kernel vs the projected IVF-router
kernel vs the n·κ floor. Measured solid (from kernel_speed_measured.json), projected dashed (from the P0
component fits in cost_profile.json + the P4 IVF cost ratios). Shows the program's result: the IVF router
moves the kernel off the flat n^1.3 curve down toward the linear floor at 12M.

Run: python3 -m ssa.p5_synthesis_figure  ->  paper/figures/p5_synthesis.png
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
fit = json.load(open("paper/figures/cost_profile.json"))["fit"]      # P0 power laws
pa = lambda key, x: fit[key][1] * x ** fit[key][0]

m = n_m >= 16384
pd_ = np.polyfit(np.log(n_m[m]), np.log(dense_m[m]), 1)
ps_ = np.polyfit(np.log(n_m[m]), np.log(ssa_m[m]), 1)
dense_law = lambda x: np.exp(pd_[1]) * x ** pd_[0]
ssa_law = lambda x: np.exp(ps_[1]) * x ** ps_[0]
block = 128
floor = lambda x: pa("attention", x)                                  # n·κ floor (linear)
flat_proj = lambda x: pa("attention", x) + pa("router", x) + pa("maskbuild", x)   # P0 component sum (captures maskbuild n^2.12)
ivf = lambda x: pa("attention", x) + pa("router", x) * (x / block) ** -0.5 + pa("maskbuild", x) * (x / block) ** -1.0

xe = np.logspace(np.log10(262144), np.log10(12e6), 120)
xall = np.logspace(np.log10(16384), np.log10(12e6), 160)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10.6, 6.4))
fig.patch.set_facecolor("#000000"); ax.set_facecolor("#000000")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(262144, 12e6, color="white", alpha=0.05)

ax.plot(n_m, dense_m, color="#e8806f", lw=2.5, marker="o", ms=4, label="dense O(n²) — measured")
ax.plot(xe, dense_law(xe), color="#e8806f", lw=1.8, ls=(0, (5, 3)))
ax.plot(n_m, ssa_m, color="#f2c14e", lw=2.5, marker="o", ms=4, label="flat-router SSA kernel — measured")
ax.plot(xe, flat_proj(xe), color="#f2c14e", lw=1.8, ls=(0, (5, 3)))
ax.plot(xall, ivf(xall), color="#3fbf90", lw=2.3, ls=(0, (4, 3)),
        label="IVF-router SSA kernel — projected (P0×P4)")
ax.plot(xall, floor(xall), color="#9a9aff", lw=1.6, ls=(0, (1, 2)), label="n·κ floor (attention-only)")

sp12 = dense_law(12e6) / ivf(12e6)
ax.annotate(f"@12M: flat kernel ~{dense_law(12e6)/flat_proj(12e6):.0f}× over dense (maskbuild-bound),\n"
            f"IVF router ~{sp12:.0f}× — at the floor ({ivf(12e6)/floor(12e6):.1f}×)",
            xy=(12e6, ivf(12e6)), xytext=(6e5, 7.0), color="#9fd9c4", fontsize=8.2,
            arrowprops=dict(arrowstyle="->", color="#9fd9c4", lw=0.9))

ax.set_xticks([16384, 65536, 262144, 1048576, 12e6]); ax.set_xticklabels(["16K", "64K", "256K", "1M", "12M"])
ax.set_xlim(16384, 1.4e7); ax.set_ylim(1, 3e6)
ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
ax.set_ylabel("Attention wall-clock per call (ms, log)", color="#bdbdbd")
ax.text(0.0, 1.10, "P0–P5 SYNTHESIS — measured (solid) + projection (dashed), one 16 GB GPU",
        transform=ax.transAxes, color="#8a8a8a", fontsize=9, fontweight="bold")
ax.set_title("Closing the gap to the floor: flat router → IVF router", color="white",
             fontsize=14, loc="left", pad=24)
ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
ax.legend(loc="upper left", fontsize=8.2, framealpha=0.12, labelcolor="#dddddd")
fig.text(0.012, 0.012, "IVF-router line is a cost projection (P0 component fits × P4 IVF ratios) on benign "
         "geometry; faiss-gpu wall-clock at 12M unmeasured on the 16 GB card. Floor = the linear attention term.",
         color="#6f6f6f", fontsize=6.5)
fig.tight_layout(rect=(0, 0.028, 1, 1))
out = "paper/figures/p5_synthesis.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"wrote {out}")
print(f"@12M: dense={dense_law(12e6):.0f}ms flat-SSA={flat_proj(12e6):.0f}ms IVF-SSA={ivf(12e6):.0f}ms "
      f"floor={floor(12e6):.0f}ms ; IVF gap to floor = {ivf(12e6)/floor(12e6):.2f}x")
