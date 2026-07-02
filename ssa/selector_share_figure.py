"""
Selector share of the forward vs context length — the DSA-comparison figure. DSA's selector eats 58% of
prefill at 1M; this plots the selector's share for the flat router, the per-layer IVF kernel, and the CCC
(single-layer and L=24-amortized), against that reference.

Reads paper/figures/{cost_profile,ivf_kernel_e2e,ccc_kernel}.json. Run: python3 -m ssa.selector_share_figure
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _flat_share():
    cp = json.load(open("paper/figures/cost_profile.json"))
    fit = cp["fit"] if isinstance(cp, dict) and "fit" in cp else None
    if not fit:
        return None, None
    n = np.logspace(np.log10(2.6e5), np.log10(12e6), 60)
    pa = lambda k, x: fit[k][1] * x ** fit[k][0]
    sel = pa("router", n) + pa("maskbuild", n)
    return n, sel / (sel + pa("attention", n))


def _ivf_share():
    r = json.load(open("paper/figures/ivf_kernel_e2e.json"))["rows"]
    n = np.array([x["n"] for x in r], float)
    sel = np.array([x["router_ms"] + x["maskbuild_ms"] for x in r])
    tot = np.array([x["total_ms"] for x in r])
    return n, sel / tot


def _ccc():
    r = json.load(open("paper/figures/ccc_kernel.json"))["rows"]
    n = np.array([x["n"] for x in r], float)
    return n, np.array([x["selector_share"] for x in r]), np.array([x["amortized_share_L24"] for x in r])


def main():
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    ax.set_xscale("log")

    fn, fs = _flat_share()
    if fn is not None:
        ax.plot(fn, fs, color="#f2c14e", lw=2.0, ls=(0, (5, 3)), label="flat (n/b)² router — share (fit)")
    inn, ivs = _ivf_share()
    ax.plot(inn, ivs, color="#5aa9e6", lw=2.4, marker="o", ms=5, label="per-layer IVF kernel — measured")
    cn, cs, ca = _ccc()
    ax.plot(cn, cs, color="#3fbf90", lw=2.6, marker="s", ms=6, label="CCC selector — single-layer (measured)")
    ax.plot(cn, ca, color="#9fe0c4", lw=2.2, marker="D", ms=5, ls=(0, (4, 2)),
            label="CCC selector — amortized over L=24 (arithmetic)")

    ax.axhline(0.58, color="#ff5d5d", lw=1.6, ls=(0, (2, 2)))
    ax.text(3.0e5, 0.60, "DSA @1M = 0.58 (reported; different hw/model;\nshare of TOTAL prefill — see the Qwen leg)",
            color="#ff9d9d", fontsize=8, va="bottom")

    ax.set_xticks([262144, 1048576, 4194304, 12582912]); ax.set_xticklabels(["256K", "1M", "4M", "12M"])
    ax.set_xlim(2.4e5, 1.5e7); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
    ax.set_ylabel("Selector share of the forward", color="#bdbdbd")
    ax.text(0.0, 1.11, "SELECTOR COST — the CCC pays for certificates + streaming per layer, wins on amortization",
            transform=ax.transAxes, color="#8a8a8a", fontsize=8.4, fontweight="bold")
    ax.set_title("Selector share vs context — flat, IVF, CCC, and DSA's 58% reference", color="white",
                 fontsize=13, loc="left", pad=28)
    ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
    ax.legend(loc="center right", fontsize=8, framealpha=0.12, labelcolor="#ddd")
    fig.text(0.012, 0.012, "Single-head rig: the single-layer share uses an attention-only denominator (harsher than "
             "DSA's total-prefill denominator). The comparable model-level number is measured in the Qwen leg "
             "(longctx_share.py, route_ms/prefill_ms).", color="#6f6f6f", fontsize=6.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("paper/figures/selector_share.png", dpi=150, facecolor=fig.get_facecolor())
    print("wrote paper/figures/selector_share.png")


if __name__ == "__main__":
    main()
