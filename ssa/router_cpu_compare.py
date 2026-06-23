"""
Same-device router comparison (both on CPU) — isolates the ALGORITHM from the GPU↔CPU device mismatch.
The earlier "faiss-cpu IVF is 75-333x slower" was unfair: the flat router ran on the GPU's blazing GEMM
while the IVF was dragged to CPU with transfers. On equal footing (both CPU), the flat GEMM is nb² and the
IVF (build + search) is ~linear, so the IVF wins from ~512K up and the gap grows — the op-count advantage
IS realized in wall-clock. Realizing it inside the GPU kernel just needs faiss-gpu (no transfer).

Run: python3 -m ssa.router_cpu_compare  ->  paper/figures/router_cpu_compare.{json,png}
"""
from __future__ import annotations
import json
import time
import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

faiss.omp_set_num_threads(faiss.omp_get_max_threads())
d, block, top_c = 64, 128, 8


def _t(fn, reps=3):
    fn()
    s = time.time()
    for _ in range(reps):
        fn()
    return (time.time() - s) / reps * 1000


def main():
    np.random.seed(0)
    rows = []
    print(f"{'n':>9} {'nb':>6} {'flat CPU GEMM':>14} {'IVF total':>10} {'winner':>10}")
    for n in (262144, 524288, 1048576, 2097152, 4194304):
        nb = n // block
        mu = np.random.randn(nb, d).astype("float32")
        qb = np.random.randn(nb, d).astype("float32")
        tf = _t(lambda: np.argpartition(qb @ mu.T, -top_c, axis=1)[:, -top_c:])
        nlist = max(4, int(nb ** 0.5))

        def build():
            ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            ix.train(mu); ix.add(mu); return ix
        tb = _t(build); ix = build(); ix.nprobe = 4
        ts = _t(lambda: ix.search(qb, top_c))
        tot = tb + ts
        rows.append(dict(n=n, nb=nb, flat_cpu=tf, ivf_build=tb, ivf_search=ts, ivf_total=tot))
        print(f"{n:>9} {nb:>6} {tf:>11.1f} ms {tot:>7.1f} ms {('IVF '+f'{tf/tot:.1f}x' if tot < tf else 'flat'):>10}")
        json.dump(rows, open("paper/figures/router_cpu_compare.json", "w"), indent=2)

    n = np.array([r["n"] for r in rows], float)
    flat = np.array([r["flat_cpu"] for r in rows])
    ivf = np.array([r["ivf_total"] for r in rows])
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(n, flat, color="#f2c14e", lw=2.4, marker="o", ms=5, label="flat router (n/b)² GEMM — CPU")
    ax.plot(n, ivf, color="#3fbf90", lw=2.4, marker="o", ms=5, label="IVF router (build+search) — CPU")
    for i in range(len(n)):
        if ivf[i] < flat[i]:
            ax.annotate(f"{flat[i]/ivf[i]:.1f}×", (n[i], ivf[i]), textcoords="offset points",
                        xytext=(0, -13), color="#9fd9c4", fontsize=8, ha="center")
    ax.set_xticks([262144, 524288, 1048576, 2097152, 4194304])
    ax.set_xticklabels(["256K", "512K", "1M", "2M", "4M"])
    ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
    ax.set_ylabel("Router wall-clock per call (ms, log)", color="#bdbdbd")
    ax.text(0.0, 1.07, "SAME-DEVICE (both CPU) — isolating the algorithm from the GPU↔CPU mismatch",
            transform=ax.transAxes, color="#8a8a8a", fontsize=8.4, fontweight="bold")
    ax.set_title("Flat (n/b)² GEMM vs IVF router — equal footing", color="white", fontsize=13.5, loc="left", pad=20)
    ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.12, labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig("paper/figures/router_cpu_compare.png", dpi=150, facecolor=fig.get_facecolor())
    print("wrote paper/figures/router_cpu_compare.{json,png}")


if __name__ == "__main__":
    main()
