"""
P8-D — a fold forces the state to GROW (`fold_not_hopfield` + `detectability_is_a_fold`).

A mid-stream distribution shift (the key distribution rotates from region A to an orthogonal region B) is
a fold: a FIXED-size memory cannot be length-robust across it. A bounded (forgetting-gated) fixed memory
fades the pre-shift facts as the post-shift burst arrives; a SLOT-BIRTH memory grows a new partition when
write-surprise crosses threshold, leaving the pre-shift slot untouched (disjoint support ⇒ exact
preservation, `forgetting_requires_shared_support`). We measure pre/post-shift recall and the state-size
trajectory as the post-shift burst lengthens.

Run:  python3 -m ssa.fastweight_shift            # -> paper/figures/fastweight_shift.json
"""
from __future__ import annotations
import json
import os
import torch
from ssa.fastweight import SlotMemory, _unit

D = 48


def _facts(direction, n, g, bias=0.6):
    """n facts whose keys are DISTINGUISHABLE within the region (a random component) but biased toward
    `direction` so the region is SEPARABLE from another (slot routing keys on the bias)."""
    K = _unit(bias * direction + _unit(torch.randn(n, D, generator=g)))
    V = _unit(torch.randn(n, D, generator=g))
    return K, V


def _recall(mem, K, V):
    if K.shape[0] == 0:
        return float("nan")
    ok = 0
    for i in range(K.shape[0]):
        o = _unit(mem.read(K[i]))
        ok += int((o @ _unit(V).T).argmax() == i)
    return ok / K.shape[0]


def run(n_post, trials=20):
    """Pre-shift facts (region A) then a post-shift burst of length n_post (region B). Returns mean
    pre/post recall and final slot count for a fixed (gated) memory vs a slot-birth memory."""
    agg = {("fixed", "pre"): 0., ("fixed", "post"): 0., ("birth", "pre"): 0., ("birth", "post"): 0.,
           ("fixed", "slots"): 0., ("birth", "slots"): 0.}
    for s in range(trials):
        g = torch.Generator().manual_seed(s)
        a = _unit(torch.randn(D, generator=g)); b = _unit(torch.randn(D, generator=g))
        Kpre, Vpre = _facts(a, 12, g)                                # the pre-shift facts to preserve
        Kpost, Vpost = _facts(b, n_post, g)
        mems = {"fixed": SlotMemory(D, rule="gated_delta", beta=1.0, decay=0.98, n_slots=1),
                "birth": SlotMemory(D, rule="gated_delta", beta=1.0, decay=0.98, n_slots=1,
                                    birth_threshold=1.1)}
        for name, mem in mems.items():
            for i in range(Kpre.shape[0]):
                mem.write(Kpre[i], Vpre[i])
            for i in range(Kpost.shape[0]):
                mem.write(Kpost[i], Vpost[i])
            agg[(name, "pre")] += _recall(mem, Kpre, Vpre)
            agg[(name, "post")] += _recall(mem, Kpost, Vpost)
            agg[(name, "slots")] += mem.n_slots()
    return {k: v / trials for k, v in agg.items()}


def main():
    out = "paper/figures/fastweight_shift.json"
    rows = []
    print("=" * 92)
    print("P8-D — a fold forces state growth: fixed (gated) memory forgets pre-shift; slot-birth preserves it")
    print("=" * 92)
    print(f"  {'n_post':>7} | {'fixed pre':>9} {'fixed post':>10} {'fixed slots':>11} |"
          f" {'birth pre':>9} {'birth post':>10} {'birth slots':>11}")
    for n_post in (8, 24, 48, 96, 192):
        r = run(n_post)
        rows.append({"n_post": n_post,
                     "fixed_pre": round(r[("fixed", "pre")], 3), "fixed_post": round(r[("fixed", "post")], 3),
                     "fixed_slots": round(r[("fixed", "slots")], 2),
                     "birth_pre": round(r[("birth", "pre")], 3), "birth_post": round(r[("birth", "post")], 3),
                     "birth_slots": round(r[("birth", "slots")], 2)})
        print(f"  {n_post:>7} | {r[('fixed','pre')]:>9.2f} {r[('fixed','post')]:>10.2f} "
              f"{r[('fixed','slots')]:>11.1f} | {r[('birth','pre')]:>9.2f} {r[('birth','post')]:>10.2f} "
              f"{r[('birth','slots')]:>11.1f}", flush=True)
    print("  -> as the post-shift burst grows, the FIXED memory's pre-shift recall decays toward 0 (a")
    print("     bounded state cannot hold both regimes); the SLOT-BIRTH memory grows a slot at the fold")
    print("     and keeps BOTH — length-robustness across a shift REQUIRES a growing state (fold_not_hopfield).")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"meta": {"d": D, "rule": "gated_delta", "decay": 0.98, "trials": 20, "birth_threshold": 1.1,
                        "note": "d=48 reference memories; recall = associative decode over that regime's values"},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
