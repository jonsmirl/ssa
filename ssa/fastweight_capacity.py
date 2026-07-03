"""
P8-B — capacity is set by the READ rule (P1) and the WRITE rule is coherence control (P2).

P1 (`softmax_capacity`, RetrievalMarginRecognition.lean): over the SAME stored pairs, a linear read
o = S q is rank-d capped (recall falls off at m ≈ d), while a softmax read over the same keys stays
near-perfect far past d (capacity ~ e^{β(1−ε)}). The read rule, not the substrate, sets the class.

P2 (`capacity_search_tension`, SearchTradeoff.lean): the delta write (erase-before-write) maintains the
key near-orthogonality that keeps capacity high; the additive write lets interference accumulate. Sweep
key coherence × write rule under the LINEAR read (which exposes the state's true capacity).

Recall = the standard associative-memory decode: predicted token = argmax_i ⟨read, value_i⟩.

Run:  python3 -m ssa.fastweight_capacity        # -> paper/figures/fastweight_capacity.json
"""
from __future__ import annotations
import json
import os
import torch
from ssa.fastweight import FastWeightMemory, random_keys, decode_recall, _unit


def _store(d, m, rule, coherence, seed):
    g = torch.Generator().manual_seed(seed)
    K = random_keys(m, d, coherence, g)
    V = _unit(torch.randn(m, d, generator=g))
    mem = FastWeightMemory(d, rule=rule, beta=1.0, keep_kv=True)
    for i in range(m):
        mem.write(K[i], V[i])
    return mem, K, V


def recall_at(d, m, rule, read, coherence=None, trials=8):
    accs = []
    for s in range(trials):
        mem, K, V = _store(d, m, rule, coherence, seed=s)
        reads = [mem.read_softmax(K[i]) if read == "softmax" else mem.read_linear(K[i]) for i in range(m)]
        accs.append(decode_recall(reads, V, list(range(m))))
    return sum(accs) / len(accs)


def main():
    out = "paper/figures/fastweight_capacity.json"
    rows = []
    print("=" * 90)
    print("P8-B — CAPACITY: the READ rule sets the class (P1); the WRITE rule is coherence control (P2)")
    print("=" * 90)

    print("\n[P1] recall vs #pairs m — linear read (rank-d wall) vs softmax read (exponential), random keys")
    print(f"  {'d':>4} {'read':>8} " + " ".join(f"m={m:<5}" for m in (8, 32, 64, 128, 256, 512)))
    for d in (32, 64, 128):
        for read in ("linear", "softmax"):
            cells = []
            for m in (8, 32, 64, 128, 256, 512):
                r = recall_at(d, m, "additive", read)
                cells.append(r)
                rows.append({"test": "P1", "d": d, "read": read, "rule": "additive", "m": m, "recall": r})
            print(f"  {d:>4} {read:>8} " + " ".join(f"{c:>6.2f}" for c in cells), flush=True)
    print("  -> linear read collapses near m≈d; softmax read over the same pairs holds far past it —")
    print("     capacity is a property of the READ, not the stored substrate (softmax_capacity).")

    print("\n[P2a] the delta rule is EXACT on orthogonal keys to m=d (d=64), where additive also holds —")
    print(f"  {'keys':>12} {'additive':>9} {'delta':>7} {'gated_delta':>12}")
    orow = {r: _orthogonal_recall(64, 64, r) for r in ("additive", "delta", "gated_delta")}
    for r, v in orow.items():
        rows.append({"test": "P2a", "d": 64, "m": 64, "keys": "orthogonal", "rule": r, "recall": v})
    print(f"  {'orthogonal':>12} {orow['additive']:>9.2f} {orow['delta']:>7.2f} {orow['gated_delta']:>12.2f}")

    print("\n[P2b] recall vs load m into the OVERLOADED regime — LINEAR read, d=64, coherence 0.1")
    print(f"  {'m':>6} {'additive':>9} {'delta':>7} {'gated_delta':>12}")
    for m in (32, 64, 128, 192, 256):
        cells = {r: recall_at(64, m, r, "linear", coherence=0.1) for r in ("additive", "delta", "gated_delta")}
        for r, v in cells.items():
            rows.append({"test": "P2b", "d": 64, "m": m, "coherence": 0.1, "rule": r, "recall": v})
        print(f"  {m:>6} {cells['additive']:>9.2f} {cells['delta']:>7.2f} {cells['gated_delta']:>12.2f}",
              flush=True)
    print("  -> the write rule matters most under OVERLOAD (m>d): the delta rule's erase-before-write bounds")
    print("     interference and degrades more gracefully; at m≤d with random keys additive is already fine.")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"meta": {"write_step": 1.0, "read_temp": 8.0, "trials": 8, "seed": "0..7",
                        "decode": "argmax over stored values",
                        "note": "d≤128 reference memories; recall is associative decode, not an LM"},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\n  wrote {out}")


def _orthogonal_recall(d, m, rule):
    """Exactly-orthogonal keys (identity basis, m=d) — the clean witness for the delta rule's exactness."""
    g = torch.Generator().manual_seed(0)
    K = torch.eye(d)[:m]
    V = _unit(torch.randn(m, d, generator=g))
    mem = FastWeightMemory(d, rule=rule, beta=1.0, keep_kv=False)
    for i in range(m):
        mem.write(K[i], V[i])
    reads = [mem.read_linear(K[i]) for i in range(m)]
    return decode_recall(reads, V, list(range(m)))


if __name__ == "__main__":
    main()
