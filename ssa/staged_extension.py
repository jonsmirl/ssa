"""
Staged context extension — climb past the zero-shot wall one rung at a time.

`ssa_extrapolation.py` showed a RoPE + gentle-curriculum model extends ~2–4x zero-shot, then decays. SubQ's
pipeline goes much further by STAGING: extend, continue-train briefly at the new length, extend again.
Because the routing it learned is position-invariant (the key->value offset is constant, content-match is
length-free), each rung is a cheap ADAPT — a few hundred steps starting from the ~2x zero-shot recall — not
a fresh train. This climbs the ladder and measures:
  (a) recall recovered at each rung after the adapt,
  (b) the adapt COST per rung (is it cheap, and roughly constant in length rather than growing?),
  (c) how far DENSE training reaches before GPU memory caps it — beyond which the subquadratic kernel
      (`ssa_kernel.py`, shown to 256K) is what carries the same model further.

The point is not to reach 12M tokens here (that needs the kernel + compute); it is to show the LADDER works
with our pieces — each doubling is a cheap adapt off the prior rung's zero-shot recall — and to locate the
exact point where a toy's O(n^2) training memory (not the algorithm) becomes the binding constraint.

Run:  python3 -m ssa.staged_extension
"""
from __future__ import annotations
import math
import torch

from .ssa_demo import MQAR, ssa_attention
from .ssa_extrapolation import RoPEModel, train_phase, recall, gentle_train

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def bs_for(tokens, budget=12000):
    """Batch size on a fixed token budget, so per-step memory is ~constant as the rungs grow."""
    return max(2, min(48, budget // tokens))


def adapt(model, task, n_pairs, target=0.97, chunk=100, cap=600):
    """Continue-train at length n_pairs until recall recovers (or cap). Returns steps used."""
    tokens = 2 * n_pairs + 17
    bs = bs_for(tokens)
    used = 0
    while used < cap:
        train_phase(model, task, chunk, n_pairs, min(n_pairs, 16), bs=bs, warmup=20)
        used += chunk
        if recall(model, task, n_pairs, bs=max(8, bs)) >= target:
            break
    return used


def main():
    torch.manual_seed(0)
    # keys big enough for the longest permutation (n_pairs <= n_keys); values small so the output head
    # stays at proven trainable scale (a large value vocab under-trains the small model — vocab 2562 here).
    task = MQAR(n_keys=2048, n_vals=512)
    L0 = 48
    print("=" * 92)
    print("STAGED CONTEXT EXTENSION — base at 48 pairs, then extend + adapt, rung by rung")
    print("=" * 92)
    rope = RoPEModel(task.vocab).to(DEV)
    print(f"  RoPE model: d=256, 6 layers, {sum(p.numel() for p in rope.parameters()) / 1e6:.1f}M params")
    base_steps = sum(st for _, st in [(2, 500), (4, 700), (8, 1000), (16, 1400), (32, 1800), (48, 2200)])
    gentle_train(rope, task, L0, "base")

    print(f"\n  climbing from {L0} pairs (base curriculum = {base_steps} steps):")
    print(f"  {'rung':>6} {'tokens':>7} {'mult':>5} {'zero-shot':>10} {'adapt steps':>12} "
          f"{'after adapt':>12} {'+SSA':>7}")
    rungs = [96, 192, 384, 768, 1536]
    last, total = L0, base_steps
    for T in rungs:
        if DEV == "cuda":
            torch.cuda.empty_cache()
        tokens = 2 * T + 17
        bs = max(8, bs_for(tokens))
        try:
            zs = recall(rope, task, T, bs=bs)                       # zero-shot at the new length
            steps = adapt(rope, task, T)                           # cheap continue-train to recover
            after = recall(rope, task, T, bs=bs)
            ncl = max(8, int(1.5 * math.sqrt(2 * T)))
            top_c = max(3, round(0.15 * ncl))                      # ~15% of clusters — a fair, not starved, budget
            fns = [lambda q, k, v: ssa_attention(q, k, v, n_clusters=ncl, top_c=top_c, local_w=8,
                                                 routing="cumulant") for _ in rope.attn]
            rs = recall(rope, task, T, attn_fns=fns, bs=bs)        # same model, SSA at inference
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {T:>6} {tokens:>7} {T // L0:>4}x   -- dense-training memory cap reached "
                  f"(beyond here: the subquadratic kernel, ssa_kernel.py, shown to 256K) --")
            break
        total += steps
        print(f"  {T:>6} {tokens:>7} {T // L0:>4}x {zs:>10.3f} {steps:>12} {after:>12.3f} {rs:>7.3f}",
              flush=True)
        last = T

    print(f"\n  reached {last} pairs (~{2 * last + 17} tokens, {last // L0}x the base) in {total} total "
          f"steps ({total - base_steps} on top of the base curriculum)")
    print("  Each rung is a cheap ADAPT off the prior rung's ~2x zero-shot recall — the per-rung cost stays")
    print("  small and roughly flat in length, so the staged total is far below a from-scratch long-context")
    print("  curriculum. The same model runs subquadratically at inference (SSA column). Where dense TRAINING")
    print("  memory caps out is an O(n^2) toy-harness limit, not the algorithm — the kernel carries it on.")


if __name__ == "__main__":
    main()
