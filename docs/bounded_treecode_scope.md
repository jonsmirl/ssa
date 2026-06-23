# Bounded-candidate treecode router — design scope

**Goal.** A hierarchical block router for `ssa_kernel` that beats the flat `(n/b)²` GEMM router *past ~2M
tokens* — where the flat router both saturates and (decisively) **OOMs** — without the 7× regression the
naive 2-level router hit at 2M (RESULTS § "treecode reality check", (b)).

**The one invariant that makes it work: a fixed beam width `W`.** The naive 2-level router regressed because
its candidate gather was `Θ(n^{1.5}·d)` — the per-query candidate set grew as `√n` (keep_coarse × group_size).
Bounding the frontier to a constant `W` at every level makes every intermediate tensor `(n/b) × (W·F) × d` —
**fixed width, linear in n** — so there is no gather blow-up and no `(n/b)²` matrix is ever materialized.

---

## Why this beats flat past 2M (the three reasons, in order of decisiveness)

1. **Memory — flat *cannot run*.** The flat router materializes the `(n/b)×(n/b)` block-score matrix:
   `nb=16384` (≈2M tokens, b=128) is a 0.5 GB/head fp16 matrix → ~4 GB across 8 heads; `nb=32768` (≈4M) is
   ~17 GB → **OOM on a 16 GB GPU**. The treecode's largest tensor is `(n/b)·W·F·d` (linear) — it runs at 4M,
   8M, 12M where flat is simply dead. *Past ~4M the treecode is not faster, it is the only option.*
2. **No 2M regression.** Fixed `W` ⇒ the gather is `Θ(n/b · W·F · d)` (linear), not `Θ(n^{1.5}·d)`. The
   naive router's blow-up is structurally removed.
3. **Cost — `O(n/b · log(n/b))` vs `(n/b)²`.** `L = log_F(n/b)` levels, each `W·F` bound-evals per
   query-block. At n=1M (nb=8192, F=8 → L≈4, W=8): ~275 evals/query-block vs 8192 for flat — ~30× fewer ops.
   The crossover with the flat GEMM's excellent constant stays near ~1M (per (b)); the point is it **keeps
   winning past 2M** instead of regressing.

---

## Design

### Tree (built once per forward, over key-blocks)
- Leaves = the `B = n/b` key-blocks; each carries summary `(μ_block, R_block, σ²_block, lo)` where `R` is the
  radius (max member distance to mean), `σ²` the diagonal variance (the cumulant term, (a)), `lo` the earliest
  token index in the block (for causality).
- Build `L` levels by grouping `F` children → one parent (contiguous, **positional**): parent mean = mean of
  children; **recursive radius** `R_parent = max_child(‖μ_child − μ_parent‖ + R_child)` (the bound the prune
  theorem needs, `SearchTradeoff.subtree_radius_bound`); parent `lo = min child lo`.
- Stored as flat tables: `children:(num_nodes,F)`, `mu:(num_nodes,d)`, `R/σ²/lo:(num_nodes,)`. `num_nodes =
  O(B)` → linear. Build = a handful of segment-reductions per level (geometric sum → O(B) total).
- **Positional grouping is deliberate**: causality becomes a range check (`node.lo ≤ q_end`), the build needs
  no permutation gather, and the admissible bound is geometric so it holds for *any* grouping. (Content
  clustering would give tighter radii / better pruning but costs a permutation + per-block causal masks — a
  quality variant, not v1; see Risks.)

### Beam descent (per query-block, fully batched, fixed width `W`)
```
frontier : (Q, W)            # Q = n/b query-blocks; start = top-level nodes (few), top-W by bound
for level in range(L):
    children = tree.children[frontier]          # (Q, W, F)      gather, fixed width
    mu, R, lo = tree.mu[children], tree.R[children], tree.lo[children]
    bound = einsum('qd,qwfd->qwf', qb, mu) + qnorm[:,None,None]*R   # admissible UB (+ ½·σ² cumulant term)
    bound = where(lo <= q_end[:,None,None], bound, -inf)            # causal mask (future nodes pruned)
    topW  = bound.reshape(Q, W*F).topk(W)                           # keep W; next frontier
    frontier = children.reshape(Q, W*F).gather(-1, topW.indices)
# frontier now holds ~W leaf blocks per query-block -> score, take top_c -> kv_idx for BlockMask
```
Every tensor is fixed-width `(Q, W·F, …)`. `L` batched `(gather → bmm → topk)` launches, no divergence — every
query-block does identical work. The score is the **admissible upper bound** (keeps the descent sound: a
high-relevance block's ancestors carry a high UB, so they survive the beam for sufficient `W`); the cumulant
`σ²` term is the (a) recall knob.

### Causality
Positional node spans `[lo, hi]`; visible to a query-block ending at `q_end` iff `lo ≤ q_end`. Mask future
nodes to `−∞` before each `topk`. Intra-block causality is handled downstream by FlexAttention's `mask_mod`.

### Decode (incremental, query_len = 1)
A single query descends the tree once — `O(L·W·F)`, ~constant. The tree is built over the cached keys and
extended as the KV cache grows (rebuild the affected leaf + its ancestors, or periodic full rebuild). Cheap
relative to the `(n/b)`-wide flat scan a decode step would otherwise pay.

### Integration
- New `hier_block_route(q,k, F, W, top_c, …) -> (kv_num, kv_idx)` feeding `BlockMask.from_kv_blocks`, drop-in
  beside `block_route`.
- **Hybrid dispatch**: `n < CROSSOVER` (~1M, from (b)) → flat GEMM (better constant); `n ≥ CROSSOVER` →
  treecode; `nb ≥ FLAT_OOM` → treecode forced. One `if` on `n`.

---

## Phased plan
| phase | deliverable | est. |
|---|---|---|
| P1 | tree build (positional F-ary, recursive radius + diagonal σ², `lo`) — pure tensor ops | ~1 d |
| P2 | batched fixed-`W` beam descent → top_c blocks (the core gather/bmm/topk loop) | ~2–3 d |
| P3 | causal masking + `BlockMask` integration + hybrid dispatch + decode path | ~1 d |
| P4 | **cost/memory benchmark** vs flat, 256K→8M: confirm no 2M regression + runs where flat OOMs | ~½ d |
| P5 | **quality**: NIAH recall of the treecode router vs flat at matched budget; tune `W`, `F`, cumulant | ~1–2 d |

~1–1.5 weeks focused. P1–P4 prove the *cost/memory* claim (the "beats flat past 2M" the question asks);
P5 is the separate *quality* axis.

## Open risks (honest)
- **Quality at small `W` is the real risk, not cost.** Positional coarse nodes span mixed content → large
  radii → loose bounds → the beam may not discriminate the needle's block (the (a) lossless-fails mechanism,
  amplified at coarse levels). Levers, in order: the **cumulant `σ²` term** (lifted approximate recall
  0.71→0.95 in (a)); larger `W` (more cost); **content clustering** of leaves (tight radii, but adds a
  permutation gather + per-block causal masks). P5 must measure this; a v1 that wins on cost but loses recall
  is not a win.
- **The constant may keep flat ahead below ~1M** — fine, that's what the hybrid dispatch is for.
- **`BlockMask` construction overhead** at huge `nb`, and decode-path tree maintenance, need their own timing.

## Success criteria
1. **Cost**: treecode router wall-clock ≤ flat for `n ≥` crossover, **monotone past 2M/4M** (no regression).
2. **Memory**: runs at 4M/8M where the flat GEMM OOMs.
3. **Quality**: NIAH single-needle recall within ε of the flat router at a matched key budget (the gate).

Only when all three hold does the measured kernel line (see `unified_scaling.png`) move from the flat
`n^{1.3}` curve toward the `O(n)` floor. The necessity that makes this worth building is machine-checked:
`flat_router_work` / `subquadratic_forces_skip` (Substrate Lean) — a flat router *cannot* be the long-context
answer.
