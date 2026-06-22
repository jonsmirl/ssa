# Retrieval-margin demonstrator — results

A controlled, NumPy-only validation of the content-addressable sparse-attention theory
(the retrieval-margin theory (paper §3)) and a direct test of the one
genuinely open piece: **can a sublinear content selector be assembled from known ANN parts, and
when does it work?**

> **On the *(proved)* claims below.** Results marked *(proved)* — and the theorem names cited inline
> (`samuelson_prune_gate`, `ellipsoidal_search_bound`, `temperedLogPartition_max_sandwich`,
> `lossless_selector_reads_every_key`, `hierarchical_prune`, `dropped_combination_error_bound`, …) — are
> machine-checked in the Substrate Lean 4 development, module
> `Substrate.Inference.PhaseTransition.Algebra.*` (axiom-pure, no `sorry`/`admit`). That formalization is
> maintained separately and is **not** included in this public repository; the theorem names are the
> pointers into it. The Python in this repo *measures* those results — it does not prove them.

Run: `python3 -m ssa.experiments` · Tests: `pytest ssa/tests/` (42 pass).
Method: synthetic keys/queries with *exactly controlled* gap, separation, dimension, and count — so
each prediction is tested against ground truth. No training; this validates the theory and the
selector mechanism, not a trained model.

## The read-side theory holds exactly

| Test | Prediction | Result |
|---|---|---|
| **E1** recovery weight | `p⋆ = 1/(1+(n−1)e^{−βΔ})` | matches measured target mass to **float precision** (err ≤ 1e−16) for n up to 1e5 |
| **E2** length generalization | margin grows only as `log n`; recall held far past threshold | recall **1.00** from n=1e3 to n=1e5 (100×); threshold `n* ~ 1e7–1e8 ≫ n` |
| **E3** truncation (sparse=dense) | `‖ô−o‖ ≤ 2V·(missed mass)`, exp-small in gap | bound holds on **every trial**; error decays with budget k |

These are confirmations of proven identities — they pass by construction, which is the point: the
read side is settled.

## The selector — the withheld piece — is buildable from known parts (under separation)

**E4(a): cost is sublinear, recall ≈ exact** (good separation, fixed budget k=64):

| n | exact (cost / recall) | centroid | SimHash-LSH |
|---|---|---|---|
| 4 000 | 4000 / 1.00 | 153 (3.8%) / 0.985 | 163 (4.1%) / **1.00** |
| 16 000 | 16000 / 1.00 | 272 (1.7%) / 0.930 | 253 (1.6%) / 0.995 |
| 64 000 | 64000 / 1.00 | 517 (0.8%) / 0.915 | 587 (0.9%) / **1.00** |

The **cost fraction falls as n grows** (3.8% → 1.7% → 0.8%) — genuinely sublinear. **SimHash-LSH** (a
2015-era technique) holds **~1.00 recall at <1% of keys scored** up to n=64 000.

**E4(b): the dependence on geometry (the doubling-dimension crux)** — centroid routing is lossless
only when basins are tight; **LSH stays near-lossless across all tested separations**:

| basin spread | centroid recall | LSH recall |
|---|---|---|
| 0.08 (tight) | 1.00 | 1.00 |
| 0.15 | 0.94 | 1.00 |
| 0.30 | 0.87 | 1.00 |
| 0.60 (diffuse) | 0.89 | 1.00 |

**Reading:** for single-needle retrieval with a clean cue, a known LSH index reproduces near-perfect
recall at ~1–2% cost, sublinearly, with no new mechanism. Centroid/routing selection is more
sensitive to whether the keys cluster well — exactly the "benign geometry" (low doubling dimension)
that training is incentivized to produce (`softmax_capacity`: capacity is exponential in separation).

## The honest boundary — recall generalizes, reasoning does not (E5)

With an **imperfect cue** (a realistic pointer is noisy, not the exact key), single-hop recall is
`ρ = 0.91`, and an h-hop chain succeeds `≈ ρ^h`:

| hops h | chain success | `ρ^h` |
|---|---|---|
| 1 | 0.92 | 0.91 |
| 3 | 0.71 | 0.75 |
| 6 | 0.60 | 0.57 |

Chain success tracks `ρ^h`. A real pointer chain (errors propagate) is no better. So the same
machinery that wins at recall **decays geometrically at composition** — efficient retrieval is not
long-horizon reasoning.

## Verdict — what this establishes, and what it does not

- **Establishes:** the read-side theory is exact; a sublinear selector built from *known* ANN parts
  (SimHash-LSH; centroid routing) captures the single-needle target **losslessly at ~1% of keys
  scored, sublinearly, to n=64 000** — *when keys are separated*. For the single-needle regime, the
  piece SubQ withheld is reproducible from public components. LSH is the stronger of the two here.
- **Does not establish:** that a *trained* model at 12 M tokens actually has separable key geometry
  (the empirical crux — no theorem gives it; it must be measured on a real model), nor anything about
  the *hard* tasks: cue noise lowers the recall ceiling for **every** selector (E5), and composition
  decays as `ρ^h` regardless of how good the selector is.
- **Bottom line:** the selector is buildable from known parts under the geometry training is driven
  toward; the wins are efficiency + single-needle recall; the limits — noisy-cue recall and multi-hop
  reasoning — are set by the task, not the selector, and no selector closes them.

## The trained-keys crux (experiment (i)) — `train.py`

The synthetic harness above couldn't settle the real question: does *training* produce keys a
sublinear selector can use? We trained a small encoder (`d_raw=64 → d=128` MLP) on a content-
addressable retrieval task (retrieve the right item from a **corrupted** cue: 40% of features masked
+ noise), then probed its keys on **held-out items at larger N than training** — vs an untrained
encoder. The answer is a sharp **"yes, but not the way the bolt-on story assumed."**

| encoder / cue | key coherence ε | exact dense acc | LSH sel-recall | LSH end-to-end | cost |
|---|---|---|---|---|---|
| untrained, corrupted (N=10k) | 0.99 (collapsed) | 0.41 | 0.84 *(but scans ~100% of n — no sparsity)* | 0.41 | ~100% |
| **trained, corrupted (N=10k)** | 0.57 (separated) | 0.72 | **0.19** | 0.18 | 2% |
| **trained, near-clean (N=10k)** | 0.57 | 1.00 | **1.00** | **1.00** | 2% |

Findings:

1. **Training *does* separate keys** — coherence collapses from 0.99 (untrained MLP maps everything
   to nearly one direction) to ~0.57, and exact retrieval rises from ~0.4 to 0.72 under corruption
   (1.00 near-clean). The "training drives separability" half of the theory is **confirmed**, and it
   generalizes to unseen items at 6× the training pool size.
2. **But post-hoc sublinear selection is *not* automatically lossless.** On the trained keys with a
   realistic **corrupted** cue, LSH recall is only 0.19 — the selector misses. The control (same
   model, same keys, **near-clean** cue) is lossless: LSH recall 1.00 at 2% cost. So the breakage is
   driven by **cue quality, not training**: a trained retriever gets the answer right via a learned
   *margin*, but a corrupted query is *not an angular near-neighbor* of its key — and angular LSH
   needs proximity, not margin. (This is exactly the E5 imperfect-cue regime, now seen end-to-end.)
3. **Therefore the selector must be *co-trained*, not bolted on.** Known ANN bolted onto a standard
   retrieval-trained model is lossless only in the near-clean-cue regime (which reproduces E4). In the
   realistic regime it fails; making it lossless requires putting the selector *in the training loss*
   so the model learns query/key geometry the selector can navigate (separate `W_q`/`W_k` aligned for
   the cue, or a jointly-trained router) — which is what NSA/SubQ-style systems actually do.

**This sharpens the earlier verdict.** "The withheld piece is buildable from known parts" holds only
with the qualifier: *with near-clean cues, or with a co-trained selector.* Bolting generic LSH onto a
plain retriever does **not** inherit losslessness under realistic cues. The plausible real content of
SubQ's withheld selector is the **co-training** — harder, and not reducible to a known index dropped
in after the fact.

## Closing the arc (experiment (ii)) — `co_train.py`

(ii) asks whether *co-training the selector* recovers what post-hoc selection lost. Three settings,
all on held-out items, same corrupted cue (40% masked), sublinear selection:

| setting | sel-recall | end-to-end acc | cost (frac of n) |
|---|---|---|---|
| post-hoc LSH on retrieval-only encoder (= (i)) | 0.18 | 0.16 | 2% |
| representation proximity term + same post-hoc LSH | 0.21 | 0.20 | 2% |
| **co-trained router (no load balance)** | **1.00** | **0.71** (= dense ceiling) | **~100%** (degenerate) |
| **co-trained router (balanced, B=64, r=4)** | 0.40 | **0.36** | **9%** |

Dense (exact) ceiling on this task ≈ 0.74. Findings:

1. **A representation proximity term barely helps (+0.04).** Pulling the cue toward its key in angle
   can't manufacture proximity when 40% of *random* features are destroyed — the cue lacks the
   information to be an angular near-neighbor. (Robustness ≠ recoverable-to-proximity.)
2. **A co-trained router recovers losslessness — but only by giving up sparsity.** With no load
   balance it reaches the dense ceiling (recall 1.00) because it routes to clusters that contain
   essentially everything (cost ~100%). The information *is* present; the difficulty is doing it
   *sublinearly*.
3. **Balanced (sublinear) co-trained routing beats post-hoc ~2.2× (0.36 vs 0.16 at 9% cost) but does
   not reach the ceiling.** There is a real **balance-vs-routing tension**: forcing uniform clusters
   (for sublinearity) makes partitions a *mean-centroid* summary can't route a corrupted cue to —
   the coarse summary washes out the cue's fine discriminative signal. This is exactly the cumulant
   point from the theory: a node summary needs **mean *and* spread**, not the mean alone.

**Verdict (the honest arc).** Post-hoc selection on a plain retriever fails under realistic cues;
co-training the selector is **necessary and helps substantially (~2.2×)**; but getting **both
lossless and aggressively sublinear** is genuinely hard with a simple co-trained mean-centroid router
— it needs richer summaries / learned hashes that preserve discriminative directions. So the withheld
selector is **real co-designed engineering, not a bolt-on index** — which is precisely why a working
selector at 12M tokens would be a genuine result, and why the lazy "known ANN dropped in" story does
not hold up when you actually build it.

## The theory's prescription, tested (experiment (1)) — `co_train.py` section [D]

(ii) diagnosed the remaining gap as a *mean-centroid summary losing the cue's fine signal*, and the
theory was explicit about the fix: a cluster's routing score should be the Laplace approximation
of its **log-partition**, the first **two cumulants** —
`s_b = log|b| + β·q·μ_b + ½ β² qᵀ Σ_b q` — where `μ_b = ∇Φ` (mean) and `Σ_b = ∇²Φ` (Fisher /
covariance). Mean-only routing keeps just `q·μ_b`; it asks "which cluster's *average* is closest."
The spread term turns it into "which cluster's *log-sum-exp* (≈ best key) is highest" — the right
question. We tested it on the *same* trained router, the *same* clusters, at *matched* cost (only the
score changes):

| routing score | end-to-end acc, r=2 (5.6%) | r=4 (10.5%) | r=8 (20.2%) |
|---|---|---|---|
| mean only (1 cumulant) | 0.261 | 0.389 | 0.521 |
| **mean + spread (2 cumulants)** | **0.334** | **0.468** | **0.582** |
| spread-term gain | **+0.073** | **+0.079** | **+0.061** |

The second-cumulant (spread) term the theory prescribes raises end-to-end accuracy by **+0.06 to
+0.08 at every operating point, at zero extra cost**, robust across seeds (~3–4σ). It narrows the gap
to the dense ceiling but does not fully close it (two cumulants ≠ the full log-sum-exp; the rest is
the genuine hard part). **The cumulant geometry the theory built — `softmax = ∇Φ`,
`Hessian = Fisher = covariance` — is load-bearing, not decorative:** it correctly predicted both the
deficiency of mean-only routing *and* the exact term that fixes it, and the fix measurably works.

This is the cleanest validation in the whole investigation: a theory made a
falsifiable engineering prediction (route by mean + spread), and a trained model confirmed it.

## The real-key test — does SubQ's bet pay off on real representations? — `real_keys.py`

Every experiment above used **worst-case synthetic keys** (uniform on the sphere, or a retrieval
encoder driven to spread its keys), so the negative results could not refute SubQ: real trained
representations might have the *benign* geometry (clumped onto a low-dimensional manifold) that turns
the admissible bound's pruning on. This is the regime SubQ actually lives in, and DeepSeek's DSA could
afford to ignore (it ate a quadratic indexer up to ~52K tokens; SubQ at 12M cannot). So we ran the
faithful test: extract a **real pretrained transformer's actual attention keys and queries** (GPT-2 on
real wikitext-2 text), characterize their geometry, and run the exact admissible branch-and-bound
selector (`SearchTradeoff.cluster_prunable`) on them — head-to-head against synthetic random keys at
matched n / d / cue-margin.

**Finding 1 — real keys ARE benign in the clumping sense.** Coherence 0.37 (vs 0.125 for random
unit vectors in d=64), effective dimension ≈ 8 of 64 (the keys live on a low-dimensional manifold).
The benign geometry SubQ's bet requires is genuinely present — real data is *not* the synthetic
worst case.

**Finding 2 — but clumping ≠ cheap LOSSLESS selection, and the split is head-structured.** Exact
admissible B&B (lossless top-1) prunes well on **sharp** heads (low effective dim — ~49% of keys
scored) but barely at all on **diffuse** heads (high effective dim — ~99%, i.e. no pruning). The
diffuse heads are the **deep, long-range** heads (layers 5/8/11) — exactly the content-retrieval heads
that matter at long context. This is `cluster_radius_floor` / `capacity_search_tension` realized on
real data: the heads with the most retrieval capacity (most key spread/separation) are precisely the
ones whose cluster radii are too large for the admissible bound to prune. The trilemma bites where it
counts.

**Finding 3 — SubQ doesn't need lossless, and the APPROXIMATE regime DOES pay off.** SubQ does
approximate top-k, not exact top-1. At a fixed *sublinear budget* on a diffuse deep head (layer 11,
head 0 — where lossless B&B failed):

| budget | recall (real keys) | recall (matched random) |
|---|---|---|
| 2%  | 0.67 | 0.36 |
| 5%  | 0.81 | 0.48 |
| 10% | 0.90 | 0.64 |
| 20% | 0.94 | 0.77 |

Real clumping buys high top-k recall at a small budget — ~1.7× the random baseline at 5%, recovering
90% of the dense argmax at a 10% budget — even on the head where exact lossless pruning was impossible.

**Verdict (calibrated).** SubQ's bet — that real-data geometry enables a subquadratic selector —
**holds, but specifically for approximate top-k selection, not exact/lossless top-1.** Real
representations are benign enough that a fixed-budget routed/clustered selector recovers most of the
dense argmax cheaply. The trilemma is **not** refuted: it still forbids cheap *lossless* selection on
the diffuse long-range heads, so a SubQ-style system must (a) accept approximation (bounded recall
loss) and/or (b) co-adapt keys to be more clumpable (clue #2 — which trades capacity, exactly as
`capacity_search_tension` and the synthetic clump-sweep showed). This pins the design space (a learned,
co-adapted, approximate top-k selector) **and** predicts the failure mode: **recall degradation on
diffuse / high-entropy / adversarial retrieval** — the heads where keys are most spread. That is a
falsifiable prediction about where SubQ-style long-context models lose accuracy.

*Caveat:* GPT-2's context caps n at 1024; the clumping/approximate-recall conclusions are robust at
that scale, but the cleanest *sublinear-scaling-in-n* claim would want a long-context model
(Qwen2.5-0.5B, 32K) — a tracked follow-up, not a change to the verdict.

## The long-context / cross-architecture check — `longctx_keys.py`, `longctx_probe.py`

The GPT-2 test capped n at 1024 with learned positional embeddings. We repeated the measurements on the
architecture SubQ-style models actually use — **RoPE + grouped-query attention** — on TinyLlama-1.1B
(2K) and **Qwen2.5-0.5B (16K context)**, capturing post-RoPE keys (exactly what attention scores).

**The trilemma deepens with effective dimension.** Real keys stay clumped by coherence at every scale,
but RoPE spreads them across more dimensions as depth/length grow: effective dim **8/64 (GPT-2) →
17.6 (TinyLlama) → 30–39 (Qwen)**. Lossless exact branch-and-bound (the admissible bound) therefore
prunes *less* as we go: GPT-2 ~49% on sharp heads, TinyLlama ~100%, Qwen ~101% (= synthetic). **Cheap
lossless selection is forbidden, and more firmly the longer/deeper the model** — `cluster_radius_floor`
/ `capacity_search_tension` realized on real keys.

**The decisive result — approximate long-range retrieval needs SECOND-ORDER routing.** On Qwen at 16K
with genuinely long-range targets (median dense-target distance **10,514**; 95% of targets ≥ n/4 back),
we routed by three cluster scores at a fixed budget (B scaled to ~64 keys/cluster, internal control —
same keys, same clusters, only the score changes):

| cluster score (deep head 18.0, long-range, 5% budget) | recall |
|---|---|
| admissible upper bound `μ·q + ‖q‖R_b` (right for lossless) | 0.00 |
| centroid relevance `μ·q` (Routing-Transformer style) | 0.01 |
| **second cumulant `μ·q + ½ qᵀΣ_b q`** (Laplace / log-sum-exp; `softmax=∇Φ`, `Hessian=Fisher=covariance`) | **0.88** |

On the deepest long-range head the obvious centroid router gets **1%**; the second-cumulant score gets
**88%** (0.96 at a 10% budget). The target is an *in-cluster outlier on a high-variance axis the mean
cannot see* — only the spread (multipole second moment) term surfaces it. This is the strongest
confirmation in the whole investigation that the second-order geometry (`Hessian = Fisher =
covariance`, the cumulant/log-sum-exp routing) is **load-bearing for long-range retrieval**, now at 16K.

**But it is head-dependent — and that is the point.** No single fixed cheap score wins on all heads
(centroid wins head 11.0 at 0.48; the cumulant wins the deep heads 18.x; both fail head 11.1). The
optimal selection function varies by head and content. **A fixed clustering index cannot be cheap and
accurate everywhere — the router must be LEARNED and co-adapted.** This is exactly why SubQ foregrounds
its three-stage training and the RL-for-global-attention stage, and why the "nominal vs functional
context" framing is theirs: vanilla keys *are* routable cheaply at long range, but only with a
content-/head-adaptive second-order selector that a vanilla model does not supply off the shelf.

**Net, across GPT-2 / TinyLlama / Qwen.** (1) Lossless cheap selection is impossible (trilemma, worse
with effective dim). (2) Cheap *approximate* long-range retrieval is geometrically available on real
keys — confirming SubQ's `O(n·k)` bet — **but requires second-order (cumulant/multipole), head-adaptive
routing**, not centroid routing. (3) Because the best score is head- and content-dependent, the selector
must be learned and co-adapted — which pins SubQ's contribution to the *training that manufactures
routability*, not the selection architecture alone, and predicts the concrete failure mode of any
pure-centroid sparse-attention scheme (collapse on the deepest long-range heads, as measured: 0.01).

## The end-to-end reproduction — `ssa_demo.py`

A self-contained program that reproduces the SSA *mechanism* and its core qualitative claims on the
canonical long-context retrieval task (multi-query associative recall, MQAR — what SSA is built for).
A 1.1M-param transformer (d=128, 4 layers, 4 heads) is trained with dense attention on MQAR (64
key→value pairs) under a length curriculum 16→32→48→64; the induction circuit "clicks" mid-curriculum
and dense recall reaches **1.000**. Then dense attention is *swapped for SSA at inference* — per query,
exact attention over a local window plus the keys of the top-`k` clusters ranked by the cumulant score
`⟨q,μ_b⟩ + ½ qᵀΣ_b q` (the second-cumulant routing result object). Results:

| demonstration | result |
|---|---|
| **[2] SSA recovers dense recall** at a fraction of keys | top_c=2 → **0.987** at 25% attended; top_c=3 → **0.997** at 33% |
| **[3] routing score** (tight budget top_c=1) | cumulant **0.882** vs centroid **0.845** |
| **[4] training manufactures routability** (mass kept ÷ coverage) | trained **1.08×** (concentrates the attended key) vs untrained **0.82×** (≈ random) |
| **[5] functional vs nominal context** (needle at distance D) | local-window-only: 0.99 at D=4, **0.00–0.02 at D≥16**; SSA: **0.98–1.00 at every D** |
| **[6] O(n·k) scaling** (clusters ∝ √n, fixed top_c=2) | attended fraction **23.6% → 1.7%** as n grows 128→8192; dense/SSA FLOP ratio **2.1× → 29.7×**, growing ~√n |

This is the SSA result reproduced end to end: a subquadratic sparse-attention layer recovers dense
long-range retrieval at `O(n·k)` cost, the speedup *grows* with context (mirroring SSA's reported
6.88×@128K → 56×@1M), and it works for exactly the two reasons the experiments identified —
**second-cumulant routing** (centroid is worse) and **training that makes keys routable** (the untrained
model is not). Far-needle recall (functional context) requires the global selection; a local window
alone fails — SSA's headline distinction. NOT reproduced: the proprietary 12M checkpoint, the absolute
benchmark numbers, or their exact learned selector — this is the mechanism at small scale, honestly bounded.

## SSA-Small — an example ~12M-parameter checkpoint — `ssa_checkpoint.py`

Scaling the reproduction to a real, saved artifact: a **12.6M-parameter** transformer trained on
long-context associative recall (MQAR, up to 128 key→value pairs / ~273 tokens) that runs as
subquadratic sparse attention. The wide model would not form the binding circuit when started at 16
pairs (it fell into a frequency shortcut — loss fell but recall stayed at chance); a **gentle curriculum
from trivial sizes** (2→4→8→16→…→128 pairs) groks every length cleanly to 1.000. A final **SSA-in-the-
loop fine-tune** co-adapts the keys to the sparse selection (SubQ's co-adaptation stage; SSA recall
1.000 after it). Saved to `ssa_small_12m.pt` (gitignored).

| demonstration | result |
|---|---|
| **[A] SSA recovers dense recall** | top_c=2 → **0.995**, top_c=3 → 0.999, top_c=4 → 1.000 (dense 1.000) |
| **[B] routing score** (top_c=2) | cumulant 0.995 ≥ centroid 0.994 — both easy here (co-adapted keys are highly routable; the cumulant's decisive edge showed in the *hard* Qwen-16K long-range regime, 0.88 vs 0.01) |
| **[C] functional context** (needle at distance D) | local-window-only: 0.92 at D=4, **0.00–0.02 at D≥16**; SSA: **1.000 at every D up to 248** |
| **[D] O(n·k) scaling** | attended fraction **17.8% → 1.7%** as n grows 256→16384; dense/SSA FLOP ratio **2.8× → 29.9×** |

So SSA-Small is a genuine ~12M-parameter checkpoint with SubQ's characteristics: it runs as
subquadratic sparse attention, recovers dense long-range recall at `O(n·k)`, retrieves from any distance
(a local window alone fails), and depends on SSA-co-adapted keys. **Honest scope:** ~12M *parameters*
(the SubQ-Small scale), **not** a 12M-*token*-context model (orders more compute); it uses learned
positional embeddings, so true beyond-training-length extrapolation (the literal 12M-token flavor) needs
RoPE/ALiBi + a length curriculum — the documented next step, not reproduced. The absolute benchmark
numbers and proprietary selector are likewise not reproduced.

## A genuine subquadratic kernel — measured wall-clock — `ssa_kernel.py`

The reproduction above counted the `O(n·k)` cost analytically (it computed the full score matrix and
masked). This is the real kernel: a fused block-sparse attention (PyTorch FlexAttention) that never
materializes the n×n scores — per query-block it routes to the top-k key-blocks by the cumulant score
and runs a fused kernel over only the selected blocks. Measured against dense FlashAttention (`sdpa`),
H=8, d=64, fp16, fixed budget (top_c=8 blocks + local):

| context n | dense | SSA | **speedup** | attended frac |
|---|---|---|---|---|
| 4,096 | 0.39 ms | 0.81 ms | 0.5× | 48.5% |
| 8,192 | 1.07 | 0.84 | 1.3× | 27.1% |
| 16,384 | 3.89 | 1.12 | 3.5× | 14.5% |
| 32,768 | 12.8 | 1.91 | 6.7× | 7.5% |
| 65,536 | 48.1 | 4.68 | 10.3× | 3.8% |
| 131,072 | 188 | 12.0 | 15.7× | 1.9% |
| 262,144 | 750 | 36.5 | **20.6×** | 1.0% |

The speedup is **measured, not asserted**, and grows monotonically with context (the shape of SubQ's
own curve; crossover ~8K — the small-n slowdown is exactly SubQ's admitted short-context overhead). A
faithfulness check (does routing find a planted relevant region?) confirms it isn't vacuous: block-hit
rate 78% at a tiny top_c=4 budget, declining at fixed budget over half-context distance (the budget must
scale modestly, or use a coarse→fine hierarchy). *Honest scope:* block-granularity routing (NSA-style —
the block-score matrix is `O((n/128)²)`, cheap but not asymptotically subquadratic; a hierarchical router
removes it, negligible in this range where the `O(n·k)` attention dominates).

## Genuine length extrapolation — train short, retrieve long — `ssa_extrapolation.py`

SSA-Small used learned positional embeddings, so its functional context was capped at its trained length.
A **RoPE** model trained (with the gentle curriculum) on 48 pairs retrieves at much longer contexts it
never saw, while a learned-positional twin collapses:

| context | mult | RoPE dense | learned-pos | RoPE + SSA |
|---|---|---|---|---|
| 48 pairs | 1× | 1.000 | 1.000 | 1.000 |
| 96 | 2× | **0.987** | 0.470 | **0.988** |
| 192 | 4× | **0.756** | 0.238 | 0.702 |
| 384 | 8× | 0.270 | 0.113 | 0.338 |
| 768 | 16× | 0.067 | 0.064 | 0.126 |

RoPE + gentle curriculum learns **position-invariant content routing** (the key→value offset is a
constant +1, content matching is position-free), so it extrapolates to ~2–4× training length with strong
recall (0.99 at 2×, 0.76 at 4×) — decisively beating learned-pos (0.47, 0.24), which fails the instant it
passes its trained length (untrained position embeddings). **SSA preserves the extrapolated retrieval at
`O(n·k)`** — even slightly better at extreme length (selection filters distractors). *Honest ceiling:*
RoPE alone gives a few× of headroom; recall decays toward chance beyond ~4×. Reaching much longer (the
literal 12M-token regime) needs position-interpolation/NTK + training at longer lengths — the documented
next step, on top of a mechanism now demonstrated to extrapolate.

## Hidden gains from the math — the routing temperature + value-aware selection

Pushing on the geometry surfaced two *forced* gains, each now proved and measured.

### (1) The routing temperature — the tempered-routing result + `tempered_routing.py`

The centroid / cumulant / exact-best-key spectrum is **one knob**: the tempered (escort) cluster score
`Φ_b(β) = β⁻¹ log Σ_{j∈b} e^{β⟨q,k_j⟩}` is the centroid at β→0, the cumulant score at β=1, and the
cluster's exact maximum at β→∞ — the q-deformation. The keystone result
`temperedLogPartition_max_sandwich` proves `max ≤ Φ_b(β) ≤ max + (log n)/β`: the best-key bias is
`(log n)/β`, which **shrinks as β grows** (`temperedLogPartition_slack_antitone`). So routing at β>1
provably tightens the β=1 cumulant estimate. Measured on the Qwen-16K deep head (long-range, 5% budget):

| routing | recall |
|---|---|
| centroid (β→0) | 0.013 |
| cumulant — cheap 2nd-order, β=1 | 0.903 |
| **cheap 2nd-order, β≈2** | **0.941** |
| exact tempered, any β≥0.5 | 1.000 |
| oracle (β→∞) | 1.000 |

The deployable gain: route the cheap moment-based score at **β≈2, not β=1** (+~4pp, free — just a
constant), plateauing where the 2-cumulant truncation saturates (a 3rd-cumulant term would close more).
The *exact* tempered score is near-oracle at any β (it reads the true best member through the smooth-max)
— confirming the family genuinely reaches exact retrieval, at the cost of a full member scan.

### (2) Value-aware selection — `the theory (see paper)`

A hint from statistical leverage / rate–distortion: the attended output is a convex combination
`o = Σ w_i v_i`, and `dropped_combination_error_bound` proves that dropping a set `D` and renormalizing
incurs error `≤ (1−W_D)⁻¹ Σ_{i∈D} w_i ‖v_i − o‖` — controlled by weight **times value-deviation**, not
weight alone. So a key whose *value* already equals the output is free to drop even at large weight,
while a distinctive-value key matters even at small weight. **Value-aware selection** (drop by smallest
`w_i‖v_i − o‖`) controls the true output error where score-only selection (drop by smallest `w_i`) does
not — a tighter sparsity at equal fidelity, and a routing signal no score-only scheme uses.

Both results are proved. They are
genuine improvements to the selection math — a tunable temperature with a proven shrinking bias, and a
value-aware drop criterion with a proven error bound — not just engineering knobs.

## Frontier-scale confirmation — Gemma-4-26B (head_dim 256) — `gemma_keys.py`

The routing findings were validated on GPT-2, TinyLlama, and Qwen — all **head_dim 64**. Gemma-4-26B-A4B
is a sharply different point: **26B parameters** (MoE, 4B active), RoPE + GQA, and **head_dim 256** (4×).
We extracted its real post-RoPE attention keys (text path, loaded as `Gemma4ForConditionalGeneration`
via CPU offload — 47 GB weights in 78 GB RAM) and re-ran the core results at n=2048, long-range, 5% budget.

Both proven gains survive — and get **stronger** as the geometry gets more diffuse:

| head | eff-dim | centroid | cumulant |
|---|---|---|---|
| layer 22 | ~37–41 / 256 | 0.77–0.84 | **0.98–1.00** |
| layer 29 (deepest) | **85** / 256 | **0.02–0.26** | **0.21–0.58** |

The deepest layer (eff-dim 85 — the most diffuse geometry measured anywhere) is exactly where centroid
routing **collapses** and the cumulant **rescues** it — the Qwen failure mode, more extreme. Temperature
sweep on the hardest head (29.3):

| β | recall |
|---|---|
| centroid (β→0) | 0.095 |
| 1.0 (cumulant) | 0.583 |
| **2.0** | **0.704** |
| 4.0 / 8.0 | 0.688 / 0.658 |
| oracle (β→∞) | 1.000 |

The **β≈2 optimum holds and is sharper** here (+12 pp over β=1, vs +4 pp on Qwen), with the predicted
decline beyond. The larger gap to the oracle is the higher-cumulant content the 2-moment truncation can't
capture at eff-dim 85.

**Takeaway:** both gains survive a 4× jump in head dimension and ~50× in scale (0.5B → 26B), and matter
*more* as the geometry gets more diffuse — exactly as `CumulantRouting`/`TemperedRouting` predict (more
spread ⟹ more outlier structure to exploit). The routing geometry is scale-invariant; the gains are not
artifacts of small models. (Needs `transformers ≥ 5.x` for the `gemma4` architecture; the model + cached
keys stay external — this script regenerates them.)

## Hierarchical (FMM treecode) routing — making the selection itself subquadratic — the hierarchical-routing result + `hierarchical_routing.py`

The block-sparse kernel's flat front-end scores every cluster per query — `O(#clusters)`, the one place
the kernel isn't asymptotically subquadratic. The fix is a tree: group clusters and skip a whole subtree
with one bound check at its parent. That is only correct if a parent's bound dominates every key in its
subtree, which needs the **recursive radius** `R_parent = max_child (‖μ_child − μ_parent‖ + R_child)`.

**Result (proved):** `parent_bounds_child_key` (the radius translates up the tree by triangle
inequality), `subtree_radius_bound` (the recursive parent radius bounds every key under it), and the
capstone `hierarchical_prune` — *one* admissible-bound check at a parent, using `R_parent`, proves no key
anywhere in its subtree can beat the best found. This is the FMM/Barnes–Hut multipole-acceptance
correctness, proved; it composes `subtree_radius_bound` with `admissible_search_bound`.

**Measured:** a 2-level tree with that recursive radius, selection cost = nodes scored per
query, across n:

| n | #fine cells | flat cost | hier cost | hier/flat | flat recall | hier recall |
|---|---|---|---|---|---|---|
| 2,000 | 169 | 169 | 52 | 30.8% | 0.927 | 0.840 |
| 8,000 | 400 | 400 | 80 | 20.0% | 0.900 | 0.740 |
| 32,000 | 1,024 | 1,024 | 128 | 12.5% | 0.867 | 0.647 |
| 128,000 | 2,500 | 2,500 | 200 | **8.0%** | 0.753 | 0.567 |

Hierarchical approximate routing scores a **vanishing fraction** of the cells — `hier/flat` falls
30.8%→8.0% (hier cost ~ `n^{1/3}`, flat ~ `n^{2/3}`): the **selection itself is now subquadratic**, the
part a flat router could not deliver. *Honest costs:* (1) it's a recall **trade** (hier < flat — keeping
only the top-3 coarse nodes sometimes misses the target's cell; the knob that narrows it is the earlier
finding — rank coarse nodes by the *cumulant* score, or keep more); (2) **lossless** hierarchical B&B
does *not* beat flat (the coarse admissible radius is too loose to prune a subtree exactly) — the
trilemma again, at the tree level. The win is approximate, which is SSA's regime anyway. This closes the
last asymptotic gap flagged for the kernel: with the tree, *both* the read (`O(n·k)`) and the *selection*
are subquadratic.

## Is the lossless selector impossible? — `the theory (see paper)`

Three strengths, not equal:

1. **Unconditionally, for all algorithms: NO.** An unconditional super-linear lower bound for a problem
   in P is beyond known complexity theory; we cannot claim it.
2. **Worst case, conditional on SETH: YES (known).** Lossless (even high-accuracy) subquadratic attention
   = all-pairs MaxIP, which reduces from Orthogonal Vectors; under SETH it needs `n^{2−o(1)}` time when
   entries are unbounded (Alman–Song, *Fast Attention Requires Bounded Entries*). The **bounded-entry /
   benign regime is the only escape** — exactly the trilemma's bet and what SSA rests on. Conditional;
   not formalizable (no SETH framework in Mathlib).
3. **Probe model, unconditionally: YES — and it admits a proof.**

The argument:
- `unexamined_argmax_invisible` — for any examined set `S` and any skipped index `j₀ ∉ S`, there is an
  adversarial key assignment agreeing with the original on all of `S` yet making `j₀` the UNIQUE argmax
  (`k_{j₀} := c·q`, `c` large). The winning key is invisible to anything reading only `S`.
- `lossless_selector_reads_every_key` — capstone: any selector whose output depends only on `S` and that
  is lossless (always returns an argmax) must have `S = univ`. It cannot skip a single key. In the probe
  model (cost = keys examined) lossless selection is therefore `Θ(n)` per query, `Θ(n²)` total.

So: **lossless cheap selection is provably impossible in the worst case** — unconditionally in the probe
model (proved here), and conditionally for general runtime under SETH (Alman–Song). It is
possible **only on benign / bounded-entry geometry**, never unconditionally for free. This is the exact
boundary SSA lives on: its 12M-token lossless claim is a bet that real-data geometry is benign enough to
sit on the *possible* side of this wall — a bet that is, by the wall itself, **not provable in general**,
which is precisely why no independent party can confirm it from the outside.

## Routes to improvement — the math, and the deepest one built — the anisotropic bound + `anisotropic_bound.py`

Deeply exploring the math surfaced a route map (ranked by depth × provability):
- **A. Anisotropic (ellipsoidal) prune bound** — built below.
- **B. Partition-function tail correction** — sparse softmax over-normalizes by `1/(1−W_D)`; the cluster
  cumulant scores already computed estimate each unselected cluster's exp-mass (`|b|·e^{μ+½σ²}`), giving
  a corrected denominator that de-biases the read for free.
- **C. Higher cumulants** — the cheap 2nd-order routing plateaus ~0.94 vs oracle 1.0; the Edgeworth term
  `+(β²/6)κ₃` (moments-only) closes more.
- **D. Doubling dimension** — the benign regime *is* low doubling dimension; cover-trees give provable
  `O(2^{ddim}·log n)` queries (the rigorous "what to train the keys toward").
- **E. JL-sketched routing** — route in a `d'=O(log n)` random projection (inner products preserved),
  routing cost `O(n·C·log n)`.
- **F. Bound-derived training regularizer** — penalize `qᵀΣ_b q` for non-target clusters, directly
  tightening the proven prune bound = manufacture benign geometry with a provable objective.

**Built — Route A.** The flat admissible bound `⟨q,μ⟩ + ‖q‖·R` is isotropic; on an unevenly-spread
cluster it's loose (why lossless hierarchical pruning didn't fire). The tightest two-moment bound is the
support function of the moment-matched ellipsoid:

- `ellipsoidal_search_bound` (proved): a key in `μ + A·(ball R)` obeys `⟨q,k⟩ ≤ ⟨q,μ⟩ + R·‖A†q‖`
  — a *directional* radius that shrinks when `q` aligns with the cluster's thin axes.
- `ellipsoidal_radius_sq` (proved): `‖A†q‖² = ⟨q,(AA†)q⟩ = qᵀΣ_b q` — **the tightest two-moment prune radius
  is the square root of the same second cumulant the routing score uses.** Routing and pruning are one
  object, seen as a radius vs a score.

**Measured (Qwen-16K deep head, lossless B&B):** the ellipsoidal bound is strictly tighter than the
isotropic one on **100% of (query,cluster) pairs** (validating the theorem), but it cuts lossless cost only
**100% → 96.6%**. The reason is deep and reinforces the impossibility wall: the bound must still *cover the
very outlier retrieval seeks* (`R'` is set by that outlier), so it covers it in a better-shaped container
but cannot avoid covering it — the trilemma persists. The win needs queries *aligned* with the clusters'
thin axes, which only training induces (Route F). So Route A is the **correct tighter object** (proven, and
always tighter), and it shows precisely why the realized gain again routes back through benign, co-adapted
geometry — never free.

## Route F — the routability regularizer: training makes the proven bound fire — `the theory (see paper)` + `prune_regularizer.py`

The capstone: turning "the correct object" into "a measured gain." A non-target cluster is prunable for
query `q` when its bound is below the best score; the bound's radius is `√((m−1)·qᵀΣ_b q)` (Samuelson's
inequality — the variance-only admissible bound, no Mahalanobis inverse). So **penalizing `qᵀΣ_b q` for
non-target clusters directly collapses the proven prune radius.**

**Result (proved):** `samuelson_centered` — for centred scores summing to zero, `m·(s_i−s̄)² ≤
(m−1)·Σ(s_j−s̄)²` (Cauchy–Schwarz on the `m−1` other deviations). The radius is the *second cumulant
itself* — the same `qᵀΣ_b q` the cumulant/tempered routing uses (`ellipsoidal_radius_sq`), so ONE
regularizer tightens routing AND pruning.

**Measured (co-trained keys, 24 clusters × 16, tight d=28, lossless Samuelson B&B):**

| λ (reg) | accuracy | B&B cost | non-target qᵀΣq |
|---|---|---|---|
| 0 | 1.000 | 26.5% | 0.0324 |
| 1 | 1.000 | 5.7% | 0.0140 |
| 4 | 1.000 | 4.7% | 0.0077 |
| 16 | 1.000 | 4.2% | 0.0026 |
| 64 | 1.000 | 4.2% | 0.0001 |
| 256 | 1.000 | 4.2% | 0.0000 |

The regularizer drives non-target `qᵀΣ_b q` → 0, collapsing the Samuelson radius, so **lossless B&B cost
falls 26.5% → 4.2% (6×) — the proven bound fires far harder as the keys co-adapt.** And the capacity trade
the trilemma threatens does **not** appear: accuracy stays 1.000, because the regularizer makes each
cluster thin to *other* queries while leaving it spread to its *own* (query-specific anisotropy) — it
removes distractors, not capacity. Benign geometry is **manufactured essentially for free, given enough
dimension** for the per-cluster subspaces (the trade reappears only as d → #clusters).

**The conclusion of the whole route exploration.** Every route bottomed out at the same wall (the proven
bounds are right; the gain is gated by benign geometry). Route F is where the wall is *crossed* — not by a
cleverer bound, but by *training* that manufactures the geometry the bounds need, navigating the trilemma
via anisotropy. That is exactly, and only, what SSA's training does — and it is *why* its keys are routable
and *why* `head_dim` must be large enough to leave room. The honest frontier was never the algorithm; it
was always the geometry training induces, and now that is built, proven, and measured.

## More geometry training tricks — and the gate that composes them — `samuelson_prune_gate`

Route F (shrink non-target `qᵀΣ_b q`) is one of six geometry-training levers, each serving a proven bound
with its own tension:

| trick | objective | bound served | tension |
|---|---|---|---|
| mean-orthogonalization | `Σ_{b≠b'}⟨μ_b,μ_{b'}⟩²` ↓ | the prune bound's MEAN term | needs dimension |
| target whitening | `qᵀΣ_target q` ↑ | intra-cluster capacity | vs collapse |
| frame / low-coherence | frame potential ↓ (Welch bound) | `capacityBound` | raises the radius floor |
| norm-equalization | `Var‖k_j‖→0` (MIPS→cosine) | bounded radius / cleaner search | loses a DOF |
| **bounded entries** | `\|⟨q,k⟩\| = O(√log n)` | the **SETH wall** (Alman–Song) | shrinks the margin |
| peakedness / low-entropy | attention entropy ↓ | `recoveryWeight→1` + prunability | = temperature |

The deepest, **bounded entries**, opens a *second* subquadratic route entirely: bounded `⟨q,k⟩` ⟹
`exp(⟨q,k⟩)` ≈ a low-degree polynomial ⟹ the attention matrix is **low-rank** (the Performer/linear-
attention route, not selection) — the same benign geometry, a different mechanism, and the precise
training-side dual of the impossibility wall.

But the cleanest *unifying* result is how the levers compose. `samuelson_prune_gate` (proved):
a cluster is pruned — no member reaches the best score `β` — **iff the squared margin beats the spread**,

    (card − 1) · Σ(s_j − s̄)²  <  card · (β − s̄)².

So the two geometry knobs — the **margin** `β − s̄` (grown by a margin objective / separation) and the
**spread** `Σ(s_j − s̄)²` (shrunk by the `qᵀΣ_b q` regularizer, Route F) — are one prune condition.
Training pulls both, pushing each non-target cluster across the gate. This is the precise statement of
what *all* the geometry tricks are for: move clusters to the prunable side of `margin² > (m−1)·spread`,
which is the benign — and provably navigable — side of the wall.

## Characterizing the geometry-training — `geometry_characterization.py`

The needed test runs over the surveyed tricks. Three measured characterizations — two of which overturned
the naive prediction, honestly.

**[1] The prune gate is a one-effective-knob surface, not two.** Sweeping a margin regularizer and the
spread regularizer (Route F):

| margin reg | spread reg | cost | acc | margin | spread | gate-fire | pruned |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 21.0% | 1.00 | 0.768 | 0.0278 | 84% | 100% |
| 8 | 0 | 25.6% | 1.00 | 0.764 | 0.0297 | 79% | 100% |
| 0 | 8 | **6.7%** | 1.00 | 0.751 | **0.0064** | **100%** | 100% |
| 8 | 8 | 12.8% | 1.00 | 0.770 | 0.0162 | 93% | 100% |

`gate ⊆ prune` exactly as `samuelson_prune_gate` says (every gate-fire is pruned; the gate is the
*provably*-prunable subset, sufficient not necessary). But the **margin knob does nothing** — retrieval
*already* maxes the margin (~0.77 everywhere), so of the gate's two terms `card·margin² > (card−1)·SS`,
training only needs to move the **spread**. The spread regularizer drives the proven-prunable fraction
84%→100% and cost 21%→6.7%. One lever, not two.

**[2] Entry magnitude splits the two subquadratic routes.** eff-rank of `exp(B·KKᵀ)` vs entry scale B:

| B | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| eff-rank (of 256) | 2.7 | 6.9 | 38 | 217 | 256 | 256 |

Small B → low-rank attention (the **linear-attention route**); large B → full rank (only **selection**).
And selection's prune gate is **scale-invariant** (margin² and spread both scale `B²`), so selection cost
is B-invariant (21%) — it works at *any* entry magnitude. Selection is the robust route; SSA's sharp,
long-context regime is large-B, hence selection, not linear attention.

**[3] The capacity trade does not appear — even at d = #clusters.** Sweeping dimension:

| d | cost (λ=0) | acc (λ=0) | cost (λ=16) | acc (λ=16) |
|---|---|---|---|---|
| 16 | 63.3% | 1.00 | 6.7% | 1.00 |
| 32 | 21.0% | 1.00 | 6.5% | 1.00 |
| 64 | 10.3% | 1.00 | 6.3% | 1.00 |

Accuracy stayed **1.00 at every d**, even d=16=#clusters — the predicted trilemma trade never bit. The
regularizer reaches ~6.5% at *every* d (its biggest win where retrieval-only geometry is worst: tight d
63%→7%). The reason: query-specific anisotropy needs only ~1 dimension per cluster, so `head_dim ≥ local
structure` is ample room. **The trade needs `d ≪ structure`, which real head_dims (64–256) sit well
above** — which is precisely why large head dimensions put real models on the wall's possible side, for
free. Characterized: one effective lever (spread), entry magnitude selecting the mechanism, and benign
geometry essentially free given enough dimension.

---

## Could we do their construction pipeline? — the attention swap (`ssa_swap.py`)

SubQ's recipe is: open-weight base → strip dense O(n²) attention → insert Subquadratic Sparse Attention →
staged context extension → continued pre-training (~1T tokens). The non-compute core of that — the
**attention swap and the adaptation** — is exactly what our pieces support. `ssa_swap.py` demonstrates it at
micro scale: GPT-2's dense attention is replaced (via an `F.scaled_dot_product_attention` patch) with our
cumulant-routed block-sparse SSA (block 64, top-4 by `⟨q,μ⟩+½⟨q²,σ²⟩` + a 2-block local window + causal,
~38% of keys); perplexity degrades; an equal-budget continued-pretrain recovers it.

**The fair control matters.** Continued pre-training on wikitext lowers held-out perplexity for *either*
attention (domain adaptation: vanilla GPT-2 is a WebText model). So "recovery" is measured against a **dense
model given the same 300 steps**, not against the off-domain start — otherwise in-domain adaptation is
mistaken for the swap closing. (Held-out 10% of wikitext-2, disjoint from the 90% train split.)

| | perplexity (held-out) |
|---|---|
| dense GPT-2, off-domain start | 32.5 |
| **+ SSA swap, no adaptation** | **45.8**  (+13.2 — the swap degrades the dense-trained base) |
| dense + 300 steps in-domain (fair control) | 23.9 |
| **SSA-swap + 300 steps in-domain** | **25.1**  (residual **+1.2** vs the control; **94%** of the swap gap closed) |

The SSA-swapped model recovers to within **+1.2 perplexity** of the dense-adapted control while attending
only ~38% of keys — the residual is the price of sparsity at this micro budget, and it shrinks the closer the
two trainings run. The swap-and-adapt construction is **sound with our algorithm**. The pipeline maps cleanly
onto pieces we already built:

| SubQ step | our piece |
|---|---|
| open-weight base | any cached model (GPT-2 here) |
| strip dense attention | the SDPA patch hook |
| insert SSA | the cumulant-routed block-sparse kernel (`ssa_kernel.py`, 20.6× at 256K) |
| staged context extension | RoPE + gentle curriculum (`ssa_extrapolation.py`) |
| continued pre-training | the LM objective + the routability regularizer (`prune_regularizer.py`, Route F) |

**Scope (honest):** GPT-2 (124M), ~10⁶ tokens, context 1024, 300 steps — *not* their frontier base, ~10¹²
tokens, or 12M context. This shows the construction is sound with our pieces; the rest is compute.

---

## Staged context extension — climbing the ladder (`staged_extension.py`)

`ssa_extrapolation.py` showed zero-shot RoPE extension dies past ~4× (recall 0.27 at 8×, 0.07 at 16×). SubQ's
pipeline goes much further by **staging**: extend → continue-train briefly at the new length → extend again.
Because the routing is position-invariant, each rung is a cheap **adapt** off the prior rung's ~2× zero-shot
recall — not a fresh train. This builds the ladder (base 48 pairs, then doubling) and measures the adapt cost
per rung. Trainable-scale MQAR (vocab 2562: keys 2048 for the permutation, values 512 so the output head
stays at proven scale); the SSA inference budget is a fair ~15% of clusters.

| rung | tokens | mult | zero-shot | adapt steps | after adapt | + SSA (inference) |
|---|---|---|---|---|---|---|
| 96 | 209 | 2× | 0.993 | 100 | 1.000 | 1.000 |
| 192 | 401 | 4× | 0.904 | 100 | 0.999 | 1.000 |
| 384 | 785 | 8× | 0.810 | 100 | 1.000 | 0.995 |
| 768 | 1553 | 16× | 0.672 | 100 | 0.996 | 0.996 |
| 1536 | 3089 | 32× | 0.516 | 400 | 0.982 | 0.979 |

**The result.** The base curriculum cost 7600 steps; the *entire* climb from 16× (where zero-shot is 0.07) to
**32× at recall 0.982** cost **800 additional steps** — ~10% of the base, and **roughly flat per rung** (100
steps each up to 16×, 400 at 32×), not growing with length. The mechanism is the elegance: each rung only
ever extrapolates 2× from the last *adapted* length, and zero-shot reliably delivers ~2× at decent recall
(the zero-shot column never drops below 0.5 here), so every adapt starts in good shape and finishes cheaply.
That is exactly why SubQ's staged ladder (262K → 512K → 1M → 2M, each a doubling) is affordable.

The **+SSA column** confirms the same trained model runs under subquadratic cumulant-routed selection with
essentially no recall loss (≥ 0.979 at every rung) once the budget is a fair ~15% — extension and
subquadratic inference are compatible all the way up.

**Scope (honest).** We reached 32× (3089 tokens) at this toy scale; the cap is the harness's O(n²) dense
*training* memory and the small model's vocab, **not the algorithm**. The subquadratic kernel
(`ssa_kernel.py`, demonstrated to 256K) removes the inference ceiling; carrying the *training* ladder to
millions of tokens is the same rung repeated + compute + a long-context corpus.

---

## What "98–100% retrieval at 1M–12M" actually shows (`niah_analysis.py`)

SubQ's one 12M number is single-target needle-in-a-haystack (NIAH) accuracy. The naive reading — "selection
caps the effective distractor count, so it's flat in n" — is true but hides the load-bearing condition, which
is this project's whole thesis. **Cheap** selection routes by block/cluster statistics (a cumulant mean +
½·var) in O(B·d), not by scoring every key. A lone needle aggregated into its block's mean is **washed out**
(its signal ÷ block size), and across the growing number of random blocks, chance fluctuations out-rank it.
So cheap moment routing **cannot** find an isolated spike in isotropic geometry. What rescues it is **benign
geometry**: a real answer sits in a *coherent span* whose neighbors also align with the query, lifting the
whole block's score. Positional blocks, no k-means — runs in ~50s.

**[1] accuracy vs length** (margin 0.55, d=64, attend ~1024 keys):

| n | dense | SSA isolated | SSA benign |
|---|---|---|---|
| 1024 | 1.00 | 1.00 | 1.00 |
| 4096 | 1.00 | 0.47 | 1.00 |
| 16384 | 1.00 | 0.20 | 1.00 |
| 65536 | 0.90 | 0.10 | 1.00 |
| 262144 | 0.83 | 0.00 | 1.00 |

Dense degrades slowly (more distractors clear the margin); the **isolated** needle **collapses** with length
(the wash-out); the **benign** span is **flat at 1.00 and beats dense at long n** (selection caps the
distractors). **[2] vs margin** (n=65536): the isolated needle barely fires at *any* margin (0.03→0.17 across
0.45→0.95), while the benign span tracks dense once the margin clears the fixed-budget floor.

**The conclusion.** 98–100%@12M is real and theory-consistent — but it is the EASY regime: a single,
HIGH-margin, **benign** target, measured as ACCURACY not losslessness, kept findable by cheap selection only
because real geometry is coherent. A lone adversarial spike defeats cheap moment routing — the impossibility
wall (`the theory (see paper)`) in miniature — which is the low-margin / multi-needle side they report no
12M numbers for (their hard benchmark, MRCR, is at 128K).

## End-to-end frozen-swap on Gemma-4-26B-A4B — the first real-model curve (`gemma_ssa.py`, `gemma_ssa_sweep.py`)

Every result above is synthetic, a routing-recall probe, or random-key kernel timing — none is an
*end-to-end quality measurement on a real model*. This is that measurement. SSA is installed via the
transformers attention interface into the **5 full-attention (global) layers** of a **frozen Gemma-4-26B-A4B**
(MoE, 4B active; head_dim 512, K=V, GQA group 2 — the 25 sliding-window layers are already linear and left
untouched), receiving post-QK-norm/RoPE q,k, and scored on needle-in-a-haystack (NIAH) retrieval vs. the
selection budget at n=2048, block 256. **This is the *analytic* swap (it materializes the score matrix), so
it measures ROUTING QUALITY, not speed** — the subquadratic-kernel / long-context regime is separate.

**A correctness fix first.** The KV-cache incremental-decode path (query_len=1, key_len=N) was indexing the
selection mask by *query* length, silently corrupting generation under sparsity. Fixed (absolute query
positions via `cache_position`) and regression-tested — necessary for any real generation.

**Baseline (cumulant β=2), and the cliff.** `1.0→1.000, 0.5→0.833, 0.25→0.000, 0.12→0.000`. The `0.000` at
budget 0.25 is **not** corruption: generation is coherent filler ("the garden path"), no NaN — a *genuine
routing miss*. A forward-only routing-rank probe localized it: the far needle's block ranks **~3rd** by the
cumulant score (just outside the kept top-2), **worst at the deepest layer (29)**, which never ranks a far
needle even by the block-max oracle — a RoPE-distance decay (distant blocks' q·k are attenuated).

**The fix — the 3rd-cumulant (Edgeworth / skew) term.** A bake-off scoring the far-needle rank under
candidate routing scores showed the **skew term is the right outlier detector**: the needle is an outlier in
its block, the 3rd cumulant rewards outlier-bearing blocks, and it even *beat the block-max oracle* on some
layers. Implementing the deployable diagonal term `r += (β²/6)·Σ_d q_d³·m3_d` (with β=4):

| budget | baseline | +Edgeworth(β4) | +Edgeworth +layer29-dense |
|---|---|---|---|
| 1.0 | 1.000 | 1.000 | — |
| 0.5 | 0.833 | **1.000** | — |
| 0.25 | 0.000 | **0.444** | **0.556** |
| 0.12 | 0.000 | 0.000 | 0.000 |

**Attribution of budget-0.25 retrieval** (the harsh keep-2-of-8 regime):
- `0.000 → 0.444` — the Edgeworth skew term (cheap, summary-only — the §5.2 outlier-routing theory realized);
- `0.444 → 0.556` — forcing the single worst layer (29) dense (the one layer no statistic could route);
- `0.556 → 1.000` — the **frozen-key ceiling** → co-adaptation training (the moat).

**Verdict** (block 256 / β4 — *partly superseded; see "Block granularity" below*). At this fixed coarse
configuration, **algorithmic fixes recover budget-0.25 retrieval to ~0.56, and the residual *looked like* a
frozen-key limit.** That attribution is **overturned by the block-granularity sweep below**: at n=2048 the
residual is a **tuning artifact**, fully recoverable to **1.000** with finer blocks and plain cumulant — no
training. The Edgeworth term —
previously a theoretical "route to improvement" — is now a *measured* gain on a 26B model, and budget 0.5
goes 0.833→1.000. *Caveats:* quality not speed (analytic O(n²), n=2048); keep-2-of-8 is a coarse, harsh
corner (**0.556 is a lower bound** — long context keeps far more blocks at 25% and is more forgiving); single
model, single needle type, ~9-probe sampling. Runs: `runs/gemma_sweep_{fixed,edge,edge_d29}.json`;
mechanism in `gemma_ssa.py` (`edgeworth`, `dense_layers`), driver in `gemma_ssa_sweep.py`.

### Block granularity is the real lever — the budget-0.25 "ceiling" was a tuning artifact

The verdict above attributed the budget-0.25 residual to a frozen-key ceiling needing co-adaptation. A
block-granularity sweep (n=2048, budget 0.25, everything else fixed; `gemma_ssa_sweep.py --block 256,128,64`)
**overturns that** — recall is *non-monotonic* in block size and, tuned correctly, reaches **1.000**:

| block | #blocks | keep@25% | routing | NIAH |
|---|---|---|---|---|
| 256 | 8 | 2 | cumulant β4 + edgeworth | 0.444 |
| 128 | 16 | 4 | cumulant β4 + edgeworth | 0.667 |
| 64 | 32 | 8 | cumulant β4 + edgeworth | **0.000** (!) |
| 64 | 32 | 8 | **plain cumulant (β2 or β4)** | **1.000** |

A controlled isolation at block=64 (one model load, five configs) pins the non-monotonicity:

| block=64, budget 0.25 | NIAH |
|---|---|
| edgeworth, β4 | 0.000 |
| edgeworth, β2 | 1.000 |
| plain cumulant, β2 | 1.000 |
| plain cumulant, β4 | 1.000 |
| edgeworth, β4, local_w=4 (bigger window) | 0.000 |

**Reading.** (1) At **fine blocks the needle dominates its small block, so plain 2nd-cumulant routing fully
retrieves (1.000) regardless of β** — no skew term needed. (2) The Edgeworth skew term is a *coarse-block
compensator* (it rescues 0.000→0.444 at block=256, where the needle is buried among 256 keys) but is
*unnecessary at fine blocks and catastrophic at high β*: its noisy 64-key 3rd-moment estimate, weighted
β²/6 ≈ 2.7×, destroys routing (the 0.000); a bigger local window does not help, confirming it is the
**skew×β interaction**, not the window. (3) **The core lever is block granularity, not the skew term I added.**

**Corrected conclusion (at n=2048).** There is **no frozen-key ceiling** at budget 0.25 — the earlier
"0.556 → needs co-adaptation training" was a **block-size / β tuning artifact**. The simplest robust frozen
config (fine blocks + plain cumulant) reaches **1.000**, with no skew term, no layer-29-dense, and no
training. **Scope (unchanged and load-bearing):** this is n=2048 (only 32 blocks at block=64). The real
long-context target (millions of tokens → 10⁵+ blocks) is a much harder routing problem *and* the
O((n/block)²) block-score computation itself goes quadratic there (the hierarchical-router gap) — so "no
ceiling" is established at **moderate context only**, not proven at 12M. The co-adaptation-training frontier
remains relevant for the long-context / aggressive-budget regime this experiment did not reach. Run:
`runs/gemma_sweep_block.json`.
