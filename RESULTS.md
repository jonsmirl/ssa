# Retrieval-margin demonstrator — results

A controlled, NumPy-only validation of the content-addressable sparse-attention theory
(the retrieval-margin theory (paper §3)) and a direct test of the one
genuinely open piece: **can a sublinear content selector be assembled from known ANN parts, and
when does it work?**

> **On the *(proved)* claims below.** Results marked *(proved)* — and the theorem names cited inline
> (`samuelson_prune_gate`, `ellipsoidal_search_bound`, `temperedLogPartition_max_sandwich`,
> `lossless_selector_reads_every_key`, `hierarchical_prune`, `dropped_combination_error_bound`, …) — are
> machine-checked in the separate Substrate Lean 4 development, with core declarations under
> `Substrate.Universal` and domain recognitions under `Substrate.Inference.Shadow` (axiom-pure, no
> `sorry`/`admit`). That formalization is
> maintained separately and is **not** included in this public repository; the theorem names are the
> pointers into it. The Python in this repo *measures* those results — it does not prove them.

Run: `python3 -m ssa.experiments` · Tests: `pytest ssa/tests/`.
Method: synthetic keys/queries with *exactly controlled* gap, separation, dimension, and count — so
each prediction is tested against ground truth. No training; this validates the theory and the
selector mechanism, not a trained model.

## Float32 recursive-ball verification

`python -m ssa.float_tree_verification` compares the actual CUDA `CausalTree` with float64 descendant
oracles. The RTX 4080 run in [`runs/float_tree_verification.json`](runs/float_tree_verification.json) covers
five ordinary and adversarial geometries at dimension 64 and fanouts 2, 4, and 16. The unguarded formulas
underestimated 8,701 of 90,105 recursive radii and 367,480 of 1,081,260 score caps. After dimension-scaled
inflation and `nextafter(+inf)`, no underestimates were observed. A 65,536-leaf fanout-16 construction took
0.433 ms guarded and 0.223 ms unguarded. This is floating-point stress evidence, not a proof for all CUDA
inputs or kernels; the fixed beam remains approximate.

## Formal selector-contract status

The relevant Substrate audit is through `130cae3e9`. Its abstract theorems now cover four contracts used here:
the exact restricted-read TV/KL and output residual with both block certificate arms; bounded selection as a
strict top set when all unreturned candidates are strictly below the threshold, with index order as the
no-margin tie repair; the `b/n` uniform-spike ceiling for grounded adaptive reads plus its finite-randomized
extension; and the complete conditional routing plan combining nine-vote retention, the uniform
`roundWidth + 128` block cap, past-boundedness, and a causal position cut. The public paper reproduces these
statements and proofs. None verifies the Python/CUDA mapping, removes the grounded-output hypothesis, turns
the nine-vote premise into a quality result, or converts a block-count cap into latency.

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
long-range retrieval at `O(n·k)` cost, the speedup *grows* with context (mirroring SubQ's reported
7.2×@128K → 52.2×@1M — uncited; see `SUBQ_ASSESSMENT.md` § "Provenance of the external SubQ figures"),
and it works for exactly the two reasons the experiments identified —
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

### Treecode reality check — co-trained keys (a), the wall-clock crossover (b), the flat-router ceiling (c)

The section above is on *synthetic* clustered keys and counts *nodes*, not wall-clock — the reviewer's
"synthetic-only, unintegrated" critique. Three follow-ups close that gap honestly; the verdict is **"the
hierarchical router is *necessary* and *conditionally* viable, not yet a *demonstrated* end-to-end win."**

**(a) On real co-trained keys, lossless treecode pruning still fails; approximate routing survives.** Build the
2-level tree on keys shaped by the routability regularizer (Route F) vs un-co-trained:

| keys | lossless nodes scored | approx recall | approx cost |
|---|---|---|---|
| un-co-trained (λ=0) | 182/182 (**100%** — no pruning) | 0.705 | 29% |
| co-trained (λ=64) | 169/182 (**93%** — barely prunes) | **0.945** | 29% |

Co-training (which shrinks *intra-cluster* spread) barely helps the **lossless** prune (100%→93%), because the
recursive parent radius `R_parent = max_child(‖μ_child−μ_parent‖ + R_child)` is dominated by *inter-centroid*
distance — which Route F does not target. But it lifts the **approximate** recall 0.71→0.95 at the same cost.
So the realized treecode win is *approximate* even on benign co-trained geometry — exactly the
"approximate-not-lossless" the real-key tests showed, now with the mechanism (the *radius*, not the spread).
Closing the lossless gap would need a *hierarchical* routability objective that also shrinks centroid spreads.

**(b) Wall-clock crossover: ~1M tokens — and a naive 2-level then regresses.** Time the kernel's flat block
router (a dense `(n/b)×(n/b)` GEMM) vs a 2-level matmul+gather router (block 128, H=8, d=64, fp16, single 16 GB GPU):

| n | n/b blocks | flat GEMM | 2-level | winner |
|---|---|---|---|---|
| 262K | 2,048 | 2.3 ms | 3.0 ms | flat |
| 524K | 4,096 | 6.2 ms | 6.4 ms | flat |
| **1M** | 8,192 | 27.7 ms | **20.7 ms** | **2-level (1.34×)** |
| 2M | 16,384 | 102 ms | **715 ms** | flat (7× regression) |

The flat GEMM's constant is excellent, so it **wins until ~1M tokens**; only there does the lower-order
hierarchical scaling overcome it. But a *naive* 2-level router **regresses 7× at 2M** — its candidate gather is
`Θ(n^{1.5}·d)` and blows up. So the crossover is real and concrete (**~1M**), but **realizing the win in the 12M
regime needs a true FMM treecode** (bounded candidate sets, no materialized gather) — the genuine, unsolved
engineering, not the naive 2-level.

**(c) Proposition (flat-router ceiling).** A router that scores every one of the `B = ⌈n/b⌉` key-blocks for
each query performs `Θ(Q·B)` block-score evaluations (the Lean's `flat_router_work`): per query (`Q=n`) that is
`Θ(n²/b)`; amortized per query-*block* as NSA/SubQ do (`Q=n/b`) it is `Θ((n/b)²)` — a hard **speedup ceiling of
`b²`** once the router dominates (the regime SubQ's own two points already enter near 12M). Either way it
materializes a `B×B` matrix — `Θ(n²)` *time and memory* for fixed `b` (the flat GEMM's nb² matrix is a memory wall). Choosing
`b = Θ(√n)` gives the `O(n^{1.5})` flat router but caps the attention budget at `Θ(√n)` keys/block. Hence **no
constant or `√n` block size makes the flat (scan-all-blocks) router subquadratic-with-growing-speedup; a router
whose speedup grows unboundedly with `n` must examine `o(B)` blocks per query — i.e. a hierarchical
(sublinear-per-query) index.** This is a counting bound, now **formalized** (axiom-pure) as `flat_router_work` /
`subquadratic_forces_skip` in the Substrate Lean, a companion to `SearchTradeoff` / `capacity_search_tension`
(the tension since strengthened to an actual collision statement, `capacity_pigeonhole_tension`: `B < n`
cells force two ε-coherent keys into one cell — pigeonhole chained through `cluster_radius_floor`).
Its force for the SubQ assessment: a *quality-preserving* "1,000× at 12M" **provably forces** a working
hierarchical indexer — exactly the part (a)+(b) show is only *approximate* and not yet *engineered*.

## Is the lossless selector impossible? — `the theory (see paper)`

Three strengths, not equal:

1. **Unconditionally, for all algorithms: NO.** An unconditional super-linear lower bound for a problem
   in P is beyond known complexity theory; we cannot claim it.
2. **Worst case, conditional on SETH: YES (known).** Lossless (even high-accuracy) subquadratic attention
   = all-pairs MaxIP, which reduces from Orthogonal Vectors; under SETH it needs `n^{2−o(1)}` time when
   entries are unbounded (Alman–Song, *Fast Attention Requires Bounded Entries*). The **bounded-entry /
   benign regime is the only escape** — exactly the trilemma's bet and what SSA rests on. Conditional;
   not formalizable (no SETH framework in Mathlib).
3. **Grounded probe model, unconditionally: YES — and the grounding hypothesis is essential.**

The argument:
- `unexamined_argmax_invisible` — for any examined set `S` and any skipped index `j₀ ∉ S`, there is an
  adversarial key assignment agreeing with the original on all of `S` yet making `j₀` the UNIQUE argmax
  (`k_{j₀} := c·q`, `c` large). The winning key is invisible to anything reading only `S`.
- `lossless_selector_reads_every_key` — capstone: any selector whose output depends only on `S` and that
  is lossless (always returns an argmax) must have `S = univ`. It cannot skip a single key. In the probe
  model (cost = keys examined) lossless selection is therefore `Θ(n)` per query, `Θ(n²)` total.
- `a_spike_outside_the_empty_trace_leaves_the_run_identical` — for an adaptive binary decision tree,
  a spike outside the all-false reference trace changes neither the trace nor its returned set.
- `the_reach_share_is_within_the_depth_share` — if the selector returns only positions it probed, a
  depth-`b` tree recalls at most `b` of `n` one-spike placements, hence at most `b/n` uniformly.
- `a_spike_is_reached_by_at_most_the_depth_share_over_the_seed` — for any finite distribution over
  grounded depth-`b` trees, one fixed placement has seed-averaged recall at most `b/n`.
- The hypothesis cannot be erased: a depth-zero leaf returning the whole carrier recalls every placement.
  A fixed-set reader attains the `b/n` ceiling, so the grounded count is sharp.

So: **lossless cheap selection is impossible for grounded adaptive probes in the worst case**, and
conditionally hard for general attention runtime under SETH (Alman–Song). The probe theorem does not cover
an arbitrary preprocessed index whose side information names positions it did not inspect at query time;
such a claim needs a model and budget for the index. SSA lives on the measured benign-geometry side of this
boundary and makes no unconditional losslessness claim there.

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
(MoE, 4B active; head_dim 256, K=V, GQA group 2 — the 25 sliding-window layers are already linear and left
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
- `0.556 → 1.000` — **(attribution retracted)** this was called a *frozen-key ceiling* crossable only by
  co-adaptation training (the "moat"); the **"Block granularity" section below overturns it** — plain cumulant
  at a finer block reaches **1.000** with no training, so this residual was a *tuning artifact*, not a ceiling.

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

> **Artifact note.** `runs/gemma_sweep_block.json` records the **edgeworth** sweep — the three rows
> 256→0.444, 128→0.667, 64→**0.000** (the JSON does not store the routing config, so a reader sees only the
> NIAH column). The load-bearing **plain-cumulant block=64 → 1.000** result and the five-config isolation
> table come from a *separate* one-model-load run that was not saved to JSON; re-run with
> `gemma_ssa_sweep.py --block 64 --beta 2` (no `--edgeworth`) to regenerate it. The headline "no ceiling /
> 1.000" therefore rests on that un-committed run, not on the cited JSON — commit it to close the gap.

### Does it extend to longer context? (forward-only routing-rank probe)

The n=2048 result raises the obvious question: does fine-block plain-cumulant routing keep the far needle
inside the budget as context *grows*? A forward-only probe answers it cheaply — capture the full layers' q,k
(delegating the attention to real SDPA, so no n² materialization) and compute the far-needle block's
plain-cumulant rank offline. Routing-rank-within-keep was the faithful predictor of retrieval at n=2048
(≈65% of layer×heads within budget ⟺ the full 1.000 retrieval), so it extends the test without generation.

Plain cumulant (β=2), block=64, budget 25%, far needle (depth 0.25):

| n | keep@25% | within-budget (layer×heads / 80) | robust layers 5/11/23 (median rank) | weak layers 17/29 |
|---|---|---|---|---|
| 2048 | 8 | 52 (65%) | 2, 5, 3 | 10, 11 |
| 4096 | 16 | 48 (60%) | 2, 4, 5 | 27, 37 |
| 8192 | — | *not measurable here — see below* | — | — |

**The within-budget fraction holds (65% → 60%) as context doubles**, and the mechanism is sharper than
"no ceiling at one length": a **robust subset of layers (5, 11, 23) routes the far needle at rank ≤5
regardless of context length** (well inside the 25% budget), while the two weak layers (17, 29) drift further
out with n — but those were never the ones carrying retrieval. So the result is **layer-driven and stable
across context**, now established through **n=4096** (2× the original).

**Measurement ceiling (itself a finding).** n=8192 was *not measurable on this hardware*: the analytic,
score-materializing swap is memory-bound on the 16 GB GPU (OOM at 4096 without a model-memory cap; a CUDA
fault at 8192 with one). This is the *measurement path* hitting a wall, not the method — and it is exactly why
genuine long context (and *any* speed claim) needs the **real subquadratic kernel** (`ssa_kernel` + the
hierarchical router) wired into the swap, not the score-materializing analytic version used for these quality
probes. **Pushing the validation to the 12M-token regime — and measuring the speed advantage — requires
substantially more compute than a single 16 GB GPU + CPU offload** (the neocloud/cluster path). That, and the
kernel integration, are the remaining work; everything in this section is the frozen-model *quality* story,
which is now characterized.

## The floor program (P0–P5): driving the kernel to the n·κ floor

The theoretical floor is `n·κ` (attending the κ selected keys); the measured kernel sits far above it because
of router cost. Full record: `FLOOR_PROGRAM.md`. Figures: `paper/figures/{unified_scaling,router_gpu_compare}.png`;
data: `paper/figures/{cost_profile,recall_floor,bakeoff,faiss_router,router_gpu_compare,router_cpu_compare}.json`.

**P0 — cost decomposition** (`cost_profile.py`): attention ~ `n^1.02` (the floor), router ~ `n^1.76`,
**maskbuild (argsort BlockMask) ~ `n^2.12`** (the largest component). At 12M the floor is ~1% of the forward —
a **128× gap**, dominated by the `(n/b)²` score GEMM + the argsort maskbuild.

**P1 — recall-vs-κ floor map** (`recall_floor.py`): κ_min = smallest budget reaching recall ≥ 0.9 (κ_min/n IS
the floor; speedup ≤ n/κ_min):

| geometry | centroid κ_min/n | cumulant κ_min/n |
|---|---|---|
| clustered (tight) | 3.0% | 3.0% |
| diffuse / random (adversarial) | 50% | 50% |
| co-trained λ=0 | 25% | 25% |
| **co-trained λ=64** | **0.4%** | **0.4%** |

**Co-training crushes the floor 25%→0.4% (60×)** — the dominant lever; geometry-bound (adversarial = no speedup).

**P2 — cheap router wins** (`router_variants.py`): narrow-kv_idx ~2×, cross-layer sharing ÷5; low-rank a bust
(5–14% agreement on high-PR keys). All constant factors → the sub-linear router is justified.

**P3 — router bake-off** (`bakeoff.py`, recall vs selection cost on benign co-trained keys): treecode wins raw
cost (0.8%), **faiss-ivf robust** (1.6%, GPU-optimized), centroid 1.8%, **LSH out** (can't reach 0.9).

**P4–P5 — FAISS-IVF block-router, MEASURED on the GPU** (`router_gpu_compare.py`, faiss-gpu, both routers on
GPU, no transfer):

| n | flat single-head `(n/b)²` GEMM | faiss-GPU IVF | |
|---|---|---|---|
| 256K | 0.21 ms | 7.3 ms | flat 35× |
| 2M | 14.0 ms | 19.5 ms | flat 1.4× |
| **4M** | **52.6 ms** (runs; 4.3 GB) | **31.4 ms** | **IVF 1.7×** |
| **8M** | **OOM** (nb²=16.0 GiB ≈ 17.2 GB > the 16 GiB card; skipped by size, not run) | **63.7 ms** | IVF only |

**Correction (was wrong):** an earlier version skipped the flat GEMM by a `mem<3.0` guard and labelled 4M
"OOM" — false; the 4.3 GB matrix *fits* (flat = 52.6 ms there, and IVF is 1.7× faster). The flat GEMM OOMs
only at **8M** (17.2 GB matrix). And note the benchmarked `flat` is a stripped *single-head* score GEMM; the
kernel's **actual** router (`block_route`, H heads + `(B,H,nb,nb)` sel + argsort) is ~10× heavier and OOMs at
**~1M** on this card (measured: 19 ms @256K, fault @1M) — so against the real router the IVF wins at every n.
The flat GEMM's constant wins below ~3M; crossover ~3M; the **IVF router runs to 8M (64 ms) — the only router
past the wall**. The 12M "kernel ~at the floor" is a **projection** (measured router + an analytic floor
extrapolated 23× past the largest measured n; no end-to-end kernel was timed). The same-device CPU control
(`router_cpu_compare.py`) agreed; the earlier "75–333× slower" was a GPU↔CPU transfer artifact.

**Recall and speed on the *same* keys (`ivf_recall_speed.json`).** The OOM/speed table above uses random
`mu`/`qb` (speed only), and the agreement numbers came from a separate set — a fair criticism. Re-measured on
the *same* benign clustered block-means: the faiss-GPU IVF gets **recall 0.97–0.99** (top-`c` agreement with
the flat router) **and** fast search (0.3 → 22 ms, 256K → 4M) on one dataset — so the win is not speed-on-noise
+ recall-on-toy. **Caveat:** these are *synthetic benign* keys; recall+speed together on **real-model** keys at
long context is still unmeasured (the deeper open gap). And FAISS-IVF is a *known* ANN — it isn't the
contribution; it works here *only because* the co-trained/benign geometry makes the coarse quantizer rank the
target's cell (the hard part is manufacturing that geometry, §"Manufacturing routability", not the index).

*Scope (updated — the integration is now measured):* the IVF router is **now wired into the FlexAttention
kernel and measured end-to-end to 12M** (`ivf_kernel.py`, next section) — the "isolation-only" and "8M→12M
extrapolated" caveats are retired. The standalone router sweep also now runs to **12M (94 ms; flat OOMs at
8M and 12M)**. What *still* remains is multi-head and **real-model** keys at the 12M endpoint (the e2e run is
single-head, synthetic-keys-speed-only). Net: both ingredients a quality-preserving 1,000×@12M needs —
floor-lowering co-training (60×) and a sub-linear indexer (the IVF router) — are demonstrated, and the
indexer is now shown to drive a **live** kernel to ~the floor, under exactly the benign-geometry condition
the floor analysis names.

## The IVF kernel end-to-end — measured to 12M — (`ivf_kernel.py`, `ivf_decode.py`)

The floor program's last caveat was that the IVF router had only been timed *in isolation* — the full-kernel
landing at 12M was a projection. This wires the faiss-GPU IVF router straight into the FlexAttention kernel
(`ssa_flex_ivf`: IVF search over the block-means emits the `from_kv_blocks` contract directly — no `(n/b)²`
score GEMM, no argsort maskbuild) and **measures the whole forward, single-head, to 12M on the 16 GB card**.
The enabling detail is `BlockMask.from_kv_blocks(..., compute_q_blocks=False)`, which skips the dense
`(nb,nb+1)` transpose that would need 38.7 GB at nb=98,304 — the one change that makes a live 12M forward fit.

**Measured prefill decomposition** (single head, d=64, block=128, top_c=8, fp16; router = full IVF
build+search each call; dense measured to 4M then fit `~2.0e-9·n^1.98`):

| n | router (ms) | maskbuild (ms) | attention = floor (ms) | **total (ms)** | dense (ms) | speedup | peak mem |
|---|---|---|---|---|---|---|---|
| 256K | 7.8 | 0.002 | 0.9 | **8.6** | 110 (meas) | 13× | 0.14 GB |
| 1M | 13.5 | 0.002 | 3.2 | **16.1** | 1,617 (meas) | 100× | 0.55 GB |
| 4M | 36.4 | 0.003 | 13.5 | **52.0** | 26,279 (meas) | 505× | 2.18 GB |
| 8M | 65.1 | 0.003 | 32.6 | **94.3** | (fit) | — | 4.37 GB |
| **12M** | **101.4** | **0.003** | **47.5** | **139.5** | **227,227 (fit)** | **1,629×** | **6.55 GB** |

**The `n^2.12` maskbuild wall is gone** — the argsort BlockMask build that P0 measured as the dominant term
(40.7 s projected at 12M) is now **sub-millisecond** (0.003 ms), because the IVF emits `kv_idx` directly. At
12M the whole forward is **139 ms in 6.55 GB**, and the gap to the theoretical floor is a **measured 2.9×**
(total/attention), not the 128× P0 measured for the flat kernel. The 101 ms router is dominated by the faiss
index **build** (73 ms; search is 16 ms) — in real serving the index is built once and reused across layers,
so the amortized per-layer router cost is closer to the search term. (Honest scope: single head — H=8 does not
fit at 12M, K alone is 12.3 GB; synthetic random keys, so this is a **speed** result — selection quality is
the P1/P3/P4 story, unchanged; the dense speedup numerator is measured, the 8M/12M denominator is the fitted
`n^1.98` dense law since dense attention is itself unmeasurable there.)

**Decode path** (`ivf_decode.py`) — the serving cost the "serving paradox" only asserted, now measured on both
sides. Per generated token: maintain block-means incrementally (add-only IVF index, so it holds only completed
past blocks ⇒ automatically causal), IVF-search the one query, gather ≈κ keys, one κ-length softmax row. The
dense reference is a **fair** one — an fp16 flash-decode row (sdpa, q_len=1) over the whole causal prefix, no
fp32 K/V copy (measurable even at 12M):

| n | SSA step (ms) | dense step (ms) | naive fp32 dense (ms) | speedup | vs naive | search (ms) | gather+attend (ms) |
|---|---|---|---|---|---|---|---|
| 1M | 0.60 | 0.52 | 2.58 | 0.9× | 4× | 0.12 | 0.54 |
| 2M | 0.56 | 0.91 | 5.17 | 1.6× | 9× | 0.13 | 0.53 |
| 4M | 0.55 | 1.80 | 10.26 | 3.3× | 19× | 0.13 | 0.55 |
| 8M | 0.54 | 3.59 | 20.48 | 6.6× | 38× | 0.13 | 0.55 |
| **12M** | **0.58** | **5.27** | **30.96** | **9.1×** | **54×** | **0.13** | **0.56** |

The SSA decode step is **flat in n** (κ fixed at ~1,280 keys: ~0.6 ms from 1M to 12M) while the fair dense
step grows linearly with the prefix (0.5 → 5.3 ms) — a **9.1× per-step gap at 12M**, with the crossover near
1M–2M (at 1M the fp16 flash-decode step is slightly *faster* than the IVF-routed step). **Correction (was
inflated):** the previously reported "55×" was measured against a dense reference that upcast the WHOLE prefix
K/V to fp32 every step — two full-prefix fp32 copies on top of the read, ~5× slower than the fair fp16 row.
That naive reference is kept in the benchmark (`dense_naive_step_ms_mean`, 2.6 → 31.0 ms — it reproduces the
old numbers) and the honest headline is 9.1×. **Honest scope:** single head; synthetic keys (speed only);
add-only index with no quantizer retrain (valid over the 128 measured steps; a real serving loop retrains
every R blocks as centroids drift, noted in the JSON meta).

## Multi-hop retrieval — the composition law, measured — (`multihop_analysis.py`)

The single-needle NIAH story (`niah_analysis.py`) predicts but never *tests* the multi-hop regime SubQ reports
65.9% on (MRCR) while NIAH@12M is ~98%. This plants an h-hop chain in the same synthetic geometry and runs it
through the *same* budgeted block-cumulant selector: hop j's needle is addressed by direction `dirs[j]` and its
payload is `dirs[j+1]` (the next hop's query — the geometric MQAR analogue), directions mutually orthogonal so
no hop's elevation helps another. Modes: all-benign, all-isolated, and **mixed** (benign hop 1 + isolated hop 2
— the falsifiable "single needles hold, the chain collapses" case). The prediction under test: measured chain
accuracy ≈ ∏ρ_j (oracle per-hop rates), i.e. the multi-hop sag is the single-needle result read h times.

**2-hop chain accuracy vs context length (margin 0.55):**

| n | dense | isolated | benign | **mixed** (benign→isolated) |
|---|---|---|---|---|
| 4,096 | 0.99 | 0.19 | 1.00 | 0.53 |
| 65,536 | 0.78 | 0.00 | **1.00** | **0.02** |
| 262,144 | 0.56 | 0.00 | **0.88** | **0.00** |

**The composition law holds** (measured chain ≈ ∏ρ, all modes): at n=65,536, mixed has ρ1=1.00 (benign hop),
ρ2=0.02 (isolated hop), ∏ρ=0.02, measured chain **0.02** — while each *single* benign needle stays at 1.00.
One weak hop divides the whole product toward zero. This is exactly the shape of SubQ's own table: NIAH@12M
~98% (benign single needle) vs MRCR 65.9% (multi-hop) — the "suspiciously perfect" retrieval score and the
MRCR sag are the *same* benign-geometry prediction read at two ends, now self-tested by the rig rather than
only argued. **Honest scope:** synthetic geometry (the mechanism, not a real-model chain); the real-model
two-hop task (`gemma_ssa_eval.two_hop_accuracy`, wired into the sweeps) is the model-level companion.

## The fast kernel inside a real model at 8K–128K — (`gemma_ssa.py` `impl="flex"` + `longctx_swap.py`)

The Gemma frozen-swap measured *quality* with an analytic `O(n²)` probe; `ssa_kernel` measured *speed* on
synthetic keys. This closes the gap the Scope section named: the fused FlexAttention kernel is wired into a
**real pretrained model** (Qwen2.5-0.5B — 24 layers, GQA 14/2, head_dim 64) behind an `SSAConfig.impl` flag,
and measured at 8K–128K for **both** NIAH + two-hop accuracy and prefill wall-clock vs the unswapped model.
`block_route_budget` adds the budget-fraction router at query-BLOCK granularity (the fast form) with the same
cumulant score as the analytic path; decode steps and CPU fall back to the analytic path (already
decode-correct via `cache_position`). Full-budget flex reproduces the dense LM loss (smoke gate, delta 6.7e-3).

**Native window (≤32K), block=128, bf16, prefill median of 3, speedup vs the stock-SDPA dense baseline:**

| n | budget | NIAH | 2-hop | prefill (ms) | dense (ms) | speedup |
|---|---|---|---|---|---|---|
| 16K | 0.25 | 1.00 | 1.00 | 295 | 329 | 1.11× |
| 16K | 0.12 | 1.00 | 1.00 | 281 | 329 | 1.17× |
| 32K | 0.25 | 1.00 | 0.78 | 689 | 913 | 1.32× |
| 32K | 0.12 | 1.00 | 0.67 | 600 | 913 | **1.52×** |
| 32K | 0.06 | 1.00 | 1.00 | 567 | 913 | **1.61×** |

The kernel preserves single-needle NIAH at **1.00** while giving a **1.5–1.6× prefill speedup at 32K**
(budget 0.06–0.12); the speedup **grows with n and with a tighter budget** (≈1.0× at 8K, the crossover, up
through 1.6× at 32K), exactly the `ssa_kernel` crossover shape now inside a real model. The speedup is modest
because attention is only a fraction of a 0.5B forward (the MLP/other layers dominate — Whedon's own Amdahl
point) — it grows with model size and context. The **two-hop chain shows the predicted budget-sensitivity**
(1.00 → 0.78/0.67 at 32K as the budget tightens), the multi-hop sag appearing in a real model; NIAH is flat
across the same cells (the single-needle/multi-hop split again). *(Two-hop is 9 samples/cell — the dip is
directional, not precise.)*

**Granularity + kernel-necessity A/B (n=8K, `--impl analytic` vs `flex`):**

| impl | budget | NIAH | 2-hop | prefill (ms) | peak mem |
|---|---|---|---|---|---|
| analytic | 0.25 | 1.00 | 0.67 | 3,480 | **10.66 GB** |
| flex | 0.25 | 1.00 | 1.00 | **130** | **1.41 GB** |

At the same config the analytic path (which materializes the `(b,hq,n,n)` scores + selection mask) needs
**10.66 GB and 3.5 s**; the fused kernel needs **1.41 GB and 130 ms** — a 7.6× memory and 27× time gap, and
the analytic path OOMs before 64K while flex reaches 128K. Both preserve NIAH at 1.00, so query-block
granularity does not cost single-needle quality; the kernel is simply the only path that scales.

**Extended window (65K/128K under YaRN ×4):** both the dense baseline and SSA run under identical YaRN (rows
tagged `rope=yarn4` in `runs/qwen_longctx.json`); the full-budget smoke gate passes under YaRN (delta 1.2e-3),
and dense prefill is **2.8 s at 65K and 11.3 s at 128K** (7.4 GB, well within the card). The speedup **keeps
widening with n** — 1.6× (32K) → **2.15× (65K, budget 0.12, NIAH 1.00)** → **3.44× (128K, budget 0.06)** — and
the kernel reaches 128K where the analytic path OOMs. At 128K single-needle NIAH holds at 0.89–1.00 while the
**two-hop chain sags under a tight budget (0.44–0.78)** — the predicted multi-hop degradation surfacing in a
real model at extended length. 0.5B is not a validated-128K model, so ≥65K rows are **mechanism + wall-clock
evidence**, not absolute-quality claims; but the widening-speedup-with-preserved-single-needle / sagging-
multi-hop pattern is exactly the theory's real-model analogue.

**Honest scope:** single model at **0.5B** scale; ≥65K under YaRN (extrapolated positions); the speedup is on
the attention component of a small model (Amdahl-limited); Gemma-26B stays on the analytic path (CPU offload
makes 32K+ prefill infeasible on a 16 GB card). This is the first rig result that is simultaneously
**real-model × long-context × subquadratic-kernel × quality-measured** — at 0.5B, moderate length.

## The Certified Causal Cascade — an optimal selector, built and measured — (`cascade_router.py`, `routing_space.py`, `ccc_quality.py`, `longctx_share.py`)

The assessment's largest open question was the selector: DSA's eats **58% of prefill at 1M**; SubQ attacks
that number but never states its own. This composes the five ingredients an optimal selector needs — a
shared low-dim routing space, sub-block max-pool summaries, a chunked-causal streaming index, an exact
outlier side-channel, and per-query admissible certificates with escalation — into one selector
(`CausalCascade`), and measures which components pay off. Full record: `FLOOR_PROGRAM.md` § P7.

**Certificates are sound.** `test_ccc_certificates.py` pins zero violations of *certified ⇒ selection ==
the parent-index-tie-broken top-κ under the routing metric* on **both** clustered and random geometry (the certificate uses an
admissible bound over unprobed IVF cells + a search-truncation check; the cascade owns its centroids so the
probed set and radii match faiss exactly). Fire-rate is geometry-dependent: **0.89 clustered / 0.50 random**
at 1M (benign certifies; adversarial escalates). The full cascade runs end-to-end to **12M (980 ms, 6.67 GB,
single-head)** — heavier than the plain IVF kernel (139 ms), the honest cost of the certificate GEMMs,
streaming rebuilds, and the 4× sub-block index.

**Which component rescues which regime** (`ccc_quality.py`, needle-block recall at 64K, unit-norm background):

| regime | ivf (block) | +sub-block | +outlier |
|---|---|---|---|
| benign (coherent span) | 1.00 | 1.00 | 1.00 |
| **isolated** (unit-norm needle) | 0.00 | 0.05 | 0.05 |
| spike c=2 (modest high-norm) | 0.23 | 0.38 | **1.00** |
| spike c=4 | 0.35 | 1.00 | 1.00 |
| spike c=8 (large) | 1.00 | 1.00 | 1.00 |

**Reading.** Sub-block granularity rescues large spikes; the outlier channel *uniquely* rescues the moderate
c=2 spike (0.38→1.00) that sub-block still washes out; **isolated unit-norm needles stay hard for every tested
cheap selector (0.05)** — the grounded-probe obstruction in miniature, not an unrestricted indexing lower
bound. No tested component escapes it; the honest boundary is named, not hidden.

**The trained routing space — the P2 rebuttal** (`routing_space.py`, real Qwen keys). P2 concluded "low-rank
routing is a bust (5–14%)" from an *untrained random* projection. Measured on real keys:

| projection (d_r=16) | block-selection Jaccard vs full-d |
|---|---|
| untrained random (the P2 control) | 0.32 |
| PCA (unsupervised) | 0.46 |
| **trained (listwise KL)** | **0.65** (0.77 at d_r=32; held-out code doc 0.58) |

So low-rank routing is **not a bust** — a trained d_r=16 projection roughly doubles the untrained one and
generalizes across register. **But** the honest boundary: driving the real model, the d_r=16 projection
*collapses NIAH to 0.00* — 0.65 Jaccard ranks blocks approximately but is too lossy (and centroid-vs-cumulant
metric-mismatched) to drive attention losslessly. Low-rank routing is viable for *approximate ranking*, not
yet for lossless selection at d_r=16. Cross-layer transfer (`transfer_matrix`): a per-layer projection scores
0.78 on its own layer but only **0.41** median off-diagonal — the first *measurement* of the transfer the
analytic "÷5" assumed.

**Cross-layer sharing — the ÷5 measured** (`longctx_share.py`, Qwen2.5-0.5B, budget 0.12). The first
measurement of what `router_variants.py` only asserted:

| arm | n | NIAH | two-hop | route / prefill |
|---|---|---|---|---|
| per-layer full-d routing | 8K | 1.00 | 1.00 | **0.61** |
| per-layer full-d routing | 32K | 1.00 | 0.67 | 0.16 |
| share from layer 0 | 8K | **0.00** | 0.00 | 0.02 |
| **share from layer 4** | 8K | **1.00** | **1.00** | **0.06** |
| **share from layer 4** | 32K | **1.00** | 0.89 | 0.04 |

**Reading.** Per-layer routing costs **~59% of prefill at 8K** — right at DSA's 58%, because attention is a
small fraction of a 0.5B forward and the per-layer router (block stats + argsort mask) is heavy. **Sharing the
selection from a mid layer (donor=4) cuts it to ~6% with NIAH and two-hop preserved** — a measured ~10×
reduction that validates and exceeds the analytic ÷5. But **the donor choice matters**: sharing from layer 0
collapses NIAH to 0.00 (early layers route positionally, not by content) — the analytic claim assumed any
layer transfers; only mid layers do. So the lever that makes the selector cheap is **cross-layer sharing from
the right donor**, not the low-dim projection (which is too lossy at d_r=16).

**Honest scope.** Single model at 0.5B; the cascade rig result is single-head + synthetic keys (a cost/soundness
result, not a real-model speed claim); the trained-projection real-model result is a *negative* at d_r=16 (the
projection ranks but does not preserve retrieval); sharing quality is measured at one trial-budget (two-hop
0.67–0.89 shows the multi-hop sensitivity). What this settles for the assessment: SubQ's selector, if isomorphic
to CCC, has a routing share that is single-digit % **once shared from a mid layer** and preserves single-needle
retrieval while sagging on isolated/multi-hop — the NIAH≫MRCR split they report. If their share is large, they
have not solved it (explaining the gated access).

## Zero-attention memory — the other corner, measured — (`fastweight*.py`)

P0–P7 measured the SELECTION corner (route to the few keys that matter, at read time). P8 builds the
COMPRESSION corner — a fixed- or growing-state memory written at inference time (the Mamba/DeltaNet/Titans
family, SubQ's "zero attention") — as small EXACT reference implementations (d ≤ 128, no training) and
measures them against predictions from the companion Lean development (`Substrate/Inference/*.lean`,
sorry-free): **five have a machine-checked anchor** (P1, P2, P4, P5, P6); P3 — the load-bearing
selection/compression split — is a purely empirical prediction with no theorem. Recall is the standard
associative decode (argmax over stored value embeddings). These are mechanism measurements, not a trained LM.

**P1 — the READ rule sets the capacity class** (`softmax_capacity`, RetrievalMarginRecognition.lean; the
linear side is now proved too — `read_capacity_le_dim` / `rank_d_read_wall` + a tightness witness,
LinearReadCeiling.lean — so the contrast is machine-checked on both sides, not softmax-only). Over
the SAME stored pairs, the zero-attention linear read `o = S q` is rank-d capped; a softmax read over the
same keys holds far past m=d (measured to m=512; `softmax_capacity` gives the exponential capacity):

| read | m=8 | m=64 | m=128 | m=256 | m=512 |
|---|---|---|---|---|---|
| **linear** (d=32) | 1.00 | 0.82 | 0.45 | 0.16 | 0.05 |
| **softmax** (d=32) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **linear** (d=64) | 1.00 | 1.00 | 0.98 | 0.76 | 0.33 |

Recall collapses near m≈d for the linear read and holds far past it for softmax — capacity is a property
of the READ, not the stored substrate. This is why "drop attention" replaces a *function*, not a component.

**P2 — the write rule is coherence control** (`capacityBound_antitone`, SearchTradeoff.lean — separation
raises capacity; the delta rule maintains it). On exactly orthogonal keys the delta rule is exact to m=d
(so are additive and gated_delta — orthogonality makes any outer-product write exact). Under overload
(m>d, mild coherence):

| m | additive | delta | gated_delta |
|---|---|---|---|
| 64 | 0.85 | 0.82 | 0.64 |
| 128 | 0.55 | 0.49 | 0.34 |
| 192 | 0.30 | **0.33** | 0.24 |
| 256 | 0.22 | **0.25** | 0.18 |

**Honest, nuanced:** the delta rule's advantage is real but *small* on random keys and only in deep overload
(m≥192); at m≤d additive is already within capacity, and on orthogonal keys all three rules are equally exact.
The delta rule's distinctive win is same-key conflict resolution (P4), not overload capacity per se.

**P3 — write-time vs read-time relevance (the load-bearing result; no theorem — empirical)** (n=512, d=64):

| regime | surprise-gated fixed memory | selection (attention) |
|---|---|---|
| needle salient **at write time** | 0.95 | 1.00 |
| needle salient **only at read time** | **0.10** | **1.00** |

**Reading.** The gate keeps only the top-10% of writes by write-time surprise. A needle salient *at write
time* (an off-distribution key) is in that top tier and is kept (0.95); a needle salient *only at read time*
is i.i.d. with the background, so its surprise rank is uniform and the gate keeps it with only ~0.1
probability (hence recall 0.10) — while selection over the full kept stream recovers it (1.00). The mechanism
is exactly *compression ≠ selection*: the write rule must commit to what to keep *before the question exists*,
so no write-time signal can retain a needle only the read-time query makes relevant. (The gate keeps ~52 < d=64
facts, so this is a write-time-commitment failure, not a capacity-overflow one.) It is why a zero-attention
model will keep posting near-perfect NIAH (write-salient) and sag on query-only / multi-hop retrieval.

**P4 — a same-key conflict needs a tag** (`tag_resolves_conflict`, ContinualLearning.lean). Write one key,
two values: additive averages (0.42 / 0.25 — neither cleanly), delta keeps only the latest (0.00 / 1.00),
an episodic tag k⊕bucket (a *salient* tag, scaled to 2× the key norm so the two tagged keys are
well-separated) recovers **both** (0.95 / 1.00; with a plain unit-norm tag it is 0.60 / 1.00). One map cannot
store two values for one key (`same_input_conflict_unservable`); the tag makes them distinct keys.

**P5 — a fold breaks a fixed memory; growth restores it** (`fold_not_hopfield` + `detectability_is_a_fold`).
A mid-stream distribution shift (keys move to a fresh, near-orthogonal random region), pre-shift recall as
the post-shift burst grows:

| post-shift writes | fixed (gated) pre-shift | slot-birth pre-shift | slot-birth #slots |
|---|---|---|---|
| 8 | 0.90 | 0.95 | 8.3 |
| 48 | 0.40 | 0.75 | 8.7 |
| 192 | **0.10** | **0.65** | 8.7 |

The fixed baseline here is a *forgetting-gated* memory (gated_delta, decay 0.98): as the post-shift burst
grows it fades the pre-shift facts (0.90→0.10) — partly the forgetting gate, and, in deep overload
(204 facts ≫ d=48), capacity. The slot-birth memory grows a partition at the fold and keeps the pre-shift
slot untouched (0.65). `fold_not_hopfield` proves the fold's selected state is *discontinuous in the driver*,
so no fixed-energy continuous descent tracks it; growing the state is *one* remedy (a larger fixed state or a
branch jump are others the theorem doesn't exclude) — and the one measured here.

**P6 — the composition law** (`chain_le_weakest`, RetrievalMarginRecognition.lean). With per-hop rates
measured **independently** (each hop its own fresh noisy cue): ρ1=0.51, ρ2=0.49, so the composition
prediction ∏ρ=0.25. `chain_le_weakest` proves **∏ρ ≤ min hop** (0.25 ≤ 0.49 ✓) — the theorem is about the
product. The *measured joint chain* (feeding hop-1's noisy output into hop-2) is 0.15 — it tracks ∏ρ and sags
below it because chaining propagates hop-1's error. So the multi-hop sag the selection rig measured holds
for this corner too (NIAH≫multi-hop), but the machine-checked part is ∏ρ ≤ min hop, not the measured chain.

**Honest scope.** d ≤ 128 reference implementations, synthetic keys/values, recall via associative decode —
mechanism measurements, not a trained language model, and no wall-clock claims. The Lean theorems are
Layer-A structural results (finite-dimensional, idealized); they rule out corners and characterize the write
side, they do not hand you hyperparameters. Five predictions have a machine-checked anchor and reproduced
(P2's overload-capacity edge for the delta rule is weaker than folklore, reported as measured); P3, the
load-bearing selection/compression split, is empirical (no theorem).

## The trained comparison — does a learned write gate close the compression gap? — (`p9_*.py`)

**Bottom line.** Zero attention works where relevance is fixed *at write time* and fits the state
(within-capacity recall, write-salient needles) and fails, by capacity and not by tuning, where the query
decides relevance *after* the write (read-time-only, past-capacity, multi-hop). The training-dependent half of
the recipe — a learned write gate + a future-prediction loss — does **not** move that boundary (D2/D4).
Training sharpens the selection/compression split; it does not close it. The measurements:

P8 measured the compression corner with **hand-built, untrained** memories and found the load-bearing split:
write-time compression cannot serve read-time-only relevance (recall 0.10 vs selection's 1.00). P8 could not
reach the **training-dependent** half of the zero-attention recipe — a *learned* write policy plus an
auxiliary future-prediction objective ("rethink the objective function"). P9 trains a small micro-LM
end-to-end whose token-mixer is swappable between the two corners, at matched state, on MQAR, and measures
that half directly. Four mixers, all `attn_fn(q,k,v)→o` on the shared `ssa_demo.Block` backbone (d=128, head_dim
**dh=16** = the DeltaNet state dim; SSA budget **κ≈dh**): `dense` (selection, κ=n), `ssa` (selection at budget
κ), `deltanet` (compression — a differentiable delta-rule scan `S_t=S_{t-1}+β_t(v_t−S_{t-1}k_t)k_tᵀ`, `o=q̂S`,
with an optional *learned* write gate β_t=σ(w_g·k_t)), `linear` (the simplest compression write: additive, no
erase, no gate). Recall = masked-CE accuracy at query positions; multi-query MQAR (min(m,16) queries, the proven
`ssa_checkpoint` recipe); a long first curriculum stage so every mixer groks the retrieval op (vanilla linear
is slowest — ~2.5k steps, no erase term); 2 seeds, mean reported.

**D1 — trained capacity vs load (the frontier).** Recall vs #pairs m at fixed state (dh=16):

| mixer | m=2 | m=4 | m=8 | m=16 | m=24 | m=32 |
|---|---|---|---|---|---|---|
| **dense** — selection (κ=n) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **ssa** — selection (κ≈dh) | 1.00 | 1.00 | 1.00 | 0.98 | 0.94 | 0.91 |
| **deltanet** — compression (state dh=16) | 1.00 | 1.00 | 1.00 | 0.85 | 0.75 | 0.68 |
| **linear** — compression, no gate | 1.00 | 1.00 | 0.99 | 0.96 | 0.92 | 0.89 |

**Reading.** Trained selection (dense) is flat at ~1.0 across the whole sweep; SSA at a matched budget κ≈dh
holds nearly flat (a gentle dip only at the tightest overload). Both **compression** variants hold *within*
the state (m≤dh) and degrade past it — training does NOT dissolve the rank-dh wall P8 measured untrained (it
moves under training, it does not vanish). An honest surprise on *which* corner degrades faster: at these mild
overloads (m≤2·dh) with trained near-orthogonal keys, **the delta rule (DeltaNet) degrades *more* than
additive linear** — its erase-before-write forgets the older pairs that later get queried, where additive
accumulation keeps them all (with interference). P8's delta-rule advantage was measured only in *deep* overload
(m≫d) on random keys; it does not carry to this trained, mild-overload regime. Frontier: `p9_frontier.png`.

**D2 — read- vs write-salient at overload (m=32 > dh), the headline (trained P3).** Does a *learned* write
gate + the JEPA aux loss lift DeltaNet at overload?

| regime | ssa (selection) | deltanet no-gate | deltanet +gate | deltanet +gate+aux |
|---|---|---|---|---|
| **read-salient** (stock MQAR) | 0.89 | 0.59 | 0.68 | 0.61 |
| **write-salient** (reserved marker keys) | 1.00 | 1.00 | 1.00 | 1.00 |

**Reading — the learned gate does not close the gap, for two opposite reasons.** On **write-salient** (the
keep-worthy pairs use reserved marker keys, identifiable at write time), the no-gate delta rule *already*
solves it (≈1.0) — training shapes the ≤dh marker keys to be robust, so an explicit write gate adds nothing. On
**read-salient** (stock MQAR, the query is unknown at write time), the gate and the aux loss leave DeltaNet
near its capacity floor (≈0.6, a small and seed-noisy bump over the no-gate baseline) — **far below selection**
(SSA ≈0.9, dense ≈1.0). So the specific training-dependent ingredient P9 set out to test — a learned write
gate — does not close the gap: where keeping is possible training already does it, and where relevance is
read-time-only no write policy approaches selection. This is the trained mirror of P8's 0.10-vs-1.00 split.
Figure: `p9_gate.png`.

**D3 — trained 2-hop composition (trained P6).** Single chain among distractors, chain recall vs load (4
layers; a progressive-load curriculum is what groks 2-hop at this scale — cold-start does not):

| mixer | m=2 | m=3 | m=4 | m=6 | m=8 |
|---|---|---|---|---|---|
| dense | 1.00 | 1.00 | 0.91 | 0.98 | 1.00 |
| ssa | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 |
| deltanet | 1.00 | 1.00 | 0.99 | 0.95 | 0.84 |

**Reading.** Selection (dense, ssa) chains the two hops cleanly across the sweep; the compression corner holds
at low load but **sags as distractors raise the capacity pressure on top of the chaining** — consistent with
`chain_le_weakest` (the composition compounds the per-hop loss). Selection chains; compression sags.

**D4 — the JEPA aux-weight λ sweep (read-salient DeltaNet+gate, overload).** The "rethink the objective
function" ingredient, swept: λ=0→0.64, λ=0.3→0.57, λ=1→0.64 — **flat in λ**. The future-prediction auxiliary loss does not lift the
read-time-relevance wall (reported as measured; λ was not tuned to force a story).

**Honest scope.** Still synthetic MQAR (not natural language), but TRAINED end-to-end — the half P8 could not
reach. d=128 micro-LM, head_dim 16, one RTX 4080; recall at query positions; no wall-clock claims. The
measured verdict sharpens rather than closes the split: the learned write gate + future-prediction aux — the
training-dependent half of the zero-attention recipe — help exactly where a write-time signal already exists
(and there training alone suffices) and cannot manufacture one where relevance is read-time-only. Selection's
advantage on read-time relevance survives training.

---

## The summary-only floor — what the partition costs vs what the summary costs (2026-08-22)

**Module:** `ssa/bound_floor.py` · **Tests:** `ssa/tests/test_bound_floor.py` (19, each section
carrying a negative) · **No GPU, no model, ~2 min on CPU.**

### The proposition: Samuelson is the tightest summary-only bound

Samuelson's inequality is not merely valid but **attained**, by one point at
`μ + s√(m−1)` and `m−1` points at `μ − s/√(m−1)`. Verified numerically at `m ∈ {2,4,8,16,64,256}`:
mean error `≤ 7e-18`, sd error `≤ 2.2e-16`, max deviation equal to the bound to `<1e-9` in every case.

So any bound strictly below `⟨q,μ_c⟩ + √((b−1)·qᵀΣ_c q)` is violated by a block whose summary the
router cannot distinguish from the one it read. **No admissible bound computable from `(μ_c, Σ_c, b)`
alone prunes more.** The prune gate's non-necessity is therefore a property of summary-only routing,
not slack in that particular gate — every further improvement must come from a better *partition* or
from *reading keys*.

### The decomposition it makes measurable

The tightest admissible bound that exists at all is the oracle `U_c = max_{k∈c}⟨q,k⟩` — not a router
(computing it reads the block), but the floor the **partition** imposes on any correct
branch-and-bound. Clustered synthetic keys, `n=4096`, `d=64`, `B=64` k-means blocks, 120 queries per
row, query noise 0.15. **All four bounds are admissible, so recall is 1.000 in every cell** — cost is
the discriminator, not accuracy.

| spread | oracle | ellipsoidal | Samuelson | summary ÷ floor |
|---:|---:|---:|---:|---:|
| 0.02 | 89.0 | 187.6 | 720.3 | **8.1×** |
| 0.05 | 81.0 | 453.6 | 975.4 | 12.0× |
| 0.10 | 74.6 | 1340.2 | 2165.0 | 29.0× |
| 0.20 | 70.2 | 3343.4 | 3685.8 | 52.5× |
| 0.40 | 64.1 | 3946.1 | 4045.3 | **63.1×** |

At the two headline geometries (200 queries, isotropic row included):

| geometry | oracle | ellipsoidal | isotropic | Samuelson |
|---|---:|---:|---:|---:|
| benign (clustered, spread 0.15) | 71.4 | 2678.0 | 4061.8 | 3261.0 |
| isotropic (random unit) | 65.3 | 3923.7 | 4096.0 (100%) | 4034.2 |

### Three readings

1. **The routability programme has a measurable ceiling.** Driving geometry benign moves the summary
   price from **63× to 8×** the partition floor, monotonically — and not to 1×, which the tightness
   proposition says it cannot. The regularizer's measured 26.5% → 4.2% is real and it is bounded.
2. **Reading keys is worth ~4× at benign geometry** (187.6 vs 720.3 at spread 0.02). `ρ_c` is not a
   minor sharpening of the bound; it is most of the distance to the floor. First absolute
   justification for the anisotropic refinement — `anisotropic_bound.py` compares the two bounds to
   each other and never to what is achievable.
3. **The geometry is a fact about summaries, not partitions.** The partition price is nearly flat in
   the spread (89.0 → 64.1) while the summary price moves by a factor of six.

### Scope — stated because the result is easy to over-read

k-means blocks on synthetic clustered keys at one `(n,d,B)` and one query-noise level: the
adaptive/IVF regime, **not** the contiguous-position blocking of the flat kernel. The oracle is a
reference, not an achievable router. The floor is about **lossless** selection, so a budget-κ lossy
router may sit below it and SSA's does. The proposition itself is geometry-free and carries none of
these caveats.

Not measured here: real key banks (the Qwen/Gemma caches), the effect of `B`, and whether the
plateau near 8× is a property of the geometry family or of k-means. Each is a one-line change to
`bound_floor.sweep`.

### Provenance

Instantiates `Universal/Potential/PartialScore.lean` · `score_error_ge_of_reach_split` and
`not_exactOnReach_of_reach_split` — a single score standing for a fibre of completions carries an
error floor set by the objective's spread across that fibre. Partial object = the summary,
completions = blocks carrying it, objective = the block's true max logit. **The instantiation is
prose, not machine-checked**; see `docs/substrate_math_imports.md`.

### Correction and extension (same day)

**Correction — `ρ_c` is query-independent, so the ellipsoidal bound is summary-only too.** The first
write-up called it "reading keys", which is wrong: `ρ_c = max_j‖Σ_c^{-1/2}(k_j−μ_c)‖` does not
depend on `q`, so it is precomputed at index-build time and stored as **one extra scalar per block**.
The hierarchy is therefore about **summary size**, not summaries versus keys — which makes the
finding stronger, not weaker: at spread 0.02, one extra stored scalar takes the bound from **8.7× to
2.5×** the floor. The `ρ_c` premium shrinks with spread (3.4× at 0.02, 1.0× at 0.40): the anisotropic
refinement pays exactly where the geometry is already benign.

**Two algorithms tested against the floor. Both refuted; both informative.**

1. **Champion-seeded incumbent** — store `t` query-independent representatives per block (the points
   farthest from `μ_c`, the paper's own "centroid routing is blind to outliers" turned into storage)
   and seed `s★` with `max_c⟨q, champ_c⟩` before opening anything. **Exactly 0% saved at every
   geometry and every `t`.** Reason: B&B opens blocks in decreasing `U_c`, so the first block opened
   is usually the target and sets `s★` to the true maximum immediately. **Cost here is
   bound-driven, not incumbent-driven** — which is what the floor decomposition already said, since
   the oracle differs from Samuelson only in the bound.
2. **Axis-aligned box in a shared PCA basis** (`2d` scalars/block, `O(d)` query, exact support
   function). **Worse than the ellipsoid nearly everywhere** — 797.8 vs 218.4 keys at spread 0.02,
   and no pruning at all (4096) at spread ≥ 0.20. `min(ellipsoid, box)` buys **8%** at the most
   benign geometry and nothing elsewhere. Blocks are tight clusters whose principal directions differ
   from the global ones, so a shared frame is a loose body.

**The reading that survives, and it improves the paper's claim.** At benign geometry the shipped
ellipsoidal bound reads **5.33% of keys against a floor of 2.10%** — within **2.5×** of the
partition's own limit. There is not much room left *in the bound*; the remaining lever is the
**partition**. Both refuted ideas were attempts to tighten the bound further, and the floor says why
they had little to find.

**Formalized (`~/substrate`, `54a207fd7`).** `Universal/Potential/AdmissibleBound.lean` proves the
abstract skeleton, axiom-pure: `a_dropped_part_holds_no_member_above` (the drop loses nothing),
`a_higher_bound_drops_a_subset` + `dropped_card_mono` (**a tighter bound drops a superset of the
parts** — the paper's qualitative "cost is governed by tightness", proved),
`familyMax_le_of_isAdmissible` (the maximum is the least admissible bound — the floor),
`no_bound_below_an_attained_one` (an attained bound admits nothing smaller — the abstract form of the
Samuelson proposition), and `min_isAdmissible` (**licensing the implementation's "take the minimum of
the two"**, which had no warrant before). `at_the_floor_a_maximal_threshold_drops_every_part`
predicts result 1 above: at the floor, a maximal threshold drops everything, so a search still
reading parts is paying for bounds above the floor and improving the incumbent cannot help it.

---

## Variance-sensitive mass tree — large synthetic win, no real-Qwen pruning (2026-09-09)

**Module:** `ssa/bennett_mass_experiment.py` · **Record:** `runs/bennett_mass_tree.json` · **Formal
sources:** Substrate `6b3da713a` (trace), `da13ebeba` (full covariance), `9c6b1ad35` (outlier peel), and
`fad55ff82` (Inference consumer) · **Hardware:** RTX 4080 for
the Qwen forward; float64 CPU tree/oracle.

The certified tree now optionally stores the mean squared key radius $s^2$ and evaluates the deterministic
Bennett mass cap using either the safe trace lift $v=\lVert q\rVert^2s^2$ or the exact directional variance
$v=q^TCq$ from a stored full covariance. Evaluation is stable in log space, including a series for
`exp(x)-1-x` near zero, and returns the minimum of Bennett and the existing worst-radius cap. The trace mode
adds one scalar per node; full covariance adds $d^2$ (4,096 at $d=64$) and makes each bound evaluation a
quadratic form. The default reader remains the radius mode. Tests cover zero temperature/query, exact-zero
spread, large temperature, random geometry, causal prefixes, dense-output agreement, and the fact that each
minimum never exceeds the radius cap.

On a deliberately concentrated 8,192-key geometry with rare radius-setting extremes, both Bennett variants
were strictly tighter on all 8,160 sampled query/node pairs. Median excess over exact node log mass fell from
**0.816** to **0.0081** with trace and **0.000026** with covariance. At a certified 10% omitted-mass target,
both reduced mean work from **61.2 to 9.2 blocks** and from **3,884 to 556 keys**; at the much stricter 1%
target the improvement nearly vanished (96.4 to 95.0 blocks).

The real-model control is the important result. For 32 causal queries from the latter half of an 8,192-token
Qwen2.5-0.5B layer-18 head, both Bennett variants again tightened every one of 8,160 node caps and none of the
three caps had an observed float64-oracle underestimate. Trace reduced the cap by a median **0.730 log
units** and full covariance by **6.057**, but median excess above exact node log mass still remained **33.42**
and **28.03**, respectively (radius: 34.04). At fixed 5% block budget every certified omitted-mass upper bound
remained numerically 1. To certify either 10% or 1% omitted mass, all three modes opened **all 98.2 visible
blocks on average** (6,247 keys, including partial causal blocks). Thus even exact within-node covariance does
**not** make exact mass certification sparse on the tested raw post-RoPE Qwen geometry.

This is not a failure of the Bennett inequality or its implementation. Full covariance confirms the active
obstruction is the exponential worst-radius term, not merely the dimension factor in the trace lift. The next
credible exact route needs a substantially better partition or a richer tail summary than two moments plus a
single maximum; approximate routing remains the practical route used by the 10M demo. The 8,160-pair audit is empirical floating-point
verification with a $2\times10^{-11}$ log-space comparison tolerance, not an interval proof, and one
layer/head is not a population claim about all transformers.

The outlier-peel follow-up directly tested that tail hypothesis. Each node deterministically exposes the
$t$ largest Euclidean residuals, scores their mass exactly, recentres the remaining core, and applies its
Bennett cap. On the same Qwen pairs, median log-cap slack was **29.83, 29.33, 28.80, and 27.92** for
$t=1,2,4,8$ with trace moments. Combining $t=4$ with core covariance reached **23.39**, a **11.96-log-unit**
median tightening versus radius. All variants had zero observed node-cap underestimates. Nevertheless every
variant still opened all 98.2 visible blocks for both stopping tolerances, and even the fixed-5% budget's mass
certificate remained numerically 1. The $t=8$ trace summary adds 578 scalars per node at $d=64$; $t=4$ plus
covariance adds 4,418. These are fixed-depth comparisons, not a claim that raw peel caps are monotone; a
running minimum is the safe monotone construction. Small query-independent peeling therefore helps
substantially but does not cure the real-tree certificate, and larger peels approach storing/scoring the
underlying keys rather than a useful sparse summary.

The theorem's force-kept formulation was tested separately at the active causal frontier by promoting each
exposed key's containing leaf block before traversing the certified core. It seeded **3.69, 4.50, 8.34, and
12.63 blocks on average** for $t=1,2,4,8$. At $t=4$, bound evaluations fell from 190.63 to 166.78 on average,
but both 10% and 1% stopping targets still read all **98.16 blocks / 6,247.25 keys**. This block-level
realization preserves the existing attention kernel; it is not the finer individual-key side channel that
the theorem also permits.

---

## Geometry-routed score-tail certificate (2026-09-09)

**Modules:** `ssa/score_tail_certificate.py`, `ssa/score_tail_experiment.py` · **Record:**
`runs/score_tail_certificate.json` · **Fixture:** `/tmp/ssa_bennett_qwen_8192.npz` · **Execution:** CPU
float64 · **Command:**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m ssa.score_tail_experiment \
  --cache /tmp/ssa_bennett_qwen_8192.npz \
  --out runs/score_tail_certificate.json
```

The reference reader uses CCC/IVF-style geometry selection only to seed exact blocks. It independently
certifies every unopened block's attention logits with the direct mean-plus-Euclidean-radius cap, bins exact
unopened key counts under 16 score thresholds, and transports the resulting exponential-mass bound through
the existing TV, selected-to-dense KL, and value-output certificate. Routing-score certification is not
attention-score certification. The implementation preserves arbitrary causal prefixes, reads a partial
boundary block exactly, and falls back to a dense read.

### Qwen-8K checkpoint

The cached Qwen2.5-0.5B layer-18, KV-head-0 geometry uses 32 identical causal query/prefix pairs in every
mode, $\beta=1/\sqrt{64}$, 64-key blocks, and 16 tail levels. The cache has no values; output-bound checks use
the declared deterministic proxy `tanh(K[:,:8])`, while every mass result uses the real cached post-RoPE Q/K.

The original contiguous radius tree at a nominal 10% block budget opens **10.43%** of visible blocks and
leaves **71.30% mean / 74.50% median** actual attention mass, reproducing the reported roughly-70% finding.
For the side-by-side fixed-budget comparison, block-mean routing seeds improve actual omitted mass to
**59.08%** at **683.25 keys / 11.22 blocks** in every mode:

| bound | certified mass upper | hard margin | bound evals | stored scalars | scalar work units | certified |
|---|---:|---:|---:|---:|---:|---:|
| radius tree | 1.000000000000 | — | 20.50 | 33,150 | 45,040 | 0/32 |
| Bennett trace | 1.000000000000 | — | 20.50 | 33,405 | 45,040 | 0/32 |
| Bennett covariance | 1.000000000000 | — | 20.50 | 1,077,630 | 127,696 | 0/32 |
| peeled covariance, $t=4$ | 0.999999999995 | — | 20.50 | 1,160,760 | 132,944 | 0/32 |
| one residual threshold | 0.99999999999988 | 34.54 | 97.16 | 16,642 | 50,729 | 0/32 |
| 16-level tail profile | **0.99999999999885** | **31.81** | 97.16 | 16,672 | 50,744 | 0/32 |

The profile gains **2.73 log units** over a single threshold, but remains nowhere near the required
nonpositive margin. When allowed to continue to a certificate, every mode opens **6,247.25 visible keys /
98.16 blocks** on average at both $\eta=0.10$ and $0.01$. Tail profiles reduce bound evaluations relative to
the tree (97.16 vs 173.81) but do not reduce key reads.

The oracle contrast is the main diagnosis:

- Exact top keys need a median **3.384%** of visible keys to retain 90% of true mass (249.72 mean keys).
- Exact top keys plus one exact residual maximum need a median **19.825%** (1,175.91 mean keys).
- Constructing either oracle scores all 6,247.25 keys, so neither is an implemented router.

Thus the mass is sparse, and several thresholds are genuinely better than one, but the contiguous block
partition plus its admissible mean/radius caps hides the individual-key score tail. On the concentrated
synthetic control all six methods certify $\eta=0.10$ after **214.5 keys / 3.84 blocks**; on the random and
centroid-hidden-extreme controls they revert to the full visible prefix (**1,536.5 keys / 24.5 blocks**).
Across all synthetic and Qwen rows there were **zero mass violations and zero output violations** at the
$2\times10^{-11}$ audit tolerance. Maximum positive mass deficit was $5.6\times10^{-16}$ and maximum output
deficit was below $4.6\times10^{-15}$. These are empirical floating-point checks, not interval proofs.

### Certificate-margin training

**Module:** `ssa/score_tail_training.py` · **Record:** `runs/score_tail_training.json` · **Hardware:** local
RTX 4080 · **Command:**

```bash
python -m ssa.score_tail_training --steps 600 \
  --out runs/score_tail_training.json
```

The controlled 384-key task compares baseline retrieval, the existing non-target $q^T\Sigma_bq$
regularizer, `softplus(M_eta)`, and a hybrid. Evaluation discards the smooth training summaries and rebuilds
hard block means, radii, bands, counts, and margins on held-out noisy queries.

| objective | keys to hard certificate | full-read queries | dense-argmax retrieval | 25%-cap certification | 25%-cap retrieval |
|---|---:|---:|---:|---:|---:|
| baseline | 384.0 | 100.00% | 1.000 | 0.00% | 0.602 |
| variance | 369.0 | 92.19% | 1.000 | 0.00% | 1.000 |
| certificate only | 384.0 | 100.00% | 1.000 | 0.00% | 0.664 |
| hybrid | **247.5** | **29.69%** | 1.000 | **1.56%** | 1.000 |

The hybrid is the only substantial certified-work improvement in this seed, but it still fails the desired
25% operating point on 98.44% of queries. Certificate-only training is worse than the established variance
regularizer here. All hard evaluations had zero mass-bound deficits. No claim is made that optimization finds
a good partition or that held-out geometry generalizes.

There is one correction to the requested test: `softplus(M_eta)` is strictly positive, so it can never be
zero or nonpositive. The implementation tests and reports the exact equivalent boundary
`softplus(M_eta) <= log(2)` iff `M_eta <= 0`; only the latter is the hard deterministic certificate.

The GPU/FlexAttention path remains unchanged. Its integration gate (dense-oracle soundness) passed, but the
Qwen checkpoint offers no sparse certified operating point, so wiring this reference into the GPU kernel
would not yet improve measured execution. Conditional subquadraticity still requires subquadratic evaluated
bounds, opened keys, and tail levels. No real model is assumed to satisfy a favorable profile, and a
probabilistic calibrated bound has not been substituted for the deterministic guarantee.

Verification for the landed prefix:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest ssa/tests -q
240 passed, 41 skipped, 14 warnings in 21.34s

cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error subquadratic_attention.tex
Output written on subquadratic_attention.pdf (50 pages).
```

## Bounded-state recurrent repair (2026-09-10)

**Modules:** `ssa/recurrent_repair.py`, `ssa/recurrent_repair_experiment.py` · **Record:**
`runs/recurrent_repair.json` · **Hardware:** local RTX 4080 · **Command:**

```bash
python -m ssa.recurrent_repair_experiment \
  --cache /tmp/ssa_bennett_qwen_8192.npz \
  --out runs/recurrent_repair.json
```

The architecture keeps the raw KV archive and uses a bounded recurrent state only as a read controller.
Every newly selected key is scored with the **original attention query**. A streaming state containing the
maximum logit, scaled partition sum, and scaled value numerator—`d_v + 2` scalars—merges disjoint batches
exactly into softmax attention on their union. A separate bounded retained-id list prevents duplicate mass;
at fixed round count $R$ and per-round budget $\kappa$, its capacity is at most $R\kappa$. Changing the
routing query is allowed, but changing the attention query would define a new hop rather than recover mass
from the original dense distribution.

The controlled pointer task gives the positive condition the hypothesis needs. One initial routed value is
an address clue for a high-attention target. An 8-scalar controller and one-key reads select ids 0 then 47;
actual retained mass rises from **0.0000008315** after round one to **0.99994845** after round two, and proxy
output error falls from **1.41418** to **0.00005155**. This is not magic reconstruction: when two worlds give
the controller the same first observation but hide the target at different unread addresses, its deterministic
second query is identical in both worlds and cannot solve both at unit budget.

The isolated training test exposes a harder boundary for raw CE. A linear controller maps a one-hot clue to
one of 32 hard-routed addresses; only the selected archive value reaches the downstream classifier. Results
after 300 steps, evaluated with the same exact hard argmax router:

| training path | initial hard top-1 | final hard top-1 | zero controller-gradient steps |
|---|---:|---:|---:|
| raw downstream CE through hard selected value | 0.0625 | 0.0625 | **300/300** |
| explicit route CE | 0.0625 | **1.0000** | 0/300 |
| hard-forward straight-through routing surrogate | 0.0625 | **1.0000** | 0/300 |

This does **not** prove that raw CE cannot improve a full transformer: selected-token attention logits,
residual paths, and shared representations still carry gradients and can reshape routing incidentally. It
does establish that an exact hard selection index supplies no derivative telling a routed-only repair
controller which unopened address should have won. At top-$k$, CE can train scores among already selected
items, but a missed item still supplies no direct routing gradient. A router loss, smooth/straight-through
surrogate, exploration estimator, or teacher residual can directly train that isolated controller. This
does not prevent raw CE from training the continuous tail contribution described below.

On 32 identical causal queries from the cached Qwen2.5-0.5B layer-18/head-0 Q/K fixture, four static
block-mean rounds at 2.5% of visible blocks per round retain mean actual mass **0.1952, 0.2844, 0.3492,
0.4088**. Repeating the first read four times remains at **0.1952**. The disjoint four-round output agrees
with one static read of the same union to **6.7e-16**, so static recurrence buys only the summed budget.
At the same final key count, the exact-top-key oracle retains **0.9584** mean mass, again locating the gap in
routing rather than in the accumulator. The Qwen values are a declared deterministic proxy and this cache
does not contain the user's trained router; these numbers must not be substituted for its reported top-1 or
top-$k$ recall.

The practical experiment for the user's checkpoint is therefore a three-arm, equal-total-budget comparison:
one-shot $R\kappa$ routing, $R$ static disjoint retries, and $R$ state-conditioned retries. Report recovery
conditioned on an initial miss, retained mass, output error/CE, unique keys, duplicate rate, and latency. Train
the controller first with the base and router frozen, comparing raw CE against route supervision or the hard
certificate-margin surrogate, and always evaluate with hard routing. Visible filler tokens are unnecessary;
the repair rounds can be latent. No theorem guarantees a useful first-read clue or training convergence;
the trainable experiment below supplies a clue explicitly and tests fresh held-out address banks.

Verification:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest ssa/tests -q
292 passed, 14 warnings in 36.58s

cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error subquadratic_attention.tex
Output written on subquadratic_attention.pdf (52 pages).
```

## Trainable repair and fixed-state tail correction

**Status:** a trained hard-token-tree controller works on the controlled address task. Fixed-state tail
correction improves short-context Qwen CE, but the full-scale run below shows degradation at 8K/32K and
failed long-context retrieval. These are distinct experiments:
there is no claim that a Qwen GRU discovers address repair, nor that this reproduces the user's router.

### Learned individual-token retries

`runs/trainable_repair/results.json`: three seeds, 600 training steps, 256 training addresses, fresh held-out
banks and fresh random answer labels, 512 queries per seed/length. A first value contains the target's
rotated routing address; its label is stored only at that target. The controller has 64 GRU scalars,
the attention accumulator 42, and duplicate suppression eight ids. It sees no target id or dense scores
at evaluation. Route-supervision logits are dense during training and are charged separately.

| Held-out addresses | Learned answer accuracy | Equal-eight-key one-shot/static control |
|---|---:|---:|
| 256 | 100% | near 16-class chance |
| 1024 | 100% | near 16-class chance |
| 4096 | **1535/1536 = 99.935%** | **6.315%** |

Route CE and 300-step route warmup followed by 300 steps raw CE have the same final success counts.
The continuation preserves the router but cannot improve this isolated discrete policy by CE gradient.
With the clue removed, learned 1024-address accuracy is 6.445%. Raw CE alone stays near chance. The
state-only linear head also stays near chance; it is a control, not a universal limit on state models.
Two four-key reads at 4096 use 1020 node-bound evaluations, **plus 64 final routing candidate scores**
and eight original-query attention scores. `node_evaluations` excludes those final routing rescoring
operations. The immutable tree/archive is O(nd), not fixed-size. Maximum selected-union output difference
against the float32 oracle was 1.43e-6. This is measured numerical agreement, not an IEEE proof.

### Real Qwen head: fixed-state output correction

`runs/tail_state_calibrated/results.json` uses `/tmp/ssa_qwen_qkv_8192.npz`, real Q/K/V from Qwen2.5-0.5B
layer 18, KV head 0. Q and K match the existing Bennett fixture exactly. Centers are fitted on the first
512 keys only; queries 512–3071 train, 3072–4095 validate, and 4096–8191 test, sampled every 16 positions.
All variants share the same 256 final queries and mean 153 exact selected keys (actual selected mass 0.21197).
The append-only state holds 32 counts and 32 value sums, 2080 scalars plus centers.

| Output variant | Test MSE |
|---|---:|
| Sparse only | 0.133598 |
| Centroid tail | 0.110316 |
| Learned query-dependent MLP tail | 0.110164 |
| Validation-calibrated centroid tail (log gain 0.5) | **0.104541** |
| Unweighted omitted-value mean only | 0.110000 |
| Oracle per-cell mass, still unweighted within-cell values | 0.099372 |

Calibration reduces MSE 21.75% and improves L2 error on 98.83% of queries. The MLP's best validation
checkpoint is its first update; it does not substantively improve initialization. Learned mass MAE is
0.6017: useful output correction does **not** imply accurate mass estimation. Oracle cell mass is a
diagnostic, not a proven output-error floor.

### Complete-model raw CE training

`runs/qwen_tail_final/results.json`: frozen bf16 Qwen2.5-0.5B, all 24 layers, 14 query heads and two KV
heads; only 336 per-head log-mass gains train. Two past 64-key blocks plus the causal current block are
exact. Sixteen fixed cells per KV head collect every incoming key/value; selected approximate contributions
are removed before exact substitution. The state is additional information beyond sparse observations.
First-block queries remain dense and later centers use only that completed block.

Train: first 64 nonoverlapping 512-token windows of official WikiText-2 train, 100 Adam steps at 0.05.
Validation: first four official validation windows; select initialization from log gains -2,0,2,4,6 and
best checkpoint every ten steps. Test: first eight official test windows at each length, used only after
selection. Longer-length windows overlap the same document stream across lengths and are not independent
datasets. The protocol is a small fixed slice, not whole-corpus perplexity. Training peak allocation is
7.247 GB on the RTX 4080 with attention activation checkpointing.

| Context | Dense CE / PPL | Sparse CE / PPL | Trained tail CE / PPL |
|---|---|---|---|
| 512 | 2.88720 / 17.94 | 3.57777 / 35.79 | **3.01358 / 20.36** |
| 1024 | 2.83638 / 17.05 | 4.04788 / 57.28 | **3.31427 / 27.50** |
| 4096 | 2.46816 / 11.80 | 4.30722 / 74.23 | **3.96012 / 52.46** |

The 512-token correction recovers 81.7% of the sparse-to-dense CE gap. Transfer at 4096 recovers only
18.9%: length robustness is unresolved. The untrained validation-selected scalar gives test PPL 37.17,
worse than sparse. Earlier disjoint development windows likewise gave sparse 46.35, untrained tail 65.38,
and trained tail 26.12 (dense 21.62), preserved in `runs/qwen_tail_pilot` and `runs/qwen_tail_ce_pilot`.
These negative controls explain why single-head MSE and untuned summaries are insufficient.

Exact scored-key means, derived from the fixed causal budget, are 136.5, 148.5, and 157.5 at the three
lengths (max 192). The persistent aggregate state is 49,920 float scalars across the model, plus 49,152
center scalars and 336 gains. Training materializes prefix states; this is not a constant-memory trainer.
The default flat routing costs O(n²/block). Reference eight-window evaluation seconds at 4096 are dense
0.81, sparse 12.36, and trained tail 20.82: **no speedup is claimed**. The optional existing-tree adapter
bypasses the flat scan; fixed-beam work is conditional and does not certify attention mass.

The saved gains were then evaluated unchanged with SSA's **existing append-only center/radius tree**
(`runs/qwen_tail_tree/results.json`), fanout four, beam 32, individual query tokens, two past blocks:

| Context | Tree sparse CE / PPL | Tree + trained tail CE / PPL |
|---|---|---|
| 512 | 3.57640 / 35.74 | **3.01358 / 20.36** |
| 1024 | 4.04965 / 57.38 | **3.31460 / 27.51** |
| 4096 | 4.31005 / 74.44 | **3.95901 / 52.41** |

This connects the state to hierarchical routing in a complete model, rather than merely accepting an
untested external route format. Routing is over blocks with per-token queries; individual-token **leaves**
are tested in the separate recurrent controller experiment. No gains were tuned against these test runs.
Eight-window 4096 evaluation takes 131.57 s tree sparse and 138.83 s tree with tail. This launch-heavy
adapter is a correctness/quality reference, not the optimized 10M streaming kernel or a speed result.

### Reproduction and limits

```bash
python -m ssa.trainable_repair_experiment --steps 600 --seeds 0,1,2 \
  --lengths 256,1024,4096 --queries 512 --out runs/trainable_repair
python -m ssa.tail_state_experiment --cache /tmp/ssa_qwen_qkv_8192.npz \
  --steps 1000 --cells 32 --seed 0 --out runs/tail_state_calibrated
python -m ssa.qwen_tail_demo --context 512 --steps 100 --test-examples 8 \
  --official-splits --eval-contexts 1024,4096 --out runs/qwen_tail_final
python -m ssa.qwen_tail_demo --context 512 --test-examples 8 --official-splits \
  --eval-contexts 1024,4096 --router tree \
  --load-gains runs/qwen_tail_final/results.json --out runs/qwen_tail_tree
```

Qwen uses cached model/tokenizer and WikiText-2, with runtime downloads disabled. The real-head extraction
fixture is local; its SHA256 is recorded in the result. Base commit for these uncommitted-source runs is
`3382c0dfffbf6c4d8244902033b08cc1dd5f31c8`; each JSON records experiment source hashes. Later changes add
checkpoint loading and the tree adapter without overwriting the original measurements. The public final
JSON contains all 336 trained gains; `.pt` checkpoints are local regenerable artifacts excluded by git.

The paper gives a self-contained signed kernel-error identity and bound. No new Substrate theorem is
claimed: its read-chain accounting is energy/displacement accounting, not a node-operation bound.
The learned tail is not an admissible mass cap and never enters the deterministic reader as one. Causal,
full-selection, gradient, and dense-oracle tests validate implementation invariants, not learned generalization.
There is no guarantee of useful clues, convergent training, dense-equivalent output, or 10M quality for
this architecture, and no benchmark on the user's 0.998-recall checkpoint.

Verification of the delivered implementation:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest ssa/tests -q
310 passed, 14 warnings in 34.66s  (RTX 4080 / CUDA enabled)
269 passed, 41 skipped, 14 warnings in 21.59s  (CPU-only sandbox)

cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error subquadratic_attention.tex
Output written on subquadratic_attention.pdf (55 pages).

git diff --check
PASS
```

## RTX 6000 full-corpus and long-context tail evaluation

**Result: negative length transfer.** The saved 512-context gains improve full-corpus quality at 512/4K,
but worsen sparse-only quality at 8K/32K. This overturns any interpretation of the small pilot as a working
long-context correction. All runs completed; the negative result is not an OOM or failed execution gate.

**Artifact:** `runs/kaggle_tail_v1/ssa_tail_fullscale.json`; exact remote source snapshots and manifest are
in the same directory. **Notebook:** `jonsmirl/ssa-tail-fullscale-rtx6000`, version 1, private, ARC3 attached,
internet OFF. **Hardware:** RTX PRO 6000 Blackwell Server Edition, 94.97 GiB GPU memory, 176.88 GiB visible
host memory. Torch 2.10.0+cu128; pinned offline Transformers 5.12.1 and FAISS 1.14.1 wheels. No FAISS GPU
search is used. Driver elapsed time is **1754.4 s (29.2 minutes)**.

Protocol: unchanged frozen Qwen2.5-0.5B and 336 saved gains, all 24 layers and 14 query heads. The entire
official WikiText-2 test stream contains **298,938 tokens**. Nonoverlapping windows reset context and
state; each length includes the partial last window, and NLL is summed and divided by the actual number
of predicted tokens. Window-boundary transitions are excluded. This is our stated tokenization/windowing
protocol, not a claim of direct comparability to other published WikiText perplexities. There is no fitting
or test-driven selection in this run; the previously inspected first test windows remain part of the full
corpus, so this enlarges rather than replaces the holdout.

| Context | Windows / scored targets | Dense CE / PPL | Sparse CE / PPL | Frozen tail CE / PPL |
|---|---|---|---|---|
| 512 | 584 / 298354 | 2.84114 / 17.14 | 3.56388 / 35.30 | **2.98248 / 19.74** |
| 4096 | 73 / 298865 | 2.49299 / 12.10 | 4.30522 / 74.09 | **4.05773 / 57.84** |
| 8192 | 37 / 298901 | 2.44504 / 11.53 | 4.35865 / 78.15 | **4.59922 / 99.41 — worse** |
| 32768 | 10 / 298928 | 2.40206 / 11.05 | 5.20952 / 183.01 | **5.56623 / 261.45 — worse** |

Tail beats sparse CE on 584/584, 62/73, 7/37, and 1/10 paired windows respectively. Token top-1 accuracy
(not retrieval recall) follows the same reversal:

| Context | Dense token accuracy | Sparse | Tail | Whole-corpus seconds: dense / sparse / tail |
|---|---:|---:|---:|---|
| 512 | 45.61% | 37.33% | 43.67% | 5.32 / 83.44 / 102.74 |
| 4096 | 49.54% | 29.43% | 31.32% | 2.71 / 106.04 / 124.84 |
| 8192 | 50.08% | 28.59% | 25.34% | 2.67 / 132.20 / 150.94 |
| 32768 | 50.68% | 22.33% | 16.55% | 3.56 / 160.24 / 178.98 |

The batched tree starts every token from a disjoint cover of **completed past blocks**. Only those nodes
and their descendants can be searched; future-bearing summary nodes never enter the candidate forest.
Radius caps use the existing outward guard. Fixed fanout four and beam 32 keep the tested-slot count
conditional on logarithmic height; beam pruning is still approximate and not a mass certificate. It is a
new batched traversal, not a promise of bitwise route equality to every earlier traversal under pruning.
All three quality modes use the same model and causal data; sparse and tail share this routing algorithm.
Exact attention still scores two past 64-key blocks plus the causal current block, maximum 192 keys.

Tail counts/value sums now advance by query chunks, rather than materializing all prefix summaries at
once. Vocabulary projection is also chunked, with exact target-count-weighted CE. The raw KV archive and
hidden states still grow with sequence length; this is not a fixed-total-memory transformer. Whole-corpus
peak allocations are at most 1.63, 1.69, 1.75, and 2.21 GB at the four lengths. This reference remains
substantially slower than dense attention on the RTX 6000; no speed win is inferred from the bounded
selected set.

### Long-context retrieval and execution

Nine fixed direct-word semantic probes use depths 0.1, 0.5, and 0.9 at each length. They all query the
same `walnut` fact against four candidates; this is a limited mechanism test, not a broad retrieval suite.

| Context | Dense correct | Sparse correct | Tail correct |
|---|---:|---:|---:|
| 8192 | 3/3 | 2/3 | 1/3 |
| 32768 | 3/3 | 2/3 | 0/3 |
| 131072 | 1/3 | 0/3 | 0/3 |
| Total | **7/9** | **4/9** | **1/9** |

Every mode executes the complete transformer at **131,072 tokens**. Tail time is 91.0–91.1 s per 128K
probe, sparse 82.9–83.0 s, dense 2.6–2.8 s; peak allocation is **5.807 GB**. The attached model config has
`max_position_embeddings=32768`, default RoPE, theta 1e6. No YaRN or positional rescaling is applied.
Thus 128K failure includes base-model positional extrapolation and is not solely a router/state result;
the tail's deterioration is already present within the configured 8K/32K range. These tests do not update
or repeat the earlier 10M capacity claim, which used a different execution/router/YaRN configuration.

### Reproduction and verification

```bash
python kaggle_tail/build_notebook.py
kaggle kernels push -p kaggle_tail -t 10800
kaggle kernels status jonsmirl/ssa-tail-fullscale-rtx6000
python kaggle_tail/watch.py
kaggle kernels output jonsmirl/ssa-tail-fullscale-rtx6000 -p runs/kaggle_tail_v1
```

Base commit is `35c2c0b69338d67e9301da798ea1d04361b6d6a2`; the manifest records the exact uncommitted
test-run source hashes. All twelve source/input SHA256 checks match the downloaded artifacts. Input token
bundles and generated notebooks remain local/private rebuildable artifacts, not public corpus copies.
The saved gain hash matches the committed small-run gain artifact; no gains were changed.

Remote gate: dense fallback max last-logit delta **0.25** (bf16 tolerance 0.5), CE delta **0.001509**
(tolerance 0.02); chunked projection versus Hugging Face CE delta **3.83e-8** (tolerance 1e-5). All passed.
The tolerances validate approximate bf16 numerical wiring, not mathematical dense identity or an IEEE
rounding proof. CPU tests additionally compare the batched prefix forest to exhaustive routing, test
future isolation at a pruned beam, deterministic ties, and chunk invariance.

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest ssa/tests -q
317 passed, 14 warnings in 38.91s  (CUDA enabled)

cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error subquadratic_attention.tex
Output written on subquadratic_attention.pdf (56 pages).
```

What remains unproved and unmeasured: reliable long-context tail approximation, a training objective that
finds it, preservation of retrieval when correction is enabled, production throughput, and quality on the
user's high-recall router/checkpoint. No deterministic certificate or Substrate theorem predicted this
learned approximation would generalize; the measured failure does not contradict the conditional math.

## Mean-consistent tail and bounded-influence revision

**Development validation, not a new holdout.** Two official validation windows at each length compare
seven fixed candidates, with the same causal tree, exact key budget, base model, and saved gains.
Lengths overlap in token coverage; the validation split has prior development use. The only proposed
influence cap is 0.25; no cap grid or gain refitting is used here.

| Context | Dense PPL | Sparse | Old tail | Actual means, zero gains | Actual means, clipped saved gains | Prototype + 25% cap | Actual means + 25% cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8.70 | 16.71 | 10.63 | 11.88 | 11.93 | 12.14 | 13.16 |
| 8192 | 8.45 | 60.69 | 74.47 | 53.79 | 59.07 | 47.36 | 53.46 |
| 32768 | 10.28 | 180.91 | 246.34 | 210.30 | 214.19 | 145.03 | 169.90 |

The actual-mean estimator obeys a real-arithmetic Jensen **lower** mass inequality but still worsens
32K CE versus sparse-only. The prototype with capped influence wins mean validation CE among the four
revisions and improves sparse-only CE at each length; it sacrifices some of the old tail's 512-context
benefit. This is not recovery of dense quality, a mass upper certificate, or a retrieval result.

`runs/tail_revision_selection.json` freezes that configuration before the version-2 regression test.
The rule is recorded after examining validation: require improvement over sparse at every length,
then minimize the unweighted mean of the three context CEs. Only the two capped variants qualify.
The full test corpus and probes were already inspected in version 1, so version 2 is not a pristine
holdout. No version-2 result is used to select the correction.

### Why this change, and what the math does not promise

The assignment prototypes are selected from the first 64 keys, not the actual means of the keys assigned
to them. On the previously inspected Qwen layer-18/KV-head-0 cache, a 32-query float64 diagnostic finds
that the prototype estimate overstates actual omitted mass on 29 queries: median ratio 30.17, maximum
102.78. Replacing it with actual omitted-cell means gives median ratio 0.0543 and maximum 0.2022,
with no overestimate. These one-head data do not prove the cause of the whole-model failure.

Actual means require counts, key sums, and value sums, with selected contributions subtracted. For a
partition into nonempty cells, uniform-within-cell weights proportional to the exponential of each
cell's mean logit uniquely minimize reverse KL to dense softmax within that coarse family. The exact
KL gap is the log partition ratio. Substrate commit
`206193290e5aa1b0465d518f12625c10952b0953` now supplies the exact projection, unique optimum, refinement,
selected-key summaries, gain accounting, and conditional output/movement theorems. The public proof
specification is included at `docs/coarse_read_projection.md`; the original build request is retained
at `docs/substrate_coarse_tail_request.md` with its completion status.
Neither reverse-KL optimality nor partition refinement guarantees lower output error or model CE.

The selected revision instead retains the prototype estimate and limits its normalized mixture share
to 0.25. This bounds movement from the selected output, not error relative to dense output. A correction
can still point in the wrong direction. The numerical audit replaced subtraction of a large log-mass
offset with a directly bounded scalar mixture, fixing extreme-logit cancellation and empty-tail NaN
gradients. Validation used the earlier, algebraically equivalent cap arithmetic; deployment uses the
stable implementation. Bitwise validation/deployment equivalence is not asserted.

### Artifacts and verification

```bash
python -m ssa.tail_mass_diagnostic --cache /tmp/ssa_qwen_qkv_8192.npz --out runs/tail_mass_diagnostic.json
python -m ssa.tail_revision_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

Artifacts: `runs/tail_mass_diagnostic.json`, `runs/tail_revision_validation.json`,
`runs/tail_revision_selection.json`, and `runs/tail_revision_verification.json`. The base SHA is
`35c2c0b69338d67e9301da798ea1d04361b6d6a2`; source and input hashes identify the uncommitted experiment.
The CUDA suite reports **349 passed, 14 warnings in 56.08 s**, with no skips. Suite timing overlaps
the validation run and is not an isolated performance measurement.

### Reader-weighted training objective (training not run)

Substrate's `ReadWeightedError` distinguishes raw residual norm from error seen by a fixed linear
reader. For Qwen's frozen attention output projection, an exact local target is
`||W_O (o_dense - o_corrected)||^2`, with all heads concatenated before applying `W_O`. Separate
per-head MSE misses its cross-head terms. This measures immediate residual-stream discrepancy;
later normalization, MLPs, and CE remain nonlinear and are not covered by that identity.

Writing `e=o_dense-o_S`, `d=u-o_S`, a training-only shared scalar oracle gate is
`clip(dot(W_O e, W_O d)/||W_O d||^2, 0, rho)`, or zero for a zero denominator. A future learned gate
must use only inference-available summaries, with dense errors restricted to training labels. Multiple
independent head gates require a coupled box-constrained quadratic, not independent scalar optima.
Gate training has not been run. A bounded oracle diagnostic has now tested its local headroom; see below.
Existing transported fitting budgets require assumptions such as
fixed targets, independent convex updates, and exact compatible transport; they do not establish a
guarantee for shared-network SGD or downstream CE.

### Reader-weighted oracle diagnostic

`python -m ssa.reader_weighted_tail_diagnostic` captures all fourteen query heads and two KV heads at
layer index 18 on the first 8,192 official-validation tokens. Sparse and capped outputs share the same
dense hidden inputs and causal routes. It evaluates 8,000 queries after the trivial fully selected prefix.
The oracle chooses one scalar in `[0,1]` multiplying the already capped multi-head correction; it cannot
amplify the correction beyond the cap. All heads are concatenated before applying the actual frozen
output projection. This is not an end-to-end CE evaluation or an inference-time mechanism.

| Output | Mean squared projected L2 error |
|---|---:|
| Sparse | 33.66724 |
| Prototype + cap | 24.55520 |
| Raw-error oracle suppression | 24.50256 |
| Reader-weighted oracle suppression | 24.48491 |
| Joint 14-head feasible oracle suppression | 22.97759 |

The cap improves this local error by 27.07%, but even the dense-informed reader-weighted oracle adds only
**0.286%** improvement. It selects the full capped correction on 88.5% of queries. Cross-head terms
contribute roughly 32.3% of capped projected error, so ignoring them changes the objective substantially;
nevertheless, this particular scalar suppression problem has little remaining headroom. This does not
exclude gains from a different correction direction, richer state, new reads, amplification, or joint
multi-layer training. The extension to fourteen independently controlled head gates, solved jointly
through the full output-projection Gram matrix, reduces projected error by **6.42% versus the cap**.
Raw error increases versus the shared reader-weighted oracle. Thus the small shared-scalar gain must
not be presented as a limit on independent head gating. The feasible numerical box solution takes
30 coordinate sweeps; maximum projected KKT residual is `8.73e-6`, and maximum first-order gap is
`4.70e-6`. These are floating-point diagnostics, not an outward-rounded certificate of an exact optimum.
Dense targets remain oracle inputs, not available at inference; no gate has been trained on these labels.
Artifact: `runs/reader_weighted_tail_diagnostic.json`, with input/source hashes;
three focused oracle-optimality and cross-head tests pass. No new Substrate theorem is needed to run this
diagnostic: the existing fixed-reader algebra and scalar quadratic identity suffice.
The final CUDA-enabled suite including this diagnostic reports **352 passed, 14 warnings in 36.49 s**,
with no skips and no concurrent local model run; its record is in `runs/tail_revision_verification.json`.

## Capped tail: full-corpus improvement does not preserve retrieval

**Measured status:** the validation-selected 0.25 influence limit improves full-corpus perplexity over
sparse-only at every tested length, but strict retrieval deteriorates from 4/9 to 2/9. This is a useful
CE improvement, not a retrieval-safe solution. No further setting was selected from these test results.

Artifact: `runs/kaggle_tail_v2/ssa_tail_fullscale.json`. Private notebook
`jonsmirl/ssa-tail-fullscale-rtx6000`, version 2, completed in **1750.94 s (29.18 minutes)** on the same
94.97-GiB RTX PRO 6000. ARC3 was attached, internet OFF, and the same two pinned public wheels were
installed with `--no-index --no-deps`. All twelve corpus arms and twenty-seven probe arms completed.
The input stream, saved gains, model, selected-key budget, and causal router match version 1. The tail
remains the prototype estimator, not the Jensen estimator; only its applied mixture is capped at 0.25.

| Context | Targets | Dense PPL | Sparse PPL | Original tail PPL | Capped tail CE / PPL | Capped token top-1 |
|---|---:|---:|---:|---:|---|---:|
| 512 | 298354 | 17.14 | 35.30 | 19.74 | 3.15383 / **23.43** | 41.52% |
| 4096 | 298865 | 12.10 | 74.09 | 57.84 | 3.92787 / **50.80** | 32.39% |
| 8192 | 298901 | 11.53 | 78.15 | 99.41 | 4.28877 / **72.88** | 29.03% |
| 32768 | 298928 | 11.05 | 183.01 | 261.45 | 5.04194 / **154.77** | 22.66% |

The 512-context cap loses some original-tail benefit. The 8K improvement over sparse is much smaller
than on development validation. Token accuracy is not retrieval recall. Capped whole-corpus execution
takes 103.98, 126.19, 151.95, and 179.90 seconds at the four lengths, versus dense 5.38, 2.72, 2.68,
and 3.56 seconds. No speedup is claimed. Peak allocation at 128K is 5.807 GB; the new rule adds neither
selected keys nor cell state. The KV archive still grows with context.
Paired capped CE beats sparse on 584/584, 73/73, 23/37, and 9/10 windows; it beats the old tail on
3/584, 56/73, 32/37, and 9/10. The mean gain is not an every-window guarantee.

### Strict retrieval, including the tie

| Context | Dense strict wins | Sparse strict wins | Original tail strict wins | Capped strict wins | Capped gold ties |
|---|---:|---:|---:|---:|---:|
| 8192 | 3/3 | 2/3 | 1/3 | 2/3 | 1 |
| 32768 | 3/3 | 2/3 | 0/3 | 0/3 | 0 |
| 131072 | 1/3 | 0/3 | 0/3 | 0/3 | 0 |
| Total | **7/9** | **4/9** | **1/9** | **2/9** | **1** |

At 8192/depth 0.5, capped `walnut` and `lantern` both score **4.15625**. The existing candidate-order
rule reports this as correct, giving 3/9 ordered successes; it is not a strict win. Version 1 has no
gold ties in any probe arm. At 32K, corpus CE improves while all three capped retrieval probes fail.
At 128K, native unscaled RoPE exceeds the model's configured 32K range, but that cannot explain the
already observed within-range 32K regression. The repeated probes are narrow mechanism tests, not a
broad retrieval benchmark or the user's separate high-recall checkpoint.

### Reproduction, provenance, and mathematical boundary

```bash
python kaggle_tail/build_notebook.py --tail-mode prototype --gain-mode saved \
  --max-tail-share 0.25 --selection runs/tail_revision_selection.json
kaggle kernels push -p kaggle_tail -t 10800
python kaggle_tail/watch.py
kaggle kernels output jonsmirl/ssa-tail-fullscale-rtx6000 -p runs/kaggle_tail_v2
```

The selection artifact is embedded in the run. Base commit:
`35c2c0b69338d67e9301da798ea1d04361b6d6a2`; the manifest identifies the exact uncommitted deployment
sources and inputs. The downloaded source snapshot is the reproduction target; subsequent public
docstring clarification of weak versus strict improvement does not change its numerical algorithm.
All fourteen manifest hashes verify; tokens and gains are byte-identical to version 1. Dense/sparse
per-window losses, accuracies, layouts, and baseline probe records reproduce version 1 exactly.
`python runs/kaggle_tail_v2/audit.py` regenerates `comparison.json`, including paired-window and strict
tie-aware retrieval summaries.
Dense fallback and chunked-loss gates pass with the same measured deltas as version 1. This run is a
frozen regression comparison on a previously inspected test set, not a pristine holdout.

Substrate `206193290` establishes the actual-mean projection and conditional local movement/readout
theorems. It does not make the prototype cap variationally optimal or guarantee its CE, retrieval, or
CUDA behavior. The route-aware follow-up is now formalized at
`68968fc7aae0a024172288f090f25c4d61451324`: exact restricted-read overlap, full executed-trace stability,
signed nonlinear propagation with shared state radii, and strict prediction margins on a supplied
candidate set. Its public specification is `docs/routed_correction_certificate.md`; the original
request is retained at `docs/substrate_routed_perturbation_request.md` with completion status.
This is a conditional theorem, not a certificate already evaluated for these Qwen runs. Uniform
derivative/guard bounds, a valid full-state model, final readout identification, and all runtime costs
still need to be supplied by SSA. It preserves a reference prediction, not correctness or CE, and
does not turn the observed retrieval failures into successes.

## Routed-correction implementation and computed-endpoint fallback

Two implementations answer different questions. `ssa/routed_correction_certificate.py` checks the
conditional nonlinear path certificate from Substrate `68968fc7a`; its concrete provider is a complete
small CPU transformer. `ssa/endpoint_acceptance.py` instead compares two computed model outputs
directly; the Qwen experiment uses this second policy, not uniform Qwen Taylor bounds.

### Analytic-bound complete transformer reference

Artifact: `runs/routed_certificate_reference.json`. The model has eight positions, width four, two
heads, two layers, seven vocabulary entries, token/position embeddings, causal hard top-two attention,
RMSNorm, tanh feed-forward layers, and a final normalized vocabulary projection. A correction reads
omitted-key cell means; those scans are charged. Full-sequence states include all positions. Every
executed insertion-sort comparison is recorded, including ties; actual-state branch replays and
charged union reads handle route changes. Bounds are analytic uniform RMSNorm/softmax/product/tanh
bounds, not estimates from sampled Hessians. Seeded weights scaled by 0.2 and RMS epsilon 0.5 make
this a deliberately bound-friendly toy, not a Qwen analogue with useful certified constants.

Four seeds and strengths `0,0.0001,0.01,0.25,1,4,64` give:

| Check | Measured result |
|---|---:|
| Trials | 28 |
| Algebraic margin passes, before endpoint veto | 18 |
| Accepted nonzero corrections | 14 of 24 |
| Observed prediction-changing proposals | 2, both rejected |
| Endpoint vetoes after algebraic pass | 0 |
| Numerical interval violations / maximum interval deficit | 0 / 0 |
| Maximum Taylor-remainder deficit | 0 |
| Maximum raw radius-tube deficit | 1.07553e-16 |
| Accepted prediction violations | 0 |

The raw tube deficit is within the float64 consistency tolerance, **not exactly zero**. The generic
checker trusts supplied complete states, correct derivatives/branch evaluations, and uniform bound
hypotheses; a provenance string is not a proof. Independent tests stress the analytic provider but do
not prove IEEE arithmetic. Rejection often reflects loose bounds rather than a changed prediction.
Each run charges five Jacobians with 5,120 scalar entries, five forced-branch replays, routing scans,
guards, and read unions; per-trial counts/timing are in the artifact. This implementation has no
subquadratic runtime claim. Its deployed result explicitly falls back to the reference on rejection.

### Qwen: target-free endpoint acceptance, with both forwards charged

Artifact: `runs/routed_acceptance_qwen.json`, status `complete`. Local RTX 4080, frozen cached
Qwen2.5-0.5B in bfloat16, saved 336 gains from `runs/qwen_tail_final/results.json`; no training or
new threshold selection. Each arm uses the existing causal batched tree, two past 64-key blocks,
current causal block, and 16 prototype cells per KV head. The candidate has the previously selected
25% mixture cap. Both independent 24-layer prefills run with `use_cache=False`.

The gate sees no target labels: use the candidate logits only if reference and candidate have the
same unique winner; otherwise retain reference logits. Ties reject. Nonfinite candidates reject and
nonfinite references error. Corpus decisions use all 151,936 vocabulary entries. Gold labels enter
only subsequent scoring. Preservation of the computed reference argmax is **by construction**, not
independent evidence of a path theorem, reference correctness, or recovery of missing information.

The first two nonoverlapping complete windows from official WikiText-2 validation are used at each
length; windows overlap across lengths, and this development split has already been inspected.

| Context | Targets | Reference PPL | Candidate PPL | Accepted PPL | Accepted fraction | Reference = accepted token accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 1,022 | 16.70847 | 12.07519 | 14.90995 | 75.8317% | 44.3249% |
| 8,192 | 16,382 | 60.68602 | 47.52214 | 55.30998 | 54.4622% | 31.2904% |
| 32,768 | 65,534 | 180.91186 | 144.96062 | 166.22768 | 45.2773% | 22.1564% |

All 82,938 scored positions preserve reference argmaxes. The empirical CE benefit is smaller than
the ungated candidate's; rejecting every argmax change also rejects beneficial prediction changes.
Each length charges four full forwards for its two windows, and 8/128/512 logit projections in
256-position chunks. Total measured corpus times are 3.83/41.97/201.85 seconds, including both arms,
gate, and diagnostic scoring. This is not a matched latency benchmark or a speedup claim.

Six fixed walnut NIAH probes use a **four-candidate** gate, not a full-vocabulary guarantee:

| Context | Reference strict wins | Candidate strict wins | Accepted strict wins | Proposals accepted |
|---|---:|---:|---:|---:|
| 8,192 | 3/3 | 2/3 | 3/3 | 2/3 |
| 32,768 | 3/3 | 0/3 | 3/3 | 0/3 |

Depths are 0.1, 0.5, and 0.9. These are a separate same-device paired run: sparse probe scores and
success counts differ from the prior RTX 6000 execution, so its baselines must not be reused here.
The combined corpus/probe experiment charges 24 full model forwards and 603.56 seconds of measured
evaluation work, with peak allocated memory 2.2653 GB. Offline local caches suffice; Kaggle was not
needed. No new dense run was performed. The six probes are not a broad retrieval evaluation.

### Reproduction and remaining work

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.routed_certificate_experiment
python -m ssa.routed_acceptance_demo
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

Both experiment artifacts record base commit `35c2c0b69338d67e9301da798ea1d04361b6d6a2` and exact
source hashes: the new implementation is an uncommitted working-tree addition to that base, not code
already present at the base SHA. All seven Qwen source hashes, its gain hash, and both CPU provider/core
hashes match the measured files. Verification output is recorded in
`runs/routed_acceptance_verification.json`.

The path checker is implemented for the small transformer, but useful uniform Qwen bounds, scalable
derivative evaluation, and verified floating-point margins remain open. The Qwen fallback is not
dual-cache autoregressive serving and does not preserve the sampling distribution or fix wrong
reference predictions. Once both endpoints have been computed, direct comparison avoids the extra
path-certificate work; useful path certificates would need amortization or partial-execution savings
not demonstrated here. Neither the small reference nor the endpoint experiment establishes cheap
retrieval-safe tail correction, a general CE guarantee, or quality preservation at 10M context.

## Shared-memory fitting: capacity, interference, and attention diagnostics

`ssa/span_memory.py` and `ssa/span_memory_experiment.py` compare shared linear value decoders with
cell summaries, motivated by Substrate `22d6b0c4c`'s held-span fitting theorem and `6d7ae4257`'s
single-key interference witness. This tests a proposed implementation family; it does not assume
that Qwen's values are realizable by one linear function of its keys.

### Protocol and mathematical scope

The input is the previously inspected layer-18/KV-0 fixture `/tmp/ssa_qwen_qkv_8192.npz`, containing
8,192 Q/K/V rows of width 64. SHA256:
`5656b2725bbdb29a09c7c13c93126ac3f2cfb44ba0c32e60768ffc49fc54895a`.
Feature normalization, fixed random tanh features, and k-means centers use the first 512 keys only.
Every value decoder sees the same first 4,096 values during fitting; no suffix value enters the
frozen fit. There is no hyperparameter selection on the suffix.

Two representations use the same update comparisons: normalized raw keys plus an intercept, and
fixed random tanh features plus an intercept. These features are **not learned**. Joint updates are
`W <- W + 0.5 pinv(X_batch) (V_batch-X_batch W)`, with relative SVD cutoff `1e-10`; sequential updates
apply normalized delta steps of rate 0.5 in stream order. Batch size is 64. Joint fitting acts on
each supplied batch, **not all historically held keys**. The full held-span theorem must not be
claimed for this update schedule or for inconsistent target values. Full-prefix least-squares
fits are separately labelled oracle capacity diagnostics.

The frozen protocol predicts the unseen second-half values without state updates. The online
protocol admits every intervening value before its query, so it measures causal reconstruction,
not unseen-value generalization. Persistent update batches use a fixed grid; queries inside a
batch fit a temporary copy which is discarded. Additional measurement queries therefore cannot
change the future persistent state, and repeated temporary reads/solver work are charged.

At each of 16 common suffix query positions, all modes select the same two past 64-key blocks by
centroid score and the current causal block: mean 160.5 exact selected keys. Predicted values are
replaced by exact values at selected positions. The main attention comparison supplies the **true
dense attention weights**, isolating value approximation; it explicitly scans all visible keys
and is not a deployable sparse reader. A separate online cell-centroid arm approximates both
weights and values, with count/sum subtraction for selected keys. It is the cell-summary baseline
architecture, not the calibrated 16-cell full-model tail with saved gains.

### Measured negative result for the fitting variants

Mean attention-output L2 error, averaged over the same 16 queries and three seeds:

| Online reconstruction | 8,192-scalar cap | 16,384-scalar cap |
|---|---:|---:|
| Sparse exact selected attention | 2.3583 | 2.3583 |
| Linear batch-joint, oracle weights | 36.7952 | 36.7952 |
| Linear sequential, oracle weights | 3.6549 | 3.6549 |
| Random-tanh batch-joint, oracle weights | 194.1890 | 6.1325 |
| Random-tanh sequential, oracle weights | 3.2521 | 3.2815 |
| Cell means, oracle weights | **1.9987** | **1.9760** |
| Cell means, approximate centroid weights | 2.0924 | 2.0178 |

The raw linear models are identical across seeds/budgets, not independent replications. Random-feature
and cell seeds are 0, 1, 2; they do not produce additional independent documents or query sets.
Oracle weights are not an output-error lower bound: weight errors can sometimes compensate value
errors. No new fitting variant wins this comparison, so no variant was advanced to full-model CE
or retrieval testing. These negative results were retained without a suffix-selected rate/rank sweep.

Frozen suffix relative Frobenius value errors are 14.9875 for linear batch-joint, 1.2232 for linear
sequential, 16.6179/1.7141 for tanh batch-joint, 1.0752/1.1078 for tanh sequential, and
0.8987/0.9051 for cells at the two caps. Good fitting on the latest batch is not good retention:
the report records old-anchor errors separately from current-batch contraction and irreducible error.
Ill-conditioning is also exposed: seed-0 prefix batches reach retained condition numbers about
50,988 for raw features and 115,146 for the smaller tanh representation. The exact-real span theorem
does not assert stable pseudoinverse fitting on such data. These measurements do not exclude better
regularization, learned features, a different fitting schedule, or useful output cancellation.

### Exact obstruction and synthetic regression gates

`linear_obstruction` converts stored floating-point numbers to exact dyadic rationals modulo the
fixed prime 2,147,483,647. On the first 65 fixture positions, the matrix consisting of all 64 key
columns and the first value column has modular rank 65. Consequently its rational determinant is
nonzero: **no real linear key-to-value map fits even that value coordinate on those positions**.
This checks the supplied stored numbers, not unrounded activations. The routine's failure to find
such a minor is explicitly inconclusive; it does not certify realizability. This obstruction does
not rule out approximate attention outputs, richer features, or nonlinear memories.

The synthetic tests reproduce realizable joint contraction, off-span invariance, and the proved
single-key interference witness: candidate residual-norm sum 2 to 2.5 while the selected squared
error falls 1 to 0.25. The candidate squared-error sum rises 2 to 4.25; these two aggregates are
not conflated. Tests also show inconsistent-target error floors and failure on an unseen orthogonal
target despite exact training fit. Protocol gates cover leakage, deterministic ties, full-selection
dense recovery, query-schedule invariance, selected NaNs, and stable sparse normalization when
selected dense-normalized weights underflow to zero.

### Resource accounting and reproduction

Caps cover **representation state**, including normalization/feature parameters and value weights,
or cell centers/counts/sums. Raw-linear state is 4,288 scalars; tanh state is 8,190/16,317 scalars
(63/126 features); cell state is 8,127/16,383 scalars (63/127 cells). Each joint/sequential pair has
identical features, state size and observed values. The comparison does not equate total resident
memory or FLOPs. The full K/V diagnostic archive, materialized feature arrays, solver/batch workspace,
temporary query copies, anchor-error checks and dense truth calculations are additional resources.

Each case charges 2,568 exact query-selected values, 98,312 shared oracle key logits and dense truth
value rows, plus 97,792 key rows scanned while rebuilding the reference routing means. Online state
observes all 8,192 values, with extra temporary-batch processing counted separately. Cell calibration,
assignment and repeated prediction work, update-operation estimates and SVD work proxies are also
reported. Estimates are not allocator measurements or a runtime guarantee.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  ssa/tests/test_span_memory.py ssa/tests/test_span_memory_experiment.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.span_memory_experiment
```

Artifact: `runs/span_memory_comparison.json`; source hashes and fixture hash are embedded. The base
commit is `35c2c0b69338d67e9301da798ea1d04361b6d6a2`, with these new working-tree source files identified
by hash rather than claimed present at that commit. This is a CPU float64/exact-integer diagnostic;
Kaggle and a new model forward were unnecessary. Verification is in `runs/span_memory_verification.json`.
The focused tests report **46 passed in 0.18 s**. The full CUDA-enabled regression suite reports
**449 passed, 14 existing deprecation warnings in 40.93 s**. Both experiment source hashes match
the measured files, and `git diff --check` passes.

## Cell-summary minimax and persistent all-prefix ridge

**Status: implemented and measured; stability improves, but useful tail recovery is not established.**
The experiment in [`summary_recovery_experiment.py`](ssa/summary_recovery_experiment.py) consumes
the sharp fixed-summary recovery and persistent-ridge mathematics described self-containedly in
[`summary_recovery_experiments.md`](docs/summary_recovery_experiments.md). Its reference cores are
[`cell_summary_minimax.py`](ssa/cell_summary_minimax.py) and
[`persistent_ridge.py`](ssa/persistent_ridge.py). No GPU or new model forward is needed for these
cached-head diagnostics. They do not measure perplexity, retrieval accuracy, or efficient inference.

### Matched Qwen measurements

The fixture is `/tmp/ssa_qwen_qkv_8192.npz`, SHA-256
`5656b2725bbdb29a09c7c13c93126ac3f2cfb44ba0c32e60768ffc49fc54895a`: the previously inspected
Qwen2.5-0.5B layer-18, KV-head-0 document. All modes share 16 suffix queries, the same causal
prefixes, and exactly the same selected keys. Entries below average the same queries across
three seeds; seeds do not constitute independent held-out documents.

| Head-output mean L2 error | 8,192-scalar cap | 16,384-scalar cap |
|---|---:|---:|
| Sparse selected attention, locally normalized | 2.3583 | 2.3583 |
| Persistent projected-linear ridge, online | 2.3021 | 2.1338 |
| Persistent tanh ridge, online | 1.9266 | 1.8589 |
| Earlier cell-value prediction, oracle weights | 1.9987 | 1.9760 |
| Unread-mean coefficient decoder, oracle weights | 2.0023 | 1.9804 |
| Median coefficient decoder, oracle weights | 0.9934 | 0.9822 |
| **Selected-only oracle, true weights and zero tail** | **1.0661** | **1.0661** |
| One global value sum, oracle median coefficient | 1.0262 | 1.0262 |

The selected-only oracle is essential: it uses the *true globally normalized selected weights*,
which already require a dense denominator here. Much of the apparent improvement over sparse
attention is available without estimating tail values at all. Ridge does not beat that control;
its tail estimate increases mean error. Median cell decoding improves on the selected-only and
one-global-sum oracles modestly, but is still dense-oracle evaluation, not a deployable router.
None of these comparisons establishes retrieval preservation.

Ridge does fix the severe instability of the tested batch-update rule. At matching tanh features,
joint-update mean errors are 964.5762/194.1890, versus persistent ridge 1.9266/1.8589; sequential
delta controls give 3.5299/3.2521. Those controls have the same features but smaller state because
they do not retain Gram/cross statistics. The artifact also reruns the earlier full-budget controls
with their wider features; these two comparison classes are explicitly separate.

### What the minimax diagnostic says

For fixed cell sums and exact selected values, the optimal arbitrary-decoder worst-case error
over scalar values in `[-1,1]` is the sum of unread coefficient deviations from their cell median.
It is attained by a decoder and by an indistinguishable pair of balanced integer witnesses.
This is a **fixed-observation, full-cube** result, not a lower bound on observed Qwen error or
accuracy. Actual Qwen values are correlated vectors and the witnesses need not be model-realizable.

| Mean scalar-cube recovery radius | 8,192 cap | 16,384 cap |
|---|---:|---:|
| No summary, supplied selected set: omitted mass | 0.74909 | 0.74909 |
| One global value sum | 0.74281 | 0.74281 |
| Geometry-cell sums: sharp optimum | 0.73655 | 0.73464 |
| Same cell sums: unread-mean coefficient decoder | 1.28321 | 1.27573 |

The geometry cells only slightly reduce the worst-case information loss at this route budget.
Across all 96 Qwen query/seed/cap instances, radii range from 0.45205 to 0.99031; **zero** satisfy
radius at most 0.10 or 0.01. These are absolute scalar-cube error targets, **not omitted-mass
certificates**. The median optimum is below the mean-coefficient decoder's worst-case error,
but neither a probability-normalized decoder nor an observed-value optimum is required by the theorem.

### Fitting, validation, and resources

Normalization and feature bases use the first 512 keys. Penalties `{0.01,1,100,10000}` are
compared by frozen value MSE: fit `[0,3072)`, validate `[3072,4096)`. All twelve Qwen family/seed/cap
fits select **100**, which remains fixed for the sum-loss objective as context grows. Suffix
values are not used for selection. After refitting the first 4096 rows, frozen suffix relative
Frobenius errors are 0.93645/0.94160 for projected-linear and 0.90713/0.91133 for tanh. Online
measurements instead admit every value before its query; they are not unseen-value prediction.

The state cap includes normalization/basis, Gram, cross statistics, cached decoder, count and
penalty. Ridge uses **35/63 features and 8,045/16,193 scalar-equivalents**, rather than pretending
that only its decoder is stored. Cell state uses 63/127 cells and 8,127/16,383 scalars. Solvers,
reference archives, materialized feature arrays and simultaneous experimental replicas are extra.

Each case charges 2,568 exact query-selected values, 98,312 shared oracle logits/dense truth
value rows, and 97,792 key rows scanned by reference routing. Ridge admits 8,192 distinct values
but processes 8,648 fit rows including discarded partial-query copies. Validation rereads are
additional. Gram/cross operation estimates, factorizations, reader-gain solves, diagnostic
least-squares fits, old-anchor checks, control SVDs, feature work, cell scans and sorts are itemized.
These are work/storage estimates, not measured allocator peaks or equal-FLOP comparisons.

Online fixed-reader gains span 0.01672–0.15270 and 0.02185–0.20630 for projected-linear at the
two caps; tanh spans 0.01414–0.11176 and 0.01818–0.15074. No mismatch-radius or target-bias
bound is supplied for Qwen, so these are amplification diagnostics, not output certificates.

### Synthetic and numerical checks

Predefined synthetic cases use 256 positions, eight-dimensional keys, four-dimensional values,
eight queries and a 512-scalar cap. They are controls, not favorable replacements for Qwen.

| Synthetic case | Sparse L2 | Selected-only oracle L2 | Median-cell L2 | Projected-linear ridge L2 | Tanh ridge L2 |
|---|---:|---:|---:|---:|---:|
| Realizable linear values | 1.13609 | 1.53563 | 0.88823 | 0.000314 | 0.31791 |
| Independent values | 0.41346 | 0.20722 | 0.13945 | 0.20077 | 0.15643 |
| Concentrated attention, independent values | 1.72320 | 1.34486 | 1.20817 | 1.34392 | 1.34093 |

Across Qwen and synthetic cases there are 120 cell instances and 480 frozen/online ridge
instances. Integer witness and state-cap violations: **zero**. Maximum primal/dual pairing
discrepancy is **1.11e-15**; maximum signed-output identity discrepancy is **2.38e-14** rounded
up. Maximum row-norm-scaled output-bound and universal reader-gain-bound deficits are both zero.
The numerical audit tolerance is `1e-10`; this is not an IEEE or interval proof of softmax/solves.

### Reproduction and remaining scope

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.summary_recovery_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  ssa/tests/test_cell_summary_minimax.py ssa/tests/test_persistent_ridge.py \
  ssa/tests/test_summary_recovery_experiment.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

Artifact: [`runs/summary_recovery_comparison.json`](runs/summary_recovery_comparison.json).
Verification: [`runs/summary_recovery_verification.json`](runs/summary_recovery_verification.json).
The artifact identifies base commit `35c2c0b69338d67e9301da798ea1d04361b6d6a2` plus exact hashes
of the working-tree source files; these new files are not claimed present at that base commit.
The mathematical source build is `940cee72ffb2d8ccd3c78bb70a1c2b5ce06382fb`.

The full suite reports **527 passed, 14 existing warnings in 38.85 s**; the focused run reports
**78 passed in 0.32 s**. Source and fixture hashes match the artifact and `git diff --check` passes.
Tests cover future-data
isolation, validation selection, input immutability, full-selection dense recovery, integer
witnesses, ridge objective/gain/bias identities and the repeated-residual counterexample.
The global-sum control also uses fixed row-order accumulation so extra query measurements do
not change its later output through a different floating-point reduction order.

Still unproved/unimplemented: efficient non-oracle coefficient/normalization construction,
query-uniform accessible residual bounds, model-realizable recovery lower bounds, and end-to-end
retrieval or CE benefit. These results do not justify promoting ridge as a working long-context
tail-recovery solution. They identify normalization and the observation channel as issues separate
from regression stability.

## Partial-coordinate certified attention reads

**Status: a positive logical-read result, not a runtime or semantic-retrieval result.**
The [public experiment report](docs/partial_coordinate_attention.md) gives the complete math,
protocol, commands, provenance, comparisons and work-accounting limits. A CPU implementation
supplies attention-logit intervals from unread key-coordinate extrema to the existing 16-band
score-tail certificate. No new covariance/Bennett/outlier cap or learned predictor is used.

On 16 cached Qwen2.5-0.5B layer-18/KV0 queries with matched causal prefixes:

| Target omitted mass | Initial coordinates / 64 | Key-coordinate reads | Value rows | Combined logical K/V | Certified |
|---|---:|---:|---:|---:|---:|
| 10% | 16 | 83.57% | 78.09% | 80.83% | 16/16 |
| 10% | 32 | 66.47% | 32.95% | 49.71% | 16/16 |
| 10% | 64 | 100% | 9.86% | 54.93% | 16/16 |
| 1% | 32 | 81.16% | 62.32% | 71.74% | 16/16 |
| 1% | 64 | 100% | 32.64% | 66.32% | 16/16 |

At r=32 and the 10% target, actual omitted mass averages 0.8344%, versus a
9.055% certified upper bound. The existing radius reader reads all keys and values.
The exact-score oracle value floor averages 5.94%/27.83% at the 10%/1% targets,
but requires every exact key logit and ignores real seed/batch constraints.

Fixed 128-plus-boundary value budgets average 2.65% of visible keys. Their top-16
coverage reaches 97.27% at r=16 and 100% at r=32, yet actual omitted mass is
20.27%/17.89%; only 0/16 and 1/16 queries respectively certify 10% mass. Token 0
is the true top-1 key in all queries, making 100% top-1 recall trivial sink retention.
These are attention-key diagnostics, not semantic retrieval accuracy.

The 10%-value-cap progressive policy certifies 9/16 queries at 10% mass and 2/16
at 1%, after reading all key coordinates. Failure is reported, not accepted.
Synthetic concentrated/random/adversarial cases are included, with identical queries
across modes. Random geometry largely requires full reads.

Audit: **880 trials, zero violations above 1e-9**, maximum interval deficit
7.11e-15 and output deficit 1.98e-15; zero mass/KL deficits or false strict top-set
certificates. Focused tests: **102 passed in 0.24 s**. Full suite: **652 passed,
14 existing warnings in 46.98 s**. This is not a formal IEEE soundness proof.

The fraction excludes dense summary construction, value-norm scanning, dense NumPy
interval refreshes and oracle computations, all separately reported. The implementation
still scans every key; across n queries this is not subquadratic. No controller was
trained and no GPU speedup, held-out quality or new complete-model result was measured.

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.partial_coordinate_experiment`.
Artifact: [partial_coordinate_attention.json](runs/partial_coordinate_attention.json).
Verification: [partial_coordinate_verification.json](runs/partial_coordinate_verification.json).
Base commit `35c2c0b69338d67e9301da798ea1d04361b6d6a2`; exact uncommitted source and
fixture hashes are embedded in the artifacts.

## Partial-coordinate GPU latency on fresh multi-head geometry

**Status: numerical certificates pass; actual acceleration fails.**
The [GPU implementation and protocol](docs/partial_coordinate_gpu.md) measures a fused-kernel
prototype on the RTX 4080, using two fresh named WikiText-2 test articles, three Qwen layers,
four query heads and four causal prefixes. Each timed call processes two query heads sharing
one KV head, not all 14 heads or complete model inference. There are 96 head queries per setting.

| Initial coordinates | Mass target | Mean requested K/V bytes / dense | Mean value rows / dense | Median sparse call | Median dense BF16 call |
|---|---:|---:|---:|---:|---:|
| 32/64 | 10% | 56.00% | 41.33% | 8.570 ms | 0.079 ms |
| 32/64 | 1% | 81.25% | 75.01% | 9.914 ms | 0.079 ms |
| 64/64 | 10% | 55.14% | 10.28% | 4.783 ms | 0.079 ms |
| 64/64 | 1% | 67.41% | 34.81% | 7.429 ms | 0.079 ms |

Every setting certifies all 96 head queries; all **384 numerical trials** have zero observed
interval, mass, supported-KL or output-bound deficits and no false stops. Full value reads
are needed in 5/96, 37/96, 0/96 and 8/96 cases respectively. The requested byte counts are
masked/gather scalar loads, **not measured physical DRAM traffic**. Sparse calls are slower
than native BF16 in all 192 measured two-head/configuration calls, and slower than the
FP32 dense control in 191/192 calls; that control's median is 1.444 ms. Sparse output
is FP32; native dense output is BF16.

At 32 coordinates and 10% mass, actual omitted mass averages 0.3584%, versus a 5.0157%
upper bound; mean head-output L2 error is 0.00654. The output certificate remains loose,
averaging 1.3643. Attention sink token 0 wins 75/96 head queries, so attention top-key
coverage still must not be called semantic retrieval accuracy.

Timings are warmed, synchronized wall times including host control, sorts, adaptive decisions,
GPU launches and sparse output; index construction and dense oracles are excluded and
reported separately. Including index construction raises medians to 8.998, 10.474, 5.378
and 7.911 ms. This is not a cold-cache bandwidth benchmark or a complete-model serving test.

The implementation uses cached upper-score order and suffix log-masses. It also removes
16-band rounding by summing individual upper caps, and doubles read batches instead of
the CPU reference's fixed increments. Thus fresh-data work differences cannot be attributed
solely to GPU fusion. The certificate/controller is not a single fused device kernel.

Before the sweep, a large-common-logit softmax normalization bug was corrected using
centered scores; CUDA regressions cover both signs at 1e8 and 1e9. Float64 queries are
rejected rather than silently rounded. Float32 guards remain empirical engineering
allowances, not a formal floating-point theorem.

Commands:

```sh
python -m ssa.partial_coordinate_fixture --out-dir /tmp/ssa_partial_fresh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.partial_coordinate_gpu_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

Sources and model/fixture hashes are in [the complete artifact](runs/partial_coordinate_gpu.json).
The [uncached control](runs/partial_coordinate_gpu_uncached.json) covers four matched groups;
[verification](runs/partial_coordinate_gpu_verification.json) records source hashes and tests.
The full GPU-enabled suite passes **700 tests, with 14 existing warnings in 44.64 s**.
No training, perplexity, semantic retrieval, full-model sparse rollout, or Kaggle run is claimed.
The result does not justify scaling this host-adaptive design to larger hardware; a genuinely
device-resident controller and fewer global scans would be a new implementation experiment.

## Fixed-stage device certificate and conditional dense fallback

**Status: the agreed latency stop/go test fails; no crossover through 32K.**
The [device-resident pipeline](docs/device_coordinate_attention.md) uses a fixed sparse proposal,
GPU-side mass decision, selected-value reads on acceptance, and full K/V reads only for rejected
heads. No query-dependent host branch or scalar extraction is used. Rejected proposals' extra
key reads are charged. Both sparse and dense paths use the same warmed CUDA Graph harness.

The sweep reuses the two named 8K articles and adds a nonrepeated natural-text 32K WikiText
test stream excluding those articles. Three layers, four query heads and matched causal prefixes
give 132 head queries, each tested at eight fixed settings: 32/64 initial coordinates, 25%/50%
proposal budgets, and 10%/1% omitted-mass targets. Each call contains two heads sharing one KV
head. This is a frozen dense-geometry attention test, not a sparse-model serving evaluation.

On the growth stream alone, aggregated across the eight settings and six two-head groups:

| Context | Pipeline GPU-event median | Dense GPU-event median | Median paired slowdown | Mean fallback fraction | Mean requested K/V bytes / dense |
|---|---:|---:|---:|---:|---:|
| 8,192 | 0.287 ms | 0.011 ms | 25.80× | 36.46% | 91.28% |
| 16,384 | 0.314 ms | 0.019 ms | 16.26× | 29.17% | 85.16% |
| 32,768 | 0.366 ms | 0.026 ms | 13.78× | 14.58% | 73.31% |

All **1,056 head-query/configuration trials** passed the numerical audit with zero observed
proposal-mass, final-mass, interval or output-bound deficits, no false acceptance and no invalid
or uncertified final result. Nevertheless, there are **zero wins in 528 matched comparisons**
under either synchronized wall timing or CUDA-event timing. Even the best individual event-time
ratio is 11.02× slower than dense. The relative gap narrows with length, but no measured point
approaches parity; extrapolating a future crossover would be unsupported.

Conditional fallback matters: a rejected head reads both its proposal key coordinates and all
dense keys, while reading its values only once. With 32 initial coordinates and a 25% proposal,
accepted K/V requests are about 43.75% of dense; rejected requests are 131.25%. These counts
exclude separately charged summary/intermediate traffic and are not physical DRAM measurements.
Favorable average requested-byte counts alone do not establish favorable runtime.

Fixed-stage means no adaptive host loop, not one monolithic fused kernel. Stable global sorting,
mass reductions, status masks and output merges remain device operations. Build, preparation,
compilation/capture costs are outside warmed replay and separately recorded. The result is negative
even with that favorable amortization. Deterministic real-arithmetic bounds are not replaced by
calibrated probabilities; floating-point allowances still lack a formal IEEE proof.

```sh
python -m ssa.device_coordinate_fixture --out-dir /tmp/ssa_device_growth
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.device_coordinate_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

[Complete artifact](runs/device_coordinate_attention.json),
[source hashes and verification](runs/device_coordinate_verification.json), and
[self-contained protocol](docs/device_coordinate_attention.md).
Focused tests: **45 passed in 2.23 s**. Full GPU-enabled suite: **745 passed, 14 existing
warnings in 44.19 s**, with the new CUDA graph tests executed rather than skipped.
No new model training, semantic retrieval score, larger-context success or Kaggle result is claimed.
The recommendation is to stop this in-GPU design rather than repeat hardware-scale experiments;
this does not prove all deterministic sparse attention, or slower-memory/offloaded KV designs,
cannot succeed.
