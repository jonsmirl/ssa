# Does the Subquadratic (SubQ) math survive scrutiny? — an independent assessment

**TL;DR — the two things the coverage missed.** SubQ launched in May 2026 with a 1,000×-at-12M claim, and in
June brought "receipts": a paid third-party (Appen) evaluation reporting **98% needle-in-a-haystack at 6M and
12M tokens**, widely read as validating the architecture. This rig — an *adversarial* reconstruction built
only from the public claims and the prior art — reaches two conclusions that **neither the journalists nor the
paid evaluator state**:

1. **The 98% NIAH "receipt" tests the easy case.** A needle-in-a-haystack hit is a single, high-margin,
   *benign* target — precisely the regime cheap summary routing is built to ace. The same mechanism
   **collapses** on isolated spikes and multi-hop retrieval, which is exactly why SubQ's *own* numbers split:
   RULER / NIAH (single-needle) ≈ 95–98%, but **MRCR (multi-hop) = 65.9%**, with a 17-point research→production
   drop. The benchmark spread is a *prediction* of the theory — and 98% NIAH@12M says **nothing** about the hard
   regime where a quality-preserving 1,000× actually has to hold. (§ "Why the benchmark spread is the predicted
   shape".)
2. **1,000×@12M exceeds SubQ's *own* scaling.** Fit SubQ's two published speedups (7.2×@128K, 52.2×@1M) to the
   linear rate `n/κ` → κ≈19k → 12M predicts **≈630×**. So 1,000× is **~1.6× above SubQ's own implied rate**, and
   a quality-preserving version provably *forces* a hierarchical indexer the flat NSA/DSA family lacks (an
   elementary counting argument, **machine-checked in Lean**). This is Will Depue's *"the scaling and speedup
   numbers don't line up,"* made exact. (§ "The sharp independent finding".)

Neither is a debunking. **Calibrated verdict: the bet is sound; the strong claims are unproven** — and the
specific places the evidence thins out are nameable. Everything below is the argument and the evidence.

> **Sources & provenance.** The SubQ figures and quotes here are taken from the public launch coverage —
> Michael Nuñez, *"Miami startup Subquadratic claims 1,000× AI efficiency gain with SubQ model"* (May 5, 2026)
> and Will Douglas Heaven, *"A startup claims it broke through a bottleneck that's holding back LLMs,"* MIT
> Technology Review (June 19, 2026) — plus the X commentary they quote (Depue, Rysana, McAteer) and SubQ's own
> technical blog. They are reproduced faithfully against those pieces but are **second-hand**: this rig has no
> access to SubQ's code, weights, or internal numbers. The rig's **own** measurements (`RESULTS.md`,
> `paper/figures/*.json`) stand independently; the SubQ-specific anchors do not, and the load-bearing
> conclusions (e.g. "1,000× is ~1.6× above SubQ's own rate," computed from 7.2/52.2) **move if a cited figure is
> wrong.**

This repository is an **independent-evaluation rig** for Subquadratic's SubQ model. It reverse-engineers the
public mechanism (Subquadratic Sparse Attention) from Subquadratic's stated claims and the prior art,
formalizes the read-side theory, and runs it on real pretrained models — *without* access to SubQ's code,
weights, or production system. `RESULTS.md` is the evidence log; this file is the consolidated verdict.

## Headline verdict — calibrated-skeptical, not debunking

**SubQ's mechanism is real and reproducible; SubQ's *strong framing* — "fully subquadratic," "linear,"
"1,000× at 12M" — is exactly where this rig's own evidence says the claim weakens.** That lands close to the
public skeptic-who-believes-it position (Rysana: *"subquadratic attention done well… odds of it being BS are
extremely low"*) while independently substantiating each specific skeptical objection. The rig reconstructs
the *recipe*, not SubQ's code — so the verdict is **"the bet is sound; the strong claims are unproven,"** not
"they're lying."

## Claim-by-claim

| SubQ public claim | What this rig's evidence shows | Verdict |
|---|---|---|
| "Fully subquadratic / compute grows linearly" | **Robust finding:** ~1,000×@12M is **~1.6× above the linear rate** SubQ's own two points imply (`n/κ`, κ≈19k). A flat per-query-block router *also* ceilings at `b²`, but **how high depends on the block size** — `b≈22` (a fragile 2-point fit) → 490×, yet a realistic `b=64`→4096× or `b=128`→16384×, so the "exceeds the ceiling" version is brittle and the **linear claim is the one to lean on**. Either way a *growing*-speedup, quality-preserving 1,000× **forces a sub-linear-per-query index** — `flat_router_work`/`subquadratic_forces_skip` (an elementary counting argument; its formal version is machine-checked in the *separate, unshipped* Lean development, see `README.md` § Scope) — the component the NSA/DSA family lacks. The rig has that index only in *reference* form: the flat→IVF crossover is ~3M (measured), the treecode regresses naively, and its pruning is approximate + benign-conditional (RESULTS § "treecode reality check") | **Refutes the strong form, and indicates what 1,000× requires.** Not "subquadratic," but **"1,000× is above SubQ's own rate, and a quality-preserving version forces the hierarchical index the flat family lacks"** — identified, but unbuilt at scale and benign-conditional: *necessary, not beaten.* |
| "1,000× at 12M tokens" | reached only by *analytically extrapolating* the recovery floor `√(2 log κ/d)` on planted needles; largest real synthetic run = 262K, largest real-model = 2–4K | **Unverified extrapolation**, contingent on benign geometry surviving to 12M — and inconsistent with SubQ's own scaling (below) |
| Speedup "7.2× @128K → 52.2× @1M" (Appen later measured **56× vs FlashAttention**, at long context) | reproduced ≈20–22× @256K vs FlashAttention (apples-to-apples, router build included); crossover ~8K matches SubQ's admitted short-context overhead. Appen's 56× is at large `n` — consistent with the *growing* curve (52.2×@1M), not a separate claim | **Supports the curve shape, at both ends** |
| "Content-dependent selection by meaning" | reproduced via cumulant routing; works for benign / coherent-span targets, collapses for isolated needles (1.00 → 0.00 by 256K) | **Supports the mechanism, exposes the omitted caveat** |
| "Built on open-source weights" (first guessed Kimi/DeepSeek; Whedon later confirmed a **Qwen** finetune) | the construction pipeline (swap dense→SSA + adapt) reverse-engineers exactly this; the Gemma frozen-swap is a sparse-attention retrofit of open weights — +13.2 ppl swap cost, recovering to +1.2 with equal-budget adaptation. (The rig's own real-key probes use Qwen-16K deep-head keys — the same base family SubQ used) | **Strongly supports "it's a sparse-attention finetune," and quantifies the quality cost** |
| "RL stage to use distant context" | maps to the routability regularizer / co-training; training drives lossless B&B cost ~6× down — but the `capacity_search_tension` theorem says geometry-shaping trades against capacity | **Plausible, reproduced — but not "free"** |

## The sharp independent finding: SubQ's 12M headline exceeds SubQ's own scaling — and a flat router cannot reach it

Two models, both anchored on *only* SubQ's two published speedups (7.2×@128K, 52.2×@1M):

- **(1) Linear / attention-bound** (`speedup = n/κ`, the regime where routing is cheap relative to the
  `κ`-key read): κ ≈ **19,100**, consistent across both points. Extrapolating to 12M → **≈630×**, so SubQ's
  ~1,000× is **~1.6× above its own rate.**
- **(2) Saturating, per-query-block router** (`speedup = n/(κ + n/b²)` — NSA/SubQ amortize routing over a query
  block, so the router costs `(n/b)²` and the speedup **ceilings at `b²`**): fitting both points gives κ≈17,900
  and block size **`b≈22`, a hard ceiling `b² ≈ 490×`**. Under this fit SubQ at 12M is already *past* the
  router↔attention crossover (~8.7M tokens), and **~1,000× is above the flat-router ceiling entirely.**

The two models bracket one conclusion: **1,000× at 12M is not supported by SubQ's own measured points** — 1.6×
too high if routing stays cheap, and above the hard ceiling if it doesn't (the observed slight sub-linearity,
`α≈0.95` over 128K→1M, is the router term already biting). Model (2) gives the *mechanism*: a flat router's
speedup saturates at `b²`, and lifting that ceiling needs either **larger blocks** (coarser summaries → the
MRCR / multi-hop quality collapse SubQ already reports) **or a hierarchical, sublinear-per-query index** (fine
blocks, no ceiling). So a *quality-preserving* 1,000× **provably forces a hierarchical indexer** —
`flat_router_work` / `subquadratic_forces_skip`, machine-checked in the unshipped Lean development
(`README.md` § Scope). (`b≈22` is a fragile 2-point fit; the robust
claim is "above SubQ's own extrapolation," and the *necessity* claim is model-independent — any fixed-block
flat router saturates.)

This is Depue's *"the scaling and speedup numbers don't line up,"* made quantitative and given its mechanism —
**but it proves what 1,000× would *require*, not that this rig delivers it** (the treecode is unbuilt at scale;
see the "fully subquadratic" row and `README` § Scope).

## The July 2026 interview: the anchors drift toward the rig, and the strong claim quietly narrows

A July 2026 interview (The New Stack, Lardinois; plus the full video transcript) moves several things, none in
SubQ's favor on the strong framing:

- **The @1M speedup anchor now has three values** — 52.2× (May coverage), "like a 60×" (spoken), 64.5× (June
  model card, *compute* not wall-clock). The rig's provenance note flagged exactly this risk; here it *helps*
  the finding. Across the range, the implied per-query budget is κ ∈ [16.3K, 20K], and the linear extrapolation
  to 12M lands **600–780×** — every variant of SubQ's own numbers falls short of the headline, and Whedon now
  says **"almost 1,000×"** on tape. The sharp finding is robust to their figure drift. (The 64.5× FLOPs vs 52.2×
  wall-clock split at the same n implies ~24% non-attention overhead — the router term the rig measures.)
- **The claim narrowed.** May's "1,000× efficiency gain" became July's *"attention compute reduction at close to
  1,000×"* — per-attention-layer FLOPs, hedged — the walkback the serving-paradox section predicted. Whedon's own
  Amdahl argument against hybrids ("80% non-quadratic layers → max 5×") also cuts against his headline at short/mid
  context, where MLP compute dominates; it is only honest at 12M where attention swamps everything.
- **The no-trade-off claim, on tape.** *"It is novel to do so **without quality trade-offs** in particular on
  long context retrieval."* Stated that baldly, it is contradicted by SubQ's *own* model card (MRCR 65.9% with a
  17-pt research→prod drop) and by the rig's now-measured multi-hop collapse. "No trade-off on long-context
  retrieval" is true only if "retrieval" is quietly the single-needle regime.
- **SubQ now makes the rig's router-cost argument.** On GLM 5.2: *"at 1 million tokens, 58% of the prefill
  latency comes from that selection mechanism… a quadratically scaling component."* That is the flat-router
  ceiling the rig's Lean `subquadratic_forces_skip` formalizes — deployed reflexively against a competitor. The
  reflexive question goes unasked both times: **what fraction of SubQ's *own* prefill is its selector at 1M and
  12M, and what is its complexity class?** Their growing-speedup curve is only possible with the hierarchical
  indexer the necessity proof requires (≈ the rig's IVF router, now measured — previous section).
- **The pretraining "insight nobody's published" is the rig's §8.2 in miniature.** *"Really long multi-million-
  token pre-training extends the model's ability to generalize mid-context post-training to super-long-context
  versions of that task."* That is the staging ladder (`staged_extension.py`, §8.2: extend → cheap adapt →
  extend) plus manufacturing-routability (§7) — training creates the benign geometry that makes summary routing
  lossless — stated as company strategy.
- **Hardware ↔ the memory floor.** Whedon's aside about *"chip types other than NVIDIA that offer differentials
  in SRAM/DRAM relative to FLOPs"* matches the floor program's finding that the `n·κ` floor is a **memory** floor,
  not a compute floor (the 17 GB mask wall, the gather-bound IVF path). A κ-key sparse read is exactly the
  workload where SRAM-heavy parts beat FLOPs-heavy ones.
- **Provenance flag (unreconciled).** This assessment's claim table says Whedon "later confirmed a **Qwen**
  finetune"; the July article says the company *"has not said which open-weight model it started from."* One of
  these is stale — the second-hand Qwen anchor should be re-sourced before it is leaned on. (Also updated from
  the model card: RULER 95% → **99.12%@128K**, "**<100B parameters**", "**~1T tokens** continued pretraining".)

Net: the interview strengthens three rows (figure-drift robustness of the scaling finding, the explicit
no-trade-off claim vs SubQ's own MRCR, first-party confirmation of the geometry-manufacturing recipe) and opens
one new thread (the memory-floor hardware framing). It moves nothing that changes the verdict.

## Why the benchmark spread is the predicted shape — and why the Appen NIAH@12M "receipt" doesn't settle it

This is the rig's most useful contribution, so state it plainly. The June coverage treats the **Appen
evaluation — 98% needle-in-a-haystack at 6M and 12M tokens** — as the receipt that "validated their
architecture." It validates *less than it looks like*, and the rig predicted exactly this.

NIAH and RULER are **single-needle, high-margin, benign** retrieval: one planted fact, coherent, with a large
score margin over the distractors. That is the precise regime where cheap summary routing is *supposed* to be
near-perfect — and the rig reproduces it (98–100% holds for benign targets). But the same mechanism **collapses
when the target is an isolated spike or requires multi-hop chaining**, because no benign span lifts the block's
summary score (the rig measures the collapse: 1.00 → 0.00 by 256K on isolated needles). So the *theory predicts
the shape of SubQ's own table*:

| SubQ / Appen result | regime | theory says |
|---|---|---|
| RULER@128K **95%**, NIAH@6M/12M **98%** | single benign needle | near-perfect — the easy case |
| MRCR v2 **65.9%** (research **83** → prod **65.9**, −17 pts) | multi-hop / isolated | sags — the predicted failure mode |
| LiveCodeBench **89.7%**, SWE-Bench **81.8%** | local-ish, short-range | frontier, unaffected |

The "suspiciously perfect" NIAH numbers and the MRCR sag are **not** independent facts to be explained away one
by one — they are the *same* benign-geometry prediction, read at two ends. The consequence for the headline:
**98% NIAH@12M demonstrates speed-at-long-context on the easy regime; it does not establish the quality the
quality-preserving 1,000× claim actually rests on** — which lives in the multi-hop / isolated regime MRCR
already shows sagging. A genuinely decisive receipt would be **multi-hop retrieval (MRCR-style) at 6M–12M**,
not single-needle NIAH. SubQ has not published that measurement — but the rig now *builds* it at both levels
(`multihop_analysis.py` synthetic + `gemma_ssa_eval.two_hop_accuracy` real-model), and it comes out exactly as
predicted: benign single needles hold at ~1.00 while a **mixed 2-hop chain (one benign + one isolated hop)
collapses to 0.02**, with measured chain ≈ ∏ρ (the composition law). The NIAH↔MRCR split is no longer only
argued from SubQ's table — it is reproduced on the rig's own mechanism. (Falsifiable prediction, now
self-tested; `RESULTS.md` § "Multi-hop retrieval".)

## The serving paradox, answered

*"If it's 1,000× cheaper, why gate access?"* — because the 1,000× is a best-case analytic floor on benign
geometry, while the realized system has a superlinear router, an `O(n)`-per-step decode path, and faithfulness
that declines with needle distance (block-hit ~78% at a tiny budget, falling over half-context distance on
random keys). The `O(n)`-per-step decode cost — the part prefill speedups hide — the rig now **measures**
(`ivf_decode.py`): with a fixed κ the IVF-routed decode step is *flat in n* (~0.53 ms from 1M to 12M) while a
dense step grows with the prefix (2.6 → 29.6 ms), so the decode advantage is real **but only ~55× at 12M, not
1,000×**, and it rests on the same benign, add-only-index assumptions. The production speedup is well below the
headline — exactly why you would meter it. (The Magic.dev parallel — 100M tokens, claimed 1,000×, $500M raised,
then public silence — is the cautionary base rate.)

## The floor program: both levers demonstrated — and the gap now closed end-to-end (measured)

A follow-on program (`FLOOR_PROGRAM.md`, phases P0–P5) drove the rig's own kernel toward the `n·κ` floor and,
in doing so, demonstrated the two things SubQ's 1,000× requires. (1) **Lower the floor:** co-training drops
κ_min from 25% to **0.4%** of n (60×) — the mechanism behind SubQ's RL/co-training stage, and the answer to
"is a tiny κ viable" — *on benign geometry only* (diffuse/adversarial stays at a 50% floor = no speedup).
(2) **Close the gap:** a FAISS-IVF block-router (O(√nb)) is now **wired into the FlexAttention kernel and
measured end-to-end** (`ivf_kernel.py`) — not projected. A full **12M-token forward runs in 139 ms and 6.55 GB,
single-head**, on the 16 GB card; the argsort maskbuild that dominated the projected gap collapses to
**sub-millisecond** (the IVF emits `kv_idx` directly), and the residual gap to the floor is a **measured 2.9×**,
down from the 128× the flat kernel paid — the sub-linear indexer the necessity proof (`subquadratic_forces_skip`)
said was *required*, now shown driving a live kernel to ~the floor. So SubQ's claim is **achievable in practice
on one GPU under exactly the benign-geometry condition the floor analysis names** — pinned, not refuted. (What
remains: this is single-head and synthetic-keys — a **speed** result; multi-head, real-model keys at the 12M
endpoint still need more than a 16 GB card. Selection quality at long context is the separate P1/P3/P4 story.)

## What this rig cannot settle

It reconstructs the recipe, not SubQ's code. It cannot say whether SubQ solved the router-cost problem this
repo couldn't wire up, what their real kernel does, or their true production speedups / pricing. **"The bet is
sound and the strong claims are unproven"** is the honest verdict — calibrated scrutiny, not a debunking.

---
*Evidence and reproduction: `RESULTS.md` (per-experiment tables) and the `ssa/` reference implementations.
Scope and known limitations of the rig itself: `README.md` § "Scope — what is and isn't demonstrated."*
