# Does the Subquadratic (SubQ) math survive scrutiny? — an independent assessment

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
| "Fully subquadratic / compute grows linearly" | a flat **per-query-block** router costs `(n/b)²`, so its speedup **ceilings at `b²`**. Fitting SubQ's two published points to `speedup = n/(κ+n/b²)` gives κ≈19k and `b≈22` → a ceiling **≈490×**, so ~1,000× at 12M sits *above what a flat router can reach*; lifting the ceiling needs either coarse blocks (the MRCR quality collapse SubQ reports) or a **hierarchical index**. So a quality-preserving 1,000× **provably forces a hierarchical (sublinear-per-query) indexer** — `flat_router_work`/`subquadratic_forces_skip`, machine-checked — the one component the NSA/DSA family lacks. But the rig has that index only in *reference* form: a wall-clock benchmark puts the flat→hierarchical crossover at ~1M tokens, a *naive* 2-level *regresses* 7× by 2M, and its pruning is approximate + benign-conditional (RESULTS § "treecode reality check") | **Refutes the strong form, and proves what 1,000× requires.** Not "subquadratic," but **"a flat router ceilings below the claim, so 1,000× forces the hierarchical index it lacks"** — proven necessary and identified, but unbuilt at scale and benign-conditional: *necessary, not beaten.* |
| "1,000× at 12M tokens" | reached only by *analytically extrapolating* the recovery floor `√(2 log κ/d)` on planted needles; largest real synthetic run = 262K, largest real-model = 2–4K | **Unverified extrapolation**, contingent on benign geometry surviving to 12M — and inconsistent with SubQ's own scaling (below) |
| Speedup "7.2× @128K → 52.2× @1M" | reproduced ≈20× @256K vs FlashAttention (apples-to-apples, router build included); crossover ~8K matches SubQ's admitted short-context overhead | **Supports the curve shape** |
| "Content-dependent selection by meaning" | reproduced via cumulant routing; works for benign / coherent-span targets, collapses for isolated needles (1.00 → 0.00 by 256K) | **Supports the mechanism, exposes the omitted caveat** |
| "Built on open-source weights" (Kimi/DeepSeek finetune) | the construction pipeline (swap dense→SSA + adapt) reverse-engineers exactly this; the Gemma frozen-swap is a sparse-attention retrofit of open weights — +13.2 ppl swap cost, recovering to +1.2 with equal-budget adaptation | **Strongly supports "it's a sparse-attention finetune," and quantifies the quality cost** |
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
`flat_router_work` / `subquadratic_forces_skip`, machine-checked. (`b≈22` is a fragile 2-point fit; the robust
claim is "above SubQ's own extrapolation," and the *necessity* claim is model-independent — any fixed-block
flat router saturates.)

This is Depue's *"the scaling and speedup numbers don't line up,"* made quantitative and given its mechanism —
**but it proves what 1,000× would *require*, not that this rig delivers it** (the treecode is unbuilt at scale;
see the "fully subquadratic" row and `README` § Scope).

## Why SubQ's benchmark spread is the predicted shape, not (necessarily) cherry-picking

SubQ's own table — **RULER@128K 95%** (single-needle, easy) vs **MRCR v2 65.9%** (multi-hop, hard), with a
~17-point research→production drop on MRCR — is exactly what the benign-geometry theory predicts: cheap summary
routing stays near 1.00 for coherent single targets and degrades on isolated / multi-hop retrieval, where no
benign span lifts the block score. The "suspiciously perfect" numbers are the **benign regime**; the multi-hop
sag and the research→prod gap are the **predicted failure mode** showing through — not (necessarily) fraud. The
rig names it as a falsifiable prediction about where SubQ-style models lose accuracy (`RESULTS.md`).

## The serving paradox, answered

*"If it's 1,000× cheaper, why gate access?"* — because the 1,000× is a best-case analytic floor on benign
geometry, while the realized system has a superlinear router, an `O(n)`-per-step decode path, and faithfulness
that declines with needle distance (block-hit ~78% at a tiny budget, falling over half-context distance on
random keys). The production speedup is well below the headline — exactly why you would meter it. (The
Magic.dev parallel — 100M tokens, claimed 1,000×, $500M raised, then public silence — is the cautionary base
rate.)

## The floor program: both levers demonstrated (in projection)

A follow-on program (`FLOOR_PROGRAM.md`, phases P0–P5) drove the rig's own kernel toward the `n·κ` floor and,
in doing so, demonstrated the two things SubQ's 1,000× requires. (1) **Lower the floor:** co-training drops
κ_min from 25% to **0.4%** of n (60×) — the mechanism behind SubQ's RL/co-training stage, and the answer to
"is a tiny κ viable" — *on benign geometry only* (diffuse/adversarial stays at a 50% floor = no speedup).
(2) **Close the gap:** a FAISS-IVF block-router (O(√nb), 0.93–0.97 selection agreement) projects to put the
kernel within **1.10× of the floor at 12M** (a 128× → 1.1× gap) — the sub-linear indexer the necessity proof
(`subquadratic_forces_skip`) said was *required*. Its wall-clock is **measured on the GPU** (faiss-gpu, no
transfer): the flat `(n/b)²` GEMM OOMs at 4M while the IVF router runs linearly to **8M (62 ms)** — the only
router past the flat OOM — landing the kernel ~at the floor. So SubQ's claim is **achievable in principle under exactly
the benign-geometry condition the floor analysis names** — pinned, not refuted. (Both are cost projections on
the 16 GB card; the faiss-gpu 12M wall-clock is unmeasured.)

## What this rig cannot settle

It reconstructs the recipe, not SubQ's code. It cannot say whether SubQ solved the router-cost problem this
repo couldn't wire up, what their real kernel does, or their true production speedups / pricing. **"The bet is
sound and the strong claims are unproven"** is the honest verdict — calibrated scrutiny, not a debunking.

---
*Evidence and reproduction: `RESULTS.md` (per-experiment tables) and the `ssa/` reference implementations.
Scope and known limitations of the rig itself: `README.md` § "Scope — what is and isn't demonstrated."*
