# Subquadratic Sparse Attention (SSA)

Content-routed sparse attention for long contexts — **exact softmax over a *selected subset* of keys** (a
bounded, dropped-mass error versus full attention), with the theory that says when it is sound and the
experiments that measure it. (Read the **Scope** note under "Selected results" before the headlines.)

**This repository doubles as an independent-evaluation rig for Subquadratic's SubQ model** — the consolidated
verdict on what the reverse-engineering establishes (and cannot) about SubQ's public claims is in
[`SUBQ_ASSESSMENT.md`](SUBQ_ASSESSMENT.md): *the mechanism is real and reproducible; the strong framing
("fully subquadratic," "1,000× at 12M") is where the evidence weakens* — including a quantitative check that
SubQ's 12M speedup exceeds what its own two published points imply, so a *quality-preserving* 1,000× provably
requires a hierarchical indexer (the component the NSA/DSA family lacks) that this rig has only in reference form.

![Attention compute vs context length: dense O(n²) rising at the top, our flat-router kernel (measured) staying
below but speedup-capped by the argsort BlockMask build, the measured faiss-GPU IVF router dropping the kernel
onto the n·κ floor, and SubQ's two published points plus its 1,000×@12M claim — measured (solid) + projection
(dashed), one 16 GB GPU.](paper/figures/unified_scaling.png)

Dense self-attention costs `O(n²)` in the sequence length `n`. SSA replaces it with a mechanism that, for each
query, (i) routes to a small content-dependent set of key **blocks** using only per-block summary statistics,
(ii) adds a local window, and (iii) computes **exact** softmax attention over the selected keys. The per-query
work is `O(κ)` in a fixed budget `κ ≪ n` plus a sublinear routing cost, so the layer runs in `O(n√n)` flat, or
near-linear with a hierarchical router.

The full writeup — algorithm, all the supporting math, and the measured results — is in
[`paper/subquadratic_attention.pdf`](paper/subquadratic_attention.pdf) (compiled PDF), with the
[LaTeX source](paper/subquadratic_attention.tex) and a [Markdown copy](paper/subquadratic_attention.md)
alongside it. This README is the short version plus how to run everything.

## The idea in one screen

A long-context attention layer is, operationally, an **associative recall**: a query must place most of its
softmax weight on the few keys that matter. For a target key with score margin `Δ` over `μ` competing
distractors, the recovered weight is

```
w★ = σ(βΔ − log μ),     β = 1/√d
```

so the target is recovered when `βΔ > log μ` — degradation is only **logarithmic** in the distractor count.
(Here `β = 1/√d` is the *nominal* softmax temperature that makes the law dimension-free; in the routing
experiments `β` is treated as a tunable inverse-temperature knob — the identity holds for any `β`, and measured
routing quality peaks near `β ≈ 2`, not at `1/√d`, so no experiment instantiates the literal `1/√d` value.)
Selection's whole value is that it cuts `μ` from `n` to a fixed budget `κ`, which makes retrieval **flat in
context length**. The catch is that the selector must contain the target in its budget while reading only
summaries. SSA routes each block `c` by the second-cumulant (tempered) score

```
r_c(q) = ⟨q, μ_c⟩ + (β/2) · qᵀ Σ_c q
```

— the block mean plus the variance of the logit across the block. The variance term is what lets routing see
an in-block outlier that the mean alone washes out. Summary-only routing is provably lossless **when** the
key **geometry is benign** (off-target blocks have small spread `qᵀΣ_c q`) — a *sufficient* condition (the
prune gate fires), not a necessary one — and training, or simply a finer block size, can manufacture that
geometry. None of this is free in the worst case: **cheap and lossless** selection cannot hold for arbitrary
keys (proved); length-robustness is the third axis — the *trilemma* (at most two of the three) — see the
paper's impossibility argument.

## Repository layout

```
paper/        the paper (LaTeX + Markdown)
ssa/          the Python package (NumPy/PyTorch reference implementations + experiments)
ssa/tests/    pytest suite
RESULTS.md    the experiment log (every measured table, with context)
```

## Quickstart

```bash
pip install -r requirements.txt          # numpy + torch are enough for the core demos
python -m ssa.ssa_demo                    # the end-to-end SSA mechanism on a retrieval task
python -m ssa.niah_analysis               # what the reported "98–100% at 12M" (a selection-based figure) shows (≈1 min)
python -m ssa.staged_extension            # staged context extension, rung by rung
pytest ssa/tests                          # the test suite
```

Everything runs on CPU; a CUDA GPU makes the trained demos and the kernel benchmark much faster. The
`real_*`/`longctx_*`/`gemma_*` and `ssa_swap` scripts additionally need `transformers`+`datasets` and download
pretrained models on first run.

## What each module does

| module | what it demonstrates | paper |
|---|---|---|
| `ssa_demo.py` | the full SSA layer end to end: cumulant block routing + local window + exact attention, `O(n·k)`, recovering dense accuracy at a small budget | §4 |
| `ssa_kernel.py` | a fused block-sparse kernel (PyTorch FlexAttention) that never materializes the `n×n` scores (≈20.6× over dense at 256K) | §4.4 |
| `ivf_kernel.py` | the faiss-GPU IVF router wired into the kernel (`ssa_flex_ivf`), measured end-to-end to **12M tokens** single-head (139 ms, 6.55 GB; maskbuild ~0) | §4.4 |
| `ivf_decode.py` | the decode path: per-step IVF-routed SSA vs dense, both measured to 12M (SSA step flat ~0.53 ms; 55× at 12M) | §4.4 |
| `multihop_analysis.py` | multi-hop chained retrieval through the block selector — the composition law `chain ≈ ∏ρ` and the mixed-mode collapse | §10 |
| `longctx_swap.py` | the fast kernel inside a real model (Qwen2.5-0.5B) at 8K–128K: NIAH/two-hop quality + prefill wall-clock vs budget | §9 |
| `cascade_router.py` | the **Certified Causal Cascade** — one streaming selector composing sub-block max-pool, a warm-start IVF index, an outlier channel, and per-query **sound certificates** + escalation | §10 |
| `routing_space.py` | a trained low-dim routing projection — the P2 "low-rank is a bust" rebuttal (untrained 0.32 → trained 0.65 block-Jaccard on real Qwen keys) | §7 |
| `longctx_share.py` | cross-layer routing sharing, **measured** — routing's share of prefill 59% → 6% (donor=4), NIAH preserved; the analytic ÷5 made real | §9 |
| `ccc_quality.py` | which cascade component rescues which regime (sub-block ↔ spikes, outlier ↔ moderate spikes, isolated stays hard) | §10 |
| `fastweight.py` | zero-attention / fast-weight memory (additive / delta / gated-delta writes; linear + softmax reads; slot birth) — the compression corner | §11 |
| `fastweight_capacity.py` | the READ rule sets the capacity class (linear rank-d wall vs softmax exponential); the write rule is coherence control | §11 |
| `fastweight_recall.py` | write-time vs read-time relevance (**compression ≠ selection**), same-key conflicts need tags, the 2-hop chain bound | §11 |
| `fastweight_shift.py` | a fold breaks a fixed memory — it forgets across a shift; a growing (slot-birth) state preserves it | §11 |
| `p9_microlm.py` | the trained comparison: a swappable micro-LM (dense / SSA / DeltaNet+learned-gate / linear token-mixers) + a JEPA future-prediction aux loss | §11 |
| `p9_tasks.py` | the trained-MQAR suite — load sweep, write-salient (marker keys) vs read-salient, 2-hop chains | §11 |
| `p9_compare.py` | trained selection-vs-compression sweeps: the capacity frontier, the null gate/aux result, the composition sag | §11 |
| `ssa_checkpoint.py` | trains a small (~12M) SSA model with the gentle curriculum | §8 |
| `ssa_extrapolation.py` | zero-shot length extrapolation under rotary position vs a learned-positional control | §8.1 |
| `staged_extension.py` | the staging ladder: extend → cheap adapt → extend, reaching 32× the trained length | §8.2 |
| `ssa_swap.py` | the construction pipeline: swap dense attention for SSA in a pretrained model and adapt | §9 |
| `niah_analysis.py` | the benign-geometry condition behind single-needle retrieval at long context | §10 |
| `prune_regularizer.py` | the routability regularizer (shrink off-target `qᵀΣq`) + lossless branch-and-bound | §7.2 |
| `geometry_characterization.py` | the prune gate, the entry-magnitude split (selection vs linear attention), the capacity trade | §5.3, §7.3 |
| `tempered_routing.py` | the temperature family interpolating centroid ↔ cumulant ↔ exact-max | §5.2 |
| `hierarchical_routing.py` | tree routing with a recursive radius (FMM / Barnes–Hut style) | §4.4 |
| `anisotropic_bound.py` | the ellipsoidal (covariance) search bound on real keys | §5.1 |
| `core.py` | the closed-form theory (recovery weight, detectability, truncation) + NumPy primitives | §3 |
| `train.py` | trains the retrieval encoder — manufacturing routable geometry | §7 |
| `adaptive.py` | branch-and-bound exact selection with the admissible bound; k-means | §5.1 |
| `co_train.py` | co-trained selector experiments | §7 |
| `experiments.py` | the recovery-margin numerical suite | §3 |
| `real_keys.py`, `longctx_keys.py`, `longctx_probe.py`, `gemma_keys.py` | extract real query/key geometry from pretrained models (GPT-2, Qwen, TinyLlama, Gemma) and test routing/selection on it | §7 |

## Selected results

- **Mechanism.** Second-cumulant routing recovers targets where centroid routing collapses; routing quality
  peaks near `β ≈ 2`.
- **Kernel.** ≈20.6× wall-clock speedup over a dense exact kernel at `n = 262,144`.
- **Routability.** Co-training the `qᵀΣq` regularizer drove *lossless* branch-and-bound selection **cost** from
  26.5% to 4.2% of keys. (Accuracy is 1.000 *by construction* — admissible B&B always returns the exact argmax
  — so this measures the cost, not task quality.)
- **Length.** Staged extension reached **32× the trained length** (recall **0.979 under SSA**; 0.982
  dense-adapted) for ~800 adaptation steps — a toy MQAR task at ~3k absolute tokens, not a real long context.
- **Construction.** Swapping SSA into a 124M dense model costs +13.2 perplexity; an equal-budget adaptation
  recovers to within **+1.2** of a dense model given the same training, while attending ~38% of keys.
- **Long-context retrieval.** 98–100% single-needle accuracy at long context holds for **benign** targets
  (coherent spans) and *collapses* for isolated spikes — the benign-geometry condition made explicit.
- **Real-model frozen-swap (Gemma-4-26B-A4B, a 4B-active MoE).** SSA swapped into a *frozen* 26B model (no
  retraining; bit-exact dense gate) reaches **full** single-needle retrieval at a 25% budget once the block
  size is tuned (plain cumulant); an earlier apparent "frozen-key ceiling" at coarse blocks was a *tuning
  artifact*. This is a routing-**quality** result at n≤4096 — **not** a speed result.
- **The compute floor (P0–P5).** The kernel's gap to the theoretical `n·κ` floor is the router (the `(n/b)²`
  score GEMM + the argsort `BlockMask` build). Co-training lowers the floor itself **60×** (κ_min 25%→0.4% of
  keys), and a **faiss-GPU IVF router** — measured on the GPU, running linearly to **8M** (the only router past
  the flat router's memory wall: the score GEMM OOMs at 8M and the kernel's real `block_route` at ~1M).
  Full record: [`FLOOR_PROGRAM.md`](FLOOR_PROGRAM.md).
- **The IVF kernel end-to-end — MEASURED to 12M (`ivf_kernel.py`).** The IVF router wired straight into the
  FlexAttention kernel (`from_kv_blocks(compute_q_blocks=False)` skips the 38.7 GB dense transpose) runs a
  **full 12M-token forward in 139 ms and 6.55 GB, single-head** — the argsort maskbuild `n^2.12` wall is now
  **sub-millisecond**, and the gap to the floor is a *measured* **2.9×** (was a 128× projection). The decode
  path (`ivf_decode.py`) is **flat in n** (~0.53 ms/step) vs dense's growing prefix read — **55× at 12M**.
  (Single-head, synthetic keys = speed only.)
- **Multi-hop retrieval — the composition law self-tested (`multihop_analysis.py`).** A chained retrieval
  through the same block selector obeys `chain ≈ ∏ρ`: benign single needles hold at **1.00** while the **mixed**
  chain (one benign + one isolated hop) collapses to **0.02** — the NIAH@12M (~98%) vs MRCR (65.9%) split the
  assessment predicted, now measured by the rig.
- **The fast kernel in a real model at 8K–128K (`longctx_swap.py`).** The fused kernel swapped into Qwen2.5-0.5B
  (`impl="flex"`) largely preserves NIAH while its prefill speedup **grows with context** — **1.6× at 32K,
  2.15× at 65K, 3.44× at 128K** (under YaRN) — the first result that is real-model × long-context ×
  subquadratic-kernel × quality-measured (at 0.5B scale). Single-needle retrieval holds where the two-hop chain
  sags under tight budget (the predicted multi-hop split), and at matched budget the analytic O(n²) path needs
  7.6× the memory and OOMs before 64K where the kernel reaches 128K.
- **The Certified Causal Cascade — an optimal selector, measured (`cascade_router.py` + P7).** One streaming
  selector composing five ingredients, with **sound per-query certificates** (zero violations; certified ⇒
  selection == exact routing-metric top-κ; fire-rate 0.89 clustered / 0.50 random). The trilemma table names
  what each part does: sub-block granularity + the outlier channel rescue spikes, **isolated unit-norm needles
  stay hard for every cheap selector** (the impossibility wall). The selector's biggest lever is measured:
  **cross-layer sharing cuts routing from 59% of prefill to 6% (donor=4) with NIAH preserved** — the analytic ÷5
  made real. The trained low-dim routing space rebuts P2's "bust" (0.32 → 0.65 block-Jaccard on real keys) but
  is too lossy at d_r=16 to drive retrieval losslessly — an honest boundary.
- **The other corner — zero-attention memory, measured against the theorems (P8, `fastweight*.py`).** Small exact
  fast-weight memories (DeltaNet/Titans family) measured against six predictions (five with a machine-checked
  anchor; the load-bearing one is empirical): a needle salient **only at read time** is lost by a surprise-gated
  fixed memory (0.10) but recovered by selection (1.00) — **compression ≠ selection**. Also measured: the READ
  rule sets the capacity class (linear rank-d wall vs softmax holding to m=512); a fold breaks a fixed memory
  (forgets 0.90→0.10 across a shift) where slot-birth holds 0.65; and the composition prediction ∏ρ ≤ min hop
  (proved, `chain_le_weakest`) with the measured chain sagging below — so NIAH≫multi-hop holds for the
  compression corner too.
- **The trained comparison — a learned write gate does not close the gap (P9, `p9_*.py`).** A micro-LM trained
  end-to-end on MQAR with a swappable token-mixer at matched state (head_dim dh=16): trained selection
  (dense/SSA) is flat in load while trained **DeltaNet still walls at m≈dh** — training moves the rank-d wall,
  it does not remove it. The specific training-dependent lever — a *learned* write gate — is a **null
  ingredient**: write-salient (marker keys) is solved *without* it, read-salient walls *with* it, and the JEPA
  future-prediction aux loss is flat in its weight. Training sharpens P8's compression≠selection split rather
  than closing it.

## Scope — what is and isn't demonstrated

The four-way conjunction is now *split across two measured results*, not held in one: `ivf_kernel.py` is
**subquadratic × long-context (12M) × end-to-end-measured** but single-head and synthetic-keys (speed only);
`longctx_swap.py` is **real-model × long-context (to 128K) × subquadratic-kernel × quality-measured** but at
0.5B scale and moderate length. No *single* run is yet all four at the 12M endpoint on a frontier-size model —
read the headlines with that seam in mind.

- The **≈20× flat-kernel speedup** is on *synthetic* keys at a *fixed* budget with an `O((n/block)²)` block-score
  router. That router is **now wired out**: `ivf_kernel.py`'s IVF router drives the kernel end-to-end and
  **measured to 12M** (139 ms, single-head) — the maskbuild `n^2.12` wall is gone (sub-ms) and the gap to the
  floor is a measured 2.9×. Still single-head (H=8 does not fit at 12M) and synthetic-keys (**speed**, not
  selection-quality — that is the P1/P3/P4 story).
- The **Gemma-26B frozen-swap** stays an *analytic `O(n²)`* routing-**quality** probe (no speed claim); the
  **fused-kernel speed+quality** result is now `longctx_swap.py` on Qwen2.5-0.5B (`impl="flex"`), which reaches
  128K under YaRN and shows a **1.5–1.6× prefill speedup at 32K** with NIAH preserved. The speedup is modest
  because attention is a fraction of a 0.5B forward (Amdahl) — it grows with model size and context.
- The **124M perplexity** demo attends a *constant* ~38% of keys at n=1024 — a constant fraction is still `O(n²)`.
- The `O(n√n)` / near-linear complexity is the **algorithm's**; the subquadratic **win is now demonstrated
  end-to-end** (`ivf_kernel.py`, single-head to 12M). Validating it *multi-head, real-model, at 12M*
  simultaneously still needs substantially more than a single 16 GB GPU.
- **Multi-hop is the honest failure mode:** the composition law (`multihop_analysis.py`) and the real-model
  two-hop task (`longctx_swap.py`, `gemma_ssa_eval.py`) both show the chain sagging where single needles hold —
  the benign-geometry condition is load-bearing, not incidental.
- The **"(proved)"** results are machine-checked in a *separate* Lean development
  (`Substrate.Inference.PhaseTransition.Algebra.*`) that is **not shipped in this repo**; the prose maps to
  *lower bounds / sufficient conditions*, not equalities (see the paper's bibliography).
- The **zero-attention (P8)** results are `d ≤ 128` reference implementations on synthetic keys — mechanism
  measurements against the Lean predictions, **not a trained language model** and with no wall-clock claims.
  Five of six predictions carry a machine-checked anchor (P3, the load-bearing selection/compression split,
  is empirical); P2's overload-capacity edge for the delta rule is reported as measured (weaker than folklore),
  and P6's machine-checked part is ∏ρ ≤ min hop — the measured joint chain is reported separately.
- The **trained comparison (P9)** is a `d=128` micro-LM (head_dim 16) trained end-to-end on **synthetic MQAR
  (not natural language)**, one RTX 4080, recall at query positions, no wall-clock claims. It reaches the
  training-dependent half of the zero-attention recipe that P8 could not; the null gate/aux result (D2/D4) is
  empirical, and the DeltaNet wall / composition sag echo the read-side `softmax_capacity` / `chain_le_weakest`
  anchors without being new proofs.

See [`RESULTS.md`](RESULTS.md) for the full tables and context.

## Building the paper

```bash
cd paper && latexmk -pdf subquadratic_attention.tex
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
