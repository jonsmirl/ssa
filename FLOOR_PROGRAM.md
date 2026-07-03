# Driving SSA to the n·κ floor — the P0–P5 program

**Goal.** The measured SSA kernel sits far above the theoretical floor `n·κ` (attending the κ selected
keys). This program *sized the gap*, *mapped the floor*, and *closed the gap* with a sub-linear router —
all on a 16 GB GPU, with the 12M regime projected (the scope), every result reproducible (`ssa/*.py` +
`paper/figures/*.json`). Figures: `paper/figures/unified_scaling.png` (the result — dense O(n²), our kernels, the
floor, and SubQ's claim) and `router_gpu_compare.png` (the measured GPU router).

| phase | question | result |
|---|---|---|
| **P0** instrument | where does the time go? | `cost_profile.py`: attention ~ n¹·⁰² (the floor), router ~ n¹·⁷⁶, **maskbuild ~ n²·¹²** (the *largest*). At 12M the floor is ~1% of the forward — a **128× gap**, dominated by the `(n/b)²` score GEMM **and** the argsort BlockMask build. |
| **P1** map the floor | how low can κ go? | `recall_floor.py`: κ_min vs geometry × difficulty. **Co-training crushes the floor 25% → 0.4% (60×)** — the dominant lever. Geometry-bound: tight 3%, **diffuse/adversarial 50% (no speedup)**; long-range raises it. This is the SubQ κ-viability test. |
| **P2** cheap wins | do constant-factor fixes reach it? | `router_variants.py`: narrow-kv_idx ~2× on maskbuild; cross-layer sharing ÷5 (Gemma); **low-rank routing a bust** (5–14% agreement on high-PR keys). All constant factors — the `(n/b)²` remains. **Gate: sub-linear router justified.** |
| **P3** bake-off | which sub-linear router? | `bakeoff.py` (recall vs cost on benign co-trained keys): **treecode wins raw cost (0.8%)**, **faiss-ivf robust (1.6%, best recall, GPU-optimized)**, centroid 1.8%, **LSH out** (can't reach 0.9). Decision: **faiss-ivf** for the kernel — competitive cost + an optimized constant (sidesteps the treecode's wall-clock-constant problem). |
| **P4** integrate | how close does it get? | `faiss_router.py`: IVF over block-means scores **O(√nb) blocks (8→64× fewer), 0.93–0.97 agreement**, emits kv_idx directly. **MEASURED on the GPU** (`router_gpu_compare.py`, faiss-gpu, both on GPU, no transfer): the single-head flat GEMM's constant wins below ~3M (crossover ~3M; **IVF 1.7× faster at 4M**); that GEMM OOMs only at 8M (17 GB nb² matrix), and the kernel's *real* `block_route` (H heads + sel + argsort) OOMs far earlier (~1M, measured), while the **IVF router runs linearly to 8M (64 ms) — the only router past the wall**. The 12M kernel "~at the floor" is a **projection** (the measured router + an analytic floor extrapolated 23× past the largest measured n, no end-to-end kernel). (Same-device CPU control `router_cpu_compare.py` agreed.) |
| **P5** synthesis | the picture | `unified_scaling.png` (SubQ's compute orientation): dense O(n²) rises; our flat-router kernel stays below but is **speedup-capped** (maskbuild ~n²·¹²); the **projected** IVF-router kernel (measured router + analytic floor) approaches the floor. SubQ's published points + 1,000× claim are shown for *reference* — plotted as compute = our dense fit / its speedup (cross-hardware, **not** a head-to-head). **Necessary, not beaten.** |

## What it establishes
- **The gap to the floor is the router** (score GEMM + argsort maskbuild, both `(n/b)²`), and a sub-linear
  IVF router that emits `kv_idx` directly removes *both* — the kernel reaching **~1.1× the floor at 12M** is a
  projection. The router wall-clock is **measured on the GPU** (`router_gpu_compare.py`, faiss-gpu): the
  single-head flat GEMM's constant wins below ~3M, IVF is 1.7× faster at 4M, that GEMM OOMs at 8M (17 GB) and
  the kernel's real `block_route` OOMs at ~1M, while the **IVF router runs to 8M (64 ms) — the only router past
  the wall**. The router is small vs the analytic floor, so the *projected* kernel lands ~at the floor.
- **The floor itself is set by geometry and crushed by co-training** (25%→0.4%). So the two multiplicative
  levers — lower the floor (co-train) and close the gap (IVF router) — *both* work, on benign geometry.

## P6 — the IVF router wired into the kernel, measured end-to-end to 12M (`ivf_kernel.py`, `ivf_decode.py`)
The isolation caveat is retired. The IVF router now emits the `from_kv_blocks` contract directly and drives a
full FlexAttention forward (`ssa_flex_ivf`), built with `compute_q_blocks=False` so the dense `(nb,nb+1)`
transpose — 38.7 GB at nb=98,304 — is skipped. **Measured single-head, to 12M, on the 16 GB card:**

| n | router (ms) | maskbuild (ms) | attention=floor (ms) | total (ms) | gap to floor | peak |
|---|---|---|---|---|---|---|
| 1M | 13.5 | 0.002 | 3.2 | 16.1 | 5.0× | 0.55 GB |
| 4M | 36.4 | 0.003 | 13.5 | 52.0 | 3.9× | 2.18 GB |
| **12M** | **101.4** | **0.003** | **47.5** | **139.5** | **2.9×** | **6.55 GB** |

The projected 40.7 s maskbuild at 12M is now **sub-millisecond**; the gap to the floor is a *measured* 2.9×
(was a 128× projection for the flat kernel). The standalone router sweep also now runs to **12M (94 ms; flat
OOMs at 8M and 12M)**. Decode (`ivf_decode.py`): the IVF-routed step is **flat in n** (~0.53 ms, 1M→12M) vs a
dense step's growing prefix read (2.6→29.6 ms) — **55× at 12M**, both measured.

## Honest scope
- The end-to-end 12M result is **single-head** (H=8 does not fit — K alone is 12.3 GB) and on **synthetic
  random keys**, so it is a **speed** result; selection quality is the P1/P3/P4 story, on **benign** geometry.
  Adversarial / multi-hop geometry breaks the benign assumption (the P1 50% floor; the SubQ MRCR sag, now
  reproduced by `multihop_analysis.py`). What remains: **multi-head, real-model keys** at the 12M endpoint.
- The decode index is add-only (no quantizer retrain) — valid over the measured 128 steps; a serving loop
  retrains every R blocks as centroids drift.
- The treecode wins raw selection cost but its wall-clock constant is the known blocker; faiss-ivf is the
  pragmatic choice and the one now wired in. A **multi-head, real-model ≥40 GB-GPU 12M run** is the next step.

## Tie to the SubQ assessment
SubQ's "1,000× at 12M" needs both pieces this program demonstrates: a **floor-lowering training stage** (their
RL ≈ our co-training, 60×) **and** a **sub-linear indexer** (≈ our IVF router). The second is no longer a
projection — it drives a **live kernel to a measured 2.9× of the floor at 12M** — and both hold only on
**benign geometry**. So the claim is *achievable in practice on one GPU under exactly the benign-geometry
condition the floor analysis names* — not refuted, pinned (and the geometry condition is load-bearing: the
multi-hop measurement shows the chain collapsing off it).

## P7 — The Certified Causal Cascade (CCC): an optimal selector, built and measured

The floor program showed the *ingredients* of a cheap selector; P7 composes them into one selector and
measures which parts pay off. Five components (`cascade_router.py`, `routing_space.py`, `longctx_share.py`):
(1) a shared low-dim **routing space**, (2) **sub-block max-pool** summaries (spike sensitivity), (3) a
**chunked-causal streaming index** (prefill/decode one path), (4) an **outlier side-channel** (the k=c·q
impossibility case), (5) per-query **certificates + escalation**. Design rationale: the assessment's open
question is "what does SubQ's selector cost?" — DSA's eats 58% of prefill at 1M; this asks what an optimal
selector costs, and which components are load-bearing.

**The certificate is sound** (`test_ccc_certificates.py`, zero violations on clustered AND random): certified
⇒ the selected top-κ parent blocks equal the exact top-κ under the routing metric. Fire-rate is
geometry-dependent — **0.89 clustered / 0.50 random at 1M** (benign geometry certifies; adversarial escalates).
The full cascade runs end-to-end to **12M (980 ms, 6.67 GB)** single-head.

**Which component rescues which regime** (`ccc_quality.py`, needle-block recall at 64K):

| regime | ivf (block) | +sub-block | +outlier |
|---|---|---|---|
| benign (coherent span) | 1.00 | 1.00 | 1.00 |
| **isolated** (unit-norm needle) | 0.00 | 0.05 | 0.05 |
| spike c=2 (modest high-norm) | 0.23 | 0.38 | **1.00** |
| spike c=8 (large high-norm) | 1.00 | 1.00 | 1.00 |

Sub-block granularity rescues large spikes; the outlier channel uniquely rescues the *moderate* c=2 spike
(0.38→1.00) that sub-block still washes out; **isolated unit-norm needles stay hard for every cheap selector
(0.05)** — the impossibility wall in miniature, exactly as the trilemma predicts. No component is free of it.

**The trained routing space — the P2 rebuttal** (`routing_space.py`): P2's "low-rank routing is a bust
(5–14%)" tested an *untrained random* projection. On real Qwen keys, untrained random reproduces the bust
(**0.32** block-Jaccard), but a **trained d_r=16 projection reaches 0.65 (0.77 at d_r=32)** — ~2× — and
generalizes to a held-out code doc (0.58). So low-rank routing is viable for approximate ranking, *not* a
bust. **But** driving the real model, the d_r=16 projection collapses NIAH to 0.00 — 0.65 Jaccard is too
lossy (and it is centroid-vs-cumulant metric-mismatched): the honest boundary is that low-rank routing ranks
blocks approximately but is not accurate enough to drive attention losslessly at d_r=16.

**Cross-layer sharing — the ÷5 measured** (`longctx_share.py`, the first measurement of what
`router_variants.py` only asserted): per-layer full-d routing costs **~59% of prefill at 8K on Qwen2.5-0.5B**
(comparable to DSA's 58%!); **sharing the selection from a mid layer (donor=4) cuts it to ~6% with NIAH and
two-hop preserved at 1.00** — a measured ~10× reduction. But the donor choice matters: sharing from **layer 0
collapses NIAH to 0.00** (early layers route positionally, not by content) — the analytic ÷5 assumed any layer
transfers; the measurement shows only mid layers do. The DSA-comparable number (routing share of total
prefill) is thus **measured**, and the lever that makes the selector cheap is cross-layer sharing from the
right donor, not the low-dim projection.

**Tie to the assessment.** The open "what does SubQ's selector cost?" now has a constructive answer and a
falsifiable signature: a selector isomorphic to CCC has a routing share that (a) is single-digit % once shared
from a mid layer, and (b) preserves single-needle retrieval but sags on isolated/multi-hop — exactly the
NIAH≫MRCR split SubQ reports. If SubQ's selector share is large, they haven't solved it (explaining the gated
access); if small with the predicted quality split, it is isomorphic to this.

## P8 — The other corner: zero-attention / fast-weight memory, measured

P0–P7 mapped the SELECTION corner of the trilemma; P8 builds the COMPRESSION corner (fixed-/growing-state
memory written at inference time — the Mamba/DeltaNet/Titans family, SubQ's "zero attention") as small
exact reference implementations and measures them against six predictions from `Substrate/Inference/*.lean`
(sorry-free) — **five with a machine-checked anchor**, one (P3) purely empirical. Files:
`fastweight{,_capacity,_recall,_shift}.py`.

| prediction | Lean anchor | measured |
|---|---|---|
| the READ rule sets the capacity class | `softmax_capacity` | linear read collapses at m≈d; softmax over the same pairs holds to m=512 (theorem: exponential) ✓ |
| the write rule is coherence control | `capacityBound_antitone` | delta exact on orthogonal keys (so are additive/gated there); a *small* overload edge on random keys — additive competitive at m≤d ~ |
| **write-time vs read-time relevance** | *(empirical — no theorem)* | a read-time-salient needle is LOST by a gated fixed memory (0.10) but recovered by selection (1.00) — **compression ≠ selection** ✓ |
| a same-key conflict needs a tag | `tag_resolves_conflict` | additive averages, delta keeps latest, a salient tag recovers BOTH (0.95/1.00; plain tag 0.60/1.00) ✓ |
| a fold breaks a fixed memory | `fold_not_hopfield` | gated fixed pre-shift recall 0.90→0.10; slot-birth holds 0.65 (growth is one remedy the theorem allows) ✓ |
| the composition law | `chain_le_weakest` | ∏ρ ≤ min hop (0.25 ≤ 0.49, the theorem); the measured joint chain sags to 0.15 ✓ |

**What it settles.** The read rule sets the capacity class (not the substrate); the write rule is coherence
control; write-time compression *cannot* serve query-only relevance (the load-bearing result, empirical — why
NIAH stays easy and multi-hop sags for this corner too); episodic tags help and a fixed memory cannot track a
fold (growth is one remedy). The trilemma does not evaporate when you drop attention — it tells you which
corner you moved to.

**Tie to the assessment.** SubQ's "zero attention" pivot now has a measured rig and a sharpened falsifiable
prediction: any write-time-compression model will lead its benchmark table with NIAH/RULER (write-salient)
and be quiet on multi-hop / query-only retrieval — because for this corner the multi-hop sag is the proved
`chain_le_weakest` (∏ρ ≤ min hop) plus the empirical write-time-commitment failure, not a training gap.
Pinned, not refuted.
