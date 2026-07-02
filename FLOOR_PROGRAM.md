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
