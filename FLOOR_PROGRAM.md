# Driving SSA to the n·κ floor — the P0–P5 program

**Goal.** The measured SSA kernel sits far above the theoretical floor `n·κ` (attending the κ selected
keys). This program *sized the gap*, *mapped the floor*, and *closed the gap* with a sub-linear router —
all on a 16 GB GPU, with the 12M regime projected (the scope), every result reproducible (`ssa/*.py` +
`paper/figures/*.json`). Figures: `paper/figures/p5_synthesis.png` (the result), `unified_scaling.png`
(claim vs floor).

| phase | question | result |
|---|---|---|
| **P0** instrument | where does the time go? | `cost_profile.py`: attention ~ n¹·⁰² (the floor), router ~ n¹·⁷⁶, **maskbuild ~ n²·¹²** (the *largest*). At 12M the floor is ~1% of the forward — a **128× gap**, dominated by the `(n/b)²` score GEMM **and** the argsort BlockMask build. |
| **P1** map the floor | how low can κ go? | `recall_floor.py`: κ_min vs geometry × difficulty. **Co-training crushes the floor 25% → 0.4% (60×)** — the dominant lever. Geometry-bound: tight 3%, **diffuse/adversarial 50% (no speedup)**; long-range raises it. This is the SubQ κ-viability test. |
| **P2** cheap wins | do constant-factor fixes reach it? | `router_variants.py`: narrow-kv_idx ~2× on maskbuild; cross-layer sharing ÷5 (Gemma); **low-rank routing a bust** (5–14% agreement on high-PR keys). All constant factors — the `(n/b)²` remains. **Gate: sub-linear router justified.** |
| **P3** bake-off | which sub-linear router? | `bakeoff.py` (recall vs cost on benign co-trained keys): **treecode wins raw cost (0.8%)**, **faiss-ivf robust (1.6%, best recall, GPU-optimized)**, centroid 1.8%, **LSH out** (can't reach 0.9). Decision: **faiss-ivf** for the kernel — competitive cost + an optimized constant (sidesteps the treecode's wall-clock-constant problem). |
| **P4** integrate | how close does it get? | `faiss_router.py`: IVF over block-means scores **O(√nb) blocks (8→64× fewer), 0.93–0.97 agreement**, emits kv_idx directly. **Projected @12M: 128× gap → 1.10× — at the floor.** Wall-clock **CONFIRMED same-device** (`router_cpu_compare.py`: flat GEMM is nb², IVF ~linear → IVF beats it from 512K, **6.7× at 4M on CPU**). The naive faiss-cpu-in-a-GPU-pipeline measured 75–333× *slower* — a **device-mismatch artifact** (GPU↔CPU transfer), not the algorithm; the GPU realization needs **faiss-gpu**. |
| **P5** synthesis | the picture | `p5_synthesis.png`: at 12M the **flat kernel is only ~20× over dense** (maskbuild-bound), the **IVF-router kernel ~2336× — on the floor** (op-count, CPU-validated). The router moves the kernel off n¹·³ onto the linear floor. |

## What it establishes
- **The gap to the floor is the router** (score GEMM + argsort maskbuild, both `(n/b)²`), and a sub-linear
  IVF router that emits `kv_idx` directly removes *both* — projecting the kernel to **~1.1× the floor at 12M**.
  The op-count→wall-clock step is **validated same-device** (`router_cpu_compare.py`: IVF beats the flat GEMM
  6.7× at 4M on CPU); realizing it inside the GPU kernel needs **faiss-gpu** (the naive faiss-cpu-in-GPU is
  transfer-bound — an artifact, not the algorithm).
- **The floor itself is set by geometry and crushed by co-training** (25%→0.4%). So the two multiplicative
  levers — lower the floor (co-train) and close the gap (IVF router) — *both* work, on benign geometry.

## Honest scope
- The IVF-router 12M number is a **cost projection** (P0 fits × P4 IVF ratios) on **benign** geometry, with
  the op-count→wall-clock step **validated same-device** (`router_cpu_compare.py`). What is unmeasured is the
  **GPU pipeline**: faiss-gpu (no transfer) is needed to realize it on-device — a naive faiss-cpu router inside
  a GPU kernel is transfer-bound (75–333× slower, an artifact). Adversarial / multi-hop geometry breaks the
  benign assumption (the P1 50% floor; the SubQ MRCR sag).
- The treecode wins raw selection cost but its wall-clock constant is the known blocker; faiss-ivf is the
  pragmatic choice. A fused faiss-gpu (or treecode) kernel + a real ≥40 GB-GPU 12M run is the next step.

## Tie to the SubQ assessment
SubQ's "1,000× at 12M" needs both pieces this program demonstrates in projection: a **floor-lowering
training stage** (their RL ≈ our co-training, 60×) **and** a **sub-linear indexer** (≈ our IVF router,
reaching the floor) — and both only on **benign geometry**. So the claim is *achievable in principle under
exactly the benign-geometry condition the floor analysis names* — not refuted, pinned.
