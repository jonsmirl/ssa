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

## Scope — what is and isn't demonstrated

No single result here is simultaneously *real-model × long-context × subquadratic × quality-matched*; read the
headlines with that in mind.

- The **≈20× kernel speedup** is on *synthetic* keys at a *fixed* budget; its flat router still builds an
  `O((n/block)²)` block-score matrix, and the hierarchical router that removes it is a *separate, un-wired* module.
- The **Gemma frozen-swap** is an *analytic `O(n²)`* routing-**quality** probe (it materializes the score
  matrix and masks it) — no speed claim.
- The **124M perplexity** demo attends a *constant* ~38% of keys at n=1024 — a constant fraction is still `O(n²)`.
- The `O(n√n)` / near-linear complexity is the **algorithm's**; the subquadratic **win is designed, not yet
  demonstrated end-to-end**. Validating it at real long context (and the 12M-token regime) needs substantially
  more compute than a single GPU.
- The **"(proved)"** results are machine-checked in a *separate* Lean development
  (`Substrate.Inference.PhaseTransition.Algebra.*`) that is **not shipped in this repo**; the prose maps to
  *lower bounds / sufficient conditions*, not equalities (see the paper's bibliography).

See [`RESULTS.md`](RESULTS.md) for the full tables and context.

## Building the paper

```bash
cd paper && latexmk -pdf subquadratic_attention.tex
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
