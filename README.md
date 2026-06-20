# Subquadratic Sparse Attention (SSA)

Content-routed **exact** attention for long contexts, with the theory that says when it is sound and the
experiments that measure it.

Dense self-attention costs `O(n²)` in the sequence length `n`. SSA replaces it with a mechanism that, for each
query, (i) routes to a small content-dependent set of key **blocks** using only per-block summary statistics,
(ii) adds a local window, and (iii) computes **exact** softmax attention over the selected keys. The per-query
work is `O(κ)` in a fixed budget `κ ≪ n` plus a sublinear routing cost, so the layer runs in `O(n√n)` flat, or
near-linear with a hierarchical router.

The full writeup — algorithm, all the supporting math, and the measured results — is in
[`paper/subquadratic_attention.tex`](paper/subquadratic_attention.tex) (and a Markdown copy,
[`paper/subquadratic_attention.md`](paper/subquadratic_attention.md)). This README is the short version plus
how to run everything.

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
an in-block outlier that the mean alone washes out. Summary-only routing is provably lossless exactly when the
key **geometry is benign** (off-target blocks have small spread `qᵀΣ_c q`), and training can be made to
manufacture that geometry. None of this is free in the worst case: cheap, lossless, length-robust selection
cannot hold simultaneously for arbitrary keys — see the paper's impossibility argument and the trilemma.

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
python -m ssa.niah_analysis               # what "98–100% at 12M" actually shows (≈1 min)
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
- **Routability.** Co-training the `qᵀΣq` regularizer drove *lossless* branch-and-bound selection cost from
  26.5% of keys to 4.2%, at no accuracy cost.
- **Length.** Staged extension reached **32× the trained length at recall 0.982** for ~800 adaptation steps on
  top of a 7600-step base.
- **Construction.** Swapping SSA into a 124M dense model costs +13.2 perplexity; an equal-budget adaptation
  recovers to within **+1.2** of a dense model given the same training, while attending ~38% of keys.
- **Long-context retrieval.** 98–100% single-needle accuracy at long context holds for **benign** targets
  (coherent spans) and *collapses* for isolated spikes — the benign-geometry condition made explicit.

See [`RESULTS.md`](RESULTS.md) for the full tables and context.

## Building the paper

```bash
cd paper && latexmk -pdf subquadratic_attention.tex
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
