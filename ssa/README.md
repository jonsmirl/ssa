# `ssa/` — the implementation and experiments

This package holds the reference implementations of Subquadratic Sparse Attention (SSA) and the experiments
behind every claim in the paper (`../paper/`). It is two layers:

- **A closed-form / NumPy layer** that states and tests the theory directly (the recovery-weight law, the
  selection bounds, the prune gate). Fast, CPU-only, no training.
- **A PyTorch layer** that trains small transformers and runs the actual sparse-attention mechanism, the
  kernel, length extension, and the dense→subquadratic construction.

**Every module is runnable** and prints its results:

```bash
python -m ssa.<module_name>      # e.g.  python -m ssa.ssa_demo
```

Core demos need only `numpy` + `torch`. The real-model probes (`real_keys`, `longctx_keys`, `longctx_probe`,
`gemma_keys`) and `ssa_swap` additionally need `transformers` + `datasets` and download pretrained models on
first run. GPU is optional but speeds up the trained demos and the kernel benchmark.

The notation below matches the paper: `q,k` are query/key vectors, `d` head dimension, `β = 1/√d` the inverse
temperature, a key block `c` has mean `μ_c` and covariance `Σ_c`, and the cumulant routing score is
`r_c(q) = ⟨q,μ_c⟩ + (β/2)·qᵀΣ_c q`.

---

## 1. Theory core and primitives

**`core.py`** — the closed-form predictions and the NumPy building blocks everything else reuses.
- `recovery_weight(beta, gap, mu)` = `σ(βΔ − log μ)` (paper Eq. 3.1); `threshold_n(beta, gap)` the
  detectability threshold (Eq. 3.2).
- `softmax`, `dense_read`, `read_over` (exact attention over a candidate set).
- Selectors: `ExactSelector`, `CentroidSelector`, `LSHSelector` — the baseline ways to pick the candidate set.
- Key generators: `random_unit_keys` (isotropic, worst case), `clustered_keys` (benign), and `coherence`
  (a scalar geometry diagnostic).

**`experiments.py`** — the recovery-margin numerical suite; each function tests one prediction.
`e1_recovery_weight`, `e2_length_generalization`, `e3_truncation`, `e4_selector` (is a sublinear lossless
selector buildable?), `e5_recall_vs_reasoning` (what selection does *not* buy you). Paper §3.

---

## 2. The SSA layer and the kernel

**`ssa_demo.py`** — a self-contained, end-to-end reproduction of the mechanism on multi-query associative
recall (MQAR). Defines the task (`MQAR`), a small transformer (`Attention`, `Block`, `Tiny`), and the layer
itself: `ssa_attention(q,k,v, n_clusters, top_c, local_w, routing)` does cumulant-routed top-`c` cluster
selection + a local window + exact attention over the selected keys (`O(n·k)`). `dense_mass_captured` measures
how much of the true softmax mass the selection recovers. **Start here.** Paper §4.

**`ssa_kernel.py`** — the *real* subquadratic kernel: a fused block-sparse attention (PyTorch FlexAttention)
that never materializes the `n×n` scores. `block_route` builds the per-query-block top-`c` selection,
`ssa_flex` runs the fused kernel, and `benchmark_speed` / `benchmark_needle` measure wall-clock speedup vs
`dense` (≈20.6× at 256K) and retrieval. Paper §4.4.

**`ssa_checkpoint.py`** — trains and saves a ~12M-parameter SSA model (`build_and_train`) with SSA *in the
training loop* so the keys co-adapt to the sparse selection, then checks functional long-context recall.
Writes `ssa_small_12m.pt` (gitignored, regenerable). Paper §8–9.

---

## 3. Length generalization

**`ssa_extrapolation.py`** — train short, retrieve long. A rotary-position (`RoPE`) model (`build_rope`,
`apply_rope`, `RoPEAttn`, `RoPEModel`) trained with a `gentle_train` curriculum extrapolates to lengths it
never saw, while a learned-positional control collapses. Paper §8.1.

**`staged_extension.py`** — climb past the ~2–4× zero-shot wall by doubling the length and doing a cheap
`adapt` at each rung. Reaches 32× the trained length at recall 0.982 for ~800 adaptation steps. Paper §8.2.

---

## 4. The construction pipeline

**`ssa_swap.py`** — turn a dense pretrained model into a subquadratic one. `ssa_masked` is a differentiable
block-sparse attention installed by patching `scaled_dot_product_attention` (`_patched_sdpa`); the script
swaps it into GPT-2, measures the perplexity hit, then `continue_train`s to recover — against a *fair* dense
control given the same in-domain training. Paper §9.

---

## 5. Long-context retrieval, characterized

**`niah_analysis.py`** — what "98–100% retrieval at 1M–12M tokens" actually shows. `niah_trial(..., mode)`
runs single-needle retrieval for `dense`, an `isolated` needle, and a `benign` needle (a coherent span).
Isolated needles collapse with length; benign ones stay flat — the benign-geometry condition made explicit.
Runs in ~1 minute. Paper §10.

---

## 6. Routing theory and selection

**`adaptive.py`** — exact selection via branch-and-bound on the admissible bound `U_c(q) = ⟨q,μ_c⟩ + ‖q‖·R_c`
(paper Eq. 5.1). `kmeans` builds the blocks; `probe_bandb` measures the adaptive cost. The result is identical
to scanning every key, but only blocks whose bound clears the running best are read.

**`tempered_routing.py`** — the routing-temperature family. `_tempered(s, beta, mode)` is the escort score
`β⁻¹·log Σ e^{β·s}`, which is the centroid at `β→0`, the cumulant at `β=1`, and the exact max at `β→∞`, with
best-key bias `(log n)/β`. `route_recall_tempered` sweeps `β`; recall peaks near `β≈2`. Paper §5.2.

**`hierarchical_routing.py`** — make the *selection* itself subquadratic with a tree (`build_tree`) and a
recursive radius `R_parent = max_child(‖μ_child − μ_parent‖ + R_child)`, so one parent bound check prunes a
whole subtree (`hier_approx` vs `flat_approx`, `lossless_cost`). Paper §4.4.

**`anisotropic_bound.py`** — the ellipsoidal (covariance) bound `⟨q,μ⟩ + R'·√(qᵀΣq)` (paper Eq. 5.2) prunes
more than the isotropic radius bound. `cluster_stats` builds the per-cluster `(μ, Σ)`; `bnb_cost` compares the
two bounds' branch-and-bound cost. Paper §5.1.

---

## 7. Manufacturing routability (training the geometry benign)

**`train.py`** — does *training* drive keys toward routable geometry? Trains a retrieval `Encoder` and
measures, on held-out items at larger scale, whether an LSH/centroid selector stays lossless at sublinear
cost vs an untrained encoder. Paper §7.

**`co_train.py`** — co-train the selector with the model (`RouterModel`, `train_router`, `probe_router`):
a learned margin, not a generic ANN index, is what makes selection robust under corrupted cues. Paper §7.

**`prune_regularizer.py`** — **Route F**: co-train the `qᵀΣ_c q` regularizer (`train(..., lam)`) and watch the
Samuelson prune bound (`samuelson_bnb`) start to fire — lossless selection cost falls from 26.5% to 4.2% of
keys at no accuracy cost. Paper §7.2.

**`geometry_characterization.py`** — characterizes the trained geometry: the two-knob prune gate, the
entry-magnitude split between the two subquadratic routes (`eff_rank` of `e^{B·KKᵀ}` grows with `B` → linear
attention for small `B`, selection for large `B`), and where the capacity trade bites. Paper §5.3, §7.3.

---

## 8. Real-model probes (need `transformers` + `datasets`)

These extract real query/key geometry from pretrained models and re-run the routing/selection tests on it —
the evidence that the geometry is benign in practice, not just in synthetic benign setups.

- **`real_keys.py`** — GPT-2 (learned positions, n≤1024). `extract_qk`, exact/budgeted branch-and-bound
  (`bandb_exact`, `bandb_budget`), `route_recall`, and a `matched_synthetic` control.
- **`longctx_keys.py`** — the rotary + grouped-query architectures real long-context models use (Qwen2.5,
  TinyLlama) at much longer context. `extract_qk_hf`, `target_distance_stats`.
- **`longctx_probe.py`** — disentangles "the geometry fails" from "the selector was naive" by ordering on
  relevance + the cumulant score and scaling the cluster count.
- **`gemma_keys.py`** — the frontier-scale, deep-head check (Gemma, head_dim 256): re-runs centroid-vs-cumulant
  routing and the temperature sweep on a 256-dimensional head.

---

## Tests

`tests/` holds the pytest suite (fast CPU assertions, with GPU-gated kernel checks). Run `pytest ssa/tests`.
See `tests/README.md` for what each test locks in.
