# `ssa/tests/` — the test suite

Fast, deterministic assertions that lock in the load-bearing claims of the paper and the package. Most run on
CPU in seconds with no training and no model downloads; the GPU-gated files (the block-sparse kernel, the IVF
kernel/decode, and the flex-swap kernel path) skip cleanly when CUDA / FlexAttention / faiss is unavailable.

## Running

```bash
pytest ssa/tests                 # everything (CPU; the kernel test self-skips without a GPU)
pytest ssa/tests/test_core.py    # one file
pytest ssa/tests -k samuelson    # by keyword
pytest ssa/tests -q              # quiet
```

Expected: **54 passed** on CPU (**64** with CUDA + faiss — 10 GPU-gated kernel tests self-skip on CPU).

## What each file checks

**`test_core.py`** — the theory predictions and the baseline selector (paper §3).
- `test_recovery_weight_is_exact_target_mass`, `test_recovery_threshold_at_half` — the recovery-weight law
  `σ(βΔ − log μ)` and its `½` crossing.
- `test_length_generalization_logarithmic` — the needed margin grows only as `log n`.
- `test_truncation_bound_holds` — the sparse-vs-dense output error is bounded by the missed mass.
- `test_centroid_selector_sublinear_and_lossless_under_separation` — selection is lossless at sublinear cost
  when keys are separated (benign).
- `test_recall_degrades_under_crowding`, `test_reasoning_decays_as_rho_pow_h` — recall erodes as distractors
  crowd, and an `h`-hop chain succeeds like `ρ^h` (recall ≠ reasoning).

**`test_ssa.py`** — the SSA layer (`ssa_attention`), no training.
- `test_full_budget_recovers_dense` — at full budget SSA equals dense attention.
- `test_attended_fraction_below_one`, `test_fraction_falls_with_context` — it really is sparse, and the
  attended fraction falls as context grows.
- `test_kmeans_assignments_valid` — the block/cluster assignment is well-formed.

**`test_ssa_kernel.py`** — the block-sparse FlexAttention kernel (GPU-gated; skips on CPU).
- `test_full_budget_matches_dense` — the fused kernel equals dense at full budget.
- `test_routing_is_sparse_and_causal` — routing selects a strict subset and never attends the future.

**`test_ivf_kernel.py`** — the IVF-routed kernel (`ivf_kernel.py`; GPU + faiss gated).
- `test_ivf_full_budget_matches_dense` — exhaustive IVF selection equals dense causal attention.
- `test_ivf_selection_is_causal_and_owns_its_block` — tight budget is a strict causal subset; own block kept.
- `test_ivf_agreement_with_flat_router` — IVF vs flat `block_route` block selection Jaccard ≥ 0.6 (clustered).

**`test_ivf_decode.py`** — `test_decode_step_matches_prefill_last_row`: the decode primitive
(`ivf_decode.decode_attend`) reproduces the IVF kernel's prefill row for the same selected blocks (GPU+faiss).

**`test_multihop_analysis.py`** — the synthetic multi-hop rig (CPU).
- `test_dense_two_hop_near_perfect_at_high_margin`, `test_composition_law_sanity` — dense chains, and measured
  chain ≈ ∏ρ (never above its weakest hop).
- `test_isolated_chain_collapses_benign_holds`, `test_mixed_tracks_weak_hop` — the falsifiable single-needle-
  holds / chain-collapses prediction.

**`test_flex_swap.py`** — the fast kernel inside the real-model swap (`block_route_budget` + `impl="flex"`).
- `test_budget_router_causal_and_counts`, `test_budget_router_monotone_in_budget` (CPU) — the budget router's
  causality, own-block, and monotone-in-budget invariants.
- `test_flex_full_budget_matches_dense_sdpa`, `test_flex_matches_analytic_at_full_budget`,
  `test_flex_no_future_leak`, `test_flex_pad_nonmultiple_length` (GPU) — the fused path equals dense / the
  analytic path, never leaks the future, and handles non-block-multiple lengths.

**`test_ssa_extrapolation.py`** — the RoPE model (paper §8.1), no training.
- `test_rope_preserves_norm` — the rotary map is an isometry.
- `test_rope_model_forward_shape`, `test_rope_runs_at_arbitrary_length` — the model runs at lengths beyond any
  trained length (the precondition for extrapolation).

**`test_tempered.py`** — the routing-temperature identities (paper §5.2).
- `test_limits_and_second_order` — the escort score reduces to centroid / cumulant / max in the right limits.
- `test_tempered_sandwich`, `test_higher_temperature_tightens_to_max` — the `max ≤ score ≤ max + (log n)/β`
  sandwich, and the bias shrinks as `β` grows.

**`test_anisotropic.py`** — `test_both_bounds_are_admissible`: the isotropic and ellipsoidal search bounds are
both valid upper bounds on every key's score (paper §5.1).

**`test_hierarchical.py`** — tree routing (paper §4.4).
- `test_recursive_radius_bounds_the_subtree` — the recursive parent radius really bounds all descendants.
- `test_lossless_returns_true_argmax` — hierarchical selection returns the exact best key.
- `test_hierarchical_scores_fewer_nodes` — the tree scores fewer nodes than the flat scan.

**`test_prune_regularizer.py`** — Route F and the Samuelson gate (paper §5.3, §7.2).
- `test_samuelson_bound_is_admissible` — `max_j s_j ≤ s̄ + √((m−1)·Var)` holds.
- `test_samuelson_bnb_is_lossless` — branch-and-bound with that bound returns the true argmax.
- `test_prune_gate_predicts_pruning` — the closed-form gate predicts which blocks are actually pruned.

**`test_real_keys.py`** — the selector logic behind the real-model probes, on synthetic geometry (no model
load).
- `test_exact_bandb_is_lossless` — exact branch-and-bound matches the dense ceiling by construction.
- `test_clumped_keys_prune_more_than_spread` — benign (clumped) geometry prunes more than diffuse geometry.
- `test_budget_recall_monotone_and_benign_beats_random` — recall rises with budget, and benign ordering beats
  random.

**`test_geometry_characterization.py`** — `test_eff_rank_grows_with_entry_magnitude`: the effective rank of
`e^{B·KKᵀ}` grows with the entry magnitude `B` — the split between the linear-attention and selection regimes
(paper §7.3).

## Conventions

- Tests import from the installed package path (`from ssa.core import …`), so run them from the repository
  root (or with the repo on `PYTHONPATH`).
- Randomized tests fix seeds for determinism.
- CPU-only by default; the kernel test is the single exception and skips gracefully without a GPU.
