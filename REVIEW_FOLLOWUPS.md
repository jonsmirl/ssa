# Review follow-ups — the GPU-heavy batch (deferred)

A three-track external review (theory / methodology / code) ran on 2026-07-06. The zero-GPU corrections
are already in: `dd852d1` (fair decode baseline **9.1× at 12M, not 55×**; starved-probe guard;
`install_ssa` reset — each with a regression test) and `2a40b2f` (paper regime split, diagonal-gate
caveat + sound surrogate, tempered-score normalization, Alman–Song wording, trilemma-as-taxonomy,
Lean-namespace cites, SUBQ symmetry note, packaging pins). What follows is the remaining batch — every
item needs GPU hours, a model download, or the separate Lean repo. Ordered by value.

## 1. Close the speed/quality seam at 12M (the review's top remaining threat)

**Why.** The 12M result is speed-only (single-head, synthetic, fitted dense denominator past 4M); all
quality evidence lives at ≤128K on different models. `SUBQ_ASSESSMENT.md`'s verdict composes across that
seam. No selection-quality number has ever been measured *through the actual routed kernel* past 128K.

**Do.**
- New module (suggest `ivf_quality.py`): plant `niah_analysis.py`-style needles — benign coherent-span
  AND isolated unit-norm AND spike c∈{2,8} — into the synthetic single-head key stream at n ∈
  {1M, 4M, 12M}; run the *real* `ssa_flex_ivf` forward (not the oracle selector); measure (a) needle-block
  routing recall, (b) probe-output ≈ needle-value (the `test_routing_finds_the_needle` criterion). Memory
  is fine: the 12M forward ran in 6.55 GB.
- One point that is speed+quality simultaneously on real geometry: multi-head, real-Qwen-keys
  (`qwen_keybank.py` / `longctx_keys.py` extractions) through the IVF kernel at 1–4M with NIAH probes.
  H=8 at 12M needs ≥40 GB (K alone 12.3 GB) — rent if the full endpoint is wanted; 1–4M fits locally.

**Accept when:** RESULTS gains a table "recall through `ssa_flex_ivf` vs n × geometry", and the
assessment's "achievable … pinned" paragraph can cite a single run that is subquadratic × long-context ×
quality-measured. Expect the isolated row to fail (the trilemma predicts it) — report it; that *is* the
result. ~1–2 h local GPU.

## 2. Statistical floor — variance and sample sizes (review W3/W4)

**Why.** No timing variance anywhere (committed JSONs drift ~8–10% from prose tables — e.g.
`kernel_speed_measured.json` 22.2× vs RESULTS' 20.6× at 256K); NIAH/two-hop cells are 9-probe binomial
estimates (SE ≈ ±0.15) yet attribution steps lean on 0.444-vs-0.556 differences.

**Do.**
- `longctx_swap.py` / `longctx_share.py`: raise to ≥50 probes/cell at n ≤ 32K (`--niah-trials 17
  --twohop-trials 17` ≈ 51/cell), add Wilson 95% intervals to the JSON rows and doc tables.
- Timing benchmarks (`ssa_kernel.benchmark_speed`, `ivf_kernel`, `ivf_decode`): ≥5 process-level repeats,
  report median±IQR, interleave dense/SSA arms; note WSL2 clock jitter in the meta. Then regenerate the
  RESULTS/README tables *from the committed JSONs* (small script, e.g. `tools/tables_from_json.py`) so
  prose and artifacts cannot drift again.
- Re-run the marginal Gemma comparisons (the budget-0.5 Edgeworth claim) at ≥30 probes before keeping the
  "measured gain on a 26B model" phrasing.

**Accept when:** every headline table cites an interval, and the two tables the review caught drifting are
regenerated. ~2–4 h GPU + the Gemma download (item 3 shares it).

## 3. Commit the missing Gemma block=64 run (review W5)

**Why.** The load-bearing retraction ("no frozen-key ceiling — plain cumulant, block=64 → **1.000**")
rests on a run never saved to JSON; `runs/gemma_sweep_block.json` only records the *edgeworth* sweep
(and the schema doesn't store routing config). Disclosed in RESULTS' artifact note (~line 926) — close it.

**Do.** Extend the sweep-row schema to record the routing config (block, beta, edgeworth, dense_layers);
then `python3 -m ssa.gemma_ssa_sweep --block 64 --beta 2` (no `--edgeworth`) and commit the JSON
(suggest `runs/gemma_sweep_block64_plain.json`). Prereqs: `transformers>=5`, google/gemma-4-26B-A4B
download, device_map=auto on the 16 GB card. If the 1.000 does not reproduce at ≥30 probes (item 2),
update the retraction — that is the point of committing it.

## 4. P9 robustness — the wall-tracks-dh control (review W6)

**Why.** "Trained DeltaNet walls at m≈dh=16" is the capacity interpretation, but the single decisive
control — showing the wall *moves* with dh — was never run; the gate/aux null results ride on one LR
(6e-4) and 2 seeds.

**Do.** In `p9_microlm.py`/`p9_compare.py`: replicate D1 at dh=32 (wall should move to m≈32 — if it
doesn't, the capacity reading is wrong and D1 needs a rewrite); 3-point LR sweep per mixer {3e-4, 6e-4,
1.2e-3} + a 2×-steps arm to certify the D2/D4 nulls aren't optimization artifacts; `--seeds 0 1 2 3 4`
with std in the JSON. ~2–6 h on the 4080 (the existing curriculum is the budget driver).

**Also (code-review finding 6, deferred because it perturbs P9 artifacts):** `ssa_swap.ssa_masked` lets
zero-pads of the final partial block enter mean/var (unlike `gemma_ssa._block_stats`) and lets a query's
own partial block into routing stats. Selection-only, near-zero effect, but fix it *in the same PR as the
P9 re-run* so code and committed artifacts stay consistent.

## 5. One standard long-context benchmark (review W7)

**Why.** Every quality claim is synthetic or template-probe; the NIAH≫MRCR split that anchors the SubQ
critique is reproduced only on home-grown analogues.

**Do.** Run RULER (or a public MRCR-style multi-needle task) on the swapped Qwen2.5-0.5B at 32–128K,
dense vs `impl="flex"` at matched budget, same harness both arms. Even a small subset converts the
composition-law claim from analogue to standard measurement. ~2–4 h GPU.

## 6. Route-share estimator cleanup (code-review finding 7 — re-measure to apply)

`longctx_share.py` divides a warmup-contaminated *mean* (`ROUTE_MS`) by a *median* prefill, and
`gemma_ssa._flex_mask` puts two `torch.cuda.synchronize()` inside the timed region of the SSA arm only.
Both inflate the SSA side (conservative for the paper's claims, so it never blocked publication) — but
clean it: time route with the same median-of-reps discipline outside warmup, drop the in-region syncs
(or sync both arms), then re-run the share table (59%→6% is the headline it feeds). Do together with
item 2's Qwen runs.

## 7. Lean additions (in `~/substrate`, not this repo)

- Randomized-selector impossibility: the averaging/Yao step over a uniformly-planted spike (the paper now
  states it as an unformalized remark — close that).
- Adaptive (decision-tree) selector model for `lossless_selector_reads_every_key` (current statement is
  non-adaptive; the paper's in-text argument already covers adaptive).
- The pigeonhole step in `SearchTradeoff.capacity_search_tension` (B < n cells force two ε-coherent keys
  into one cell), turning a pair of monotonicities into an actual tension statement.
- The rank-d linear-read wall (easy linear algebra), so the P8/P9 "capacity is a property of the read"
  contrast is proved on both sides rather than softmax-side-only.
- After any of these land: update the paper's formalization note and RESULTS' anchor table.

## 8. Small code follow-ups (no re-measurement needed, batch with any of the above)

- `ivf_kernel._t` swallows exceptions matching "assert/device/handles" into missing-cell/OOM rows —
  record the exception string in the row so a genuine kernel failure can't masquerade as an OOM point.
- `test_ccc_certificates.py`: add a duplicated-key / near-tie stress geometry (the tie-parity assumption
  documented in `_assign`), and a `cert_margin>0` row showing it absorbs the ties.
- NIAH harness: assert the tokenizer's `max_length=n_tokens+64` never clips the trailing question
  (currently unguarded; string-level tests only).

## Suggested order on a free GPU day

1 (seam, local part) → 2+6 together (Qwen statistics + share cleanup) → 3 (Gemma download day) →
4 (P9 overnight) → 5 (RULER) → 7 (Lean, no GPU) whenever. Items 1–3 change what the assessment can
claim; 4–5 harden the trained-comparison and benchmark story; 7–8 are polish.
