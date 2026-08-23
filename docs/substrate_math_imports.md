# Substrate math that applies to SSA — an import list

`~/substrate` gained ~877 new files under `Universal/` since this repo last synced its Lean anchors
(2026-07-08, `REVIEW_FOLLOWUPS.md` §7). Four of them bear directly on SSA's theory, and one supplies
a bound SSA's paper currently states only in the sufficient direction.

**Register warning, binding on this file.** Every declaration below was opened and read; the name,
path and line are verified. **The MAPPING from SSA's objects to the substrate's is asserted by hand
and is not machine-checked.** Nothing here is a formalisation of an SSA claim — it is a statement
that SSA's object is an instance of a proved abstract one, and the instance arrow is prose. Cite it
as such, exactly the way the existing anchors in `README.md` §"(proved)" are cited.

Paths are relative to `~/substrate/lean/Substrate/Universal/`.

---

## 1. The block-summary error floor — the necessary direction SSA is missing

`Potential/PartialScore.lean`

SSA routes block `c` from summaries `(μ_c, Σ_c)` and argues that summary-only routing is **lossless
when the geometry is benign** — off-target blocks have small spread `qᵀΣ_c q`. The paper is explicit
that this is *sufficient*, "not a necessary one". `PartialScore` supplies a necessary-side bound of a
different shape, and it holds for **every** summary, not just the second-cumulant one.

**The mapping.** A partial object `p` is a **block**; its completions `E p c` are the **keys in that
block**; the objective `V c` is the **true attention logit** `⟨q, k_c⟩` for a fixed query; the partial
score `s p` is the **router's block score** `r_c(q)`.

| declaration | what it gives SSA |
|---|---|
| `reach_spread_le_two_mul_score_error` (`:117`) | if a router is uniformly `ε`-accurate on a block, the true logit spread across that block is at most `2ε` |
| **`score_error_ge_of_reach_split`** (`:130`) | **half the true logit spread across a block is a LOWER BOUND on the uniform error of EVERY block score.** A property of the partition and the query; no choice of summary moves it |
| `score_error_pos_of_reach_split` (`:138`) | two keys in one block with different logits force strictly positive router error |
| `not_exactOnReach_of_reach_split` (`:146`) | **no exact summary score exists on a block whose keys disagree** — the `ε = 0` corner |
| `exists_exactOnReach_iff_contentRead` (`:160`) | welds the ceiling to `Accumulation/StoreContentRead.lean` · `contentRead_iff_separatesContent` (`:102`): an exact block score exists exactly when the objective never differs across two keys sharing a block |

**Why this is worth importing.** SSA's `qᵀΣ_c q` term is an *estimate of the in-block logit spread*,
and this says the spread is an **error floor for any router that reads only summaries**. So the
second-cumulant score is not a heuristic that happens to work — it estimates precisely the quantity
that lower-bounds every summary router. That is a stronger statement about the design than the paper
currently makes, and it is a **per-query, per-block** bound, so it composes with the existing prune
gate rather than replacing it.

**It also generalises the impossibility.** The paper's "cheap and lossless selection cannot hold for
arbitrary keys" is proved against a planted-spike adversary. `not_exactOnReach_of_reach_split` gets
the lossless half from the partition alone: **any** block containing two keys of different logit
admits no exact summary score, adversary or not. The adversarial construction is then about making
the *spread large*, which is the quantitative statement, not the existence one.

**Honest limit.** These bound the error of the block **score**. SSA's operational question is whether
the target lands in the top-κ *selected set*, which is a rank question, not a score-accuracy one. A
score error of `ε` implies a selection error only when `ε` exceeds the score gap to the κ-th block —
that step is not in the substrate and would have to be supplied.

---

## 2. Top-κ is not softmax at any temperature — the dropped mass is irreducible

`Accumulation/SelectionWeights.lean`

The README describes SSA as "exact softmax over a *selected subset* of keys (a bounded, dropped-mass
error versus full attention)". This file makes the "versus" precise.

| declaration | what it gives SSA |
|---|---|
| `softmaxAt_le_of_le` (`:163`) | the weights are monotone in the score at any non-negative `β` — softmax expresses the ranking |
| **`selection_excludes_nothing_at_finite_scale`** (`:178`) | **at every finite `β`, each candidate keeps strictly positive weight.** Softmax never implements an exclusion |
| `selection_is_flat_at_zero_scale` (`:186`) | at `β = 0` the ranking is gone entirely — the selection is carried wholly by the scale |
| `softmaxAt_tendsto_top` (`:225`) | concentration on a **strict** top only in the sharp limit; on a tied face the limit is uniform on the face, not a point |

**What it buys.** Hard top-κ selection is **not** softmax at any temperature — not a sharp one, not a
tuned one. So the dropped mass is not an approximation error that a better `β` shrinks; it is the
weight softmax would have assigned to the excluded keys, and it is structurally unavailable to the
selected-subset operator. That is a cleaner statement of SSA's error term than "bounded" and it says
where the bound must come from: the *tail* of the score distribution outside the budget, never the
temperature.

The tie caveat is worth carrying into `test_ccc_certificates.py`'s documented tie-parity assumption:
`softmaxAt_tendsto_top` needs a **strict** top, and on a tied face the sharp limit is uniform. The
existing near-tie stress geometry in `REVIEW_FOLLOWUPS.md` §8 is testing exactly this corner.

---

## 3. An unrouted key is indistinguishable from an absent key

`Accumulation/AddressLoss.lean`

The file's own framing is SSA's situation: **content is never lost, access is monotonically lost.**

| declaration | what it gives SSA |
|---|---|
| `the_held_set_stays_constant` (`:113`) | every key stays in the KV cache — "held" is invariant |
| `the_reached_set_only_shrinks` (`:121`) | what a bounded read can reach only shrinks as the store grows |
| **`the_read_cannot_separate_absent_from_held`** (`:133`) | **a read cannot distinguish a key that is absent from one that is held but not reached** |
| `a_fully_held_store_can_be_read_by_nothing` (`:265`) | the degenerate corner: everything retained, nothing reachable |

**What it buys.** This is the right abstraction for "lost in the middle" and for SSA's routing
failure mode generally: the needle is present in the cache and the output cannot tell that from its
absence. It gives the `REVIEW_FOLLOWUPS.md` header note on He et al. a formal home — position bias is
a statement about which keys are *reached*, on a *held* set that does not change with position.

---

## 4. The dense-vs-routed crossover, as an exact object

`Accumulation/RouteChargeCrossover.lean` (built 2026-08-22, the newest of these)

The README's headline figure is a crossover plot: dense `O(n²)` against the routed kernel, with the
speedup capped by the argsort BlockMask build until the IVF router drops it onto the `n·κ` floor.
This file is that comparison as a theorem, over **two routes with two prices** — a fidelity term and
a carriage term.

| declaration | what it gives SSA |
|---|---|
| `crossoverSize` (`:281`) | `⌊F / rate⌋₊ + 1` — the crossover is **computable**, not asymptotic |
| **`the_crossover_is_exact`** (`:333`) | **an IFF**: the routed route is cheaper at size `n` **iff** `crossoverSize ≤ n`. Below the count the dense route is the one to take |
| `the_fidelity_price_carries_no_size` (`:249`) | the quality term carries no content size — **every** size-dependence of the comparison sits in the compute charge |
| **`no_crossover_without_a_charge_rate_margin`** (`:379`) | if the routed route's per-unit rate is no better, **no crossover exists at any `n`** |
| `a_fixed_charge_margin_has_no_crossover` (`:360`) | a flat margin is a **dichotomy** — one route wins throughout, decided by comparing the constant against the fidelity price |

**What it buys, and the caution.** The last two are the sharp ones for this repo. A crossover is
decided by the **margin's slope**, not by its level at one measured `n` — so a single speedup point
establishes a level and never a direction of travel. Applied here: the argsort-capped flat-router
regime and the IVF regime are *different charge functions*, and the README already treats them
separately; this says that is not a presentational choice but the thing the comparison turns on.

**A third quantity is required and SSA does not currently name it.** Comparing dropped-mass quality
against compute needs an **exchange rate** between them (`RoutePrice.lossPrice`), and
`at_a_zero_price_for_loss_the_charge_settles_it` proves it load-bearing: at a zero rate the quality
term leaves the comparison entirely. Any speed/quality verdict — including
`SUBQ_ASSESSMENT.md`'s composed one — is **underdetermined until that rate is stated.** This is the
cheapest actionable item in this document.

---

## 5. Budget bounds that may apply — flagged, not claimed

`Accumulation/ChannelWidth.lean` · `no_carrier_carries_below_the_part_count` (`:167`) and
`Accumulation/ChannelBudget.lean` · `the_width_and_the_carrier_bound_the_part_count` (`:269`,
`k ≤ n · log_{|A|}|V|`) bound how many distinguishable parts a channel of a given width and
per-position budget can carry. The shape matches "κ blocks, each read through a fixed-size summary",
but the substrate's addressing discipline is **position-addressing**, and whether SSA's router is an
instance has **not** been checked. Listed so it is not lost; do not cite it until the discipline is
matched.

---

## 6. The one outstanding item may now be tractable

`REVIEW_FOLLOWUPS.md` §7 leaves the **randomized-selector impossibility** (average the deterministic
planted-spike adversary over a uniform spike location — Yao) open, deferred because "no probability
scaffold exists in Inference yet — from-scratch via bare `Finset.sum/card`."

That premise has weakened: the tree now carries `Universal/Stochastic/` (e.g. `MartingaleFate.lean`,
`SphereCoordinateMoments.lean`), `Potential/Entropy/ContinuousKL.lean`, and
`Carrier/Quantizer/SparsityCapacity.lean`. **Whether any of them supplies the averaging step has not
been checked** — this is a note that the blocking reason should be re-examined before the item is
deferred again, not a claim that the scaffold fits.
