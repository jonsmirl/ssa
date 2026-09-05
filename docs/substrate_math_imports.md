# Substrate math that applies to SSA — an import list

This assessment maps inspected Substrate results to SSA's selection, attention error, and cost models.
The actionable extension is an adaptive output certificate in `ssa/certified_attention.py` (paper §5.7).
It combines existing log-sum-exp and barycenter bounds with the support-restriction interpretation made
explicit by Substrate's `UniformSupport.lean` and finite-vector `TotalVariation.lean` results.
This is an application of established inequalities, not a claim of a new mathematical inequality.

The focused source review covers `PartialScore`, `AdmissibleBound`, `LogSumExpBound`, `SelectionGeometry`,
`ValueAwareSelection`, `UniformSupport`, `TotalVariation`, and `ApproximateSelection`, alongside SSA's
current implementations and imported results. It is not an exhaustive review of the Substrate tree.
`Carrier/Simplex/ApproximateSelection.lean` concerns continuous selections of correspondences and does
not provide a sparse-attention selector or a runtime improvement here.

**Formalization scope.** Source theorems are machine-checked in Substrate. **The mapping to SSA and the
composition of these theorems are not machine-checked.** Paper §5.7 supplies an ordinary mathematical
proof; Python tests compare the implementation to a dense oracle. Float64 evaluation with an outward
score cushion is not interval arithmetic. Declaration names are the stable lookup keys; line references
in the exploratory entries below are not a guarantee about a concurrently changing Substrate checkout.

Paths are relative to `~/substrate/lean/Substrate/Universal/`.

---

## Implemented: adaptive mass, KL and value-aware output certificates

| source | inspected declaration | SSA use and scope |
|---|---|---|
| `Potential/Entropy/UniformSupport.lean` | `klDiv_uniformSupport`, `klDiv_uniformSupport_lt_of_strict_subset` | At equal logits, `KL(subset || dense) = log(n / kept_count)`; shrinking the kept set increases this divergence. General nonuniform softmax restriction is derived in the paper. |
| `Potential/Entropy/TotalVariation.lean` | `totalVariation_le_sqrt_klDiv` | Allows zero weights in the first distribution, but requires a strictly positive second one. Applies to subset-to-dense KL. The exact TV identity for restriction is stronger, so Pinsker is not used to set the stop threshold. |
| `Potential/Entropy/LogSumExpBound.lean` | `logSumExp_max_sandwich`, `softMax_sandwich` | An admissible maximum-logit bound `U_c` gives unopened block partition mass at most `b_c exp(beta U_c)`; actual block counts matter. |
| `Generator/Dissipative/SelectionGeometry.lean` | `barycenter_truncation_bound` | Value-dependent output control. The computable form centered at the kept output is derived separately in the paper, rather than substituting an estimate for the unknown dense output in this theorem. |

For kept partition sum `Z_S` and omitted partition sum `Z_D`, restriction gives exactly

```
delta = Z_D / (Z_S + Z_D)
TV(subset, dense) = delta
KL(subset || dense) = -log(1 - delta)
```

If the unopened summaries give `L_c <= Z_c <= A_c`, define `L=sum L_c`, `A=sum A_c`, and
`d_c=norm(value_mean_c - kept_output) + value_radius_c`. Then

```
delta <= A / (Z_S + A)
KL(subset || dense) <= log(1 + A/Z_S)
norm(dense_output - kept_output)
    <= min(max(d_c) * A/(Z_S+A), sum(A_c*d_c)/(Z_S+L))
```

The lower block bound is Jensen's `b_c exp(beta dot(q,key_mean_c))`; the upper uses a key ball.
The reverse divergence `KL(dense || subset)` is infinite for any proper restriction of finite-logit
softmax. Lean's finite real-valued `klDiv`, whose `log 0` is totalized, must not be read as encoding this
infinity. Both support theorems and the implementation retain the correct direction.

The implementation opens blocks in descending upper partition mass, accepts seed blocks from another
router, checks all requested tolerances, and doubles the opened count on failure. It returns
`certified=False` if a block cap is reached without meeting them. The partial causal block is opened
using only visible keys, so future keys do not influence its bounds or decision.

The deterministic fixture (`python -m ssa.certified_attention`, seed 0, `n=4096`, `d=32`, `d_v=8`, `b=64`,
`beta=4`) reads 64 keys for concentrated logits at mass tolerance 0.001, with measured output error
`1.79e-6 <= 4.62e-5`. Flat logits require a full scan at that tolerance. Identical values permit an
output-only certificate after 64 keys even with omitted mass 0.984375. The first and last cases each
evaluate 64 key bounds and 63 value bounds; the full-scan case uses seven checks and 321 value bounds.

Construction reads the full cache once. Per query, the reference pays `O(Bd)` for key bounds,
`O(B log B)` for ordering, up to `O(B d_v log B)` for output certification, and the cost of opened
keys/values. At fixed block size this is not a sublinear router. It improves the error contract and
adaptive selection on the demonstrated geometries, not the existing GPU kernel's measured speed.
Six checks in `ssa/tests/test_certified_attention_gpu.py` compare the certificates against float64
CUDA SDPA on an RTX 4080, covering concentrated and flat logits, equal values, and both full and
partial causal blocks. These validate numerical agreement with an independent attention implementation;
the selector still runs on CPU. GPU routing speed and real-model quality remain unmeasured.

## 1. What the partial-score floor does and does not establish

`Potential/PartialScore.lean`

`PartialScore` bounds the error of one scalar used to approximate an objective over all completions of
a partial object. It does not by itself bound a router's selection error or computational cost.

**The mapping.** A partial object `p` is a **block**; its completions `E p c` are the **keys in that
block**; the objective `V c` is the **true attention logit** `⟨q, k_c⟩` for a fixed query; the partial
score `s p` is the **router's block score** `r_c(q)`.

| declaration | what it gives SSA |
|---|---|
| `reach_spread_le_two_mul_score_error` (`:117`) | if a router is uniformly `ε`-accurate on a block, the true logit spread across that block is at most `2ε` |
| **`score_error_ge_of_reach_split`** (`:130`) | **half the true logit spread across a block is a LOWER BOUND on the uniform error of EVERY block score.** A property of the partition and the query; no choice of summary moves it |
| `score_error_pos_of_reach_split` (`:138`) | two keys in one block with different logits force strictly positive uniform individual-logit approximation error |
| `not_exactOnReach_of_reach_split` (`:146`) | **no scalar equals every individual logit in a block whose logits disagree** — the `ε = 0` corner |
| `exists_exactOnReach_iff_contentRead` (`:160`) | welds the ceiling to `Accumulation/StoreContentRead.lean` · `contentRead_iff_separatesContent` (`:102`): one scalar reproduces every individual logit exactly when logits never differ within a block |

**Limit of this mapping.** A scalar cannot equal two distinct individual logits. It can still equal
their maximum, or upper-bound every logit, and a router can still retain the correct block. Thus
within-block spread does not prove failure of lossless selection. The cumulant score remains a
heuristic; the theorem does not prove its optimality.

**The mapping relevant to summary ambiguity.** Take the partial object to be a summary tuple, its
completions to be all blocks sharing that tuple, and the objective to be each completion's true maximum
logit. A pair of indistinguishable blocks with different maxima then gives a maximum-score error floor.
An attained upper-bound construction establishes a different claim: no universally admissible upper
bound on that summary can fall below the attained maximum. Paper §5.4 uses this second argument with
Samuelson's witness and `AdmissibleBound`, rather than deducing it from within-block spread.

Uniform score error `epsilon` is sufficient to preserve a strict top-k set when the true gap at its
boundary exceeds `2*epsilon`. Failure of that sufficient condition does not imply misselection.

---

## 2. Proper restriction leaves positive omitted mass at finite temperature

`Accumulation/SelectionWeights.lean`

The README describes SSA as "exact softmax over a *selected subset* of keys (a bounded, dropped-mass
error versus full attention)". This file makes the "versus" precise.

| declaration | what it gives SSA |
|---|---|
| `softmaxAt_le_of_le` (`:163`) | the weights are monotone in the score at any non-negative `β` — softmax expresses the ranking |
| **`selection_excludes_nothing_at_finite_scale`** (`:178`) | **at every finite `β`, each candidate keeps strictly positive weight.** Softmax never implements an exclusion |
| `selection_is_flat_at_zero_scale` (`:186`) | at `β = 0` the ranking is gone entirely — the selection is carried wholly by the scale |
| `softmaxAt_tendsto_top` (`:225`) | concentration on a **strict** top only in the sharp limit; on a tied face the limit is uniform on the face, not a point |

**What it buys.** A proper hard top-κ restriction differs from full softmax at every finite
temperature because omitted weights remain positive. Increasing `β` can shrink that mass when all
maximizers are retained with a strict gap, but does not make it exactly zero at finite `β`. If a tied
maximizer is omitted, even the sharp limit can retain nonzero omitted mass. Identical values can give
zero output error despite positive dropped mass; an output guarantee and a weight guarantee differ.

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

`Accumulation/RouteChargeCrossover.lean`

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

**A scalar price comparison needs an exchange rate.** Combining a quality penalty and compute cost
into a single weighted objective requires a coefficient such as `RoutePrice.lossPrice`. At zero loss
price the comparison ignores quality. A constrained comparison is another valid choice: minimize
work subject to a stated error tolerance, as in the adaptive certificate above. Pareto dominance also
needs no exchange rate. Thus the theorem does not make every speed/quality comparison underdetermined;
it identifies an assumption needed for its particular scalar objective.

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
