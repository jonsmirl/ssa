# Partial-coordinate reads with an attention-mass certificate

Status: implemented and measured CPU reference. On 16 previously inspected Qwen-8K
queries, reading 32 of 64 coordinates initially and then completing selected keys
certifies a 10% omitted-mass target using 49.71% of dense **logical K/V scalar reads**
on average. This is not a measured speedup, a semantic-retrieval result, or a
subquadratic algorithm. The reference scans every key and uses dense temporary arrays.

The highest-attention key is token 0 for all 16 queries. Its 100% top-1 recall is
therefore attention-sink retention, not evidence of semantic retrieval accuracy.

## Self-contained mathematical bridge

Let the visible prefix contain keys K_j, query q, and positive scale beta. After
reading coordinate set D, the partial attention logit is

\[
\widetilde s_j=\beta\sum_{a\in D}q_aK_{ja}.
\]

With certified global coordinate bounds B_a >= max_j |K_ja|, triangle inequality gives

\[
|s_j-\widetilde s_j|\le\sum_{a\notin D}|\beta q_a|B_a.
\]

The common-B specialization is Substrate's partial-pairing bound: B times the
unread query L1 norm. This is **not omitted attention probability**. Largest-|q|
coordinates minimize that norm at fixed coordinate count; this need not minimize
the weighted bound when B_a varies.

The tighter implemented alternative stores each block's coordinate minima m_ba
and maxima M_ba. For key j in block b,

\[
L_j=\widetilde s_j+\sum_{a\notin D}\min(\beta q_am_{ba},\beta q_aM_{ba}),\qquad
U_j=\widetilde s_j+\sum_{a\notin D}\max(\beta q_am_{ba},\beta q_aM_{ba}).
\]

Thus L_j <= s_j <= U_j. Revealing a coordinate replaces its interval by its
actual product. Intersecting successive intervals retains monotone bounds.
Completing all coordinates recovers the exact logit in real arithmetic.
No covariance, Bennett, outlier, learned, or probabilistic cap is introduced.
These are attention-logit bounds, not CCC routing-metric certificates.

Two independent certificates follow:

1. Strict top-set separation: min selected L > max unselected U. Positive scaling
   and positive-temperature softmax preserve the top set. This does not transport
   the numerical margin to probability space. Boundary ties fail strict separation;
   implementation index tie-breaking does not change that theorem.
2. Attention-mass certificate: feed unopened U_j to the existing score-tail reader.
   Disjoint bands with count n_b and ceiling u_b >= each member U_j give
   A = sum_b n_b exp(u_b) >= sum_unopened exp(s_j). With exact selected mass Z_S,

\[
\delta\le\bar\delta=\frac{A}{Z_S+A},\qquad
M_\eta=\log A-\log Z_S-\log\frac{\eta}{1-\eta}\le0.
\]

The last inequality is exactly the requested stopping condition for 0 < eta < 1.
For the selected, renormalized read, TV equals actual omitted mass, supported
KL equals -log(1-delta), and output L2 error is at most 2 B_V delta when all
visible value norms are <= B_V. The implementation returns the corresponding
upper bounds from bar-delta, using log arithmetic. Full selection has A=0.

A minimum of earlier tail caps stays admissible as the unopened set shrinks.
One band is the residual maximum-score cap. An unquantized sum exp(U_j) is
tighter than band rounding, but is exact residual mass only when U_j=s_j.
Even after every coordinate is read, a fixed 16-band profile generally has slack.

Private mathematical provenance: PartialPairingTop `7395e8a8a`, SparQ recognition
`495dcb05f`, and TopSetMonotone `f0ce0f345`. The latter supplies dedicated softmax
order equivalence, not an application of a pointwise g-composition theorem.
The formulas above and the public code suffice without access to Substrate.
The new Python composition is not itself Lean-verified. Guarded float64 arithmetic
is dense-oracle tested, **not a formal IEEE interval-arithmetic implementation**.

## Reproducible protocol

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.partial_coordinate_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests/test_partial_coordinate_attention.py ssa/tests/test_partial_coordinate_experiment.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

Base commit: `35c2c0b69338d67e9301da798ea1d04361b6d6a2`, plus uncommitted source
identified by SHA256 in the artifact. Fixture `/tmp/ssa_qwen_qkv_8192.npz`:
`5656b2725bbdb29a09c7c13c93126ac3f2cfb44ba0c32e60768ffc49fc54895a`.
It contains Qwen2.5-0.5B layer 18, KV head 0, Q/K/V dimension 64; beta=1/sqrt(64).
The cache is not included in the repository. No fresh model forward, training,
or held-out document evaluation was performed.

Queries are the 16 positions `linspace(4096,8191,16,dtype=int)` of this cache.
Every mode uses the same query and visible prefix ending at that query. Prefix
snapshots are built before routing; future keys cannot affect summaries.
Synthetic concentrated, random and centroid-cancelled extreme-key cases use seed
71, 1024 keys, key dimension 64, value dimension 8 and eight matched queries each.

Block size is 64, tail levels 16, coordinate counts 0/8/16/32/64. A partial causal
boundary block is force-kept. Other seed keys are chosen by partial logit, with
lower index breaking ties. Seed size is 128 plus the partial boundary length.
Selected keys are completed exactly; only their values contribute to the read.
If needed, fetch batches of 128 unopened keys with largest U until certified or
the stated value budget is exhausted. Dense oracles run **after** this policy stops.
Value norms are scanned for build-time metadata, not used to select a route.

Comparisons include fixed seed-only value budgets for both global-absolute and
block-interval bounds, unrestricted certificate stopping for block intervals,
and a progressive-coordinate policy capped at 10% of value rows (or the seed
budget, if larger). The progressive policy refines all coordinates before fetching
additional values; it is deliberately conservative, not a cost-optimal policy.
The existing radius reader uses identical queries/prefixes/targets but fetches
whole blocks, so its comparison changes routing granularity as well as the bound.

## Measurements

Percentages below are means over 16 Qwen queries. Every unrestricted stopping
row certifies 16/16 queries. K means all queried coordinate scalars, including
completion of fetched keys; V means value rows. Combined K/V weights dimensions
equally here because d_K=d_V=64.

| Mass target | Initial coordinates | K read | V read | Combined K/V | Actual omitted mass |
|---|---:|---:|---:|---:|---:|
| 10% | 8 | 97.68% | 97.35% | 97.51% | 0.00053% |
| 10% | 16 | 83.57% | 78.09% | 80.83% | 0.02865% |
| 10% | 32 | 66.47% | 32.95% | 49.71% | 0.83443% |
| 10% | 64 | 100% | 9.86% | 54.93% | 5.28720% |
| 1% | 16 | 94.31% | 92.41% | 93.36% | 0.00239% |
| 1% | 32 | 81.16% | 62.32% | 71.74% | 0.08496% |
| 1% | 64 | 100% | 32.64% | 66.32% | 0.67970% |

Zero-coordinate and existing radius stopping both read all keys/values at both
targets. The exact-score oracle minimum value fractions are 5.94% for 10% mass
and 27.83% for 1% mass. Constructing this oracle scores every key; it also lacks
the real protocol's seed/batch constraints. It is a floor, not a cheap router.

At 32 coordinates and the 10% target, mean certified mass upper bound is 9.055%,
head-output L2 error 0.01989, and output-error upper bound 2.1716. The latter is
loose: this is a mass guarantee, not a promise of comparably small value error.

The fixed value budget averages 156.5 keys (2.65% of visible rows):

| Initial coordinates | Top-16 coverage in fetched set | Block mass upper | Actual omitted mass | 10% certified queries |
|---|---:|---:|---:|---:|
| 8 | 83.98% | 99.96% | 26.86% | 0/16 |
| 16 | 97.27% | 97.60% | 20.27% | 0/16 |
| 32 | 100% | 68.69% | 17.89% | 1/16 |
| 64 | 100% | 23.99% | 17.45% | 2/16 |

None of these fixed-budget rows certifies 1% mass. Top-16 coverage is inclusion
in a larger fetched set, not precision@16 or semantic retrieval accuracy.
At r=32 the top-1 strict certificate succeeds for all queries, yet the mass
certificate succeeds for only one. High top-key recall is not low omitted mass.
At that setting the global-absolute bound is 89.27%, versus block intervals'
68.69%; the single residual maximum gives 94.22%, and unquantized block caps give
60.32%. At r=64, those last two are 73.92% and the exact residual 17.45%, while
the 16-band result is 23.99%. Bands materially help but remain conservative.

The progressive policy with a 10%-value cap certifies 9/16 queries at the 10%
mass target and 2/16 at 1%. It reads **all key coordinates** in both cases.
The remaining queries are explicitly uncertified, not silently accepted.

Synthetic r=32 stopping uses combined logical K/V fractions 55.86%/56.79%
(concentrated), 99.39%/100% (random), and 57.66%/99.49% (adversarial), at the
10%/1% targets respectively. These have d_V=8; fractions are dimension-weighted,
not directly equivalent to the Qwen K/V average. Favorable geometry is not assumed.

## Work accounting and verification

For a fixed r and k completed keys, unique logical key reads are exactly
`n*r + k*(d-r)`; value reads are `k*d_V`. These are not memory-controller byte
measurements. Index construction reads all n*d key scalars; value-bound
construction scans every value row. Both are rebuilt per reference query here.
The index stores `2*ceil(n/64)*d+d` scalar extrema (both bound modes retained).
The reference also stores the full archive and O(n*d) query terms/masks. Each
refresh visits n*d coordinate slots; tail profiles and sorts inspect unread keys.
Oracle scans, final diagnostic caps, build storage and refresh work are separately
itemized. Slot counts are not a complete FLOP or cache-traffic count; resident
state excludes transient arrays and is not a peak-memory measurement.

For example, the final 8192-key Qwen query at r=32, eta=.1 reads 405,504 logical
key scalars and 4,480 value rows. It also performs 37 dense refreshes (19,398,656
coordinate slots), 303,104 key-bound evaluations and 206,080 profile-item visits.
The index summary is 131,584 bytes; reported resident query state is 4,924,416
bytes, excluding the archive and transient arrays. Thus the reference can do much
more arithmetic than dense attention despite its smaller logical read ledger.

All 880 partial-coordinate query/mode trials pass the 1e-9 numerical audit:
maximum logit-interval deficit 7.11e-15; mass and KL deficits zero; output-bound
deficit 1.98e-15; false strict top-set certificates zero. Tiny discrepancies include
canonical elementwise-sum versus BLAS-dot rounding at exact/full-read endpoints.
This finite audit does not establish soundness of arbitrary floating-point inputs.
Focused tests: **102 passed in 0.24 s**. Full suite: **652 passed, 14 existing
deprecation warnings in 46.98 s**. Tests include poisoned future prefixes, equal
logits, strict ties, extreme keys, value-independent routing, full recovery,
monotone retained caps, separate top-set/mass outcomes and the exact read ledger.

Artifacts: [measurements](../runs/partial_coordinate_attention.json),
[verification](../runs/partial_coordinate_verification.json). The Qwen portion of
this full multi-mode CPU reference run took 109.96 seconds including builds and
oracles; it is not a benchmark against an optimized dense kernel.

Next practical test: fuse coordinate bounds and tail accumulation, charge summary
traffic, and measure end-to-end GPU latency against exact dense attention on fresh
documents and multiple heads. No result here proves that this will be faster,
that all heads have favorable tails, or that fixed-r all-key scans become
subquadratic across n query positions. The learned corrective-read controller
remains untrained and unused by this experiment.
