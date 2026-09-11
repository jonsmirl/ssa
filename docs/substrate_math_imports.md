# Substrate math that applies to SSA — an import list

This assessment maps inspected Substrate results to SSA's selection, attention error, and cost models. The
relevant audit includes the routed comparison-path review at `68968fc7a`, the cell-mean review at
`206193290`, and earlier review at `1cad53d99`
(2026-09-10), including the potential-store precursor
`88a4c7012`, the six subsequent SSA commits, the variance-sensitive mass-tree core, and its Inference
consumer named below.
The adaptive output certificate in `ssa/certified_attention.py` (paper §5.7) combines log-sum-exp bounds with
restricted-read identities and value geometry. Its abstract mass, divergence, exact residual, and two output
arms are now machine-checked; the Python instantiation remains a tested implementation rather than extracted
Lean code.

The actual-cell-mean tail now has its own formal owner at
`206193290e5aa1b0465d518f12625c10952b0953`: `CellMeanProjection`, `SelectedCellMeanRead`,
`ResidualStep`, appended output transport in `RestrictedReadOutputBound`, and
`Attention/CellMeanReadRecognition`. The public [specification](coarse_read_projection.md) reproduces
the definitions, proofs, and streaming identities without private dependencies. The central identity
is `KL(r || p) = KL(r || q_P) + log(Z/Z_P)` for normalized nonnegative cell-uniform competitors.
It gives a unique reverse-KL optimum, monotone partition refinement, and exact gain accounting.
Selected singletons and actual omitted count/key/value sums realize the read; full selection is dense.

`Z_P` is a **lower** mass bound. Conditional KL/TV/output transport requires a separate supplied upper
bound `U >= Z`; it is useful only when `U/Z_P` is close to one. Residual-step theorems characterize
improvement through alignment, and a positive normalized omitted-read counterexample rules out
unconditional safety. The scalar movement identity applies per head or to a shared scalar step;
independent head gates must not be commuted through the output projection. The selected experimental
prototype-plus-cap mode is not the actual-mean variational optimum. None of these results certifies
its floating-point implementation, improves CE by theorem, or establishes subquadratic query work.

The connected route-aware certificate is formalized at
`68968fc7aae0a024172288f090f25c4d61451324`. Its [public specification](routed_correction_certificate.md)
includes definitions, proof arguments, and nonlinear/affine witnesses. The owners are:

| Owner | What is established | Application boundary |
|---|---|---|
| `RestrictedReadOverlap` | `TV(p_S,p_T)=1-Z_(S∩T)/max(Z_S,Z_T)` and output/union-read bounds | Same actual logits and values at one state; nonempty selected sets; neither read is thereby dense-equivalent |
| `ComparisonTrace` | Stability of every executed comparison and its leaf; hard-top-one discontinuity witness | Finite unrolling and all real decision functions must represent the implementation; tie-breaking alone gives no positive stability radius |
| `QuadraticRemainder` | Sharp `H/2` vector remainder from uniformly Lipschitz derivatives | A derivative at one checkpoint is insufficient; the bound holds on the stated convex neighborhood |
| `LinearPath` | Signed pullback identity, remainder intervals, changing state spaces, affine exactness | Signed jumps may be retained or charged by their bounds; neither is free |
| `ComparisonPath` | One hypothesis set closes radii, trace/jump alternatives, and terminal strict margins | Include cache/summary dependence and the actual correction at the perturbed state; later nonlinear maps need their own stages/bounds |
| `ComparisonPathWitness` | Complete nonlinear and affine route-change instances | Satisfiability examples, not evidence that Qwen bounds are useful |
| `Attention/RoutedCorrectionRecognition` | Strict reference-winner preservation and charged attention-stage instantiation | Only the supplied candidate set, not untested vocabulary; preservation does not imply correctness |

SSA implements a float64 whole-path checker and a complete small causal-transformer provider with
analytic uniform derivative/guard bounds and charged reference/union work. See paper §5.16 and
`runs/routed_certificate_reference.json`: 18/28 algebraic passes, 14 nonzero accepted corrections,
and zero numerical interval violations. Useful Qwen path bounds and verified floating-point execution
remain open. The separate Qwen experiment in §5.17 compares two computed endpoints directly; its
reference-winner preservation is by construction, not validation of the nonlinear path hypotheses.
No model-quality or runtime claim follows from the Lean build alone.

`ssa/score_tail_certificate.py` now supplies a new concrete admissible mass cap to that same reader: exact
unopened block counts are grouped under separately certified attention-logit upper edges and their
exponential band sums are added. The restricted-read/output transport and “minimum of admissible caps” step
reuse the audited Substrate results. The finite disjoint-band inequality itself is proved in public paper
§5.7 and Appendix B.20 but is **not currently a Substrate theorem**. CCC remains only a routing-metric seed;
the Python reader derives a distinct mean-plus-radius attention-logit cap before invoking the mass algebra.

The focused source review covers `PartialScore`, `AdmissibleBound`, `LogSumExpBound`, `SelectionGeometry`,
`ValueAwareSelection`, `RestrictedReadBound`, `RestrictedReadOutputBound`, `BoundedTopSelection`,
`BoundedReadMiss`, `ComposedSelectionPlan`, `StorePotentialRead`, `PotentialStoreProcess`,
`CumulantEnclosure`, `SequentialErrorComposition`, `GroupedPairingClaim`,
`DistributionWeightedPartition`, `IndexReadMiss`, `BoundedExponentialMass`, `MomentMassTree`,
`MassTreeRead`, `MassTreeComplexity`, `OutlierMassTree`, `AddressLoss`, `PairingReadWriteBack`,
`ScoreWriteBackExecution`, `ReadRunChain`, `ParameterReadChain`, and their recognitions including
`Attention/OutlierPeeledReadRecognition`, alongside SSA's current
implementations and imported results. It is not an exhaustive review of the Substrate tree.
`Carrier/Simplex/ApproximateSelection.lean` concerns continuous selections of correspondences and does
not provide a sparse-attention selector or a runtime improvement here.

**Formalization scope.** Source theorems are machine-checked in Substrate. **The mapping to SSA is not
machine-checked.** The previously separate restricted-read and output-bound composition is now checked in
`RestrictedReadOutputBound.lean`, and the reservoir/cap/causality composition is checked in
`ComposedSelectionPlan.lean`. Paper §5.7 and Appendix B remain self-contained for public readers; Python
tests compare the implementation to dense and float64 oracles. Float64 evaluation with an outward
score cushion is not interval arithmetic. Declaration names are the stable lookup keys; line references
in the exploratory entries below are not a guarantee about a concurrently changing Substrate checkout.
All five universal modules and four inference recognitions from `130cae3e9` rebuilt successfully in that
audit. The eleven theorem and recognition modules named by `88a4c7012` through `77cd1b8c6` were rebuilt
together in the current audit (8,760 jobs); their emitted `#print axioms` reports contain no dependencies
beyond the standard `[propext, Classical.choice, Quot.sound]`, with several elementary declarations using
fewer or none. The later mass-tree core was reported clean at 10,868 jobs, its covariance extension at
15,101 jobs, and the outlier-read recognition at 15,103 jobs with the same standard axiom footprint.

Paths are relative to `~/substrate/lean/Substrate/Universal/`.

## Current applicability audit

This is the present disposition of the Substrate activity after `908ec0d6d`, not a change log. “Imported”
means the public SSA paper now gives a self-contained statement and proof or the implementation already
realizes the invariant. It does not mean that the Python/CUDA program has been extracted from Lean.

| Commit and result | SSA disposition | Boundary carried into SSA |
|---|---|---|
| `466f3e543` — `SelectedPrefixRead`, `TwoHopBlockCover`, `SignedGainTracking` and recognitions | **Imported in three places.** Algorithm 1 and `_stream_mask` cut routed reads at original position; the paper compares the complete two-hop cover and the conditional `w=sqrt(n)` split; the gain-sign result is recorded for the compression comparator. | Whole-set reads and chunk-only masking can leak; connectivity is not dense-softmax equivalence; nonnegative decay cannot reverse sign but says nothing about selection quality. |
| `9d6c33e30` — `PairingCapAdmissibility` and bounded-region recognition | **Imported.** The center-radius cap is an admissible skip test at every tree node. The public appendix states the threshold/drop-set consequence and the CPU tree has a parent/child monotonicity regression test. | A descendant's tighter cap drops a superset at the same threshold. This proves a pruning order, not traversal cost, termination, GPU arithmetic, or fixed-beam quality. Region counts are not key counts or wall time. |
| `a2f98ea85` — `ReachableFamilyDimension` | **Expressivity fence imported.** With either pass fixed, varying the other reaches a strict linear subspace in nondegenerate product dimensions; at equal block/slot counts the parameter ceiling is cubic while the target is quartic. | Nothing here bounds the family with both passes free, and no rank bound follows; products of the two differently blocked factors can be full rank. |
| `27e89ead3` — `TerminatingTransferSum` and multi-hop recognition | **Architecture boundary imported.** A strictly below-diagonal linear interaction is nilpotent and `(I-A)^{-1}=sum_{k<n} A^k` with no decay assumption, so the inverse-shaped linear operator contains every finite hop. | Ordinary causal softmax includes the diagonal and a transformer layer includes normalization and nonlinear maps. The identity neither turns current SSA into a one-layer multi-hop solver nor proves the source's depth or cost claims. |
| `9fdca844a` — `LogConcaveRatio` and representative-probe recognition | **Relevant, not instantiated.** A positive log-concave score-spread function has an antitone ratio across a fixed block displacement. It can justify a representative-probe heuristic only after its empirical shape hypothesis is measured for SSA's model and block size. | It is not an admissible upper bound and licenses no exact skip. The second-order condition is stronger than log-concavity; a falling ratio at one displacement is weaker and says nothing about another displacement. SSA routes with block summaries, not one sampled position. |
| `036cd2914` — `BaseCostOptimum` | **Candidate, not retuned.** In the scalar node-work model `f log_f B`, the continuous optimum is `e` and integer fanout 3 beats every other integer `f>=2`. | The model omits GPU vectorization, memory traffic, fixed-beam recall, and tree-build cost. SSA's measured fanout 16 is unchanged until a controlled end-to-end sweep supports a replacement. |
| `21e49cbf3` — `ReflectionCorner` | **Comparator-only import.** A rank-one delta correction reverses its key line past `beta*||k||^2=1` and is exactly a norm-preserving reflection at equality 2. | This informs the repository's DeltaNet comparison, not sparse key selection. The larger reflection-product expressivity theorem supplies no routing, attention-quality, or long-context execution guarantee. |
| `183c4824d` — `WindingBridge` | **Length-generalization bridge imported.** Under the no-half-turn anti-aliasing hypothesis, the lifted phase-loop turn count agrees with the discrete winding degree, coordinatewise for a torus of rotary bands. | RoPE does not by itself prove extrapolation to unseen offsets. The theorem connects two exact winding representations; it supplies no quality or kernel result. |
| `a8d37a176` — audited `PhaseWindingRecognition` wording | **Status correction imported.** The public paper separates relative-offset algebra from winding stability: a nonzero turn needs at least one wavelength, and a changed turn count forces the anti-aliasing margin to fail somewhere. | This commit repairs a stale absence claim—the bridge already existed from the other side. It adds no kernel primitive. The 10M YaRN result remains empirical, especially at its roughly 306x scale factor. |
| `130cae3e9` — restricted reads/output, bounded top selection, bounded read miss, and composed plan | **Imported.** The paper's mass/KL/output certificate is now backed by exact residual and block-bound theorems; CCC names an index-tie-broken top set; the no-free-selection claim is narrowed to the proved grounded adaptive-read model; and the 14×70/9/128 reservoir, uniform layer cap, past bound, and causal cut are one checked theorem. | Positive weights and a nonempty kept set are required. Strict top-set exactness needs strict exclusion; otherwise an explicit index order pins the result. The `b/n` ceiling needs outputs contained in the probe trace and does not cover arbitrary preprocessed indexes. The composed plan remains conditional on nine votes and proves neither quality nor speed. |
| `88a4c7012` — `StorePotentialRead` and hierarchical attention recognition | **Imported.** The paper defines one finite score/value store whose log-partition derivative is its scalar softmax read, identifies the attention/Hopfield instance, and connects region score caps to the selected-output certificate. | The hierarchy, partition, kept set, bounds, and traversal are supplied. The two-cumulant score is not the potential, and no work or quality claim follows unless the displayed certificate is small. |
| `fae7dad0f` — `PotentialStoreProcess` | **Imported.** The exact fresh-append mass, potential, surviving-weight, and read laws are stated and proved publicly, including the decomposition when the score surface also changes. | Append-only storage does not imply stable access weight or degraded output. No score update, cumulative drift, recovery, or convergence result is supplied. |
| `1459a8781` — `CumulantEnclosure` and cumulant-routing recognition | **Imported.** A block logit range turns the mean-plus-variance routing statistic into a deterministic interval for normalized log-sum-exp. | The cubic remainder needs a third-central-moment bound along the whole tilt segment; its value at zero alone is insufficient. The interval selects no route and proves no retrieval accuracy. |
| `7ecd7832f` — `SequentialErrorComposition` and multi-hop recognition | **Imported.** Routed-read state error obeys the Lipschitz-weighted recurrence, with additive and exact corners and a counterexample to any final bound from local errors alone. | Lipschitz gains and same-input local output errors are hypotheses; score error does not automatically become state error, and this is not a probability-product law. |
| `51a71137f` — `GroupedPairingClaim` and query-group/transport recognition | **Imported.** The query-group top pairing is the least shared safe claim, retains a maximizer for each query below its maximum, loses drop power monotonically as queries are added, and is fixed by one common linear isometry. | No cheap query summary or algorithm is provided. Distinct position-dependent transports need not preserve even pairing order, and no attention-output claim follows. |
| `6a88f33e5` — `DistributionWeightedPartition` and certificate-aware partition recognition | **Imported as an objective.** The paper gives the weighted node-read cost, safe-partition monotonicity, strict distribution sensitivity, and least member of a supplied finite certified family. | This neither constructs nor learns a partition, optimizes globally, proves generalization, supplies an asymptotic, nor identifies node count with latency. |
| `77cd1b8c6` — `IndexReadMiss` and indexed/randomized recognition | **Imported.** The probe lower bound now accounts for a finite index: `K` states, depth `b`, and unread reference-output width `a` reach at most `K(b+a)` placements; finite randomization yields one fixed low-recall placement. | The identity index shows why `K` is necessary. State count is not charged as bits, construction, lookup, or arithmetic cost, and the one-spike equality model is not general scored retrieval. |
| `6b3da713a` — bounded exponential mass and moment-tree certificates | **Imported and experimentally instantiated.** `CertifiedTreeAttention(..., mass_bound="bennett")` stores trace spread, evaluates the stable Bennett cap, and takes its minimum with the radius cap. The experiment found zero cap underestimates and a large synthetic work reduction, but no sparse exact certificate on the measured Qwen head. | The vector trace lift can be much looser than directional covariance. The work theorem assumes a per-level active-node bound and an arithmetic subquadratic inequality; it does not derive either from geometry or count total kernel work. Proper-frontier attainment, tree learning, and real-model usefulness are not theorems. |
| `da13ebeba` — full-covariance moment-mass certificate | **Imported and experimentally instantiated.** The covariance identity removes the trace lift's dimension loss. `mass_bound="bennett_covariance"` recursively stores the matrix and evaluates $q^TCq$. It reduced the median real-Qwen cap by 6.06 log units versus 0.73 for trace, with zero observed underestimates. | Storage is $O(d^2)$ per node and evaluation is a quadratic form. Median cap slack still measured 28.03 log units and the traversal opened every visible block, isolating the worst-radius exponential tail as the remaining obstruction on this head. |
| `9c6b1ad35` — deterministic outlier-peeled mass tree | **Imported and experimentally instantiated.** The reference peels the largest residual norms with stable index ties, scores their mass exactly, recentres the core, and applies trace or covariance Bennett. At $t=4$ plus covariance it tightened the median Qwen cap by 11.96 log units with zero observed underestimates. Its force-kept form was also tested by promoting exposed keys' containing blocks. | Median slack was still 23.39 log units and every visible block opened, including with force-kept seeds; at $t=4$ those seeds reduced mean bound evaluations from 190.63 to 166.78 but not keys read. Fixed peel depths cost $O(td)$ node storage/query work; the experiment does not claim raw-depth monotonicity, implement an optimized builder or individual-key side channel, or derive a bounded active frontier. |
| `fad55ff82` — outlier-peeled read recognition | **Imported as the Inference consumer.** Eleven direct re-exports connect deterministic tied peeling, residual geometry, three mass caps, peeled-set TV/KL/output/causal certificates, and work/storage accounting without duplicating the Universal proofs. | The quality certificate does not earn the runtime conclusion by itself. Subquadratic work still assumes bounded active width, the per-node certificate charge, and the explicit total-work inequality; moments, radii, construction cost, and finite-precision behavior remain supplied or external. |
| `23f670cab` — `AddressLoss` | **Imported as the repair boundary.** A held-but-unreached item and an absent item can give the same thresholded read. The recurrent repair reference therefore does not claim its fixed state can infer arbitrary omitted content from sparse output alone. | The richer cost observable in the source can distinguish the witness. This is an indistinguishability statement for the supplied read, not an impossibility for every indexed router or side channel. |
| `141011487`, `9ebd1fc2f` — pairing-compatible read/write-back and its execution | **Imported as a candidate controller mechanism.** Site offsets may be updated without replacing archived payloads; relative winner access changes exactly under the supplied update. SSA uses hard exclusion only in the CPU reference to prevent duplicate mass. | The source self-reinforces the old winner for nonnegative rates and proves no later-query recall improvement. Finite score offsets are not hard exclusion; query-specific suppression can break the particular pairing-energy compatibility. |
| `d388de772`, `fa62feca4` — finite read-run and parameter-read chains | **Imported as the recurrent-read architecture.** A boundary may update carrier, query, and parameters while the raw archive is retained, and local energy/displacement costs compose over the finite chain. This matches a fixed-state controller steering later sparse reads into the unchanged KV archive. | These are not computational operation counts; the separate opened-key bound follows from finite-union cardinality. Boundary maps, useful query updates, favorable geometry, and convergence are supplied rather than derived. |
| `1cad53d99` — finite read-energy floor and score-loss correction | **Relevant accounting, not yet a training guarantee.** A bounded finite support gives an explicit read-energy floor, and a pointwise one-sided score-loss bound controls the same-support energy increase. | The floor retains a log-cardinality term. Neither arbitrary CE updates nor contraction of a fitting loss supplies the required pointwise score comparison, so this does not prove repair training converges. |

The other commits in this interval are registry, sweep-generation, economics-citation, or unrelated carrier
maintenance. They were reviewed but do not change SSA's claims or implementation.

## Implemented: hierarchical adaptive certificates

Substrate gained `Potential/Entropy/BlockLogPartition.lean` and
`Potential/Entropy/TreeLogPartition.lean` after the first review.  The declarations
`blockSoftmax_one` and `treeSoftmax_unitScale` prove that a unit-scale block/tree decomposition has
exactly the same site weights as ordinary flat softmax.  This licenses a useful implementation
distinction: SSA may organize *bounds* hierarchically without changing the target distribution.

`ssa/hierarchical_certified_attention.py` combines that identity with the existing admissible key
and value balls.  An unopened tree node bounds the partition mass of every key below it; best-first
refinement replaces the node by its children, and a leaf is opened only when necessary.  The dense
output contract and the mass, KL, and value-aware certificates are unchanged.  On concentrated
geometry this avoids the flat reference's mandatory evaluation of every visible leaf summary.

This is a CPU reference, not a new kernel result.  The tree is positional and its parent balls can
be loose; diffuse geometry may force all leaves, and the same `O(n)` worst case remains.  Substrate
proves the unit-scale tree identity, not this Python composition or its floating-point arithmetic;
the latter is tested against the dense oracle, including causal prefixes and block caps. Its float64
leaf and parent radii receive the same style of conservative inflation, while exact zero radii are
preserved only when member equality establishes them independently.

`CausalTree` in `ssa/cascade_router.py` is the GPU routing counterpart. It maintains an online fanout
tree over committed sub-block means. Each node stores a center and the recursively composed radius
`max_child (norm(child_center - center) + child_radius)`, so Cauchy–Schwarz gives the admissible search
priority `dot(q, center) + norm(q) * radius`. A fixed beam deliberately makes it an approximate selector,
not an output certificate; strict causality comes from searching only the committed frontier while the
current chunk is scored exactly. This backend was required because FAISS-GPU's IVF index killed the
Blackwell Kaggle process.

The production tree now inflates each float32 norm-plus-child-radius candidate by
`8 * (d + 4) * eps`, applies `nextafter(+inf)` to the candidate and selected maximum, and uses an
absolute-error allowance plus upward rounding for query/node caps. `python -m ssa.float_tree_verification`
exercises the actual `CausalTree` on an RTX 4080 against float64 descendant oracles. Across five adversarial
geometries and fanouts 2, 4, and 16 it checked 90,105 balls and 1,081,260 caps. Unguarded float32 arithmetic
underestimated 8,701 radii and 367,480 caps; guarded arithmetic had zero observed underestimates. The
65,536-leaf fanout-16 build microbenchmark was 0.433 ms guarded versus 0.223 ms raw. This closes the concrete
implementation gap empirically for the tested regimes; it is neither a directed-rounding proof nor a claim
that fixed-beam routing finds the exact top branch.

The failed first 10M quality run was not caused by an invalid center-radius inequality: exact-rank probes
found no relevant violation of that bound. Two composition choices were wrong. First, routing post-RoPE
vectors made semantic proximity depend on a huge positional rotation; the corrected executor routes on
pre-RoPE content Q/K but evaluates attention with the unchanged post-RoPE Q/K. Second, independently selected
evidence could disappear across heads and layers. The corrected bounded union stores the 128 blocks with the
most layer-1 head votes and injects them into each later sparse plan. This cap has a direct counting guarantee
for the observed case: 14 heads × at most 70 base selections = 980 votes, so at most `floor(980/9)=108` blocks
can receive the needle's nine-or-more votes; a 128-slot reservoir must retain it. The final run processed every
layer/head of Qwen2.5-0.5B at 10,000,128 tokens in 713.2 s (25.72 GB peak, 0.506% selected upper bound), and the
single semantic NIAH probe passed. This validates that measured composition, not general quality preservation.

---

## Implemented: adaptive mass, KL and value-aware output certificates

| source | inspected declaration | SSA use and scope |
|---|---|---|
| `Potential/Entropy/RestrictedReadBound.lean` | `totalVariation_restrictRead_fullRead`, `klDiv_restrictRead_fullRead`, `klDiv_fullRead_smoothRead_unbounded` | Exact nonuniform restricted-read TV and selected-to-dense KL identities; the reverse divergence is proved unbounded by smoothing the omitted support toward zero. Positive weights and a nonempty kept set are explicit. |
| `Potential/Entropy/RestrictedReadOutputBound.lean` | `fullOut_sub_restrictOut_eq_omit_residual`, `norm_fullOut_sub_restrictOut_le_block_min`, `fullOut_eq_restrictOut_of_one_value` | Exact residual centered at the available restricted output, both block certificate arms and their minimum, plus the equal-value zero-error fence. These now match the public §5.7 proposition directly. |
| `Potential/Entropy/LogSumExpBound.lean` | `logSumExp_max_sandwich`, `softMax_sandwich` | An admissible maximum-logit bound `U_c` gives unopened block partition mass at most `b_c exp(beta U_c)`; actual block counts matter. |
| `Generator/Dissipative/SelectionGeometry.lean` | `barycenter_truncation_bound` | Earlier supporting geometry. The exact computable form centered at the kept output is now owned directly by `RestrictedReadOutputBound.lean`; no substitution for an unknown dense output is needed. |

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
For a proper restriction, the finite real-valued `klDiv` still totalizes `log 0`; it is not itself infinity.
`klDiv_fullRead_smoothRead_unbounded` states the correct reverse-direction fact as a limit: leave omitted
sites an `epsilon` share, and the dense-to-smoothed-selection divergence tends to `+infinity` as
`epsilon -> 0+`.

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

## Imported: bounded top selection and explicit ties

`Carrier/Separation/BoundedTopSelection.lean` separates four facts that earlier SSA prose compressed into
“exact top-k.” An admissible cap bounds every member of its region. If every unprobed cap and every truncated
probed member are **strictly** below threshold while every return is at least the threshold, the return is
exactly the above-threshold set and every outsider is strictly below every insider. The returned count is a
hypothesis. With a boundary tie, weak top-k does not identify one set and strict top-k may not exist; ordering
equal scores by carrier index pins one without inventing a margin. `CausalCascade._ordered_candidates` now
implements that repair by preferring the larger parent index. Indexed certification retains strict `< tau`
checks; exhaustive routing uses the explicit tie order.

## Imported: the grounded adaptive-read ceiling

`Carrier/Separation/BoundedReadMiss.lean` proves the adaptive argument using the all-false reference trace.
A spike outside that trace changes no probe answer and therefore changes neither trace nor output. If the
selector returns only positions it probed, at most `b` of `n` spike placements are recalled by a depth-`b`
tree; uniform recall is at most `b/n`. Averaging finite seeded selectors puts the same ceiling on some fixed
placement. The hypotheses matter: a zero-depth leaf returning the whole carrier recalls every placement, so
no clean budget bound holds without grounded output. Preprocessed indexes carrying side information outside
the query-time probe trace are not modeled. The public paper's former unrestricted wording has been narrowed
to exactly this proved scope.

## Imported: one composed 10M routing plan

`Aggregation/ComposedSelectionPlan.lean` checks in one theorem what the earlier imports established
separately. Under 14 selectors of width at most 70, the positive vote floor 9, a full top-count reservoir of
capacity 128, bounded later-layer selections, and past-bounded inputs, every nine-vote item remains in every
layer plan; each plan has at most `roundWidth + 128` distinct blocks, remains past-bounded, and its positional
cut is causal. A concrete 140-item/two-vote construction shows the reservoir may drop an item below the vote
floor. The conjunction is not an attention-output, FLOP, or latency result and does not establish that the
measured needle received nine votes.

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

## 6. Read-budget impossibility is formalized under explicit access assumptions

The randomized-selector item is no longer open merely for lack of probability machinery.
`LosslessSelectionLimit` supplies the oblivious uniform-spike bound; `BoundedReadMiss` supplies
adaptive indistinguishability and finite randomization with the grounded-output condition; `IndexReadMiss`
accounts for a finite preprocessing index. The earlier sections give the precise hypotheses and owners.
These results do not rule out arbitrary uncharged preprocessing, richer access models, or favorable
learned geometry. They should not be advertised as an unconditional impossibility theorem for SSA.
