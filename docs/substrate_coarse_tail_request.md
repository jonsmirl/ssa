# BUILD: variational coarse-read projection, not another exponential-mass cap

**Status: completed in Substrate commit `206193290e5aa1b0465d518f12625c10952b0953`.**
The public, self-contained result with proof arguments and streaming pseudocode is
[coarse_read_projection.md](coarse_read_projection.md). The request below is retained to identify the
scope of that build; it is not outstanding work.

The result uses existing averaging, normalization, and KL operators. `CellMeanProjection` owns the
projection, uniqueness, refinement, gain, and conditional upper-to-lower certificates;
`SelectedCellMeanRead` owns selected-singleton labels and exact omitted summaries;
`RestrictedReadOutputBound` transports Pinsker to normed outputs; `ResidualStep` owns capped movement,
the weak/strict alignment criteria, and a positive-weight, positive-value counterexample.
`CellMeanReadRecognition` specializes them to attention.

The operative qualifications are:

- Competitors are normalized nonnegative cell-uniform weights and may have zero cell masses. Actual
  mean logits, not assignment prototypes, define the optimum.
- Finite gains need no sign assumption for the identities. Nonpositive gains lower the partition but
  do not improve the optimal reverse KL. Key and value dimensions may differ.
- Empty omitted cells contribute zero; an empty selected set is allowed by the coarse-read formula.
  A selected-only probability read used as the blend base requires nonempty selection.
- Positive steps use a weak alignment inequality for nonincrease and a strict inequality for strict
  improvement. The movement identity does not require the cap to be at most one; that restriction
  belongs to the convex-blend interpretation.
- Causality is supplied by the visible carrier and future-independent assignments/summaries. No
  favorable upper mass bound, useful fixed-budget certificate, subquadratic work/storage, learned
  geometry, CE/retrieval improvement, or floating-point/CUDA correctness is proved.

SSA's selected experimental revision is capped **prototype** mixing, not the zero-gain cell-mean
projection. This build proves its scalar movement/alignment algebra, not its empirical quality or a
variational guarantee for those prototype scores.

SSA is testing replacement of stale first-prefix routing prototypes with the actual mean of each unopened cell.
Please build the mathematical foundation below in Substrate. SSA will own numerical verification,
training, and real-model experiments. Audit for existing owners first; do not rederive existing results.

Inspect the existing finite Gibbs/normalization, KL, conditional partition, and output-read machinery,
especially `BlockLogPartition`, `TreeLogPartition`, `StorePotentialRead`, `InformationInequalities`,
`RestrictedReadBound`, `RestrictedReadOutputBound`, `MassTreeRead`, `DistributionWeightedPartition`,
and `ReadWeightedError`. `PartitionStoreProjection` is a compatible coordinate-projection result;
do not assume it already states the finite cell-uniform variational claim below. Reuse any applicable
existing theorem rather than building a parallel owner. The proposed mathematics is elementary real
arithmetic, not a request to prove properties of floating point or optimization.

Measured motivation and priority: the prototype estimate overstates residual mass by a median 30.17x
on the inspected Qwen head, whereas the actual-mean estimate is a safe real-arithmetic lower bound.
However, the actual-mean replacement still worsens 32K development-validation CE versus sparse-only.
SSA therefore selected a capped prototype mixture for its next full-corpus regression run, not the
variational estimator. This proof request establishes the exact coarse-family baseline and its limits;
it must not be described as a measured solution to the model-quality problem. A separate one-layer
diagnostic found only 0.286% extra projected-error reduction from an oracle scalar suppression gate,
so scalar gate training is not yet justified as a substantial next advance either.

## Object and central theorem

Let I be a nonempty finite carrier, with finite real logits s_i and a partition P into nonempty cells.
Put p_i = exp(s_i)/Z, Z = sum_i exp(s_i). For each cell C define

    m_C = |C|, a_C = (sum_{i in C} s_i)/m_C,
    Z_P = sum_C m_C exp(a_C),
    q_P(i) = exp(a_C)/Z_P, for i in C.

Let F_P be all probability distributions on I that are constant within each cell (equivalently,
uniform conditionals on cells of positive cell mass). Allow zero cell masses in competitors.

Prove the exact identity, for every r in F_P:

    KL(r || p) = KL(r || q_P) + log(Z/Z_P).

Consequences, consuming existing KL nonnegativity/equality machinery:

1. `KL(q_P || p) = log(Z/Z_P)` exactly.
2. q_P is the unique minimizer of KL(r || p) over F_P.
3. `Z_P <= Z` (also directly finite Jensen).
4. If P' refines P, then `Z_P <= Z_P' <= Z`, hence
   `KL(q_P' || p) <= KL(q_P || p)`.
5. Singleton refinement gives q_P = p and Z_P = Z. Equality in the Jensen bound has the appropriate
   cellwise constant-logit characterization; do not add an unnecessary distinctness hypothesis.

This is the useful algorithmic improvement: cellwise geometric-mean weights are variationally optimal
within this specific coarse family. Fixed assignment prototypes are NOT necessarily those mean logits.
There is no assertion of global best approximation outside F_P or of the opposite KL direction.

## Exact selected-key replacement and streaming realization

For an arbitrary selected set S, use singleton cells {i} for i in S and cells C\S for unopened keys,
discarding empty cells. This is a refinement of the original partition. Specialize the preceding results.
With s_i = beta <q,k_i>, maintain per-cell count, sum of keys, and sum of values. Subtract selected
contributions to get the actual omitted means; beta <q,mean(k)> is exactly mean(s), not a proxy.
Reproduce q_P's output from exact selected exponentials and the omitted count/key/value summaries.
At full selection recover the dense read, including the case of an empty omitted tail.

State causal restriction on the visible finite carrier; fixed assignments must not depend on future keys.
No computational subquadraticity or persistent-state lower bound follows merely from these identities.

## Connect upper certificates without confusing the direction

Given any separately supplied admissible total mass upper bound U >= Z, prove

    KL(q_P || p) <= log(U/Z_P).

Use existing Pinsker/output transport if already owned to obtain a TV/output-error certificate.
For example, with TV = half L1 and value norm <= V, error <= 2 V sqrt(log(U/Z_P)/2).
State that this is only useful if U/Z_P is close to one. Z_P is a LOWER mass bound and must never be
fed into `MassTreeRead` as an upper omitted-mass cap. No favorable U or cheap frontier is supplied here.

## Gains and safety: exact cost of leaving the variational optimum

For arbitrary finite cell gains g_C, let

    Z_g = sum_C m_C exp(a_C + g_C),
    q_g(i) = exp(a_C + g_C)/Z_g.

Prove

    KL(q_g || p) = log(Z/Z_g) + sum_C q_g(C) g_C
                = KL(q_g || q_P) + KL(q_P || p).

No sign assumption is needed for the identities. Nonpositive gains imply Z_g <= Z_P, but do NOT
claim that reducing gains improves reverse KL; q_P is already optimal in this family. In SSA selected
singletons have gain zero and omitted cells can have nonpositive gain. Suppressing an inaccurate value
approximation can still improve downstream CE, which is not the KL objective above.

For selected output o_S and any positive normalized approximate tail output u, cap its applied share at
rho in [0,1]: o_alpha = o_S + alpha (u-o_S), 0 <= alpha <= rho. Prove the movement bound

    ||o_alpha-o_S|| <= rho ||u-o_S||,

and the exact Hilbert-space error identity for e = o_dense-o_S, d = u-o_S:

    ||e-alpha d||^2 - ||e||^2 = alpha^2 ||d||^2 - 2 alpha <e,d>.

For alpha > 0, nonincrease is equivalent to `alpha ||d||^2 <= 2<e,d>`; strict improvement requires `<`.
Supply a counterexample to any unconditional improvement claim for rho > 0. Likewise do not infer
monotone actual output error, opposite-direction KL, or CE from partition refinement's reverse-KL law.

## Delivery and honest scope

Use a domain-neutral Universal owner and an Inference attention recognition, with normal imports,
index entries, axiom audit and project gates. Keep source statements self-contained enough for SSA to
reproduce publicly; its public repository cannot require access to private Substrate.

No theorem here should claim learned favorable geometry, retrieval preservation, training convergence,
held-out length generalization, a useful fixed-budget certificate, or numerical soundness on CUDA.
