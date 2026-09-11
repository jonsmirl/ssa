# BUILD: recovery limits and stable fitting for sparse attention memory

## Build priorities

Two primary questions, informed by completed SSA experiments:

1. **Information:** can ANY decoder of the available summary and exact reads meet the target?
   Build the sharp recovery/no-go results in sections 1, 2, and 5.
2. **Update stability:** when information could be useful, does joint fitting preserve it or
   amplify errors? Build section 6 for inconsistent targets and ill-conditioned keys.

Section 3's efficient streaming construction is a follow-on, not a substitute for either primary
question. Reuse existing workload-information results in section 5 where available. Do not spend
the build on another sufficient reference-prediction guard or a favorable realizable toy alone.

## The SSA question

Can a bounded streaming state retain what sparse attention misses, or is its remaining error
unrecoverable without more archive reads? Build an optimality result and a constructive estimator,
not another covariance/Bennett/outlier cap or another sufficient prediction-preservation theorem.

The current fixed-state tail improves measured Qwen perplexity but harms retrieval. The whole-path
certificate is implemented only for a small bound-friendly transformer. Qwen endpoint fallback
costs two full forwards and cannot repair a wrong reference winner. Neither solves this question.

### Measured evidence motivating this request

Read `ssa/span_memory.py`, `ssa/span_memory_experiment.py`, their tests, and
`runs/span_memory_comparison.json` in the public SSA repository. The 8K cached Qwen layer-18/KV-0
comparison uses two representation-state caps, three seeds, and the same 16 suffix queries. It
separates frozen unseen-value prediction from online admission of all preceding values. The main
attention diagnostic supplies all true attention weights and charges that oracle work.

Mean online attention-output L2 errors at caps 8,192 / 16,384 scalars:

| Method | 8,192 | 16,384 |
|---|---:|---:|
| Sparse selected attention | 2.3583 | 2.3583 |
| Linear batch-joint update | 36.7952 | 36.7952 |
| Linear sequential update | 3.6549 | 3.6549 |
| Fixed random-tanh batch-joint update | 194.1890 | 6.1325 |
| Fixed random-tanh sequential update | 3.2521 | 3.2815 |
| Cell means with oracle weights | 1.9987 | 1.9760 |

These fitting results use batch size 64, rate 0.5 and relative pseudoinverse cutoff `1e-10`.
They are NOT an experiment with arbitrary learned features, optimal regularization, or the optimal
decoder of every state. Joint fitting acts on each current batch, not all historically held keys.
Seed-0 training batches reach retained condition numbers about 50,988 for raw features and 115,146
for the smaller tanh features. Fitting a batch can therefore succeed while older associations fail.

An exact dyadic-to-prime-field check on the stored fixture gives rank 65 for a 65-by-65 matrix
formed from all 64 raw-key coordinates and one value coordinate. This rules out perfect linear
key-to-value fitting on those observations. It does NOT rule out small attention-read error,
affine or richer feature maps, nonlinear state, or average-case useful approximation. In particular,
do not confuse this raw-key witness with the experiment's 65-dimensional intercept-augmented features.

All 46 new SSA storage/protocol tests and the 449-test full regression suite pass. Passing those
tests verifies the experiment's behavior; it does not establish that the tested algorithms work.

The proposed mathematical object is **sparse plus linear-summary recovery**. It models SSA's
streamed value sums conditional on fixed keys/assignments. It does not model every nonlinear GRU.
Prove or correct the statements below; do not make a false target true by hiding an oracle in a
hypothesis. Separate the information-theoretic optimum from a cheaply realizable algorithm.

**Primary decision requirement:** the proofs should be able to rule out a specified strategy,
not merely certify a favorable example. Return a quantitative failure certificate when the
state/read budget cannot meet a stated error or retrieval target. Distinguish failure of the
current coefficients, failure of every decoder of the current summary, failure of every allowed
summary at the resource budget, and failure on a specified workload distribution. These are
different quantifiers; a result about one must not be advertised as another.

## Read existing owners first

- `CellMeanProjection`, `SelectedCellMeanRead`, and `RestrictedReadOutputBound`.
- `RestrictedReadOverlap`, `BoundedReadMiss`, and `IndexReadMiss`.
- `ReachableFamilyDimension` and the existing attention recognitions.
- `TargetValueAdjustment`, `TargetValueRead`, `TargetValueBoundary`, and
  `DenseReadTargetedUpdate`: selected fitting, overlap interference, and explicit negative witnesses.
- `SpanValueAdjustment`, `SpanReadChain`, `SpanReadBoundary`, and `SpanValueRead`:
  realizable held-span fitting and charged admission. The relevant commits include `22d6b0c4c`,
  `f9cde2b43`, `6445d7b10`, and `c762a7d24`; inspect later developments before building duplicates.
- `ResidualStep`, `ReadWeightedError`, and `LinearPath`: reuse local movement identities and signed
  linear propagation rather than constructing parallel owners.
- Public SSA specifications `docs/coarse_read_projection.md` and
  `docs/routed_correction_certificate.md`.

Reuse existing finite sums, normalized reads, norm duality, and indistinguishability infrastructure.
Do not rederive the cell-uniform KL optimum or miscite the two-pass parameter-count ceiling as a
rank theorem. The new target is an exact recovery radius for a specified information operator.

## 1. Sharp recovery radius, allowing arbitrary decoders

Fix a finite nonempty carrier of size n, a real matrix A with r rows, a target weight vector p,
and a selected coordinate set S. The matrix and target are fixed independently of the unknown
scalar values v in [-1,1]^n. The decoder observes only (Av, v restricted to S).

For attention, p is a positive normalized weight vector conditional on fixed keys and query.
The first theorem can allow any real p. Its being known in this theorem is NOT a free operation
in an attention implementation.

Establish the exact minimax identity, including attainment where justified:

    inf_D sup_{||v||_infinity <= 1} |D(Av, v_S) - p^T v|
      = inf_c sum_{i not in S} |p_i - (A^T c)_i|.

The infimum on the left ranges over arbitrary deterministic decoders, not just linear ones.
For a supplied c, the constructive decoder is

    o_hat = c^T (Av) + sum_{i in S} (p_i - (A^T c)_i) v_i.

Its error is exactly the unread residual coefficient paired with unread values. Prove the upper
bound and matching lower bound; do not stop at the triangle inequality. The intended dual witness is

    Ah = 0, h_S = 0, ||h||_infinity <= 1,

so h and -h give identical observations but opposite target outputs. Finite-dimensional duality
should identify the largest witness gap with the distance on the right. Handle rank-deficient A,
empty S, full S, zero residual, and duplicate/dependent summary rows.

Deduce exact recovery iff p belongs to row(A) plus the coordinate span of S. Also give the
uniform finite-query-family form with query-dependent selected sets; do not replace it with an
unjustified low-rank requirement when the sparse supports can vary by query.

## 2. Does deliberation help without changing the information source?

Extend the decoder to a deterministic adaptive program with at most b exact coordinate reads.
It may choose the next coordinate using p, A, Av, and all previous read results, with arbitrary
internal computation. Repeated reads count as operations and do not add information.

Target: its optimal worst-case error on the symmetric cube equals

    min_{|S| <= b} inf_c sum_{i not in S} |p_i - (A^T c)_i|.

The proposed lower-bound argument runs the program on zero summary and zero read answers, obtains
its set S, and chooses the dual witness h vanishing on that transcript. Both h and -h must follow
the same complete trace. The fixed-set decoder supplies the reverse inequality.

If correct, this rules out a worst-case advantage from value-adaptive deliberation for this one
fixed linear target. It does NOT rule out pointer chasing, changed future targets, correlated
value families, additional indexed information, or average-case gains. Do not claim it does.
Do not silently extend the result to randomized algorithms; state their quantifiers separately
if an extension is attempted.

## 3. Constructive streaming attention bridge without a dense-weight oracle

The optimum above is not deployable merely because it exists: forming all p_i, finding the best
S, and computing the coefficient residual can each require a full scan. Build a separate interface
that exposes these costs instead of assuming them away.

Use streamable key features phi(k_i) in R^r, with rows of A corresponding to those features.
Maintain

    H = sum_i phi(k_i) v_i^T,     z = sum_i phi(k_i).

For a query, let true unnormalized weights be w_i = exp(beta <q,k_i>) and approximate weights
be c(q)^T phi(k_i). Replace the approximation by the exact weight at every selected key once.
Define the resulting approximate numerator B and denominator a using only H, z, and those
selected replacements. Prove the exact identities and full-selection recovery.

Suppose separately justified, query-computable envelopes bound the remaining errors:

    E0 >= sum_{i not in S} |w_i - c(q)^T phi(k_i)|,
    Ev >= sum_{i not in S} |w_i - c(q)^T phi(k_i)| ||v_i||.

For a > E0, derive the deterministic ratio certificate

    ||o_dense - B/a|| <= (Ev + ||B/a|| E0) / (a - E0).

Signed features/coefficient errors must be allowed explicitly or excluded explicitly; a signed
approximate read is not automatically a probability distribution. Existing upper tail-mass caps
do not automatically certify this signed coefficient residual. The denominator guard is essential.

Supply at least one explicit query-uniform feature/residual construction with a nonconstant
attention family and bounded error, including how its r and work depend on geometry and accuracy.
An all-keys oracle residual, arbitrary supplied successful S, or one constant-logit example alone
does not establish the constructive part. If a low-dimensional bounded-query exponential
approximation is used, expose all rank/degree dependence; do not conceal exponential dimension cost.
General conditional interfaces and genuinely instantiated bounds must have distinct names.

Charge preprocessing over all keys, feature storage or recomputation, r times value dimension
state, per-item updates, selected archive reads, query coefficient construction, routing nodes,
and residual-bound evaluations. Give append-only/causal-prefix identities. A query-adaptive
sequence needs a bound uniform over its permitted query set, not a claim checked only at one
unrelated query. Do not declare subquadraticity without an explicit total-work condition.

## 4. Practical consequence and honest fences

The desired algorithmic consequence is to replace disjoint cell sums by a learned overlapping
feature summary where that reduces the certified residual, then spend exact reads on the remaining
residual. Prove only what the construction implies: there is no automatic cheap optimal subset,
no theorem that training finds useful features, and no guaranteed favorable Qwen geometry.

For indistinguishable v and -v with a nonzero target, show why a decoder cannot guarantee the
correct sign on both. This is a retrieval-relevant information obstruction, not merely a norm bound.
An attention-read certificate is not yet a multi-layer prediction certificate: compose with existing
owners only under their actual hypotheses and charge any reference execution.

Linear-summary dimension is not a bit-capacity theorem for arbitrary nonlinear real-valued state.
Conditioning on fixed keys while allowing all values is a mathematical class, not a claim about the
conditional distribution of trained Qwen values. No IEEE proof, runtime measurement, or optimization
experiment belongs in this Substrate build; SSA will implement and measure those.

## 5. Required no-go conclusions and usable rejection criteria

### Deterministic lower witnesses, not only existential worst cases

For fixed A, p, and S, expose a checkable h satisfying

    Ah = 0, h_S = 0, ||h||_infinity <= 1, |p^T h| > epsilon.

This refutes a uniform epsilon guarantee for EVERY decoder of those observations, regardless of
its training or internal computational depth. A numeric h will require SSA to verify feasibility
or account for rounding; approximate equality Ah approximately 0 is not exact indistinguishability
in noiseless real arithmetic. Add a separate observation-noise/quantization formulation if used.

For an adaptive policy, the indistinguishability witness must preserve its whole executed trace,
not just the final selected set. To rule out all policies at budget b, prove the quantified bound
over all such policies; testing one router is not enough. To rule out all summaries of dimension r,
one additionally needs a bound over all admissible A, with the permitted query family and timing
of choosing A explicit. Do not claim that stronger conclusion from a witness for one fixed A.

### Workload-specific impossibility, not just arbitrary adversarial values

Reuse existing finite conditional probability/information owners to supply a separate finite
decision version. Let X be the archive/input, Q the query, C the desired answer, and T the complete
observable transcript available to the decoder. Include Q, the allowed stored state, selected read
results, and every other side channel in T. Finite or quantized observations make the information
model explicit; do not count arbitrary-precision real coordinates as a finite number of bits.

For a supplied finite joint distribution, prove or delegate the best achievable top-one accuracy:

    sup_D Pr[D(T) = C] = sum_t max_c Pr[T=t, C=c].

For a list of at most k answers, use the sum of the k largest joint masses at each transcript,
handling deterministic ties and zero-probability transcripts. This is list coverage of one correct
answer, NOT automatically SSA's top-k overlap with a dense attention set. Define and match the
actual target metric before comparing numbers.

For probabilistic decoders under expected log loss, identify the optimum with conditional entropy
H(C | T), including support/zero conventions. Extra deterministic controller steps that expose no
new information cannot lower this optimum. New reads or side information can; charge them and do
not ban their benefit by assumption. Proving a budget-wide bound requires considering every allowed
adaptive read policy, not just evaluating the Bayes floor for one chosen transcript mechanism.

These results should permit a conclusion such as: under this explicitly stated workload and
observation budget, even the optimal decoder has top-one accuracy below 0.998. A valid such bound
rejects that strategy/target combination. A worst-case cube witness alone does not establish this
distributional conclusion. Conversely, a supplied finite distribution or one cached dataset does
not prove that a deployed model has that distribution; SSA owns that empirical identification.

### What the completion report must say

Return a decision table identifying which claims the build can actually refute:

- Current fitted coefficients are inadequate: approximation error only; better fitting may help.
- Current summary plus read information is inadequate: decoder-independent lower bound; training
  the same observation channel cannot meet the specified guarantee.
- Every permitted summary/read strategy at the resource budget is inadequate: requires the stronger
  quantified lower bound, not merely a failure of one summary.
- The strategy cannot meet an average retrieval target on a named distribution: requires the
  workload-specific bound, not merely a worst-case witness.

If an impossibility claim is false, supply a counterexample or constructive escape and identify
which extra information or resource makes it possible. A theorem need not condemn the strategy
to be useful, but it must make clear which proposed escape changes the problem and its costs.

## 6. Joint fitting without realizability: sharp error, interference, and stabilization

This is the second primary build, not an optional refinement of the realizable theorem. Work with
finite real matrices, observations as rows:

    X : m by r, Y : m by d, W : r by d,
    G = Moore-Penrose pseudoinverse(X),
    W' = W + eta G (Y-XW).

State dimensions, norms and admissible rates explicitly. Use Frobenius norms for matrix errors
and Euclidean norms for vector reads below. Handle empty observations if the interface admits
them, rank deficiency, zero residual, zero singular values and exact ties in any truncation.
Prove or correct the following targets; do not assume a successful final invariant.

### 6a. Exact decomposition of what fitting can and cannot remove

Let R=Y-XW and P=XG, the orthogonal projector onto the column space of X. Establish

    R' = Y-XW' = (I-eta P)R,
    ||R'||_F^2 = (1-eta)^2 ||P R||_F^2 + ||(I-P)R||_F^2,
    inf_B ||Y-XB||_F = ||(I-P)Y||_F.

Identify the exact realizability condition and a computable rank/nullspace obstruction.
Recover the existing realizable contraction theorem by specialization, not duplication.
Determine the actual rate interval for nonincrease and its strictness conditions; do not impose
eta in [0,1] where a wider interval is proved, nor use the wider interval to infer convex size bounds.

**Important:** the irreducible residual is annihilated by the pseudoinverse:

    G(I-P)Y = 0.

It is therefore wrong to call this same residual the noise amplified by G. Separate the error
that cannot be fitted, the in-column perturbation that can be amplified, and unseen-direction
ambiguity. A small or zero fitted-batch residual alone cannot bound the latter two.

### 6b. Reader-specific amplification and matching failure witnesses

For a SUPPLIED comparison map T and observation mismatch E satisfying Y=XT+E, prove

    W'-T = (I-eta GX)(W-T) + eta G E.

For a fixed row read a in R^r, establish the sharp worst-case amplification

    sup_{||E||_F <= epsilon} ||a G E||_2 = epsilon ||a G||_2

for epsilon >= 0 and a nonzero-dimensional output space. Include zero cases. For a supplied
output functional ell, give the corresponding sharp scalar bound with the factor ||ell||_2.
Exhibit rank-one perturbations attaining the bound instead of proving only a norm inequality.
This is about fixed X and fixed a; it does not include perturbations of keys or pseudoinverse
rank changes. T and a global mismatch bound are hypotheses, not quantities inferred from one
batch's residual. Choosing T as a batch least-squares minimizer does not certify unseen targets.

For old associations and arbitrary target y, identify the exact signed error change of aW-y
under the update. Reuse `ResidualStep` where possible. Give concrete witnesses for all of:

- Zero new-batch fitting error with arbitrarily bad error on a bounded unseen key. One proposed
  scalar witness is X=[sigma], Y=[epsilon], T=0, W=0, eta=1, unseen a=[1], sigma>0. Its error is
  epsilon/sigma. Keep epsilon (mismatch size) distinct from sigma (conditioning).
- A noiseless well-conditioned observed design with an unobserved/nullspace direction on which
  the target is not identified. Good nonzero singular values do not remove this obstruction.
- Large individual association errors whose tested weighted output cancels. Failure of uniform
  value reconstruction is not automatically failure of attention-output reconstruction.
- Existing single-key interference reproduced or delegated, with its norm-sum and squared-error
  aggregates distinguished. Do not claim that one such update refutes every joint policy.

### 6c. Sparse attention and multiple updates without replacing them by a global norm

For a supplied fixed attention distribution p and feature rows phi_i, with exact selected set S,
write the reconstructed output as

    o_hat = sum_{i in S} p_i v_i + a W,
    a = sum_{i not in S} p_i phi_i.

Thus the effect of changing the value map is a(W'-W), and an output projection/functional acts
on this actual read. Derive the signed identity and reader-specific amplification without first
bounding every value error separately. Preserve cancellation; do not silently identify different
head-specific maps with one common scalar correction through an output projection.

These are fixed-weight identities. If the update also changes Q/K, softmax weights, selected
sets, or downstream states, those changes require their own terms and existing route/read
theorems. True p and dense a are not automatically available to a sparse implementation. State
which resulting tests need a dense oracle, and what separately supplied summary/enclosure would
make them inference-available. The storage energy reader's tied scoring/value representation
must not be silently identified with Qwen's untied K/V attention.

For a sequence of updates, use existing signed affine/linear-path owners to propagate actual
bias and mismatch terms. Supply a bound depending on transported reader-specific gains, not
just a product of worst-case operator norms. Give a matching worst-case perturbation statement
under explicitly FIXED design/read sequences if possible. A fixed-design supremum does not
cover noise-adaptive query selection: distinguish actual-path algebra from uniform guarantees
over all adaptive executions. Per-batch contraction must not become a claim of all-history retention.

### 6d. A genuinely stabilized alternative, including its bias and repeated-update behavior

Provide at least one implementable alternative with an explicit gain/bias tradeoff, such as
truncated SVD or ridge fitting. For a chosen linear fitting operator G_tilde, first prove the
general identity

    W'-T = (I-eta G_tilde X)(W-T) + eta G_tilde E.

For a truncation retaining only singular values >= tau>0, expose the discarded-subspace bias
and establish the gain bound ||G_tau|| <= 1/tau, with the actual equality convention fixed.
For lambda>0 and G_lambda=(X^T X+lambda I)^(-1)X^T, an intended gain target is
||G_lambda|| <= 1/(2 sqrt(lambda)). Prefer the tighter reader-specific norm ||a G_tilde||
where available. Give nonzero examples in which stabilization reduces a justified worst-case
read-error guarantee, and counterexamples/tradeoffs preventing an unconditional accuracy claim.
The zero update alone is not a useful positive construction.

**Repeated-update trap:** replacing G by G_lambda in

    W <- W + eta G_lambda (Y-XW)

is preconditioned residual fitting, not automatically minimization of a persistently penalized
objective. Repetition can recover the unregularized fit and its noise gain. Prove the actual
finite-iteration spectral response, or use an explicit penalized minimizer/weight-decay update
and state its fixed point and bias. Early stopping must be part of the specified algorithm if
its guarantee relies on it. Do not advertise one-step suppression as perpetual noise control.

No particular threshold or penalty is assumed to generalize to Qwen. Distinguish a certified
choice using supplied noise/signal bounds from a parameter chosen after inspecting held-out error.
Charge feature generation, factorization/update work, any retained Gram/cross matrices, and
temporary buffers. Frozen W size alone is not an online fitting-memory bound. Integer operation
counts and finite-dimensional storage bounds are appropriate; SSA owns timing and floating-point tests.

## Deliverable and priority

Prioritize the sharp recovery/no-go core in 1, 2 and 5, and the inconsistent/stable-fitting build
in 6. These can proceed independently. Section 3 is a subsequent efficient-reader construction,
not required to disguise or replace a missing primary result. Reuse existing information
inequalities rather than building duplicate general-purpose entropy infrastructure.
If the constructive part is blocked, report that explicitly rather than presenting a supplied
residual oracle as a finished algorithm. Provide domain-neutral Universal owners, thin Inference
recognitions, axiom/gate/build checks, and a self-contained public specification with proof arguments.

Success would tell SSA both what its current summary irrevocably loses and what a different
summary-plus-read algorithm can actually recover. It would not by itself establish model quality.

The completion report must classify each result as one of:

1. an information obstruction applying to every decoder in the specified observation class;
2. an instability/interference result for a specified update rule;
3. a conditional stabilized construction with explicit assumptions and costs;
4. an open empirical identification question for SSA.

Do not substitute “the proof builds” for “the SSA strategy works.” Include exact hypotheses,
attainment/counterexample scope, public proof arguments, and what a subsequent numerical test
would need to measure to accept or reject the strategy.
