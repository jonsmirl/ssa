# BUILD: route-aware, multi-layer correction certificates

**Status: the finite mathematical build is complete in Substrate commit
`68968fc7aae0a024172288f090f25c4d61451324`.** The public result, proof arguments, and concrete nonlinear
and affine examples are copied into [routed_correction_certificate.md](routed_correction_certificate.md).
The request below records the intended scope; it is not an outstanding proof request.

`RestrictedReadOverlap` owns same-state restricted-read overlap and charged union reads;
`ComparisonTrace` owns finite ordered executed-trace stability and the discontinuity witness;
`QuadraticRemainder` derives the sharp half-factor from uniform derivative hypotheses;
`LinearPath` owns signed heterogeneous propagation and radius induction; `ComparisonPathBounds`
connects the same tubes, actual executions, stable comparisons or actual-state jumps, and final margins.
`ComparisonPathWitness` gives complete nonlinear and affine-switch instances, and
`RoutedCorrectionRecognition` supplies the attention interpretation. There are 40 new theorems; the
build reuses existing normalized reads, output transport, overlap, and transpose-pairing owners.

The implementation-relevant qualifications are:

- The path and remainder results work in real normed spaces; neither Hilbert structure nor a common
  stage dimension is needed. Backward reads are continuous linear functionals.
- Derivative variation and comparison-gap Lipschitz bounds must be uniform on the same stated closed
  balls used in radius induction. A sampled derivative is not enough.
- The alternative branch-jump bound may hold just at the actual perturbed state. Its attention instance
  evaluates both masks using the same actual logits and values there, and charges their union reads.
- Ordered finite comparison trees represent a supplied full execution trace. Extracting all relevant
  guards, hidden/recurrent/cache state, and smooth branches from SSA or Qwen is not proved.
- A strict positive final interval preserves exactly the supplied reference winner on the supplied
  candidate set. It proves neither reference correctness, recovery, CE improvement, nor preservation
  of untested vocabulary entries. Failed tests establish no error by themselves.
- Affine fixed branches have zero curvature remainder even when routes change. The nonlinear witness
  attains a nonzero remainder, so the result is not limited to linear cancellation examples.

SSA now implements the numerical path test and a complete small causal transformer with analytic
uniform bounds, full comparison traces, actual-state branch replay, and charged work. See
`ssa/routed_certificate_experiment.py` and `runs/routed_certificate_reference.json`. Its bound-friendly
toy settings and float64 checks are not a Qwen or IEEE instantiation. Useful uniform Qwen constants,
scalable derivative/guard evaluation, and floating-point verification remain open. A separate Qwen
endpoint fallback compares two independently computed outputs; its prediction preservation does not
validate these path bounds. The original capped-tail CE/retrieval measurements remain unchanged.

## Why this is the next harder problem

SSA's capped tail improves full-corpus perplexity, including 32K, yet loses all three 32K retrieval
probes where sparse-only wins two. The cell-mean projection and `ResidualStep` theorems at `206193290`
are sound: local reverse-KL optimality and bounded movement simply do not establish downstream safety.
The new object should connect **discrete route changes, signed readout effects, and nonlinear depth**.
Do not build another covariance, Bennett, outlier, or one-step descent bound.

SSA owns numerical bounds, floating-point verification, implementation, training, and model experiments.
Substrate should build exact finite-dimensional mathematics with explicit hypotheses. A supplied
derivative bound must be a uniform bound on a stated neighborhood, not a derivative sampled at one point.

## Reuse audit first

Inspect existing owners before constructing anything:

- `Universal/Accumulation/SequentialErrorComposition`: accumulated local errors and downstream gains.
- `Universal/Carrier/Operator/ReadWeightedError` and `ResidualStep`: read-weighted error and residual steps.
- `Universal/Potential/Entropy/RestrictedReadBound`, `RestrictedReadOutputBound`, and
  `SelectedCellMeanRead`: normalized finite reads and output transport.
- `Universal/Potential/GroupedPairingClaim` and `BoundedFamilyPairing`: shared-query claims and common
  isometric score transport. These are not already certificates for arbitrary learned perturbations.
- `Infrastructure/CommonMath/Program/AdjointSweep`, `LinearTranspose`, and
  `Universal/Generator/Coupling/CoupledEngines.pairing_transpose_mulVec`: existing adjoint algebra.
- `Universal/Carrier/Separation/BoundedTopSelection`, especially the distinction between returned-set
  correctness and executed-region identity. Existing finite execution and derivative/remainder owners.

Do not duplicate backpropagation or the existing product-of-Lipschitz-constants error theorem. The missing
work is their composition with actual changing hard routes and a signed terminal read with a controlled
nonlinear remainder. Report which requested pieces already exist and delegate to them.

## 1. Exact route-change cost from the union of two selected reads

On one nonempty finite visible carrier, take finite real logits `s_i`, arbitrary values `v_i`, and
nonempty selected sets `S,T`. Both reads must use the **same logits and values at the same state**.
Let `Z_A = sum_{i in A} exp(s_i)` and let `p_A` be softmax restricted to `A`, extended by zero.

Prove, or reuse an existing owner for, the exact overlap identity:

    TV(p_S, p_T) = 1 - Z_(S intersect T) / max(Z_S, Z_T),

where TV is half L1. Transport it to a value-output bound using existing machinery, e.g.

    ||o_S-o_T|| <= 2 V TV(p_S,p_T)       when every ||v_i|| <= V.

Include identical, disjoint, nested, and equal-mass sets; deterministic score ties cause no ambiguity
once the actual selected sets are supplied. Do not state a finite KL bound for arbitrary supports.
No full-carrier partition function is required: every quantity uses keys in `S union T` only.
Give the exact read-count union bound `|S union T| <= |S|+|T|`; do not hide these extra reads as free.

This is a **route-switch certificate between two restricted reads**, not a certificate that either
read approximates dense attention. It does not require CCC's routing metric to bound attention logits:
the union's actual attention logits are evaluated exactly in the mathematical object.

For multiple heads, keep the vector of actual head differences until the output projection. Independent
head perturbations must not be commuted through `W_O`, because cross-head cancellation matters.

## 2. Stability of an executed hard-routing trace

Model the finite router as a deterministic guarded program with explicitly ordered ties. Its nominal
execution records every comparison that can affect its control flow: beam pruning, queue order, skips,
top selection, hard cell assignment, and any other piecewise decision needed to make the fixed branch
smooth. Continuous but nonsmooth clamps also need a branch treatment if included in a differentiated
map; alternatively leave the actual correction outside that map as in item 3.

For each executed comparison, supply a signed nominal gap `m>0` in favor of its selected branch and a
bound on change of that gap over a stated input neighborhood. Prove by a first-divergence argument:

    every executed gap changes by less than its nominal margin
    ==> the perturbed execution has the same full decision trace and selected set.

A useful sufficient instance is `K E < m` when the gap is `K`-Lipschitz on an input tube of radius `E`.
The comparison functions and their bounds must account for previous computations on that trace;
checking only the final top-k margin is insufficient for a pruned tree search.

Prove a hard-top-1 discontinuity witness: arbitrarily close queries can select different keys with a
nonvanishing value-output jump at a score tie. This rules out treating the whole router as globally
Lipschitz without an extra assumption. A zero margin is not a positive stability radius. Do not claim
that every tie is harmful: branches can coincide in their outputs.

If the trace cannot be certified stable, permit the route-change term from item 1 (or another separately
supplied bound). A failed margin test does not by itself prove a route changed or that an error occurred.

## 3. Signed propagation through a nonlinear routed computation

Use a finite sequence of real finite-dimensional Hilbert state spaces, allowing stage dimensions to
differ. At stage `j`, let `F_(j,a)` be the smooth map obtained by fixing that stage's hard branch to `a`.
The state must include everything needed for the application, including recurrent summaries or cache
state when applicable. Do not assume they are independent of the perturbation.

Let the nominal and corrected executions satisfy

    x_(j+1) = F_(j,a_j)(x_j),
    y_(j+1) = F_(j,b_j)(y_j) + c_j,
    e_j = y_j-x_j.

Here `c_j` is the actual injected correction at the perturbed state, not silently its value at `x_j`.
Put

    J_j = derivative of F_(j,a_j) at x_j,
    h_j = F_(j,b_j)(y_j) - F_(j,a_j)(y_j),
    r_j = F_(j,a_j)(y_j) - F_(j,a_j)(x_j) - J_j e_j.

Then the exact recurrence is

    e_(j+1) = J_j e_j + c_j + h_j + r_j.

`h_j=0` under certified trace stability. Otherwise bound the actual branch difference, using item 1
when the differing branches are attention masks. If a stage includes nonlinear operations after
attention, either expose the attention output as its own stage or transport its jump through those
operations with stated bounds. An attention-output jump is not automatically a whole-layer jump bound.

Derive a Taylor remainder instance from a uniformly Lipschitz derivative on a suitable convex tube:

    ||r_j|| <= (H_j/2) ||e_j||^2.

The theorem must provide a genuine sufficient derivative hypothesis, not rename this desired conclusion
as the only premise. Reuse a stronger existing Taylor owner if one exists. Maintain and verify tube
membership inductively. One usable radius recurrence is

    E_(j+1) = ||J_j|| E_j + C_j + B_j + (H_j/2) E_j^2,

where `||c_j|| <= C_j`, `||h_j|| <= B_j`, and the stated tube contains the radius `E_j`. Integrate the
guard-margin conditions against these same radii; do not prove guards and propagation on unrelated
neighborhoods. The result is conditional if these bounds are too large to close the tubes.

For a supplied terminal linear read `ell`, use the existing adjoint machinery to propagate it backwards:

    lambda_m = ell,     lambda_j = J_j* lambda_(j+1).

Prove the exact signed identity and its remainder interval:

    ell(e_m) = lambda_0(e_0)
               + sum_j lambda_(j+1)(c_j+h_j+r_j),

    D = lambda_0(e_0) + sum_j lambda_(j+1)(c_j+h_j),
    |ell(e_m)-D| <= sum_j ||lambda_(j+1)|| (H_j/2) E_j^2.

Use Riesz vectors or continuous linear functionals consistently, rather than mixing conventions.
If only bounds on jumps are available, remove their signed contribution from `D` and charge
`sum_j ||lambda_(j+1)|| B_j` to the remainder. When the fixed branches are affine, the curvature
remainder must vanish and the signed formula must be exact, including nonzero route jumps.

Do not replace this signed read-specific result by only a global norm product. Such a fallback may
remain as a corollary, but it is not the requested advance. Supply a cancellation example showing why
the signed read can certify a smaller effect than summing the norms of all corrections.

## 4. Prediction-margin certificate and acceptance semantics

Include final normalization/readout among the smooth stages so the terminal state can be the actual
logit vector. For nominal chosen class `a` and competitor `b`, let the nominal margin be
`m_ab = logit_a(x)-logit_b(x)`. Apply item 3 to the pairwise linear read. If its signed correction is
`D_ab` and remainder bound is `R_ab`, then

    m_ab + D_ab - R_ab > 0  for every b != a

certifies preservation of the nominal unique winner. A tie is not a strict win. Generalize to a supplied
finite candidate read if that is the existing owner, without claiming that a four-candidate probe proves
full-vocabulary prediction preservation.

This preserves a reference prediction, not its correctness. It cannot prove recovery of answers that
the reference itself misses, or improvement in CE. An application may use an a-posteriori acceptance
test and fall back to its computed reference output; any extra reference execution, union reads,
derivative evaluations, and certificate work must be charged by SSA.

## Delivery and boundaries

Build a minimal coherent Universal owner/consumer chain with the normal Inference recognition, gates,
axiom audit, and a self-contained public specification. SSA cannot require private Substrate access.
Prefer staged delivery if necessary, but retain the connected object: route decisions, branch jumps,
nonlinear transport, and final margin must share one hypothesis set.

Do not claim that the supplied bounds are cheap or informative for Qwen, that hard routes stay stable
on real data, that training finds safe gates, or that the method is unconditionally subquadratic. No
floating-point, CUDA, optimization, distributional generalization, or retrieval-success theorem is
requested. The useful advance is a mathematically valid route-aware acceptance certificate that SSA
can subsequently try to make nonvacuous and efficient.
