# Device-decided coordinate reads with dense fallback

Status: implemented and measured on the RTX 4080. All 1,056 head trials passed
the numerical audit, but **none of the 528 paired comparisons beat dense
attention**, by either CUDA-event or synchronized wall timing. At 32K, the median
paired event slowdown is still 13.78x. Stop this implementation direction under
the agreed latency criterion; the experiment does not justify more tuning of
this design.

The acceptance decision no longer synchronizes with Python. This is a measured
attention-kernel result, not complete-model serving, semantic retrieval, or a
mathematical impossibility theorem.

## Algorithm and certificate

An index snapshots one visible causal prefix, including only its visible keys
and values. Block size is 64. It stores coordinate minima/maxima and a global
value-norm bound B. Index construction reads all K/V and is charged separately.
A prepared query fixes shapes, a coordinate count r and proposal fraction f.
For n keys the proposal contains max(ceil(f*n), n mod 64) keys.

The captured computation has a fixed sequence:

1. Sort query coordinates by absolute magnitude, read the first r coordinates
   of every key, and form per-key attention-logit intervals.
2. Select the fixed key budget by partial logit, force-keeping the partial causal
   boundary block. Stable token-index tie-breaking makes selection deterministic.
3. Complete the selected keys' remaining coordinates and compute a mass bound.
4. On the GPU, accept the sparse proposal if its mass bound meets eta; otherwise
   compute a full dense read for that head. Values are first loaded in this step.

No host decision selects between branches. Triton tests each head's acceptance
flag before loading selected values or fallback K/V. Accepted heads therefore
read selected values only; rejected heads skip that sparse value load and read
the full K/V archive. The dense tile output is normalized by a stable maximum
and exponential-mass merge. Sparse output also uses centered logits, including
the earlier regression fix for large common score offsets.

For j in block b, coordinate extrema m_ba/M_ba give, in real arithmetic,

\[
\widetilde s_j=\beta\sum_{a\in D}q_aK_{ja},\qquad
L_j=\widetilde s_j+\sum_{a\notin D}\min(\beta q_am_{ba},\beta q_aM_{ba}),\qquad
U_j=\widetilde s_j+\sum_{a\notin D}\max(\beta q_am_{ba},\beta q_aM_{ba}).
\]

Here beta=1/8 and L_j <= s_j <= U_j. This is a separate attention-logit bound,
not a claim that CCC's routing-metric certificate bounds attention mass. The
restricted-read bridge uses a lower retained mass and upper residual mass:

\[
Z_- = \sum_{j\in S}e^{L_j},\qquad A_+=\sum_{j\notin S}e^{U_j},\qquad
\delta\le\bar\delta=\frac{A_+}{Z_-+A_+}.
\]

Accept exactly when the guarded log margin
log A_+ - log Z_- - log(eta/(1-eta)) is nonpositive. This is an omitted-mass
condition, not top-key recall or a learned routing surrogate. The GPU reader
uses an unquantized per-key upper-cap sum, not the CPU reader's 16-band rounding.

On acceptance, ideal exact-score restricted attention has TV=delta,
KL(restricted || full)=-log(1-delta), and value-output error <=2B*bar-delta.
On rejection, the final read uses every key/value, so its ideal omitted mass
and supported KL are zero. A final full-read certificate does not mean the
sparse proposal was successful.

Floating-point score uncertainty and output arithmetic receive additional
allowances. With c the maximum final selected-score allowance, the returned
output bound has the form

\[
2B\bigl(\bar\delta_{\rm final}+\min(c,1)\bigr)+\epsilon_{\rm arithmetic}.
\]

These FP32/FP64 guards are engineering allowances verified against dense oracles,
not an IEEE rounding proof. The TV/KL identities concern the ideal exact-score
restricted distribution; computed floating weights need not form an exact
probability vector. Replay callers must keep inputs finite, preserve their
prepared shape/dtype/device, and avoid overflowing FP32 calculations. The result
exposes `numerically_valid` and `certified`; invalid results are not certificates.
The benchmark explicitly requires both flags to be true.

The public mathematical account in
[partial-coordinate attention](partial_coordinate_attention.md) and
[the adaptive GPU reader](partial_coordinate_gpu.md) gives the underlying
partial-pairing and restricted-read identities without requiring private
Substrate access. No covariance, Bennett, outlier or probabilistic cap is added.

## Work and timing protocol

This changes the adaptive algorithm, not just its launch method: there is one
fixed proposal, followed by sparse acceptance or dense fallback. Tested settings
are r=32/64, f=0.25/0.5 and eta=0.10/0.01. There is no favorable-profile tuning
based on a query's dense oracle; that oracle runs after the decision.

For proposal size k, every head pays n*r + k*(64-r) key-coordinate reads.
Acceptance adds k*d_V value reads. Rejection adds **another n*64 key reads plus
n*d_V value reads**. Rejected proposals are not free, and the reported requested
K/V byte fraction can exceed dense. These are logical archive-load counts at
the actual BF16 storage dtype, not hardware DRAM counters. Summary accesses,
sorts, certificate reductions and output bookkeeping are separately represented
in the implementation/work records; they remain part of replay latency.

Both the reader and matching native BF16 dense SDPA baseline use the same CUDA
Graph harness, the same two query heads sharing one KV head, and the same causal
prefix. Measurement order alternates deterministically. Preparation includes
validation and static-buffer setup; index construction, preparation, graph capture
and compilation warmup are outside steady-state replay timing and separately
reported. Query sorting, proposal scoring, mass decision, conditional fallback,
normalization and result bookkeeping are inside the timed graph.

After three warmup calls, each graph is captured and replayed three times before
measurement. There are seven repetitions. Each includes a synchronized individual
replay for wall latency and a 32-replay CUDA-event interval divided by 32.
Event timing includes the queued replay interval, not a promise of isolated
kernel-only execution. Wall timing includes synchronization. This is a warm,
fixed-shape attention workload, not cold-start latency or concurrent serving.

## Fixtures and context growth

The paired 8K comparison reuses the two named fixtures from the immediately
preceding experiment: Ise-class battleship and Second Battle of Naktong Bulge.
They are newly extracted relative to the older single-head cache, but are not
untouched data after that earlier GPU experiment. Their sampled zero-based
positions are 4096/5461/6826/8191, giving prefixes 4097/5462/6827/8192.

The growth fixture is one **natural 32K mixed-article test stream**, not repeated
8K text, repeated embeddings, or a single coherent 32K article. It concatenates
distinct WikiText2 raw-v1 test articles in source order, excluding Ise/Naktong:
Robert Boulter; Du Fu; Kiss You (One Direction song); Dick Rifenburg; 1933 Treasure
Coast hurricane; Hed PE; Ironclad warship; and Little Gidding (poem). A blank-line
separator joins articles; the last article is truncated at token 32,768. The
manifest records original row ranges, text hashes and the exact token-ID hash.
No source article or geometry is repeated by construction.

Growth positions 8191/16383/32767 give 8192/16384/32768-token prefixes of this same
stream. These natural positions also change query content: this is not a
controlled experiment holding the query fixed while increasing n. The purpose
is workload/context-growth timing, not semantic quality or generalization to
untouched held-out text. WikiText has already been inspected in this project.

Both protocols use frozen Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, full dense BF16 prefill, all 24 layers,
and post-RoPE capture at zero-based layers 6/12/18. Query heads 0/3 share KV head
0, and heads 7/10 share KV head 1. Each timed call contains two heads, not all
14 heads of a complete layer. Captures are saved as FP32 casts of BF16 geometry
and converted back losslessly for benchmarking. No sparse-model rollout is run.

The 32K extraction used the local RTX 4080: 24 SDPA calls, 1.6735 seconds capture
time, and 2.206 GB peak allocated memory. Kaggle was not needed. Its fixture is
`/tmp/ssa_device_growth/mixed_test_32768.npz`, SHA256
`9ca7a6754011d4edd3b265dd3618b0fdfd3fa9ee39c45b4b5b02c988643ddad9`.
The extraction manifest is `/tmp/ssa_device_growth/manifest.json`.

## Measurements

There are 132 distinct sampled head queries and eight settings per query:
1,056 trials, in 528 timed two-head group/setting comparisons. All final reads
are certified by the guarded checker, but this includes dense fallbacks.

The table below uses **only the mixed growth stream**, not a pooled comparison
between different documents. Each row contains 12 distinct head queries at that
prefix, evaluated at all eight settings: 96 trials and 48 timed calls. Latencies
are medians of per-call medians; slowdown is the median of paired ratios.
Fallback and byte fractions are means across the head trials.

| Prefix | Reader event time | Dense event time | Paired slowdown | Fallback heads | Requested K/V bytes / dense |
|---|---:|---:|---:|---:|---:|
| 8,192 | 0.28715 ms | 0.01114 ms | 25.80x | 36.46% | 91.28% |
| 16,384 | 0.31388 ms | 0.01926 ms | 16.26x | 29.17% | 85.16% |
| 32,768 | 0.36619 ms | 0.02614 ms | 13.78x | 14.58% | 73.31% |

Matching median wall times are 0.39620/0.07038 ms, 0.43883/0.08537 ms and
0.47169/0.08814 ms respectively (reader/dense). The relative event-time gap
narrows with context here, but there is no measured crossover. Extrapolating a
speedup beyond 32K is not justified by these three content-changing samples.

The separate two-article workload, across its four prefixes from 4,097 to 8,192
tokens, has median event times 0.29309 ms versus 0.01045 ms, paired slowdown
29.08x, fallback fraction 35.81%, and requested K/V fraction 90.74%. These
measurements use the same fixtures as the previous adaptive experiment, but
fixed proposals and graph replay change both policy and timing conditions.

The 32K setting breakdown makes clear that removing fallback alone does not
recover a latency win. Each row below contains 12 head trials and six timed
calls; the dense baseline is independently paired in the artifact.

| Coordinates | Proposal fraction | Mass target | Fallback | Requested K/V / dense | Reader event time |
|---|---:|---:|---:|---:|---:|
| 32 | 25% | 10% | 16.67% | 58.33% | 0.35424 ms |
| 32 | 25% | 1% | 50.00% | 87.50% | 0.35371 ms |
| 32 | 50% | 10% | 0% | 62.50% | 0.37384 ms |
| 32 | 50% | 1% | 25.00% | 81.25% | 0.36603 ms |
| 64 | 25% | 10% | 0% | 62.50% | 0.35990 ms |
| 64 | 25% | 1% | 25.00% | 84.38% | 0.36177 ms |
| 64 | 50% | 10% | 0% | 75.00% | 0.38070 ms |
| 64 | 50% | 1% | 0% | 75.00% | 0.37712 ms |

Across all measured calls, even the best individual event ratio is 11.02x slower
than dense; the worst is 33.36x. Requested archive-load savings do not account
for the sorts, reductions and many remaining kernels. These work counters are
not evidence of reduced end-to-end latency or physical bandwidth consumption.

Mean final head-output L2 errors on the growth stream are 0.003771 at 8K,
0.003043 at 16K and 0.001707 at 32K, averaging all settings including dense
fallbacks. These are errors against frozen dense attention outputs, not language
model accuracy or retrieval success.

## Reproduction and verification

From the repository root, with offline model/data caches and approved CUDA access:

```sh
python -m ssa.partial_coordinate_fixture --out-dir /tmp/ssa_partial_fresh
python -m ssa.device_coordinate_fixture --out-dir /tmp/ssa_device_growth
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.device_coordinate_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests/test_device_coordinate_attention.py ssa/tests/test_device_coordinate_experiment.py ssa/tests/test_device_coordinate_fixture.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

The runner verifies NPZ hashes against extraction manifests before measurement.
The result records source hashes, base commit, fixture provenance, all timing
samples, per-head decisions, proposal and final omitted mass, output error,
rounding-aware output bound and work counts. The independent FP64 oracle checks
proposal mass even when the proposal is rejected. Returned intervals for rejected
heads are post-fallback dense intervals. Final-mask checks require the proposal
on acceptance and the full carrier on rejection.

Base commit is `35c2c0b69338d67e9301da798ea1d04361b6d6a2`, plus source identified
by SHA256 in the artifact. The full run has zero observed interval, proposal-mass,
final-mass and output-certificate deficits; no invalid numerical result, false
acceptance, uncertified final result or incorrect final mask. No trial exceeded
the 1e-6 audit tolerance. Focused verification includes **32 CUDA tests and 13
CPU tests** (**45 passed in 2.23 s** in the combined run). The GPU-enabled full suite reports **745 passed, 14 existing warnings
in 44.19 seconds**; exact output is recorded in the verification artifact.

Artifacts:

- [Measurements](../runs/device_coordinate_attention.json).
- [Verification, exact test output and hashes](../runs/device_coordinate_verification.json).
- [Kernel implementation](../ssa/device_coordinate_attention.py).
- [Timing and oracle protocol](../ssa/device_coordinate_experiment.py).

Removing host decisions did not make this path competitive against matched
native dense attention through 32K. The measured recommendation is to stop this
implementation direction under the agreed latency criterion, not to request
another mass-bound theorem or assume larger hardware will fix it. This does not
prove that every sparse-attention algorithm, different memory/hardware regime,
larger context or future implementation must fail.
