# Partial-coordinate GPU experiment: certified reads, no latency win

Status: implemented and measured on the local RTX 4080. The fused coordinate
and selected-value kernels pass the fresh-geometry numerical audit, but the
complete adaptive reader is **slower than native BF16 dense attention in every
measured comparison**. Lower requested K/V traffic is not a speedup.

The experiment covers two named WikiText2 test articles, three layers, four query
heads and four positions: 96 distinct head queries, each tested at four settings,
for 384 audited trials. It is sampled attention on frozen dense-model geometry,
not a sparse-model rollout, semantic-retrieval evaluation, or serving benchmark.
The local card was sufficient; Kaggle was not used.

## Measured result

Each setting certified all 96 head queries. Percentages are arithmetic means
over those queries; latency is the median of 48 two-head group medians.

| Initial coordinates | Mass target | Value rows read | Requested K/V bytes / dense | Mean mass upper bound | Actual omitted mass | Reader latency |
|---|---:|---:|---:|---:|---:|---:|
| 32/64 | 10% | 41.33% | 56.00% | 5.016% | 0.3584% | 8.570 ms |
| 32/64 | 1% | 75.01% | 81.25% | 0.2189% | 0.01409% | 9.914 ms |
| 64/64 | 10% | 10.28% | 55.14% | 4.276% | 4.220% | 4.783 ms |
| 64/64 | 1% | 34.81% | 67.41% | 0.4629% | 0.4551% | 7.429 ms |

The matching dense BF16 SDPA baseline takes **0.07866 ms** median; the dense FP32
SDPA baseline takes **1.44448 ms**. The reader is slower in 192/192 group/setting
comparisons against BF16 and 191/192 against FP32. Median paired slowdowns against BF16 are
101.38x, 130.18x, 53.92x and 85.51x respectively. Index construction is separate;
including it gives median reader-plus-build times of 8.998, 10.474, 5.378 and
7.911 ms. No row is faster than the BF16 baseline.

At 32 coordinates, full value reads occur in 5.21% of the 10%-target trials and
38.54% of the 1%-target trials. At 64 coordinates the corresponding fractions
are 0% and 8.33%. Reading all 64 coordinates means scoring **every key**, even
when relatively few values are fetched.

Mean head-output L2 errors are 0.006538, 0.0002810, 0.06914 and 0.009755 in table
order. Mean output-error upper bounds are much looser: 1.3643, 0.3659, 1.1000 and
0.3925. A mass certificate is not a promise of a tight value-output estimate.

Token 0 is the highest-attention key in 75/96 queries (78.125%), versus 16/16 in
the earlier single-head cache. These fresh heads reduce, but do not eliminate,
the attention-sink confound. Top-key coverage is not semantic retrieval accuracy.

## Mathematical bridge and floating-point boundary

For a visible-prefix key j in block b, let m_ba and M_ba be that block's exact
coordinate extrema. With beta = 1/sqrt(64), and initially read coordinates D,

\[
\widetilde s_j=\beta\sum_{a\in D}q_aK_{ja},\quad
L_j=\widetilde s_j+\sum_{a\notin D}\min(\beta q_am_{ba},\beta q_aM_{ba}),\quad
U_j=\widetilde s_j+\sum_{a\notin D}\max(\beta q_am_{ba},\beta q_aM_{ba}).
\]

In real arithmetic, L_j <= s_j <= U_j. This is an attention-logit interval,
not CCC's routing-metric selection certificate. Completing selected keys S gives
their logits. To account conservatively for numerical score uncertainty, use
the **lower retained mass**, not the nominal computed denominator:

\[
Z_- = \sum_{j\in S}e^{L_j}\le Z_S,\qquad
A_+ = \sum_{j\notin S}e^{U_j}\ge Z_{S^c},\qquad
\delta\le\bar\delta=\frac{A_+}{Z_-+A_+}.
\]

The implemented stopping margin, with a numerical allowance on the log ratio,
is log A_+ - log Z_- - log(eta/(1-eta)). A nonpositive margin is the requested
mass test, not a top-k proxy. Full selection sets omitted mass and supported KL
to zero explicitly.

For the ideal exact-score restricted distribution, TV equals actual omitted
mass and KL(restricted || full) = -log(1-delta). Thus the returned ideal-read KL
bound is softplus(log A_+ - log Z_-), including the log-ratio allowance. If all
visible value norms are at most B, ideal restricted output error is at most
2 B bar-delta. The computed FP32 output additionally differs from that ideal
read. With c the largest selected-score interval allowance, this implementation
returns the conservative form

\[
E_{\rm output}\le 2B\bigl(\bar\delta+\min(c,1)\bigr)+\epsilon_{\rm arithmetic}.
\]

The score and arithmetic allowances are engineering guards, not formally
verified IEEE interval arithmetic. FP32 interval guards scale with absolute
product sums; FP64 log-mass guards include the prefix length as an allowance for
sequential suffix accumulation. The output arithmetic allowance depends on
B, key/value dimensions and log prefix length. The KL/TV identities describe
the ideal restricted distribution, not a claim that computed floating weights
form an exact probability vector.

A numerical audit found and corrected an important normalization bug: casting
an absolute log partition to FP32 before subtracting it from very large logits
can destroy normalization. For 128 identical logits of 10^9, that operation can
make the weights sum to 128 instead of 1. The kernel now subtracts a shared
score maximum before computing and applying the log partition. Regression tests
cover large common logits. This repair does not turn the remaining guards into
a general rounding proof.

The public mathematical foundations are also described in
[the CPU experiment](partial_coordinate_attention.md): partial-pairing bounds,
attention-logit intervals, and restricted-read TV/KL/output transport. Private
Substrate provenance includes PartialPairingTop `7395e8a8a`, its SparQ recognition
`495dcb05f`, and TopSetMonotone `f0ce0f345`; this document does not require private
repository access. The new kernels are not Lean-verified.

## Implementation, policy and charged work

Triton kernels fuse initial partial-coordinate reads with interval construction,
complete only newly selected keys' missing coordinates, and reduce only selected
values. Query heads route independently against one shared KV head. Stable
sorting resolves implementation ties by token index; it does not prove strict
top-set separation at a boundary tie.

After the seed is completed, unopened U_j remain unchanged: completing a selected
key does not alter any other key's coordinate summary or query. Therefore the
reader may sort remaining U_j once and precompute their reverse log-cumulative
exponential sums. Subsequent reads consume a prefix of that fixed order, and the
remaining mass bound is exactly its suffix sum in real arithmetic. This replaces
repeated tail sorting/reduction without changing the intended selection policy.
The retained lower-mass reduction and host-controlled stopping remain adaptive.

The initial seed is 128 plus the partial boundary-block length, capped by prefix
size. Boundary keys are force-kept; other seed keys follow partial logits. Each
subsequent batch doubles the selected count, subject to the remaining budget.
The dense oracle runs only after the policy stops. A full read remains available.

This GPU reader uses an **unquantized per-key upper-cap sum**, not the CPU
experiment's 16-band profile. It also doubles batches rather than adding 128
keys each time. Its seed size now matches the CPU runner, but its documents,
heads, floating precision, tail representation and batch schedule differ.
CPU-versus-GPU value fractions are not an implementation-only comparison.

For n keys, r initial coordinates and k selected keys, the logical K/V requests
are n*r + k*(64-r) key scalars and k*d_V value scalars. Requested byte fractions
count the actual BF16 archive dtype; they are **not measured DRAM traffic** and
exclude query/index-summary/intermediate accesses. The artifact separately
records bounds, sorts, suffix preprocessing/lookups, retained reductions and
adaptive stages. Reading n*r coordinates per query does not establish
subquadratic full-prefill work.

Index construction copies the visible K/V archive, scans all keys for extrema,
and scans all values for B. Its latency/storage are reported separately. It is
not a free online-maintenance algorithm. Prefix snapshots exclude future keys,
including in summaries of partial boundary blocks.

## Fresh fixtures and reproducibility

Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987` runs frozen BF16
dense prefill, all 24 layers, without producing vocabulary logits. The scoped
SDPA capture is restored afterward. Post-RoPE Q/K/V are saved as lossless FP32
casts of the BF16 tensors. Both captures reported exactly 24 SDPA calls and
1.306 GB peak allocated CUDA memory.

The named WikiText2 raw-v1 test articles are:

- Ise @-@ class battleship: rows [177,264), 9,482 article tokens, first 8,192 used.
- Second Battle of Naktong Bulge: rows [322,447), 12,444 tokens, first 8,192 used.

Head indices 0/3/7/10 are sampled at zero-based layers 6/12/18 and positions
4096/5461/6826/8191. Query heads 0/3 use KV head 0; 7/10 use KV head 1. Each timed
call contains **two heads sharing one KV archive**, not all 14 model heads. The
visible prefix ends at and includes the query token. Both dense baselines use
that same group/prefix; FP32 input casts are prepared outside baseline timing.
The oracle is a separate FP64 dense calculation on the same captured inputs.

Source base commit: `35c2c0b69338d67e9301da798ea1d04361b6d6a2`, plus the source
hashes recorded in the result. The fixture manifest records model revision,
article/text/token hashes, dataset Arrow hash, capture shapes and extractor hash.
The runner verifies fixture hashes against this manifest before evaluation.
Fixture hashes:

- `/tmp/ssa_partial_fresh/article_0.npz`: `ef17f1c7ca3507a482424a03a8b4e7b56cc2759329f8041f192e665bc0925b88`.
- `/tmp/ssa_partial_fresh/article_1.npz`: `2fe4192c04ac48aedcb94ed72fad017da6c07d9ebf816deb634696cd206c6490`.

From the repository root, with cached model/data and approved CUDA access:

```sh
python -m ssa.partial_coordinate_fixture --out-dir /tmp/ssa_partial_fresh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.partial_coordinate_gpu_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.partial_coordinate_gpu_experiment --limit-groups 4 --repeats 5 --uncached-tail --out runs/partial_coordinate_gpu_uncached.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests/test_partial_coordinate_gpu.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests/test_partial_coordinate_fixture.py ssa/tests/test_partial_coordinate_gpu_experiment.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest ssa/tests -q
```

The full run uses five repetitions after two warmups. Synchronized wall time
includes launches, sorting, host decisions, synchronization, certificate checks
and output computation. Compilation warmup and dense-oracle work are excluded.
These are warm measurements, not cold starts or throughput under concurrent load.

The four-group uncached control also passed all audits and matched selected-key
counts. Across its 16 matched setting/group pairs, median uncached/cached latency
was 1.025x; cached was faster in only 8/16. The runs were not interleaved, and the
dense baseline itself shifted by a median factor of 1.181x. This is not evidence
of a robust measured speedup from suffix caching, despite its reduced work count.

## Verification and remaining scope

The 384-trial fresh run has zero observed interval, omitted-mass, supported-KL
and output-certificate deficits, and no false stop. No trial exceeded the audit
tolerance of 1e-6. The largest numerical L2 difference from the exact-score
restricted output was 1.679e-6; this is separately covered by the output
allowance, not misreported as a mass deficit. Focused tests: **35 CUDA tests and
13 CPU tests passed** (combined run: **48 passed in 2.75 s**). The GPU-enabled full suite reports **700 passed, 14
warnings in 44.64 seconds**, with all 35 CUDA tests executed. Full-suite output
and final hashes are recorded in the verification artifact.

Artifacts:

- [Main measurements](../runs/partial_coordinate_gpu.json).
- [Uncached control](../runs/partial_coordinate_gpu_uncached.json).
- [Verification and hashes](../runs/partial_coordinate_gpu_verification.json).
- Local extraction manifest: `/tmp/ssa_partial_fresh/manifest.json`.

The measured conclusion is narrow: sparse value mass can be certified on these
fresh geometries, but this adaptive implementation does not improve latency.
Neither favorable score tails for arbitrary documents, formal floating-point
soundness, semantic retrieval quality, nor complete-model prediction preservation
has been established. A useful successor must reduce actual routing/control
overhead and demonstrate a latency win against the native dense baseline, without
replacing the deterministic real-arithmetic guarantee with a calibrated one.
