# Summary recovery and persistent ridge: public experiment specification

These CPU references distinguish **information discarded by a fixed summary**, **instability of a
particular fit**, and **actual attention-output error**. None of those quantities alone is retrieval
accuracy. All target weights in the diagnostic come from a charged dense softmax oracle. The
implementation does not solve the open cheap, query-uniform signed-residual construction.

Code: [`cell_summary_minimax.py`](../ssa/cell_summary_minimax.py),
[`persistent_ridge.py`](../ssa/persistent_ridge.py),
[`summary_recovery_experiment.py`](../ssa/summary_recovery_experiment.py).
Measurements: [`summary_recovery_comparison.json`](../runs/summary_recovery_comparison.json).

## Cell summaries: an attained arbitrary-decoder limit

Fix real target coefficients `p`, a partition of positions into cells, and exact selected positions
`S`, before varying unknown scalar values over `[-1,1]^n`. The observation is the sum of all values
in each cell, together with the selected values. For each cell `C`, let `c_C` be a median of its
**unread** coefficients, or zero when no unread positions remain. Then the sharp minimax radius is

\[
 d=\sum_C\sum_{i\in C\setminus S}|p_i-c_C|.
\]

This is an optimum over **every deterministic decoder of this fixed observation**, not just linear
decoders. Here is a self-contained proof for the partition case.

The decoder

\[
 \widehat o=\sum_C c_C\sum_{i\in C}v_i
             +\sum_{i\in S}(p_i-c_{C(i)})v_i
\]

has signed error `-sum_unread (p_i-c_C(i)) v_i`. Consequently its worst-case absolute error is
at most `d`. Medians minimize the sum of absolute deviations independently in each cell.

For the matching lower bound, sort unread positions in each cell by `(p_i, original index)`.
Assign `h=-1` to the lower half and `h=+1` to the upper half. An odd middle position and every
selected position receive zero. Each cell's integer sum is exactly zero, selected coordinates
vanish, and `|h_i|<=1`. Pairing with `p` gives precisely the median absolute-deviation sum `d`.
Thus `h` and `-h` have identical observations but targets `d` and `-d`; every decoder errs by
at least `d` on one of them. The displayed decoder attains that minimax value.

For vector values with row norm at most `B`, the same residual identity gives output L2 error
at most `B*d`. A coordinatewise cube gives a componentwise bound, not the same unscaled vector
bound. The code checks integer feasibility exactly and floating-point pairings/outputs numerically.
It does not implement an interval certificate for real-arithmetic model softmax. Algebraic
witness values need not be realizable by Qwen. We neither optimize `S` nor claim an observed
retrieval ceiling, a workload-averaged error floor, or a lower bound over all possible summaries.

The median decoder's coefficients need not sum to one. Its error-optimality is a worst-case
cube statement; it need not minimize error on actual correlated model values. The unread-mean
coefficient control is another decoder of the same sums. It is distinct from the earlier
full-cell-mean predicted-value replacement baseline; the experiment reports both separately.

## Persistent all-prefix ridge

Fix a feature map independent of values and a positive, constant penalty `lambda`. Maintain

\[
 H_t=\sum_{i<t}x_ix_i^T,\qquad B_t=\sum_{i<t}x_iv_i^T,
 \qquad W_t=(H_t+\lambda I)^{-1}B_t.
\]

Appending a row adds its outer products, without rereading old rows. At a solve, `W_t` is the
unique minimizer of the **sum-loss** objective

\[
 \|Y-XW\|_F^2+\lambda\|W\|_F^2.
\]

Writing `D=W-W_t`, subtracting the optimum objective gives exactly
`||XD||_F^2 + lambda ||D||_F^2`: the normal equation cancels the cross terms. The intercept is
penalized too; there is no moving prior and the penalty is not multiplied by prefix length.

This is not `W <- W + G_lambda(Y-XW)`. For the scalar unit-design, unit-target, unit-penalty
example, persistent ridge is `1/2`, while two residual corrections produce `3/4` and increase
the penalized objective from `1/2` to `5/8`. Tests keep these algorithms distinct.

For a fixed unread feature reader `a = sum_unread p_i x_i`, reconstruction is
`sum_selected p_i v_i + a W`. Its error is the actual signed unread value residual, not a sum
of value-error magnitudes. With `G_lambda=(H+lambda I)^-1 X^T`, the exact fixed-reader mismatch
gain is

\[
 \|aG_\lambda\|_2=\sqrt{b^THb},\qquad b=(H+\lambda I)^{-1}a.
\]

This permits gain evaluation from the Gram matrix without retaining `G_lambda`. It is at most
`||a||/(2 sqrt(lambda))`. To see the underlying operator bound, the normal equation gives
`||Xz||^2+lambda||z||^2 <= ||Xz|| ||y||` for `z=G_lambda y`; completing the square yields
`4 lambda ||z||^2 <= ||y||^2`. A mismatch matrix of Frobenius norm at most `epsilon` can produce
read error at most `epsilon ||aG_lambda||`, with rank-at-most-one attainment. But no justified
mismatch radius or target-bias bound is supplied for Qwen here, so the measured gain is **not an
attention-output certificate**. Irreducible least-squares residual and amplified in-column
mismatch are different: the pseudoinverse annihilates the former.

## Protocol and accounting

Defaults use the previously inspected Qwen2.5-0.5B layer-18/KV-0 fixture at
`/tmp/ssa_qwen_qkv_8192.npz`, with SHA-256
`5656b2725bbdb29a09c7c13c93126ac3f2cfb44ba0c32e60768ffc49fc54895a`.
The first 512 keys fix normalization, random projected-linear/tanh bases, and cell centers.
Penalty candidates `{0.01,1,100,10000}` fit rows `[0,3072)` and are selected by frozen value
MSE on `[3072,4096)`. Each chosen penalty is frozen, and the model is refitted on `[0,4096)`.
Suffix values never choose the penalty. This is a chronological split in an already inspected
document, not an independent held-out corpus. Query positions are 16 evenly spaced positions
from 4096 through 8191; seeds are 0, 1, 2.

Frozen suffix value prediction is reported separately from online evaluation. Online state admits
every preceding value before its query. Updates follow a fixed batch grid; a partial query batch
uses a discarded copy. Extra measurement queries therefore do not alter future persistent fits,
but their repeated reads and solves are charged. Both ridge families, matched-feature joint/delta
controls and earlier budget-matched models use identical queries, causal prefixes and exact selected
keys (two past 64-key blocks plus the current causal block).

For feature dimension `r`, value dimension `dv`, and key dimension `d`, ridge's cap includes
`r*r + 2*r*dv + 2` scalar-equivalents for Gram, cross statistics, cached decoder, count and penalty,
plus `2*d + (d+1)*(r-1)` for normalization, random projection and bias. Caps 8192/16384 therefore
allow **35/63 features**, occupying **8045/16193 scalars**, per family. Cells use 63/127 centers
and counts/value sums: 8127/16383 scalars. Identical-feature joint/delta controls occupy less
state; separately rerun earlier controls can use the full cap for wider features.

These are representation-state caps, **not total resident memory limits**. The reference holds
the full Q/K/V archive, materialized features, frozen/online/query replicas, multiple controls,
and solver workspace simultaneously. Reports itemize statistics operations, solves, reader-gain
refactorizations, validation rereads, old-anchor diagnostics, control SVDs, least-squares oracles,
feature generation, dense logits/truth, exact selected reads, reference routing scans, cell
membership scans and sorting. Work proxies and workspace estimates exclude library internals
and are not allocator measurements or runtime bounds.

The median profile scans `O(c*n)` membership predicates plus sorts within cells. Computing all
true weights and unread feature aggregates is also dense. No subquadratic attention guarantee
is inferred from fixed fitting-state size. The selected-only oracle control deliberately uses
the true global denominator and zero tail values; its success cannot be deployed by renaming it
sparse attention. The one-global-sum median control tests whether geometry-specific cells help.

## Provenance and remaining scope

The partition proof above specializes the general fixed-channel `LinearReadRecovery` result;
ridge uses the `PositiveFit`, `ColumnReadGain` and `SelectedLinearRead` results. The audited
Substrate build is `940cee72ffb2d8ccd3c78bb70a1c2b5ce06382fb`, following `32f3c1975`, `7c1cfaadc`,
`0e828ede0` and `0dff6a1da`. This document supplies the mathematics needed by this public code;
no private repository is needed to run or understand the experiments. These new Python
implementations and the partition-specific specialization were not themselves Lean-verified.

Still open: an accessible signed coefficient-residual enclosure; useful query-uniform features;
efficient routing/normalization without dense oracles; floating-point certification; actual
conditional value laws; and end-to-end CE/retrieval preservation. Improved regression stability
does not establish that any of these has been solved.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ssa.summary_recovery_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest \
  ssa/tests/test_cell_summary_minimax.py ssa/tests/test_persistent_ridge.py \
  ssa/tests/test_summary_recovery_experiment.py -q
```
