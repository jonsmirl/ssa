"""Shared block selection over a GROUP of queries — the query side of the summary hierarchy.

WHAT IS NEW HERE.  `bound_floor` summarises KEYS: a block of `b` keys becomes `(mu_c, Sigma_c)` and
Samuelson's bound is the tightest thing computable from that summary alone.  Every bound there is
evaluated PER QUERY, so routing a long context costs one bound evaluation per (query, block) pair.
At long context that is the dominant cost, because the number of queries grows with the context the
same way the number of keys does.

Real kernels therefore select blocks ONCE for a group of queries -- a decode chunk, the query heads
sharing a GQA group, a tile of positions -- and read the selected blocks for every query in the
group.  That is a different object from a per-query prune and it is not automatically exact:

    A bound admissible for one query says NOTHING about another query's argmax.

`Substrate/Universal/Potential/ChainedPrune.lean` settles the general form of this, machine-checked:

  * `a_prune_can_drop_the_greatest_part_of_a_second_objective` -- a prune beyond reproach for its own
    objective can drop another objective's best part outright.  The loss is not an approximation the
    second objective can bound; the part is gone.
  * `safe_for_every_stage_iff_safe_over_the_top_claim` -- a bound is safe for every member of a group
    EXACTLY when it dominates the group's TOP CLAIM, `max over the group`, pointwise per block.
  * `no_safe_bound_holds_below_the_top_claim` -- and the top claim is the LEAST such bound, so the
    price below is the price, not an artefact of one choice.
  * `the_safe_prune_drops_within_the_prune_of_one_stage` / `one_more_stage_drops_within_the_chain_drop`
    -- the shared prune drops within any single query's prune, monotonically in group size.  GROUP
    SIZE COSTS RETENTION.
  * `nothing_is_free_to_drop_when_every_part_is_within_one_reach` -- if every block is above the
    threshold for SOME query in the group, the safe prune drops nothing at all.

So the top claim is correct and is the floor.  The cost problem is that computing it needs every
query, which is the cost the grouping exists to avoid.  THE ALGORITHM HERE IS THE ANSWER TO THAT:
apply the paper's own summary hierarchy to the QUERY side as well, giving a bound above the top
claim that is computed once per (group, block).

    max_q max_k <q,k>
      <=  max_q [ <q, mu_c> ] + R_c                         key side: R_c = max_k ||k - mu_c||
      <=  <mu_Q, mu_c> + sqrt((mQ-1) mu_c^T Sigma_Q mu_c) + R_c          query side: Samuelson again

for unit-norm queries.  The middle term is the query-side analogue of the key-side one, and it
admits the SAME hierarchy:

    top_claim   max over the group of the per-query bound        O(mQ) per block   -- the least safe
    lowrank     rank-r truncation of Sigma_Q plus its tail       O(r d) per block
    isotropic   lambda_max(Sigma_Q) ||mu_c||                     O(d)  per block, mQ-independent

Each rung is admissible above the one below it, so the chain of relaxations composes and every rung
returns the true group argmax.  Cost is the discriminator, exactly as on the key side.

WHAT IS NOT CLAIMED.  These are bounds, not a kernel: nothing here is fused, batched, or tiled, and
the cost columns count KEYS SCORED and BOUND EVALUATIONS, not wall-clock.  The query-side hierarchy
inherits the key-side scope note -- it is measured on whatever key geometry it is handed, and
`ssa/bound_floor.py`'s numbers are synthetic clustered keys.
"""
from __future__ import annotations

import numpy as np

from .bound_floor import bound_ellipsoidal, bound_samuelson, kmeans_blocks


# =====================================================================================================
# the group floor, and the least safe bound
# =====================================================================================================

def group_oracle(K, mem, Q):
    """max over the group of max over the block.  The floor the PARTITION imposes on a shared
    selection -- not a router, and not computable without reading the keys."""
    return float((Q @ K[mem].T).max())


def group_top_claim(K, mem, Q, per_query=bound_samuelson):
    """The LEAST bound safe for every query in the group: the top claim of the per-query bounds.

    `safe_for_every_stage_iff_safe_over_the_top_claim` says safety for the whole group is exactly
    domination of this, and `no_safe_bound_holds_below_the_top_claim` says nothing safe sits below
    it.  Correct and minimal -- and O(mQ) per block, which is the cost grouping exists to avoid.
    """
    return max(float(per_query(K, mem, q)) for q in Q)


# =====================================================================================================
# the query-side summary hierarchy -- the point of the file
# =====================================================================================================

def query_summary(Q, eps=1e-9):
    """`(mu_Q, Y, mQ)` with `Y` the CENTRED queries, so `mu_c^T Sigma_Q mu_c = ||Y mu_c||^2 / mQ`."""
    Q = np.asarray(Q, dtype=np.float64)
    mu = Q.mean(0)
    return mu, Q - mu, len(Q)


def key_radius(K, mem):
    """`R_c = max_k ||k - mu_c||`: the key-side query-independent radius, one scalar per block,
    precomputed at index build.

    This is the term the group bound adds, and it must be a RADIUS and not `bound_ellipsoidal`'s
    `rho_c`.  `rho_c` is a radius in the WHITENED metric and is meaningful only multiplied by
    `sqrt(q^T Sigma_c q)`, which is query-dependent; dropping that factor to make the term shared
    inflates it by `sqrt(lam_max(Sigma_c))` and the bound stops pruning anything.  `R_c` bounds
    `max_k <q, k - mu_c>` directly for unit-norm `q`, and `R_c <= rho_c sqrt(lam_max(Sigma_c))`, so
    it is also the tighter of the two.
    """
    return float(np.linalg.norm(K[mem] - K[mem].mean(0), axis=1).max())


def group_bound_exact(K, mem, Q, eps=1e-3):
    """Query side EXACT: max over the group of the key-side ellipsoidal bound.  O(mQ) per block.

    This is the top claim of the ellipsoidal family and is the reference the two cheaper rungs are
    measured against -- they must never fall below it, or they would not be safe for the group.
    """
    return max(float(bound_ellipsoidal(K, mem, q, eps=eps)) for q in Q)


def group_bound_isotropic(K, mem, Q, eps=1e-3):
    """Query side ISOTROPIC: `<mu_Q, mu_c> + sqrt((mQ-1) lam_max(Sigma_Q)) ||mu_c|| + rho_c`.

    O(d) per block and INDEPENDENT OF GROUP SIZE -- one dot product against the group mean, plus two
    scalars fixed before any block is touched.  This is the rung a kernel can afford.
    """
    mu_Q, Y, mQ = query_summary(Q)
    mu_c = K[mem].mean(0)
    R = key_radius(K, mem)
    lam = float(np.linalg.svd(Y, compute_uv=False).max() ** 2 / max(mQ, 1)) if mQ else 0.0
    return (float(mu_Q @ mu_c)
            + np.sqrt(max(mQ - 1, 0) * lam) * float(np.linalg.norm(mu_c))
            + R)


def group_bound_lowrank(K, mem, Q, r=8, eps=1e-3):
    """Query side LOW RANK: the top-`r` query directions exactly, the tail isotropically.

    `mu_c^T Sigma_Q mu_c <= sum_{i<=r} lam_i <v_i, mu_c>^2 + lam_{r+1} ||mu_c||^2`, which is an upper
    bound because the discarded directions are bounded by the largest of them.  O(r d) per block.
    """
    mu_Q, Y, mQ = query_summary(Q)
    mu_c = K[mem].mean(0)
    R = key_radius(K, mem)
    if mQ <= 1:
        return float(mu_Q @ mu_c) + R
    s, V = np.linalg.svd(Y, compute_uv=True)[1:]
    lam = (s ** 2) / mQ
    r = min(r, len(lam))
    head = float((lam[:r] * (V[:r] @ mu_c) ** 2).sum())
    tail = float(lam[r] if r < len(lam) else 0.0) * float(mu_c @ mu_c)
    return float(mu_Q @ mu_c) + np.sqrt(max(mQ - 1, 0) * (head + tail)) + R


GROUP_BOUNDS = {
    "group_oracle": group_oracle,
    "top_claim": group_bound_exact,
    "lowrank": group_bound_lowrank,
    "isotropic": group_bound_isotropic,
}


# =====================================================================================================
# shared selection, and what it costs against per-query selection
# =====================================================================================================

def shared_select(K, Q, members, bound, eps=1e-3):
    """Select blocks ONCE for the whole group, then read every selected block for every query.

    Returns `(argmax_per_query, keys_scored, bound_evals)`.  `bound_evals` counts the UNDERLYING
    per-query bound evaluations, not blocks: `group_bound_exact` runs the key-side bound once per
    query per block and so costs `B*mQ` exactly as per-query selection does, while the two relaxed
    rungs cost `B`.  Counting blocks would credit the top claim with a saving it does not make, and
    that mis-accounting is what makes the query-side hierarchy look free when it is not.

    THE THRESHOLD IS THE GROUP'S WEAKEST INCUMBENT, not its strongest.  A block may be skipped only
    when it can help NO query in the group, and `ub <= best.max()` does not say that -- it can hold
    while `ub > best[j]` for a query `j` that still needs the block, and that query then returns the
    wrong argmax.  This is `ChainedPrune`'s per-stage threshold hypothesis in operational form: the
    chain is exact when NO stage's threshold has reached its own best value, so the shared threshold
    is bounded by the weakest of them.
    """
    ubs = [(float(bound(K, mem, Q)), mem) for mem in members]
    bound_evals = len(members) * (len(Q) if bound is group_bound_exact else 1)
    best = np.full(len(Q), -1e30)
    bkey = np.full(len(Q), -1, dtype=int)
    cost = 0
    for ub, mem in sorted(ubs, key=lambda t: -t[0]):
        if ub <= float(best.min()):
            break
        sc = Q @ K[mem].T                                   # [mQ, b]
        j = sc.argmax(1)
        v = sc[np.arange(len(Q)), j]
        upd = v > best
        best = np.where(upd, v, best)
        bkey = np.where(upd, np.asarray(mem)[j], bkey)
        cost += len(mem) * len(Q)
    return bkey, cost, bound_evals


def per_query_select(K, Q, members, bound, eps=1e-3):
    """The baseline: run the key-side prune independently for every query."""
    from .bound_floor import bnb_cost
    keys, cost = [], 0
    for q in Q:
        k, c = bnb_cost(K, q, members, bound)
        keys.append(k)
        cost += c
    return np.asarray(keys), cost, len(members) * len(Q)


def true_argmax(K, Q):
    return (Q @ K.T).argmax(1)


# =====================================================================================================
# the sweep -- what shared selection actually buys, and where the query-side hierarchy fails
# =====================================================================================================

def make_case(spread=0.05, qspread=0.02, mQ=32, n=8192, d=64, B=128, seed=1):
    """Clustered keys and a query group drawn around ONE cluster centre.  `qspread` is the group's
    own tightness and is the variable that decides everything below."""
    r = np.random.default_rng(seed)
    c = r.standard_normal((B, d))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    K = c[r.integers(0, B, n)] + spread * r.standard_normal((n, d))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    qc = c[r.integers(0, B)]
    Q = qc + qspread * r.standard_normal((mQ, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return K, Q, kmeans_blocks(K, B)


def work(keys_scored, bound_evals):
    """Total units of `O(d)` work.  A summary bound against a PRECOMPUTED `(mu_c, Sigma_c)` costs the
    same order as scoring one key, so the two columns add.  `ssa/bound_floor.py` reports keys alone
    because it varies only the bound; here the arms differ in how many bounds they evaluate, so
    counting keys alone would credit a saving that was paid for elsewhere."""
    return keys_scored + bound_evals


def sweep_group(mQs=(8, 32, 128, 512), qspreads=(0.02, 0.10, 0.30), spread=0.05,
                n=8192, d=64, B=128, seed=1, log=print):
    """Shared selection against per-query selection, at the same partition and the same key bound.

    Every arm is exact by construction and the assertion below checks it: an arm that returned a
    wrong argmax would be an instrument failure, not a cheaper algorithm.
    """
    out = []
    log(f"n={n} d={d} B={B} spread={spread} — total work = keys scored + bound evaluations")
    log(f"{'qspread':>8} {'mQ':>5} | {'per-query':>10} | {'top_claim':>10} {'x':>6} | "
        f"{'lowrank':>10} {'x':>6} | {'isotropic':>10} {'x':>6}")
    for qspread in qspreads:
        for mQ in mQs:
            K, Q, members = make_case(spread=spread, qspread=qspread, mQ=mQ, n=n, d=d, B=B, seed=seed)
            gt = true_argmax(K, Q)
            _, pc, pb = per_query_select(K, Q, members, bound_samuelson)
            base = work(pc, pb)
            row = {"qspread": qspread, "mQ": mQ, "per_query": base}
            cells = []
            for name, bd in (("top_claim", group_bound_exact),
                             ("lowrank", lambda K, m, Q: group_bound_lowrank(K, m, Q, 8)),
                             ("isotropic", group_bound_isotropic)):
                k, c, b = shared_select(K, Q, members, bd)
                assert (k == gt).all(), f"{name} lost the argmax — every group bound is admissible"
                w = work(c, b)
                row[name], row[name + "_x"] = w, base / max(w, 1)
                cells.append(f"{w:10d} {base / max(w, 1):5.2f}x")
            log(f"{qspread:8.2f} {mQ:5d} | {base:10d} | " + " | ".join(cells))
            out.append(row)
    return out
