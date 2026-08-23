"""
The bound floor — separating what the PARTITION costs from what the SUMMARY costs.

WHY THIS EXISTS.  The paper says branch-and-bound cost is "governed entirely by how tight the bound
is, i.e. by the geometry of the keys" (§4.1).  That is a statement about a numerator with no
denominator: it says looser bounds cost more without saying how much of the cost is *avoidable*.
`anisotropic_bound.py` compares two bounds against each other; nothing compares either against the
best a bound of that kind could possibly be.

THE DECOMPOSITION.  For a fixed query and a fixed block partition, three costs bracket the problem:

    ORACLE      U_c = max_{k in c} <q,k>      the tightest ADMISSIBLE bound that exists at all.
                                              Requires reading the block, so it is not a router --
                                              it is the floor the PARTITION imposes.
    SAMUELSON   U_c = <q,mu_c> + sqrt((m-1) qSq)   the tightest bound computable from (mu, Sigma, m)
                                              ALONE.  Tight in the strong sense: attained (below).
    ELLIPSOIDAL the bound the repo already ships, from (mu, Sigma, rho_c).  rho_c is
                                              QUERY-INDEPENDENT, so it is precomputed at index-build
                                              time and stored as ONE extra scalar per block -- this
                                              bound is summary-only at query time too.  The summary
                                              hierarchy is therefore about summary SIZE, not about
                                              summaries versus keys.

    cost(SAMUELSON) - cost(ORACLE)  =  the irreducible price of routing from summaries.
    cost(ORACLE)    - (one block)   =  the price of the partition itself.

THE TIGHTNESS CLAIM, AND WHY IT IS THE POINT.  Samuelson's inequality is not merely an upper bound
on the deviation of m points with a given mean and variance -- it is ATTAINED, by one point at
mu + s*sqrt(m-1) and m-1 points at mu - s/sqrt(m-1).  So for any candidate bound STRICTLY below the
Samuelson value there exists a block with exactly the summary the router read whose true maximum
exceeds it, and the bound is inadmissible on that block.

    ==> No summary-only router can prune more than the Samuelson bound without reading keys.

This is the NECESSARY-side companion to the paper's benign-geometry condition, which is stated (and
labelled) as sufficient only.  It is an instance of a proved abstract ceiling in the substrate tree:
`Universal/Potential/PartialScore.lean` (`score_error_ge_of_reach_split`, `not_exactOnReach_of_reach_split`)
says a single score standing for a whole fibre of completions carries an error floor set by the
spread of the objective across that fibre.  Here the partial object is the summary (mu, Sigma, m),
its completions are the blocks carrying that summary, and the objective is the block's true maximum
logit.  The mapping is asserted in prose, not machine-checked -- see `docs/substrate_math_imports.md`.

SCOPE, stated because the result is easy to over-read.  The floor is about ADMISSIBLE bounds, hence
about LOSSLESS selection.  A lossy router may beat it and SSA's budget-kappa mode does.  And the
oracle is not achievable by any router; it is a reference, not a target.

Run:  python3 -m ssa.bound_floor
"""
from __future__ import annotations

import numpy as np

from .core import clustered_keys, random_unit_keys
from .prune_regularizer import samuelson_bnb


# =====================================================================================================
# the bounds
# =====================================================================================================

def _stats(K, mem, q):
    proj = K[mem] @ q
    m = len(mem)
    mean = float(proj.mean())
    var = float(((proj - mean) ** 2).mean())          # POPULATION variance, as Samuelson needs
    return proj, m, mean, max(var, 0.0)


def bnb_cost(K, q, members, bound):
    """Best-first branch-and-bound under a caller-supplied admissible bound.

    Returns (argmax_key, keys_scored).  Identical control flow to `prune_regularizer.samuelson_bnb`
    so the arms differ ONLY in the bound -- if the loop differed, the comparison would be measuring
    two algorithms rather than two bounds.
    """
    ubs = [(bound(K, mem, q), mem) for mem in members]
    best, bkey, cost = -1e30, -1, 0
    for ub, mem in sorted(ubs, key=lambda t: -t[0]):
        if ub <= best:
            break
        sc = K[mem] @ q
        j = int(sc.argmax())
        if sc[j] > best:
            best, bkey = float(sc[j]), int(mem[j])
        cost += len(mem)
    return bkey, cost


def bound_oracle(K, mem, q):
    """U_c = max_{k in c} <q,k>.  The tightest admissible bound that exists.  NOT a router."""
    return float((K[mem] @ q).max())


def bound_samuelson(K, mem, q):
    """U_c = <q,mu> + sqrt((m-1) * qSq).  Summary-only, and tight (see `samuelson_witness`)."""
    _, m, mean, var = _stats(K, mem, q)
    return mean + np.sqrt(max(m - 1, 0) * var)


def bound_isotropic(K, mem, q):
    """U_c = <q,mu> + ||q||*R_c.  R_c is query-independent: precomputed, one scalar per block."""
    proj, m, mean, _ = _stats(K, mem, q)
    mu = K[mem].mean(0)
    R = float(np.linalg.norm(K[mem] - mu, axis=1).max())
    return mean + float(np.linalg.norm(q)) * R


def bound_ellipsoidal(K, mem, q, eps=1e-3):
    """U_c = <q,mu> + rho_c*sqrt(q^T S q).  rho_c is query-independent: precomputed, one scalar.

    THE QUADRATIC FORM IS THE REGULARISED ONE, and it has to be.  rho_c is measured in the metric of
    `S = Sigma_c + eps*I`, so the Cauchy-Schwarz step

        <q, k - mu> = <S^(1/2) q, S^(-1/2) (k - mu)>  <=  sqrt(q^T S q) * rho_c

    produces `q^T S q = q^T Sigma_c q + eps*||q||^2`, not `q^T Sigma_c q`.  Dropping the eps term
    makes the bound INADMISSIBLE by up to eps*||q||^2/(2*sqrt(...)) — small, but a bound that can
    sit below the block's own maximum can drop the block holding the argmax, and then the recall
    column is no longer 1.000 by construction.  Measured on 32-dimensional blocks of ~33 keys, the
    unregularised form fell below `bound_oracle` on 5 of 128 (block, query) pairs by up to 2.8e-3.
    """
    proj, m, mean, var = _stats(K, mem, q)
    X = K[mem] - K[mem].mean(0)
    S = (X.T @ X) / max(m, 1) + eps * np.eye(X.shape[1])
    try:
        Si = np.linalg.inv(np.linalg.cholesky(S))
    except np.linalg.LinAlgError:
        return bound_isotropic(K, mem, q)
    rho = float(np.linalg.norm(X @ Si.T, axis=1).max())
    return mean + rho * np.sqrt(var + eps * float(q @ q))


BOUNDS = {"oracle": bound_oracle, "samuelson": bound_samuelson,
          "isotropic": bound_isotropic, "ellipsoidal": bound_ellipsoidal}


# =====================================================================================================
# the tightness certificate
# =====================================================================================================

def samuelson_witness(m: int, mean: float, sd: float):
    """The configuration that ATTAINS Samuelson: one point at mean + sd*sqrt(m-1), the rest at
    mean - sd/sqrt(m-1).

    Returned so a caller can check the attainment numerically rather than trusting the algebra --
    the whole tightness claim rests on this configuration existing, and an off-by-one in the
    exponent would leave the bound merely valid rather than tight.
    """
    if m < 2:
        return np.array([mean])
    hi = mean + sd * np.sqrt(m - 1)
    lo = mean - sd / np.sqrt(m - 1)
    return np.array([hi] + [lo] * (m - 1))


def check_attainment(m: int, mean: float = 0.0, sd: float = 1.0, tol: float = 1e-9):
    """Does the witness have the stated mean and sd, and does its max sit exactly on the bound?"""
    x = samuelson_witness(m, mean, sd)
    return {"m": m, "mean_err": abs(float(x.mean()) - mean), "sd_err": abs(float(x.std()) - sd),
            "max_dev": float(x.max() - mean), "bound": sd * np.sqrt(m - 1),
            "attained": abs(float(x.max() - mean) - sd * np.sqrt(m - 1)) < tol}


# =====================================================================================================
# the sweep
# =====================================================================================================

def kmeans_blocks(K, B, iters=12, seed=0):
    """Non-empty k-means blocks. `adaptive.kmeans` returns (members, means, radii)."""
    from .adaptive import kmeans
    members = kmeans(K, B, iters=iters, seed=seed)[0]
    return [np.asarray(m) for m in members if len(m) > 0]


def run(n=4096, d=64, B=64, trials=200, noise=0.15, seed=1, log=print):
    rng = np.random.default_rng(seed)
    geoms = {
        "benign (clustered)": clustered_keys(n, d, B, spread=0.15, seed=seed)[0],
        "isotropic (random)": random_unit_keys(n, d, seed=seed),
    }
    out = {}
    for gname, K in geoms.items():
        K = np.asarray(K, dtype=np.float64)
        K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
        members = kmeans_blocks(K, B, seed=seed)
        rows = {k: [] for k in BOUNDS}
        recall = {k: 0 for k in BOUNDS}
        for _ in range(trials):
            i = int(rng.integers(n))
            q = K[i] + noise * rng.standard_normal(d)
            q = q / (np.linalg.norm(q) + 1e-9)
            tgt = int((K @ q).argmax())
            for bname, bfn in BOUNDS.items():
                bkey, cost = bnb_cost(K, q, members, bfn)
                rows[bname].append(cost)
                recall[bname] += int(bkey == tgt)
        res = {}
        for bname in BOUNDS:
            c = np.array(rows[bname], dtype=float)
            res[bname] = {"mean_keys": float(c.mean()), "frac_of_n": float(c.mean() / n),
                          "recall": recall[bname] / trials}
        # the decomposition
        orc, sam = res["oracle"]["mean_keys"], res["samuelson"]["mean_keys"]
        res["_decomposition"] = {
            "partition_price_keys": orc,
            "summary_price_keys": sam - orc,
            "summary_price_multiple": (sam / orc if orc > 0 else None),
        }
        out[gname] = res
        log(f"\n{gname}  (n={n}, d={d}, B={B}, {trials} queries)")
        log(f"  {'bound':<12} {'keys read':>10} {'frac of n':>10} {'recall':>8}")
        for bname in ("oracle", "ellipsoidal", "isotropic", "samuelson"):
            r = res[bname]
            log(f"  {bname:<12} {r['mean_keys']:>10.1f} {r['frac_of_n']:>10.4f} "
                f"{r['recall']:>8.3f}")
        ell = res["ellipsoidal"]["mean_keys"]
        log(f"  -> floor {orc:.1f} keys; (mu,Sigma,b) is {sam / orc:.1f}x the floor; "
            f"(mu,Sigma,rho) is {ell / orc:.1f}x -- one extra stored scalar buys {sam / ell:.1f}x")
    return out


def sweep(spreads=(0.02, 0.05, 0.10, 0.20, 0.40), n=4096, d=64, B=64, trials=120,
          noise=0.15, seed=1, log=print):
    """Does the SUMMARY price fall toward the partition floor as geometry becomes benign?

    This is the quantitative form of the paper's routability claim (§7): training manufactures
    geometry in which summary-only routing is cheap.  The floor says how much room that claim has.
    """
    rng = np.random.default_rng(seed)
    log(f"\n{'spread':>8} {'oracle':>9} {'samuelson':>10} {'ellipsoid':>10} "
        f"{'summary x floor':>16} {'recall':>7}")
    rows = []
    for sp in spreads:
        K = np.asarray(clustered_keys(n, d, B, spread=sp, seed=seed)[0], dtype=np.float64)
        K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
        members = kmeans_blocks(K, B, seed=seed)
        cost = {k: [] for k in ("oracle", "samuelson", "ellipsoidal")}
        hit = 0
        for _ in range(trials):
            i = int(rng.integers(n))
            q = K[i] + noise * rng.standard_normal(d)
            q = q / (np.linalg.norm(q) + 1e-9)
            tgt = int((K @ q).argmax())
            for bn in cost:
                bkey, c = bnb_cost(K, q, members, BOUNDS[bn])
                cost[bn].append(c)
                if bn == "samuelson":
                    hit += int(bkey == tgt)
        m = {k: float(np.mean(v)) for k, v in cost.items()}
        mult = m["samuelson"] / m["oracle"] if m["oracle"] > 0 else float("nan")
        rows.append({"spread": sp, **m, "summary_x_floor": mult, "recall": hit / trials})
        log(f"{sp:>8.2f} {m['oracle']:>9.1f} {m['samuelson']:>10.1f} {m['ellipsoidal']:>10.1f} "
            f"{mult:>16.1f} {hit / trials:>7.3f}")
    return rows


def main():
    print("=" * 78)
    print("SAMUELSON ATTAINMENT — the tightness certificate")
    print("=" * 78)
    ok = True
    for m in (2, 4, 8, 16, 64, 256):
        c = check_attainment(m)
        ok &= c["attained"]
        print(f"  m={m:4d}  max dev {c['max_dev']:.6f}  bound {c['bound']:.6f}  "
              f"attained={c['attained']}  (mean err {c['mean_err']:.2e}, sd err {c['sd_err']:.2e})")
    print(f"\n  => Samuelson is ATTAINED at every block size tested: {ok}")
    print("     So no bound below it is admissible on the whole fibre of (mu, Sigma, m),")
    print("     and no summary-only router prunes more without reading keys.")
    print()
    print("=" * 78)
    print("THE BOUND FLOOR — what the partition costs vs what the summary costs")
    print("=" * 78)
    run()
    print()
    print("=" * 78)
    print("DOES THE SUMMARY PRICE FALL TOWARD THE FLOOR AS GEOMETRY BECOMES BENIGN?")
    print("=" * 78)
    sweep()


if __name__ == "__main__":
    main()
