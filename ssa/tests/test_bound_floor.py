"""Tests for `ssa.bound_floor` — the summary-only floor and its tightness certificate.

Every section pairs a positive check with a NEGATIVE one that must fail if the mechanism it guards
is broken, because the dangerous failure here is not a crash but a bound that cannot be violated:
an "admissible" bound that is secretly infinite prunes nothing and passes every admissibility test
vacuously, and a "floor" computed from the same keys the bound reads is not a floor at all.
"""
import numpy as np
import pytest

from ssa.bound_floor import (BOUNDS, bnb_cost, bound_ellipsoidal, bound_isotropic, bound_oracle,
                             bound_samuelson, check_attainment, kmeans_blocks, samuelson_witness)
from ssa.core import clustered_keys, random_unit_keys


def _keys(n=512, d=32, B=16, spread=0.15, seed=0):
    K = np.asarray(clustered_keys(n, d, B, spread=spread, seed=seed)[0], dtype=np.float64)
    return K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------- attainment
@pytest.mark.parametrize("m", [2, 4, 8, 16, 64, 256])
def test_samuelson_is_attained(m):
    """The tightness claim rests entirely on this configuration existing."""
    c = check_attainment(m)
    assert c["mean_err"] < 1e-9, "the witness must have the stated mean"
    assert c["sd_err"] < 1e-9, "the witness must have the stated sd"
    assert c["attained"], "the witness's max must sit exactly on the bound"


@pytest.mark.parametrize("m", [4, 16, 64])
def test_no_bound_below_samuelson_is_admissible(m):
    """NEGATIVE side of the tightness claim: any bound strictly below Samuelson is violated by the
    witness, which has exactly the summary the router read. This is what makes the floor a floor
    rather than a coincidence of the estimator."""
    mean, sd = 0.3, 1.7
    x = samuelson_witness(m, mean, sd)
    bound = mean + sd * np.sqrt(m - 1)
    assert x.max() <= bound + 1e-9, "the witness must not exceed the bound"
    for eps in (1e-6, 1e-3, 0.1):
        assert x.max() > bound - eps * abs(bound) - 1e-12 or eps == 0, \
            "a strictly smaller bound must be violated by the witness"
    assert x.max() > bound * (1 - 1e-9) - 1e-12


def test_witness_has_a_single_high_point():
    """The extremal configuration is one outlier against a flat floor — the shape the paper's
    'centroid routing is blind to outliers' argument turns on."""
    x = samuelson_witness(16, 0.0, 1.0)
    assert (x > x.min() + 1e-9).sum() == 1


# ---------------------------------------------------------------------------- admissibility
@pytest.mark.parametrize("bname", ["oracle", "samuelson", "isotropic", "ellipsoidal"])
def test_bound_is_admissible(bname):
    """No key in a block may exceed its bound. An inadmissible bound makes B&B lossy and every
    cost number meaningless."""
    K = _keys()
    members = kmeans_blocks(K, 16, seed=0)
    rng = np.random.default_rng(3)
    for _ in range(12):
        q = rng.standard_normal(K.shape[1])
        q /= np.linalg.norm(q)
        for mem in members:
            ub = BOUNDS[bname](K, mem, q)
            assert (K[mem] @ q).max() <= ub + 1e-8, f"{bname} is not admissible"


def test_centroid_alone_is_NOT_admissible():
    """NEGATIVE: the centroid score is a heuristic, not a bound — it must be violated somewhere, or
    the admissibility tests above are not testing anything."""
    K = _keys()
    members = kmeans_blocks(K, 16, seed=0)
    rng = np.random.default_rng(5)
    violated = False
    for _ in range(20):
        q = rng.standard_normal(K.shape[1])
        q /= np.linalg.norm(q)
        for mem in members:
            if (K[mem] @ q).max() > float(K[mem].mean(0) @ q) + 1e-8:
                violated = True
    assert violated, "a centroid score that is never exceeded would make the bounds vacuous"


# ---------------------------------------------------------------------------- the ordering
def test_oracle_is_the_floor():
    """Every bound costs at least the oracle, because the oracle is the tightest admissible one."""
    K = _keys()
    members = kmeans_blocks(K, 16, seed=0)
    rng = np.random.default_rng(7)
    for _ in range(15):
        q = K[int(rng.integers(len(K)))] + 0.15 * rng.standard_normal(K.shape[1])
        q /= np.linalg.norm(q)
        _, c_or = bnb_cost(K, q, members, bound_oracle)
        for bfn in (bound_samuelson, bound_isotropic, bound_ellipsoidal):
            _, c = bnb_cost(K, q, members, bfn)
            assert c >= c_or, "a bound cannot cost less than the oracle"


def test_every_bound_is_lossless():
    """All four are admissible, so B&B returns the true argmax under each. Recall is not the
    discriminator here — cost is."""
    K = _keys()
    members = kmeans_blocks(K, 16, seed=0)
    rng = np.random.default_rng(11)
    for _ in range(15):
        q = K[int(rng.integers(len(K)))] + 0.15 * rng.standard_normal(K.shape[1])
        q /= np.linalg.norm(q)
        tgt = int((K @ q).argmax())
        for bname in BOUNDS:
            bkey, _ = bnb_cost(K, q, members, BOUNDS[bname])
            assert bkey == tgt, f"{bname} lost the argmax — it is not admissible"


def test_benign_geometry_costs_less_than_isotropic():
    """The direction the whole routability program depends on. Reported as a comparison of MEANS
    over queries; a single query can invert it."""
    d, B, trials = 32, 16, 40
    rng = np.random.default_rng(13)
    costs = {}
    for name, K in (("benign", _keys(512, d, B, spread=0.03, seed=1)),
                    ("isotropic", np.asarray(random_unit_keys(512, d, seed=1), dtype=np.float64))):
        members = kmeans_blocks(K, B, seed=1)
        c = []
        for _ in range(trials):
            q = K[int(rng.integers(len(K)))] + 0.15 * rng.standard_normal(d)
            q /= np.linalg.norm(q)
            c.append(bnb_cost(K, q, members, bound_samuelson)[1])
        costs[name] = float(np.mean(c))
    assert costs["benign"] < costs["isotropic"], costs


def test_the_summary_price_is_strictly_above_the_floor():
    """THE RESULT. Summary-only routing costs a multiple of the partition floor, and the multiple is
    strictly greater than one even at benign geometry — Samuelson's attainment is why."""
    K = _keys(512, 32, 16, spread=0.03, seed=2)
    members = kmeans_blocks(K, 16, seed=2)
    rng = np.random.default_rng(17)
    orc, sam = [], []
    for _ in range(40):
        q = K[int(rng.integers(len(K)))] + 0.15 * rng.standard_normal(32)
        q /= np.linalg.norm(q)
        orc.append(bnb_cost(K, q, members, bound_oracle)[1])
        sam.append(bnb_cost(K, q, members, bound_samuelson)[1])
    assert np.mean(sam) > np.mean(orc), "the summary price must exceed the partition floor"
