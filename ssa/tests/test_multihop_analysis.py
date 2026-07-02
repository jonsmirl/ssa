"""CPU tests for the synthetic multi-hop rig (ssa.multihop_analysis) — determinism, the composition law,
and the falsifiable single-needle-holds / chain-collapses prediction."""
from ssa.multihop_analysis import chain_stats


def test_trial_deterministic_same_seed():
    a = chain_stats(4096, 64, 0.55, "benign", 40, hops=2, seed=7)
    b = chain_stats(4096, 64, 0.55, "benign", 40, hops=2, seed=7)
    assert a == b


def test_dense_two_hop_near_perfect_at_high_margin():
    r = chain_stats(4096, 64, 0.9, "dense", 60, hops=2, seed=0)
    assert r["acc"] >= 0.95, r


def test_composition_law_sanity():
    """Measured 2-hop chain ≈ ∏ρ, and never exceeds its weakest hop."""
    r = chain_stats(8192, 64, 0.55, "benign", 400, hops=2, seed=1)
    assert abs(r["acc"] - r["pred"]) <= 0.10, r
    assert r["acc"] <= min(r["rhos"]) + 0.05, r


def test_isolated_chain_collapses_benign_holds():
    benign = chain_stats(32768, 64, 0.55, "benign", 60, hops=2, seed=2)
    iso = chain_stats(32768, 64, 0.55, "isolated", 60, hops=2, seed=2)
    assert benign["acc"] >= iso["acc"] + 0.3, (benign, iso)


def test_mixed_tracks_weak_hop():
    """Benign hop1 + isolated hop2: the chain tracks ρ1·ρ2 (dominated by the weak isolated hop)."""
    r = chain_stats(32768, 64, 0.55, "mixed", 80, hops=2, seed=3)
    assert abs(r["acc"] - r["pred"]) <= 0.12, r
    assert r["acc"] <= 0.5, r                                      # the isolated hop caps the chain low
