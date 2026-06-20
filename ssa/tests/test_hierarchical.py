"""Assertions for hierarchical routing — incl. a numerical check of the theory (see paper)'s
recursive radius bound (subtree_radius_bound)."""
import numpy as np
from ssa.core import clustered_keys
from ssa.hierarchical_routing import (
    build_tree, lossless_cost, flat_approx, hier_approx)


def test_recursive_radius_bounds_the_subtree():
    """The recursive parent radius bounds every key under it — subtree_radius_bound, numerically."""
    K, _ = clustered_keys(800, 32, B=20, spread=0.12, seed=0)
    tree = build_tree(K, 8, 8)
    for muc, Rc, fines in tree:
        for _, _, fi in fines:
            assert np.linalg.norm(K[fi] - muc, axis=1).max() <= Rc + 1e-4


def test_lossless_returns_true_argmax():
    """Exact best-first B&B (the admissible bound is exact) returns the dense argmax."""
    rng = np.random.default_rng(0)
    K, _ = clustered_keys(600, 32, B=15, spread=0.12, seed=0)
    tree = build_tree(K, 8, 8)
    for _ in range(30):
        i = int(rng.integers(600)); q = K[i] + 0.2 * rng.standard_normal(32).astype(np.float32)
        bkey, _ = lossless_cost(K, q, tree)
        assert bkey == int((K @ q).argmax())


def test_hierarchical_scores_fewer_nodes():
    """Approximate hierarchical routing scores fewer nodes than flat (the subquadratic-selection win)."""
    K, _ = clustered_keys(4000, 32, B=40, spread=0.10, seed=0)
    tree = build_tree(K, 16, 16)
    rng = np.random.default_rng(1)
    fcost = hcost = 0
    for _ in range(20):
        i = int(rng.integers(4000)); q = K[i] + 0.25 * rng.standard_normal(32).astype(np.float32)
        _, fn = flat_approx(K, q, tree, 8)
        _, hn = hier_approx(K, q, tree, 3, 8)
        fcost += fn; hcost += hn
    assert hcost < fcost
