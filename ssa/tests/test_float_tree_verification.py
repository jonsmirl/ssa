"""CUDA regression for the production tree's outward-rounded balls and caps."""
import importlib.util

import pytest
import torch


cuda = torch.cuda.is_available()
has_faiss = importlib.util.find_spec("faiss") is not None
skip = pytest.mark.skipif(not (cuda and has_faiss), reason="tree verifier needs CUDA + faiss-gpu")


@skip
def test_guarded_tree_contains_float64_descendants_and_scores():
    from ssa.float_tree_verification import _cases, verify_case

    raw_misses = 0
    with torch.no_grad():
        for name, leaves in _cases(256, 64, "cuda", seed=71).items():
            row = verify_case(leaves, fanout=4, queries=4, seed=83)
            assert row["guarded_radius_violations"] == 0, name
            assert row["guarded_cap_violations"] == 0, name
            raw_misses += row["raw_radius_violations"] + row["raw_cap_violations"]

    # The fixture must remain sensitive to the unguarded regression it is designed to catch.
    assert raw_misses > 0
