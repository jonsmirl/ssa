"""Dense-oracle checks for hierarchical adaptive attention certificates."""
import numpy as np

from ssa.certified_attention import CertifiedBlockAttention
from ssa.hierarchical_certified_attention import CertifiedTreeAttention
from ssa.tests.test_certified_attention import check_against_dense


def test_tree_certificate_matches_dense_over_prefixes_caps_and_seeds():
    rng = np.random.default_rng(41)
    K, V = rng.normal(size=(73, 7)), rng.normal(size=(73, 4))
    index = CertifiedTreeAttention(K, V, 8)
    q = rng.normal(size=7)
    for prefix in (1, 39, 64, 73):
        nb = (prefix + 7) // 8
        for cap in (1, min(3, nb), nb):
            mandatory_partial = bool(prefix % 8)
            seeds = [0] if prefix > 8 and cap > int(mandatory_partial) else []
            result = index.read(q, beta=2.3, prefix=prefix, max_blocks=cap,
                                initial_blocks=seeds, mass_tol=1e-5, error_tol=1e-4)
            check_against_dense(index, q, 2.3, result, prefix)
            assert result.blocks_opened <= cap
            assert result.certified == (result.mass_upper <= 1e-5
                                        and result.output_error_upper <= 1e-4)


def test_tree_uses_fewer_summary_bounds_on_concentrated_geometry():
    rng = np.random.default_rng(0)
    K = 0.01 * rng.standard_normal((4096, 16))
    K[:64, 0] += 5
    V = rng.standard_normal((4096, 3))
    q = np.eye(16)[0]
    flat = CertifiedBlockAttention(K, V, 64).read(q, beta=4, mass_tol=1e-5)
    tree_index = CertifiedTreeAttention(K, V, 64)
    tree = tree_index.read(q, beta=4, mass_tol=1e-5)
    check_against_dense(tree_index, q, 4, tree)
    assert tree.certified and tree.keys_scored == flat.keys_scored == 64
    assert tree.bounds_evaluated <= 2 * int(np.ceil(np.log2(64))) + 1
    assert tree.bounds_evaluated < flat.bounds_evaluated // 2


def test_tree_preserves_output_only_equal_value_shortcut():
    index = CertifiedTreeAttention(np.zeros((128, 3)), np.ones((128, 2)), 8)
    result = index.read([0, 0, 0], error_tol=0)
    assert result.certified and result.keys_scored == 8
    assert result.output_error_upper == 0
    assert result.mass_upper > 0.9


def test_tree_prefix_cannot_observe_future_nodes():
    rng = np.random.default_rng(9)
    K, V = rng.normal(size=(96, 5)), rng.normal(size=(96, 2))
    q, prefix = rng.normal(size=5), 53
    a = CertifiedTreeAttention(K, V, 8).read(q, prefix=prefix, max_blocks=3)
    K[prefix:], V[prefix:] = 1e8, -1e8
    b = CertifiedTreeAttention(K, V, 8).read(q, prefix=prefix, max_blocks=3)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_allclose(a.output, b.output)
    assert a.mass_upper == b.mass_upper
    assert a.output_error_upper == b.output_error_upper


def test_descendant_caps_tighten_and_drop_at_least_the_parent_cap():
    """At one threshold, every region pruned by a parent cap is pruned by each child cap."""
    rng = np.random.default_rng(23)
    K, V = rng.normal(size=(128, 6)), rng.normal(size=(128, 3))
    index = CertifiedTreeAttention(K, V, 8)
    p, q = rng.normal(size=6), rng.normal(size=6)
    qnorm = np.linalg.norm(q)

    for parent in index._nodes:
        if parent.children is None:
            continue
        parent_reach = np.linalg.norm(parent.key_mean - p) + parent.key_radius
        parent_cap = q @ p + qnorm * parent_reach
        threshold = np.nextafter(parent_cap, np.inf)
        for child_id in parent.children:
            child = index._nodes[child_id]
            child_reach = np.linalg.norm(child.key_mean - p) + child.key_radius
            child_cap = q @ p + qnorm * child_reach
            assert child_reach <= parent_reach + 1e-12
            assert child_cap <= parent_cap + 1e-12
            assert child_cap <= threshold
