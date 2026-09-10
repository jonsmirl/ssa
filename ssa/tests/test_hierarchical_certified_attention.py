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


def test_bennett_node_mass_caps_are_sound_and_no_worse_than_radius():
    rng = np.random.default_rng(17)
    geometries = [
        rng.normal(size=(256, 9)),
        1e-4 * rng.normal(size=(256, 9)),
        np.repeat(rng.normal(size=(8, 9)), 32, axis=0),
    ]
    for K in geometries:
        radius = CertifiedTreeAttention(K, np.zeros((len(K), 1)), 8, mass_bound="radius")
        bennett = CertifiedTreeAttention(K, np.zeros((len(K), 1)), 8, mass_bound="bennett")
        covariance = CertifiedTreeAttention(
            K, np.zeros((len(K), 1)), 8, mass_bound="bennett_covariance")
        peeled = CertifiedTreeAttention(
            K, np.zeros((len(K), 1)), 8, mass_bound="bennett_peel", peel_count=2)
        peeled_covariance = CertifiedTreeAttention(
            K, np.zeros((len(K), 1)), 8,
            mass_bound="bennett_peel_covariance", peel_count=2)
        for beta in (0.0, 0.125, 2.0, 30.0):
            for q in (np.zeros(9), rng.normal(size=9)):
                qnorm = float(np.linalg.norm(q))
                for rn, bn, cn, pn, pcn in zip(
                        radius._nodes, bennett._nodes, covariance._nodes,
                        peeled._nodes, peeled_covariance._nodes):
                    ids = slice(bn.start * 8, bn.end * 8)
                    logits = beta * (K[ids] @ q)
                    top = float(logits.max())
                    exact = top + float(np.log(np.exp(logits - top).sum()))
                    radius_cap = radius._node_log_upper(rn, q, beta, qnorm)
                    bennett_cap = bennett._node_log_upper(bn, q, beta, qnorm)
                    covariance_cap = covariance._node_log_upper(cn, q, beta, qnorm)
                    peeled_cap = peeled._node_log_upper(pn, q, beta, qnorm)
                    peeled_covariance_cap = peeled_covariance._node_log_upper(
                        pcn, q, beta, qnorm)
                    assert exact <= bennett_cap + 2e-11
                    assert exact <= covariance_cap + 2e-11
                    assert exact <= peeled_cap + 2e-11
                    assert exact <= peeled_covariance_cap + 2e-11
                    assert bennett_cap <= radius_cap + 2e-11
                    assert covariance_cap <= radius_cap + 2e-11
                    assert peeled_cap <= radius_cap + 2e-11
                    assert peeled_covariance_cap <= radius_cap + 2e-11


def test_bennett_tree_certificate_matches_dense_and_can_reduce_mass_bound():
    rng = np.random.default_rng(29)
    K = 0.02 * rng.normal(size=(1024, 12))
    K[:16, 0] += 4
    V = rng.normal(size=(1024, 3))
    q = np.eye(12)[0]
    radius = CertifiedTreeAttention(K, V, 16, mass_bound="radius").read(
        q, beta=4, mass_tol=0, max_blocks=1)
    index = CertifiedTreeAttention(K, V, 16, mass_bound="bennett")
    bennett = index.read(q, beta=4, mass_tol=0, max_blocks=1)
    check_against_dense(index, q, 4, bennett)
    assert bennett.mass_upper < radius.mass_upper


def test_bennett_rejects_unknown_bound_mode():
    with np.testing.assert_raises(ValueError):
        CertifiedTreeAttention(np.zeros((8, 2)), np.zeros((8, 1)), 2,
                               mass_bound="unproved")
    with np.testing.assert_raises(ValueError):
        CertifiedTreeAttention(np.zeros((8, 2)), np.zeros((8, 1)), 2,
                               mass_bound="bennett_peel", peel_count=0)


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
