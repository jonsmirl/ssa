"""Tests for the trained routing space (ssa.routing_space): identity sanity, loss behavior, PCA,
save/load, the route_recall routing-space hook, and trained-beats-random on clustered synthetic geometry.
Runs on whatever device is available (uses small tensors)."""
import numpy as np
import torch

from ssa.routing_space import (block_scores, block_means_qk, listwise_kl, jaccard_topk, pca_init,
                               random_proj, train_projection, eval_projection, synth_sample,
                               RoutingProjection, DEV)


def test_identity_projection_equals_full():
    qb, ks, nb, spb = synth_sample("clustered_tight", n=2048, seed=0)
    I = torch.eye(64, device=DEV)
    sf = block_scores(qb, ks, nb, spb)
    sr = block_scores(qb, ks, nb, spb, I, I)
    assert torch.allclose(sf, sr, atol=1e-4)
    assert listwise_kl(sf, sr).item() < 1e-5
    assert jaccard_topk(sf, sr, 8)[0] == 1.0


def test_kl_decreases_with_training():
    samples = [synth_sample("clustered_tight", n=2048, seed=s) for s in range(2)]
    before = eval_projection((random_proj(64, 16, 0), random_proj(64, 16, 1)), samples)[0]
    proj = train_projection(samples, 64, 16, steps=300, seed=0)
    after = eval_projection(proj, samples)[0]
    assert after > before + 0.05, (before, after)


def test_pca_columns_orthonormal():
    K = torch.randn(4000, 64, device=DEV)
    W = pca_init(K, 16)
    assert W.shape == (64, 16)
    assert torch.allclose(W.T @ W, torch.eye(16, device=DEV), atol=1e-3)


def test_save_load_roundtrip(tmp_path):
    proj = RoutingProjection(random_proj(64, 16, 0), random_proj(64, 16, 1), {"d_r": 16, "note": "x"})
    p = str(tmp_path / "rp.pt")
    proj.save(p)
    q = RoutingProjection.load(p, device=DEV)
    assert torch.allclose(proj.W_q, q.W_q) and torch.allclose(proj.W_k, q.W_k)
    assert q.meta["note"] == "x"
    t = torch.randn(5, 64, device=DEV)
    assert torch.allclose(proj.project_q(t), q.project_q(t))


def test_route_recall_routing_hook_identity():
    """K_route=K / Q_route=Q must reproduce the plain call (the routing hook is a no-op at identity)."""
    from ssa.real_keys import route_recall
    rng = np.random.default_rng(0)
    K = rng.standard_normal((1500, 32)).astype(np.float32)
    Q = rng.standard_normal((1500, 32)).astype(np.float32)
    a = route_recall(K, Q, 16, budget_abs=200, order="relevance", seed=1)
    b = route_recall(K, Q, 16, budget_abs=200, order="relevance", seed=1, K_route=K, Q_route=Q)
    assert a == b


def test_trained_beats_random_on_clustered():
    samples = [synth_sample("clustered_tight", n=4096, seed=s) for s in range(3)]
    rnd = eval_projection((random_proj(64, 16, 3), random_proj(64, 16, 4)), samples)[0]
    trained = eval_projection(train_projection(samples, 64, 16, steps=400, seed=0), samples)[0]
    assert trained >= rnd + 0.1, (rnd, trained)
