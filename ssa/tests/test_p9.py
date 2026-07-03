"""Tests for the P9 trained-comparison harness (ssa.p9_microlm, ssa.p9_tasks): the four token-mixers, the
DeltaNet scan (== the fastweight delta reference), the learned gate, the JEPA aux loss, and a training smoke.
Mostly CPU; the training smoke is GPU-gated."""
import torch
import pytest

from ssa.p9_microlm import (P9Model, deltanet_mix, ssa_mix, dense_mix, linear_mix,
                            FuturePredictor, jepa_loss, train_model, recall)

cuda = torch.cuda.is_available()


def test_deltanet_scan_matches_fastweight_reference():
    """The differentiable causal scan reproduces fastweight's delta rule exactly on orthogonal keys."""
    from ssa.fastweight import FastWeightMemory, _unit
    torch.manual_seed(0)
    dh, m = 16, 8
    K = torch.eye(dh)[:m]; V = _unit(torch.randn(m, dh))
    o = deltanet_mix(K.view(1, 1, m, dh), K.view(1, 1, m, dh), V.view(1, 1, m, dh), gate=None, beta=1.0)[0, 0]
    mem = FastWeightMemory(dh, rule="delta", beta=1.0, keep_kv=False)
    ref = torch.stack([(mem.write(K[t], V[t]) or mem.read_linear(K[t])) for t in range(m)])
    assert torch.allclose(o, ref, atol=1e-4), (o - ref).abs().max().item()


def test_ssa_mixer_full_budget_equals_dense():
    """ssa_mix with a budget covering every causal block == dense causal attention."""
    torch.manual_seed(0)
    B, H, n, dh = 2, 2, 48, 16
    q = torch.randn(B, H, n, dh); k = torch.randn_like(q); v = torch.randn_like(q)
    nb = (n + 7) // 8
    full = ssa_mix(q, k, v, block=8, top_c=nb, local=nb)
    ref = dense_mix(q, k, v)
    assert torch.allclose(full, ref, atol=1e-4), (full - ref).abs().max().item()


def test_all_mixers_forward_shape():
    torch.manual_seed(0)
    vocab = 66
    ids = torch.randint(0, vocab, (3, 40))
    for mix in ("dense", "ssa", "deltanet", "linear"):
        m = P9Model(vocab, d=32, n_layer=2, n_head=4, max_len=64, mixer=mix)
        logits, h = m(ids, return_hidden=True)
        assert logits.shape == (3, 40, vocab) and h.shape == (3, 40, 32), (mix, logits.shape)


def test_gate_in_unit_interval_and_open_init():
    """The learned write gate is in (0,1) and initialized OPEN (β≈1) so it defaults to write-everything."""
    torch.manual_seed(0)
    m = P9Model(66, d=32, n_layer=1, n_head=4, mixer="deltanet", delta_gate=True)
    k = torch.randn(2, 4, 10, 8)
    beta = torch.sigmoid(m.gates[0](k))
    assert (beta > 0).all() and (beta < 1).all()
    assert beta.mean() > 0.9, beta.mean().item()                     # open at init


def test_linear_mixer_is_causal():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 12, 8); k = torch.randn_like(q); v = torch.randn_like(q)
    o1 = linear_mix(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 7:] += 3 * torch.randn_like(k2[:, :, 7:]); v2[:, :, 7:] += 3 * torch.randn_like(v2[:, :, 7:])
    o2 = linear_mix(q, k2, v2)
    assert torch.allclose(o1[:, :, :7], o2[:, :, :7], atol=1e-5)     # future keys/values don't affect the past


def test_jepa_loss_decreases_with_training():
    torch.manual_seed(0)
    h = torch.randn(4, 20, 32)
    pred = FuturePredictor(32)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    first = jepa_loss(pred, h).item()
    for _ in range(50):
        opt.zero_grad(); loss = jepa_loss(pred, h); loss.backward(); opt.step()
    assert loss.item() < first - 0.05, (first, loss.item())


@pytest.mark.skipif(not cuda, reason="training smoke is GPU-paced")
def test_training_smoke_learns_above_chance():
    """Each mixer, on a tiny MQAR curriculum, beats chance — the harness trains end to end."""
    from ssa.ssa_demo import MQAR
    task = MQAR(n_keys=32, n_vals=32)
    for mix in ("dense", "deltanet"):
        torch.manual_seed(0)
        m = P9Model(task.vocab, d=64, n_layer=2, n_head=4, max_len=128, mixer=mix).to("cuda")
        for npr, st in [(2, 400), (4, 500)]:
            train_model(m, task, steps=st, n_pairs=npr, bs=48, warmup=40)
        assert recall(m, task, 4, trials=4) > 0.15, mix              # well above chance (1/32 ≈ 0.03)
