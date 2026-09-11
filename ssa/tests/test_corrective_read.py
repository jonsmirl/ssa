"""Selected-only information channel and bounded corrective-query checks."""
import math

import pytest
import torch

from ssa.corrective_read import (CorrectiveRouter, SelectedSummary, TokenBallTree,
                                 exact_union_read, summarize_selected)


def fixture():
    g = torch.Generator().manual_seed(73)
    q = torch.randn(3, 4, generator=g, dtype=torch.float64)
    K = torch.randn(12, 4, generator=g, dtype=torch.float64)
    V = torch.randn(12, 5, generator=g, dtype=torch.float64)
    ids = torch.tensor([[1, 3, 7], [2, 5, -1], [9, -1, -1]])
    return q, K, V, ids


def test_full_union_is_dense():
    q, K, V, _ = fixture()
    ids = torch.arange(len(K)).expand(len(q), -1)
    actual = exact_union_read(q, K, V, ids, .5)
    torch.testing.assert_close(actual, (.5 * q @ K.T).softmax(-1) @ V)


def test_selected_softmax_original_query():
    q, K, V, ids = fixture()
    summary = summarize_selected(q, K, V, ids, .5)
    for j in range(len(q)):
        selected = ids[j][ids[j] >= 0]
        scores = .5 * K[selected] @ q[j]
        weights = scores.softmax(-1)
        torch.testing.assert_close(summary.value_mean[j], weights @ V[selected])
        torch.testing.assert_close(summary.key_mean[j], weights @ K[selected])
        torch.testing.assert_close(summary.log_partition[j], scores.logsumexp(0))
        torch.testing.assert_close(summary.entropy[j], -(weights * weights.log()).sum())
    assert summary.count.tolist() == [3, 2, 1]


@pytest.mark.parametrize("field", ["keys", "values"])
def test_unselected_poison_not_consumed(field):
    q, K, V, ids = fixture()
    clean = summarize_selected(q, K, V, ids)
    unread = torch.ones(len(K), dtype=torch.bool)
    unread[ids[ids >= 0]] = False
    (K if field == "keys" else V)[unread] = torch.nan
    poisoned = summarize_selected(q, K, V, ids)
    for attr in ("key_mean", "value_mean", "entropy", "log_partition", "count"):
        torch.testing.assert_close(getattr(clean, attr), getattr(poisoned, attr))


@pytest.mark.parametrize("padding", [0, 4])
def test_empty_reads_ignore_every_archive_value(padding):
    q, K, V, _ = fixture()
    K[:] = torch.nan
    V[:] = torch.nan
    ids = torch.full((len(q), padding), -1)
    result = summarize_selected(q, K, V, ids)
    assert not result.value_mean.any()
    assert not result.key_mean.any()
    assert not result.entropy.any()
    assert torch.isneginf(result.log_partition).all()
    model = CorrectiveRouter(4, 5).double()
    rq, h = model(q, result)
    assert torch.isfinite(rq).all() and torch.isfinite(h).all()


@pytest.mark.parametrize("bad", [-2, 12, 99])
def test_invalid_indices_rejected(bad):
    q, K, V, ids = fixture()
    ids[0, 0] = bad
    with pytest.raises(ValueError, match="snapshot"):
        exact_union_read(q, K, V, ids)


def test_duplicate_union_rejected_but_padding_allowed():
    q, K, V, ids = fixture()
    ids[0, 1] = ids[0, 0]
    with pytest.raises(ValueError, match="distinct"):
        exact_union_read(q, K, V, ids)


@pytest.mark.parametrize("which", ["key", "value"])
def test_selected_nan_rejected(which):
    q, K, V, ids = fixture()
    (K if which == "key" else V)[1, 0] = torch.nan
    with pytest.raises(ValueError, match="selected keys and values"):
        summarize_selected(q, K, V, ids)


def test_equal_and_concentrated_logits():
    q = torch.tensor([[1.0]], dtype=torch.float64)
    K = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float64)
    V = torch.tensor([[2.0], [4.0], [9.0]], dtype=torch.float64)
    ids = torch.tensor([[0, 1, 2]])
    result = summarize_selected(q, K, V, ids)
    torch.testing.assert_close(result.value_mean, V.mean(0, keepdim=True))
    assert result.entropy.item() == pytest.approx(math.log(3))
    K[2] = 1000
    result = summarize_selected(q, K, V, ids)
    assert result.value_mean.item() == 9
    assert result.entropy.item() == 0


def test_proposal_state_and_movement_bounded():
    q, K, V, ids = fixture()
    model = CorrectiveRouter(4, 5, state_dim=7, max_delta=.3).double()
    summary = summarize_selected(q, K, V, ids)
    state = None
    for _ in range(20):
        rq, state = model.propose(q, summary, state=state)
        assert state.shape == (3, 7)
        assert (state.abs() <= 1).all()
        assert ((rq - q).norm(dim=-1) <= .3 * q.norm(dim=-1) + 1e-14).all()
    rq, _ = model.propose(torch.zeros_like(q), summary)
    assert not rq.any()


def test_feedback_disabled_ignores_observed_channels():
    q, K, V, ids = fixture()
    model = CorrectiveRouter(4, 5).double()
    summary = summarize_selected(q, K, V, ids)
    altered = SelectedSummary(summary.key_mean * 100, -summary.value_mean,
                              summary.entropy + 9, summary.log_partition + 33,
                              summary.count + 99)
    for x, y in zip(model(q, summary, False), model(q, altered, False)):
        torch.testing.assert_close(x, y)
    assert not torch.allclose(model(q, summary)[0], model(q, altered)[0])


def test_teacher_loss_external_and_gradients_exist():
    q, K, V, ids = fixture()
    model = CorrectiveRouter(4, 5).double()
    summary = summarize_selected(q, K, V, ids)
    rq, state = model(q, summary)
    # Full-archive teacher/score computation is explicitly outside controller.
    loss = torch.nn.functional.cross_entropy(rq @ K.T, torch.tensor([4, 8, 10]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.query_head.weight.grad.norm() > 0


def test_read_gradients_do_not_touch_unselected_values():
    q, K, V, ids = fixture()
    V.requires_grad_()
    exact_union_read(q, K, V, ids).square().sum().backward()
    unread = torch.ones(len(K), dtype=torch.bool)
    unread[ids[ids >= 0]] = False
    assert not V.grad[unread].any()


def test_tree_disjoint_retry_and_deterministic_ties():
    K = torch.zeros(16, 4, dtype=torch.float64)
    V = torch.arange(80, dtype=torch.float64).reshape(16, 5)
    q = torch.ones(2, 4, dtype=torch.float64)
    tree = TokenBallTree(K)
    first, _ = tree.route(q, 4, beam=16)
    again, _ = tree.route(q, 4, beam=16)
    assert torch.equal(first, again)
    retry, _ = tree.route(q, 4, beam=16, excluded=first)
    assert not (first[:, :, None] == retry[:, None, :]).any()
    union = torch.cat((first, retry), dim=1)
    output = exact_union_read(q, K, V, union)
    torch.testing.assert_close(output, V[union].mean(dim=1))


def test_future_prefix_not_visible_to_controller_or_tree():
    q, K, V, _ = fixture()
    prefix = 8
    def run(keys, values):
        # Slice before tree construction; no future geometry enters partition.
        tree = TokenBallTree(keys[:prefix])
        ids, _ = tree.route(q, 3, beam=8)
        return ids, summarize_selected(q, keys[:prefix], values[:prefix], ids)
    original, summary = run(K, V)
    K[prefix:] = 1e100
    V[prefix:] = torch.nan
    changed, altered = run(K, V)
    assert torch.equal(original, changed)
    torch.testing.assert_close(summary.value_mean, altered.value_mean)
    model = CorrectiveRouter(4, 5).double()
    torch.testing.assert_close(model(q, summary)[0], model(q, altered)[0])


@pytest.mark.parametrize("beta", [-1.0, float("inf"), float("nan")])
def test_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        summarize_selected(*fixture(), beta=beta)


def test_bad_state_rejected():
    q, K, V, ids = fixture()
    model = CorrectiveRouter(4, 5, state_dim=7).double()
    with pytest.raises(ValueError, match="state"):
        model(q, summarize_selected(q, K, V, ids), state=q.new_full((3, 7), 2))
