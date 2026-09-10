import pytest
import torch

from ssa.trainable_repair import RecurrentRepairAttention, TokenBallTree, PositiveTailState, ClusterTailState


def fixture(n=23):
    torch.manual_seed(11)
    keys = torch.randn(n, 5, dtype=torch.float64)
    q = torch.randn(3, 5, dtype=torch.float64)
    values = torch.randn(3, n, 7, dtype=torch.float64)
    return keys, q, values


def test_tree_full_beam_matches_exact_order_without_ties():
    k, q, _ = fixture()
    tree = TokenBallTree(k)
    ids, _ = tree.route(q, 5, beam=32)
    torch.testing.assert_close(ids, (q @ k.T).argsort(descending=True)[:, :5])
    more, _ = tree.route(q, 4, beam=32, excluded=ids)
    torch.testing.assert_close(more, (q @ k.T).argsort(descending=True)[:, 5:9])


def test_tree_partition_and_routing_ignore_future_keys():
    k, q, _ = fixture()
    a = TokenBallTree(k, prefix=13)
    k[13:] = 1e10
    b = TokenBallTree(k, prefix=13)
    torch.testing.assert_close(a.centers, b.centers)
    torch.testing.assert_close(a.route(q, 4)[0], b.route(q, 4)[0])
    assert a.route(q, 4)[0].max() < 13


def test_tree_ties_are_deterministic_and_excluded_ids_never_repeat():
    k = torch.zeros(13, 5)
    tree = TokenBallTree(k)
    a, _ = tree.route(torch.ones(2, 5), 6, beam=32)
    b, _ = tree.route(torch.ones(2, 5), 6, beam=32)
    assert torch.equal(a, b)
    c, _ = tree.route(torch.ones(2, 5), 6, beam=32, excluded=a)
    assert not (a[..., None] == c[:, None]).any()


def test_streamed_torch_output_and_archive_gradients_equal_union_oracle():
    k, q, v = fixture()
    K = k[None].repeat(3, 1, 1).requires_grad_()
    V = v.clone().requires_grad_()
    model = RecurrentRepairAttention(5, 7).double()
    r = model(q, K, V, TokenBallTree(k), q, budget=4, rounds=3, beta=1.2)
    ids = r.indices
    selected_k = K.gather(1, ids[..., None].expand(-1, -1, 5))
    selected_v = V.gather(1, ids[..., None].expand(-1, -1, 7))
    weights = (1.2 * (selected_k * q[:, None]).sum(-1)).softmax(-1)
    oracle = (weights[..., None] * selected_v).sum(1)
    torch.testing.assert_close(r.output, oracle)
    actual_grad = torch.autograd.grad(r.output.square().sum(), (K, V), retain_graph=True)
    expected_grad = torch.autograd.grad(oracle.square().sum(), (K, V))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)
    assert all(len(torch.unique(row)) == 12 for row in ids)


@pytest.mark.parametrize("n", [1, 3, 13])
def test_dense_fallback_and_exhausted_rounds_have_finite_gradients(n):
    k, q, v = fixture(n)
    K = k[None].repeat(3, 1, 1).requires_grad_()
    V = v.clone().requires_grad_()
    r = RecurrentRepairAttention(5, 7).double()(q, K, V, TokenBallTree(k), q,
                                                budget=16, beam=32, rounds=3)
    oracle = torch.einsum(
        "bn,bnv->bv", torch.softmax(q @ k.T, -1), V)
    torch.testing.assert_close(r.output, oracle)
    for grad in torch.autograd.grad(r.output.sum(), (K, V)):
        assert torch.isfinite(grad).all()
    assert (r.indices >= 0).sum().item() == 3 * n


def test_state_can_cross_query_boundary_but_attention_accumulator_resets():
    k, q, v = fixture()
    model = RecurrentRepairAttention(5, 7).double()
    tree = TokenBallTree(k)
    first = model(q, k[None].expand(3, -1, -1), v, tree, q, rounds=1)
    next_read = model(-q, k[None].expand(3, -1, -1), v, tree, -q, rounds=1, state=first.state)
    ids = next_read.indices
    score = (-q[:, None] * k[ids]).sum(-1)
    expected = torch.einsum("bn,bnv->bv", score.softmax(-1), v.gather(1, ids[..., None].expand(-1, -1, 7)))
    torch.testing.assert_close(next_read.output, expected)
    assert not torch.equal(next_read.state, first.state)


def test_positive_tail_is_exact_when_kernel_has_supplied_feature_factorization():
    torch.manual_seed(22)
    features = torch.rand(17, 4, dtype=torch.float64)
    values = torch.randn(17, 6, dtype=torch.float64)
    qf = torch.rand(4, dtype=torch.float64)
    kernel = features @ qf
    ids = torch.tensor([1, 3, 11])
    state = PositiveTailState(features, values)
    output, _ = state.read(qf, ids, kernel[ids].log())
    torch.testing.assert_close(output, (kernel / kernel.sum()) @ values)
    all_ids = torch.arange(17)
    dense, tail = state.read(qf, all_ids, kernel.log())
    torch.testing.assert_close(dense, (kernel / kernel.sum()) @ values)
    assert abs(tail) < 1e-15


def test_incremental_tail_state_matches_explicit_approximate_kernel_and_exact_replacement():
    k, q, v = fixture()
    centers = k[:4].clone()
    state = ClusterTailState(centers, 7)
    for start in range(0, 17, 3):
        end = min(start + 3, 17)
        state.append(k[start:end], v[0, start:end])
    assert state.tokens == 17
    assert state.counts.numel() + state.sums.numel() == 4 * 8
    ids = torch.tensor([1, 7, 12])
    beta = 0.7
    logits = beta * (k[ids] @ q[0])
    out = state.read(q[0], k[ids], v[0, ids], logits, beta=beta, log_gain=1.5)
    assigned = torch.cdist(k[:17], centers).argmin(-1)
    approximate = beta * (centers[assigned] @ q[0]) + 1.5
    approximate[ids] = logits
    torch.testing.assert_close(out, approximate.softmax(0) @ v[0, :17])
    # Fully opening the prefix removes the approximate state contribution.
    exact = state.read(q[0], k[:17], v[0, :17], beta * (k[:17] @ q[0]), beta=beta)
    torch.testing.assert_close(exact, (beta * (k[:17] @ q[0])).softmax(0) @ v[0, :17])


def test_tail_kernel_error_identity_and_output_bound():
    torch.manual_seed(31)
    weights = torch.rand(31, dtype=torch.float64) + 0.1
    approximate = torch.rand(31, dtype=torch.float64) + 0.1
    approximate[:5] = weights[:5]  # exact selected keys
    values = torch.randn(31, 7, dtype=torch.float64)
    true = (weights / weights.sum()) @ values
    out = (approximate / approximate.sum()) @ values
    residual = ((weights - approximate)[:, None] * (values - out)).sum(0) / weights.sum()
    torch.testing.assert_close(true - out, residual)
    bound = (weights - approximate).abs().sum() * (values - out).norm(dim=-1).max() / weights.sum()
    assert (true - out).norm() <= bound
