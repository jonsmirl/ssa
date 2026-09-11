import pytest
import torch
from ssa.tail_correction import jensen_cell_logmass, mix_tail_read
from ssa.hybrid_tail_attention import hybrid_tail_attention


def test_true_omitted_mean_is_a_lower_mass_bound_unlike_fixed_prototype():
    torch.manual_seed(53)
    for n in (1, 17, 257):
        k = torch.randn(n, 5, dtype=torch.float64)
        q = torch.randn(5, dtype=torch.float64)
        assigned = torch.arange(n) % 4
        counts = torch.bincount(assigned, minlength=4).double()
        sums = k.new_zeros(4, 5).index_add(0, assigned, k)
        actual = torch.stack([(k[assigned == c] @ q / 5**0.5).exp().sum() for c in range(4)])
        for gain in (-2., 0., 3.):
            bound = jensen_cell_logmass(q, counts, sums, gain).exp()
            assert (bound <= actual + 1e-12).all()
    # All actual keys are zero; an unrelated prototype overestimates arbitrarily.
    assert 100 * torch.exp(torch.tensor(10.)) > 100


@pytest.mark.parametrize("rho", [0., 0.1, 0.25, 1.])
def test_influence_bound_and_full_refinement(rho):
    torch.manual_seed(54)
    sparse, means = torch.randn(7, 5), torch.randn(7, 4, 5)
    lm, lz = torch.randn(7, 4) + 10, torch.randn(7)
    out, alpha = mix_tail_read(sparse, lz, lm, means, rho)
    assert (alpha <= rho + 1e-6).all()
    u = (lm.softmax(-1)[..., None] * means).sum(-2)
    torch.testing.assert_close(out, sparse + alpha[:, None] * (u - sparse))
    dense, zero = mix_tail_read(sparse, lz, torch.full_like(lm, -torch.inf), means, rho)
    torch.testing.assert_close(dense, sparse)
    assert not zero.any()


def test_error_improvement_requires_direction_alignment():
    target = torch.tensor([1., 0.]); sparse = torch.zeros(2)
    for direction in (torch.tensor([2., 0.]), torch.tensor([-1., 1.])):
        for alpha in (0.1, 0.9):
            e = target - sparse
            actual = (e - alpha * direction).square().sum() - e.square().sum()
            predicted = alpha**2 * direction.square().sum() - 2 * alpha * torch.dot(e, direction)
            torch.testing.assert_close(actual, predicted)


def test_capped_empty_tail_has_finite_zero_mass_gradients():
    torch.manual_seed(57)
    sparse = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    means = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    log_z = torch.randn(3, dtype=torch.float64, requires_grad=True)
    log_mass = torch.tensor([[float("-inf")] * 4, [1., 2., 3., 4.],
                             [float("-inf")] * 4], dtype=torch.float64,
                            requires_grad=True)
    out, share = mix_tail_read(sparse, log_z, log_mass, means, .25)
    empty = torch.tensor([True, False, True])
    torch.testing.assert_close(out[empty], sparse[empty])
    assert not share[empty].any()
    gradients = torch.autograd.grad(out.square().sum(), (sparse, means, log_z, log_mass))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert not gradients[-1][empty].any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonempty_cap_matches_original_at_realistic_logits(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(58)
    sparse = torch.randn(1000, 64, device=device)
    means = torch.randn(1000, 16, 64, device=device)
    log_mass = torch.randn(1000, 16, device=device) * 10
    log_z = torch.randn(1000, device=device) * 10
    rho = .25
    import math
    limit = math.log(rho) - math.log1p(-rho)
    shift = (log_mass.logsumexp(-1) - log_z - limit).clamp_min(0)
    weights = torch.cat((log_z[:, None], log_mass - shift[:, None]), -1).softmax(-1)
    expected = weights[:, :1] * sparse + (weights[:, 1:, None] * means).sum(-2)
    out, share = mix_tail_read(sparse, log_z, log_mass, means, rho)
    print(f"{device} stable-vs-shift max output difference: {(out - expected).abs().max().item():.9g}")
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(share, weights[:, 1:].sum(-1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("magnitude", [100., 1e4, 1e8])
def test_cap_survives_large_finite_log_masses(magnitude):
    sparse, means = torch.zeros(1, 1), torch.ones(1, 4, 1)
    out, alpha = mix_tail_read(sparse, torch.zeros(1),
                               torch.full((1, 4), magnitude), means, .25)
    torch.testing.assert_close(alpha, torch.tensor([.25]), atol=0, rtol=0)
    torch.testing.assert_close(out, torch.tensor([[.25]]), atol=0, rtol=0)


def test_cell_uniform_variational_kl_identity_and_refinement():
    torch.manual_seed(61)
    logits = torch.randn(21, dtype=torch.float64) * 3
    assignment = torch.arange(21) % 4
    def coarse(partition):
        counts = torch.bincount(partition).double()
        means = torch.zeros_like(counts).index_add(0, partition, logits) / counts
        return means[partition], means[partition].softmax(0)
    coarse_logits, qp = coarse(assignment)
    p = logits.softmax(0)
    counts = torch.bincount(assignment).double()
    cell_mass = torch.rand(4, dtype=torch.float64); cell_mass /= cell_mass.sum()
    r = cell_mass[assignment] / counts[assignment]
    kl = lambda a, b: (a * (a.log() - b.log())).sum()
    gap = logits.logsumexp(0) - coarse_logits.logsumexp(0)
    torch.testing.assert_close(kl(qp, p), gap)
    torch.testing.assert_close(kl(r, p), kl(r, qp) + gap)
    cell_gains = torch.tensor([-2., .5, 0., 1.], dtype=torch.float64)
    gained_logits = coarse_logits + cell_gains[assignment]
    qg = gained_logits.softmax(0)
    gained_gap = logits.logsumexp(0) - gained_logits.logsumexp(0)
    torch.testing.assert_close(kl(qg, p), gained_gap + (qg * cell_gains[assignment]).sum())
    torch.testing.assert_close(kl(qg, p), kl(qg, qp) + gap)
    refined = assignment.clone(); refined[:3] = torch.arange(4, 7)
    refined_logits, qr = coarse(refined)
    assert refined_logits.logsumexp(0) >= coarse_logits.logsumexp(0)
    assert kl(qr, p) <= kl(qp, p)


def test_jensen_hybrid_is_causal_dense_at_full_budget_and_differentiable():
    torch.manual_seed(55)
    q = torch.randn(1, 4, 37, 5)
    k = torch.randn(1, 2, 37, 5)
    v = torch.randn(1, 2, 37, 6)
    g = torch.full((4,), -1., requires_grad=True)
    config = dict(block=4, cells=3, tail_mode="jensen", log_gain=g, max_tail_share=.25)
    out = hybrid_tail_attention(q, k, v, top_blocks=1, query_chunk=7, **config)
    grad = torch.autograd.grad(out.square().sum(), g)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    full = hybrid_tail_attention(q, k, v, top_blocks=20, query_chunk=11, **config)
    torch.testing.assert_close(full, dense, atol=3e-6, rtol=3e-5)
    q[:, :, 19:] = 1000; k[:, :, 19:] = 1000; v[:, :, 19:] = -1000
    changed = hybrid_tail_attention(q, k, v, top_blocks=1, query_chunk=13, **config)
    torch.testing.assert_close(out[:, :, :19], changed[:, :, :19], atol=3e-6, rtol=3e-5)
