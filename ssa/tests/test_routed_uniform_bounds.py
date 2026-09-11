"""Independent implementation checks, not proofs of uniform/IEEE bounds."""
import pytest
import torch

from ssa.routed_certificate_experiment import ReferenceTransformer


@pytest.mark.parametrize("kind", ["attention", "mlp", "norm"])
def test_analytic_ball_bounds_dominate_sampled_jacobians_and_remainders(kind):
    model = ReferenceTransformer(seed=31, n=4, d=4, heads=2, layers=1)
    x = model.input.clone()
    mask, _ = model.route(x, 0)
    radius = .25
    lipschitz, curvature, _ = model.uniform_bounds(x, radius, kind, 0)
    function = lambda z: model.stage(z.reshape_as(x), kind, 0, mask).flatten()
    reference = function(x.flatten())
    jacobian = torch.autograd.functional.jacobian(function, x.flatten(), vectorize=True)
    assert torch.linalg.matrix_norm(jacobian, ord=2) <= lipschitz + 1e-12
    rng = torch.Generator().manual_seed(32)
    for fraction in (.1, .5, 1.):
        direction = torch.randn(x.shape, generator=rng, dtype=x.dtype)
        delta = direction.flatten() * (fraction * radius / torch.linalg.vector_norm(direction))
        y = x.flatten() + delta
        jy = torch.autograd.functional.jacobian(function, y, vectorize=True)
        displacement = torch.linalg.vector_norm(delta)
        assert torch.linalg.matrix_norm(jy-jacobian, ord=2) <= curvature * displacement + 1e-12
        residual = function(y)-reference-jacobian@delta
        assert torch.linalg.vector_norm(residual) <= .5*curvature*displacement**2 + 1e-12


def test_recorded_insertion_comparisons_resolve_all_ties_deterministically():
    model = ReferenceTransformer(seed=33, n=5, d=4, heads=2, layers=1)
    model.weights[0][0].zero_()
    mask, trace = model.route(model.input, 0)
    assert len(trace) == model.heads * sum(i*(i+1)//2 for i in range(model.n))
    assert all(not wins and gap == 0 for _, _, _, _, wins, gap in trace)
    for h in range(model.heads):
        for i in range(model.n):
            assert mask[h, i].nonzero().flatten().tolist() == list(range(min(i+1, model.top_k)))


def test_analytic_guard_ball_preserves_the_entire_comparison_trace():
    model = ReferenceTransformer(seed=34, n=5, d=4, heads=2, layers=1)
    x = model.input
    mask, trace = model.route(x, 0)
    _, _, constant = model.uniform_bounds(x, 1., "attention", 0)
    margin = min(entry[5] for entry in trace)
    assert margin > 0 and constant > 0
    radius = min(1., margin/(4*constant))
    rng = torch.Generator().manual_seed(35)
    direction = torch.randn(x.shape, generator=rng, dtype=x.dtype)
    y = x + direction*(radius/torch.linalg.vector_norm(direction))
    changed_mask, changed_trace = model.route(y, 0)
    assert torch.equal(mask, changed_mask)
    assert [entry[:5] for entry in trace] == [entry[:5] for entry in changed_trace]
    sx, sy = model.scores(x, 0), model.scores(y, 0)
    for head, query, key, previous, _, _ in trace:
        change = (sy[head, query, key]-sy[head, query, previous]) - (sx[head, query, key]-sx[head, query, previous])
        assert change.abs() <= constant*radius + 1e-12


def test_complete_attention_and_actual_tail_correction_are_causal():
    model = ReferenceTransformer(seed=36, n=5, d=4, heads=2, layers=1)
    x = model.input.clone()
    mask, _ = model.route(x, 0)
    before = model.stage(x, "attention", 0, mask) + model.correction(x, 0, mask, .25)
    x[3:] = 100
    changed_mask, _ = model.route(x, 0)
    after = model.stage(x, "attention", 0, changed_mask) + model.correction(x, 0, changed_mask, .25)
    torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=1e-12)


def test_union_jump_bound_dominates_actual_joint_projected_jump():
    model = ReferenceTransformer(seed=37, n=5, d=4, heads=2, layers=1)
    x = model.input.clone()
    first, _ = model.route(x, 0)
    second = torch.zeros_like(first)
    for i in range(model.n):
        second[:, i, max(0, i-model.top_k+1):i+1] = True
    jump = model.attention(x, 0, second)-model.attention(x, 0, first)
    bound = model.union_jump_bound(x, 0, first, second)
    assert torch.linalg.vector_norm(jump) <= bound + 1e-12
    assert model.union_jump_bound(x, 0, first, first) == 0
