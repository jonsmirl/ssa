import torch

from ssa.reader_weighted_tail_diagnostic import head_diagonal_error, scalar_oracle_gate, joint_head_gate


def test_scalar_oracle_is_optimal_on_box_and_zero_direction_is_zero():
    torch.manual_seed(83)
    error, direction = torch.randn(53, 9, dtype=torch.float64), torch.randn(53, 9, dtype=torch.float64)
    direction[0] = 0
    gate = scalar_oracle_gate(error, direction)
    assert gate[0] == 0
    optimum = (error - gate[:, None] * direction).square().sum(-1)
    for candidate in torch.linspace(0, 1, 101, dtype=torch.float64):
        assert (optimum <= (error - candidate * direction).square().sum(-1) + 1e-12).all()


def test_coupled_heads_can_cancel_and_change_the_optimal_gate():
    projection = torch.tensor([[1., 1.]], dtype=torch.float64)
    residual = torch.tensor([[1., -1.]], dtype=torch.float64)
    assert (residual @ projection.T).square().sum() == 0
    assert head_diagonal_error(residual, projection, 2).item() == 2
    error = torch.tensor([[1., 0.]], dtype=torch.float64)
    direction = torch.tensor([[1., -2.]], dtype=torch.float64)
    assert scalar_oracle_gate(error, direction).item() == .2
    assert scalar_oracle_gate(error @ projection.T, direction @ projection.T).item() == 0


def test_joint_head_box_oracle_uses_coupling_and_improves_feasible_start():
    error = torch.tensor([[1., 0.]], dtype=torch.float64)
    directions = torch.tensor([[[1., 1.], [0., -1.]]], dtype=torch.float64)
    initial = torch.tensor([[.25, .25]], dtype=torch.float64)
    gate, stats = joint_head_gate(error, directions, initial, iterations=100)
    assert ((gate >= 0) & (gate <= 1)).all()
    loss = lambda a: (error - (a[..., None] * directions).sum(1)).square().sum()
    assert loss(gate) <= loss(initial)
    assert loss(gate) < 1e-8
    assert stats["projected_kkt_max"] < 1e-5
    # Ignoring the coupling would choose (.5,0), not the joint optimum (1,1).
    assert loss(gate) < loss(torch.tensor([[.5, 0.]], dtype=torch.float64))
