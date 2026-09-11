"""CPU dense-oracle and ledger checks for device-fallback measurements."""
import pytest
import torch

from ssa.device_coordinate_experiment import audit, fixture_paths


def example():
    g = torch.Generator().manual_seed(134)
    q = torch.randn(2,64,generator=g,dtype=torch.float64)
    K = torch.randn(31,64,generator=g,dtype=torch.float64)
    V = torch.randn(31,5,generator=g,dtype=torch.float64)
    s = q@K.T/8
    proposal = torch.zeros(2,31,dtype=torch.bool)
    proposal[:,:20] = True
    p = s.softmax(-1)
    mass = (p*~proposal).sum(-1)
    accepted = torch.tensor([True,False])
    final = proposal | ~accepted[:,None]
    out = s.masked_fill(~final,-torch.inf).softmax(-1)@V
    final_mass = mass*accepted
    result = dict(output=out,proposed_selected_mask=proposal,selected_mask=final,
        accepted=accepted,proposed_mass_upper=mass,mass_upper=final_mass,
        lower=s,upper=s,output_error_upper=2*V.norm(dim=1).max()*final_mass+1e-12)
    return q,K,V,result


def test_mixed_acceptance_has_exact_dense_rejection():
    q,K,V,result = example()
    rows = audit(q,K,V,result,.99)
    assert all(max(r["audit"].values())==0 for r in rows)
    assert rows[0]["selected_values"]==20
    assert rows[1]["selected_values"]==31
    assert rows[1]["actual_omitted_mass"]==0


@pytest.mark.parametrize("fault",["false_accept","bad_final_selection","proposal_mass_deficit","output_deficit",
    "invalid_numerics","uncertified_final"])
def test_oracle_rejects_bad_pipeline(fault):
    q,K,V,result = example()
    eta = .99
    if fault=="false_accept":
        eta = .001
    elif fault=="bad_final_selection":
        result["selected_mask"][1,30] = False
    elif fault=="proposal_mass_deficit":
        result["proposed_mass_upper"] *= 0
    elif fault=="invalid_numerics":
        result["numerically_valid"] = torch.tensor([False,True])
    elif fault=="uncertified_final":
        result["certified"] = torch.tensor([False,True])
    else:
        result["output"] += 100
    rows = audit(q,K,V,result,eta)
    assert any(r["audit"][fault]>0 for r in rows)


def test_both_fixture_protocols_have_finite_matching_archive_if_available():
    from pathlib import Path
    roots = ["/tmp/ssa_partial_fresh","/tmp/ssa_device_growth"]
    if not all((Path(r)/"manifest.json").exists() for r in roots):
        pytest.skip("local cached fixture not required for portable tests")
    paths = fixture_paths(roots)
    assert len(paths)==3
