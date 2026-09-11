"""CPU checks of the GPU benchmark's independent oracle and reporting."""
import copy

import pytest
import torch

from ssa.partial_coordinate_gpu_experiment import audit_read, dense_read, summarize


def case():
    gen = torch.Generator().manual_seed(108)
    q = torch.randn(2,4,generator=gen,dtype=torch.float64)
    K = torch.randn(19,4,generator=gen,dtype=torch.float64)
    V = torch.randn(19,3,generator=gen,dtype=torch.float64)
    scores = q@K.T/2
    keep = torch.zeros(2,19,dtype=torch.bool)
    keep[:,:7] = True
    p = scores.softmax(-1)
    mass = (p*~keep).sum(-1)
    out = scores.masked_fill(~keep,-torch.inf).softmax(-1)@V
    result = dict(output=out,selected_mask=keep,lower=scores,upper=scores,
        mass_upper=mass,output_error_upper=2*V.norm(dim=1).max()*mass,
        certified=mass<=.1)
    return q,K,V,result


def test_oracle_checks_restricted_read_and_dense_baseline():
    q,K,V,result = case()
    heads,truth = audit_read(q,K,V,result,.1)
    torch.testing.assert_close(dense_read(q,K,V),truth)
    for h in heads:
        assert h["restricted_output_numerical_l2"]<1e-14
        assert max(h["audit"].values())<1e-14
        assert h["values_read"]==7
        assert h["actual_supported_kl"]==pytest.approx(h["kl_upper_from_mass"])


@pytest.mark.parametrize("fault,field",[("mass","mass_deficit"),("interval","interval_deficit"),
    ("output","output_deficit"),("stop","false_stop")])
def test_audit_detects_deliberately_false_certificates(fault,field):
    q,K,V,result = case()
    if fault=="mass":
        result["mass_upper"] *= 0
    elif fault=="interval":
        result["upper"] -= 1
    elif fault=="output":
        result["output"] += 100
    else:
        result["certified"][:] = True
    heads,_ = audit_read(q,K,V,result,.1)
    assert all(h["audit"][field]>0 for h in heads)


def test_full_selection_is_not_omitted_value_error():
    q,K,V,result = case()
    result["selected_mask"][:] = True
    result["mass_upper"][:] = 0
    result["output_error_upper"][:] = 1e-12
    result["output"] = dense_read(q,K,V)
    result["certified"][:] = True
    heads,_ = audit_read(q,K,V,result,.01)
    assert all(h["actual_omitted_mass"]==0 and h["actual_output_l2"]<1e-14 for h in heads)


def test_summary_reports_paired_slowdowns_not_inverse_means():
    q,K,V,result = case()
    heads,_ = audit_read(q,K,V,result,.1)
    row = dict(coordinates=32,eta=.1,heads=heads,query_timing={"median_ms":10},
        dense_native_timing={"median_ms":1},dense_fp32_timing={"median_ms":2})
    other = copy.deepcopy(row)
    other["query_timing"]["median_ms"] = 2
    other["dense_native_timing"]["median_ms"] = 4
    report = summarize([row,other])["r32/eta0.1"]
    assert report["head_queries"]==4
    assert report["median_paired_slowdown_vs_native"]==5.25
    assert report["sparse_faster_fraction"]==.5
