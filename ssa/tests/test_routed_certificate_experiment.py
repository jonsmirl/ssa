"""Dense numerical checks of the complete random-model provider, not IEEE proofs."""
import numpy as np
import torch

from ssa.routed_certificate_experiment import ReferenceTransformer, run_trial


def test_complete_nonzero_path_accepts_and_exposes_uniform_tube():
    torch.set_num_threads(1)
    result = run_trial(seed=0, strength=.01)
    assert result["accepted"] and result["prediction_preserved"]
    assert len(result["stages"]) == 5
    assert result["work"]["jacobian_evaluations"] == 5
    assert result["work"]["jacobian_scalar_entries"] == 5*32**2
    assert result["work"]["routing_scores"] == 2*2*2*8**2
    assert len(result["margins"]) == 6
    assert any(s["correction_norm"] > 0 for s in result["stages"])
    for stage in result["stages"]:
        assert stage["actual_error"] <= stage["radius"]+1e-12
        assert stage["taylor_remainder"] <= stage["taylor_bound"]+1e-12
        assert stage["jump_norm"] <= stage["union_jump_bound"]+1e-12
        assert stage["jacobian_norm"] <= stage["uniform_lipschitz"]+1e-12


def test_large_changed_prediction_rejected_and_changed_routes_charged():
    result = run_trial(seed=1, strength=64)
    assert not result["accepted"] and not result["prediction_preserved"]
    assert any(not s["same_trace"] for s in result["stages"])
    assert any(s["jump_norm"] > 0 for s in result["stages"])
    assert result["work"]["union_key_reads"] > 0


def test_bound_can_reject_unchanged_prediction_without_claiming_failure():
    result = run_trial(seed=0, strength=1)
    assert not result["accepted"] and result["prediction_preserved"]
    assert result["reason"] == "reference or certified candidate margin is not strictly positive"


def test_full_selection_recovers_dense_attention_and_zero_tail():
    model = ReferenceTransformer(top_k=8)
    x = model.input
    route, _ = model.route(x, 0)
    causal = torch.ones(8,8,dtype=torch.bool).tril().expand(2,-1,-1)
    assert torch.equal(route, causal)
    q,k,v = model.qkv(x,0)
    expected = (torch.softmax((model.beta*(q@k.transpose(-1,-2))).masked_fill(~causal,-torch.inf),-1)@v)
    expected = expected.transpose(0,1).reshape(8,4)@model.weights[0][3]
    assert torch.equal(model.attention(x,0,route),expected)
    assert torch.equal(model.correction(x,0,route,1),torch.zeros_like(x))
    result = run_trial(seed=0,strength=64,top_k=8)
    assert result["accepted"] and result["max_logit_change"] == 0


def test_union_tv_bound_covers_different_masks_at_one_common_state():
    model = ReferenceTransformer()
    a,_ = model.route(model.input,0)
    b=a.clone()
    b[:,:, :] = False
    for query in range(model.n):
        b[:,query,max(0,query-1):query+1] = True
    difference = model.attention(model.input,0,a)-model.attention(model.input,0,b)
    bound = model.union_jump_bound(model.input,0,a,b)
    assert np.linalg.norm(difference.numpy()) <= bound+1e-12
    assert model.union_jump_bound(model.input,0,a,a) == 0
