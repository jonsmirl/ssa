"""Dense-oracle checks of the conditional, not IEEE-verified, path certificate."""
from dataclasses import replace

import numpy as np
import pytest

from ssa.routed_correction_certificate import Guard, StageEvidence, certify_path, route_overlap


def stage(x,y,xn,yn,J,c,h,fay,fby,E,En,C,B,H,guards=(),same=True):
    return StageEvidence(np.atleast_1d(x).astype(float),np.atleast_1d(y).astype(float),
                         np.atleast_1d(xn).astype(float),np.atleast_1d(yn).astype(float),
                         np.atleast_2d(J).astype(float),np.atleast_1d(c).astype(float),
                         np.atleast_1d(h).astype(float),np.atleast_1d(fay).astype(float),
                         np.atleast_1d(fby).astype(float),E,En,C,B,H,guards,same,
                         "analytic polynomial derivative bounds on all real inputs")


def nonlinear_witness():
    return [stage(0,.5,0,.375,0,.125,0,.25,.25,.5,.375,.125,0,2),
            stage(0,.375,1,.625,-1,0,0,.625,.625,.375,.375,0,0,0)]


def test_nonlinear_source_witness_exact_endpoint():
    r = certify_path(nonlinear_witness(), np.array([[1.],[0.]]))
    assert r["accepted"], r
    assert r["algebraic_margin_passed"]
    assert r["numerical_interval_violations"] == 0
    assert r["max_interval_deficit"] == 0
    assert r["signed_terms"] == [-.125]
    assert r["remainder"] == [.25]
    m = r["margins"][0]
    assert m["certified_lower_margin"] == m["actual_candidate_margin"] == .625
    assert r["radii_checks"][0]["propagated_budget"] == .375


def test_affine_route_switch_source_witness_and_strict_reference_tie():
    # Source witness x=0 ->0, y=-1/4 ->3/4 has exact +3/4 signed change.
    s = stage(0,-.25,0,.75,1,0,1,-.25,.75,.25,1.25,0,1,0,
              guards=(Guard(0,1),),same=False)
    r = certify_path([s], np.array([[1.],[0.]]))
    assert not r["accepted"]  # Nominal logits tie; cannot preserve a strict winner.
    assert r["signed_terms"] == [.75]
    assert r["remainder"] == [0]
    assert r["radii_checks"][0]["route_alternative"] == "actual-state branch-jump bound"


def test_signed_cancellation_not_sum_of_norms():
    p = [stage(2,2,2,3,1,1,0,2,2,0,1,1,0,0),
         stage(2,3,2,2,1,-1,0,3,3,1,2,1,0,0)]
    r = certify_path(p,np.array([[1.],[0.]]))
    assert r["accepted"]
    assert r["signed_terms"] == [0]
    assert [x["correction"] for x in r["margins"][0]["stage_terms"]] == [1,-1]


@pytest.mark.parametrize("update,fragment",[
    ({"bounds_source":""},"provenance"),
    ({"curvature":0},"curvature remainder"),
    ({"next_radius":.2},"output tube"),
    ({"correction_bound":0},"correction bound"),
    ({"candidate_next":np.array([.4])},"execution/correction"),
    ({"jump":np.array([.1])},"same-candidate-state"),
    ({"radius":float("nan")},"finite"),
    ({"jacobian":np.zeros((2,1))},"dimensions"),
    ({"reference_at_candidate":np.array([float("inf")])},"finite"),
])
def test_invalid_or_missing_evidence_fails_closed(update,fragment):
    p = nonlinear_witness()
    p[0] = replace(p[0],**update)
    r = certify_path(p,np.array([[1.],[0.]]))
    assert not r["accepted"]
    assert not r["algebraic_margin_passed"]
    assert fragment in r["reason"]


def test_algebraic_result_separate_from_observed_prediction():
    p = nonlinear_witness()
    # A valid but loose analytic bound does not pass merely because its dense
    # endpoint oracle observes that the corrected prediction was preserved.
    p[0] = replace(p[0],curvature=20.,next_radius=2.625)
    p[1] = replace(p[1],radius=2.625,next_radius=2.625)
    r = certify_path(p,np.array([[1.],[0.]]))
    assert r["observed_checks"]["actual_prediction_preserved"]
    assert not r["algebraic_margin_passed"]
    assert not r["accepted"]
    assert r["numerical_interval_violations"] == 0
    assert r["max_interval_deficit"] == 0


def test_radius_budget_not_just_observed_states():
    p = nonlinear_witness()
    p[0] = replace(p[0],correction_bound=1.)
    r = certify_path(p,np.array([[1.],[0.]]))
    assert not r["accepted"]
    assert "radius budget" in r["reason"]


def test_wrong_state_jump_is_rejected_even_when_radius_large():
    # F_a(t)=t, F_b(t)=2t+1; jump(y)=.75, unlike jump(x)=1.
    s = stage(0,-.25,0,.5,1,0,1,-.25,.5,.25,2,0,2,0,same=False)
    r = certify_path([s],np.array([[1.],[0.]]))
    assert not r["accepted"]
    assert "same-candidate-state" in r["reason"]


def test_guard_strictness_and_jump_alternative():
    s = stage(2,2.1,2,2.1,1,0,0,2.1,2.1,.2,.2,0,0,0,
              guards=(Guard(.3,1,nominal_gap=.3),))
    r = certify_path([s],np.array([[1.],[0.]]))
    assert r["accepted"] and r["radii_checks"][0]["guard_stable"]
    tied = certify_path([replace(s,guards=(Guard(.2,1),))],np.array([[1.],[0.]]))
    assert tied["accepted"]  # Actual same-state jump is known to be zero.
    assert not tied["radii_checks"][0]["guard_stable"]
    zero = certify_path([replace(s,guards=(Guard(0,0),))],np.array([[1.],[0.]]))
    assert not zero["radii_checks"][0]["guard_stable"]
    invalid = certify_path([replace(s,guards=(Guard(.3,1,nominal_gap=.4),))],np.array([[1.],[0.]]))
    assert not invalid["accepted"]
    contradiction = certify_path([replace(s,same_trace=False)],np.array([[1.],[0.]]))
    assert not contradiction["accepted"] and "contradict" in contradiction["reason"]


def test_all_candidates_and_ties_checked():
    p = nonlinear_witness()
    assert not certify_path(p,np.array([[1.],[0.],[1.]]))["accepted"]
    assert not certify_path(p,np.array([[1.],[0.]]),reference_choice=1)["accepted"]
    assert not certify_path(p,np.array([[1.]]))["accepted"]
    assert not certify_path([],np.array([[1.],[0.]]))["accepted"]
    s = stage(1,1,1,0,1,-1,0,1,1,0,1,1,0,0)
    assert not certify_path([s],np.array([[1.],[0.]]))["accepted"]


def test_complete_state_path_consistency():
    p = nonlinear_witness()
    p[1] = replace(p[1],reference=np.array([.1]))
    r = certify_path(p,np.array([[1.],[0.]]))
    assert not r["accepted"] and "discontinuous" in r["reason"]


def test_uniform_bound_is_explicit_trusted_hypothesis_not_sample_validation():
    # H=0 would be globally wrong for t^4 on this nonzero ball, although the
    # observed x=y=0 pair cannot falsify it. The checker makes this trust explicit.
    s = stage(1,1,1,1,1,0,0,1,1,1,1,0,0,0)
    r = certify_path([replace(s,bounds_source="caller-supplied analytic bound")],
                     np.array([[1.],[0.]]))
    assert r["accepted"]
    assert "uniform analytic bounds on the stated balls" in r["trusted_hypotheses"]
    assert "not an IEEE proof" in r["semantics"]


@pytest.mark.parametrize("S,T",[([0],[0]),([0],[1]),([0,1],[1,2]),
                                ([0,1],[0,1,2]),([2,1,1],[1,2])])
def test_overlap_dense_oracle(S,T):
    rng = np.random.default_rng(173)
    logits = rng.normal(size=7)*8
    values = rng.normal(size=(7,4))
    r = route_overlap(logits,values,S,T)
    def dense(ids):
        ids = np.unique(ids)
        p = np.zeros(7)
        p[ids] = np.exp(logits[ids]-max(logits[ids]))
        return p/p.sum()
    ps,pt = dense(S),dense(T)
    assert r["tv"] == pytest.approx(.5*np.abs(ps-pt).sum(),abs=2e-15)
    np.testing.assert_allclose(r["output_difference"],(ps-pt)@values,atol=1e-15)
    assert r["output_error"] <= r["output_error_bound"]+1e-14
    assert r["keys_scored"] == len(set(S)|set(T))


def test_overlap_extreme_logits_union_only_and_invalid_inputs():
    s = np.array([10000.,9999.,-10000.,np.nan])
    v = np.array([[1.],[2.],[3.],[np.nan]])
    r = route_overlap(s,v,[0,2],[1,2])
    assert r["tv"] == 1
    assert np.isfinite(r["output_error"])
    for S,T in [([],[1]),([0.5],[1]),([0],[4]),([0],[3])]:
        with pytest.raises(ValueError):
            route_overlap(s,v,S,T)


def test_hard_top_one_discontinuity_witness():
    for epsilon in (.1,.001,1e-8):
        left,right = -epsilon/4,epsilon/4
        # First key wins a tie; its value is 1 and the competing value is 0.
        chosen = lambda q: int(np.argmax([q,0]))
        assert right-left < epsilon
        assert chosen(left) == 1 and chosen(right) == 0
        v = np.array([[1.],[0.]])
        r = route_overlap(np.array([right,0.]),v,[chosen(left)],[chosen(right)])
        assert r["output_error"] == 1
        assert r["tv"] == 1
