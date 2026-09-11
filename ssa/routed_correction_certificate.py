"""Float64 reference checker for conditional routed-correction certificates.

The real-arithmetic theorem is specified in docs/routed_correction_certificate.md.
Uniform derivative/guard bounds and complete stage/trace semantics are SUPPLIED
hypotheses, not inferred from observations. A provenance string is mandatory but
is not a proof. In particular sampled Hessians are not uniform curvature bounds.
This checker is not an interval-arithmetic or formally verified IEEE certificate.
It checks numerical evidence and preserves only a strict reference prediction on
the supplied readouts, not model correctness or cheaper execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Guard:
    """An executed, branch-oriented comparison and its uniform tube bound."""

    margin: float
    lipschitz: float
    nominal_gap: float | None = None


@dataclass(frozen=True)
class StageEvidence:
    reference: np.ndarray
    candidate: np.ndarray
    reference_next: np.ndarray
    candidate_next: np.ndarray
    jacobian: np.ndarray
    correction: np.ndarray
    jump: np.ndarray
    reference_at_candidate: np.ndarray
    candidate_branch_at_candidate: np.ndarray
    radius: float
    next_radius: float
    correction_bound: float
    jump_bound: float
    curvature: float
    guards: tuple[Guard, ...]
    same_trace: bool
    bounds_source: str


_EPS = np.finfo(np.float64).eps


def _slack(*values: float) -> float:
    return 128 * _EPS * max(1.0, *(abs(float(v)) for v in values))


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: expected nonempty finite vector")
    return a


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=128 * _EPS,
                                                  atol=128 * _EPS))


def certify_path(stages: Sequence[StageEvidence], readouts: np.ndarray,
                 reference_choice: int | None = None) -> dict:
    """Check a supplied finite path; malformed or inconsistent inputs fail closed.

    All stage states include every relevant cache/summary. ``reference_next`` is
    supplied F_a(x); the two explicit branch evaluations are F_a(y) and F_b(y).
    They prevent replacing the executed-state jump by a reference-state jump.
    The checker cannot verify these evaluations, derivatives, or uniform bounds
    without the underlying maps. ``bounds_source`` identifies that trust boundary.
    Numerical consistency uses 128-epsilon tolerances; strict prediction and guard
    tests reserve an additional epsilon-scaled margin. No IEEE guarantee follows.
    ``algebraic_margin_passed`` is recorded before the observed-endpoint veto.
    Interval deficits are raw positive excesses; violations exceed the numerical
    consistency tolerance. Counts cover only comparisons reached before any
    fail-closed return, as reported by ``interval_comparisons_checked``.
    """
    report = dict(accepted=False, reason="invalid evidence", radii_checks=[],
                  signed_terms=[], remainder=[], margins=[], observed_checks={},
                  algebraic_margin_passed=False,
                  numerical_interval_violations=0, max_interval_deficit=0.0,
                  interval_comparisons_checked=0,
                  semantics="conditional float64 reference; not an IEEE proof",
                  trusted_hypotheses=["complete state and executed comparison trace",
                                      "correct branch evaluations and Jacobians",
                                      "uniform analytic bounds on the stated balls"])
    try:
        if not stages:
            raise ValueError("at least one stage is required")
        parsed = []
        for j, s in enumerate(stages):
            if not isinstance(s.bounds_source, str) or not s.bounds_source.strip():
                raise ValueError(f"stage {j}: missing uniform-bound provenance")
            numbers = [s.radius, s.next_radius, s.correction_bound,
                       s.jump_bound, s.curvature]
            if not all(np.isfinite(v) and v >= 0 for v in numbers):
                raise ValueError(f"stage {j}: bounds must be finite and nonnegative")
            x, y = (_vector(s.reference, "reference"),
                    _vector(s.candidate, "candidate"))
            xn, yn, c, h, fay, fby = [
                _vector(v, name) for v, name in [
                    (s.reference_next, "reference_next"),
                    (s.candidate_next, "candidate_next"),
                    (s.correction, "correction"), (s.jump, "jump"),
                    (s.reference_at_candidate, "reference_at_candidate"),
                    (s.candidate_branch_at_candidate, "candidate_branch_at_candidate")]]
            J = np.asarray(s.jacobian, dtype=np.float64)
            if (x.shape != y.shape or any(v.shape != xn.shape for v in (yn,c,h,fay,fby))
                    or J.shape != (xn.size, x.size) or not np.all(np.isfinite(J))):
                raise ValueError(f"stage {j}: dimensions or Jacobian invalid")
            if j and (not _close(parsed[-1][2], x) or not _close(parsed[-1][3], y)
                      or abs(stages[j-1].next_radius-s.radius) > _slack(s.radius)):
                raise ValueError(f"stage {j}: discontinuous state path or tube radii")
            if not _close(fby - fay, h):
                raise ValueError(f"stage {j}: jump is not the supplied same-candidate-state difference")
            if not _close(fby + c, yn):
                raise ValueError(f"stage {j}: candidate execution/correction recurrence mismatch")
            if not isinstance(s.same_trace, (bool, np.bool_)):
                raise ValueError(f"stage {j}: same_trace must be boolean")
            if s.same_trace and not _close(h, np.zeros_like(h)):
                raise ValueError(f"stage {j}: same trace has a nonzero branch jump")
            e = y-x
            r = fay-xn-J@e
            if not _close(yn-xn, J@e+c+h+r):
                raise ValueError(f"stage {j}: error recurrence mismatch")
            en, rn = float(np.linalg.norm(e)), float(np.linalg.norm(r))
            cn, hn = float(np.linalg.norm(c)), float(np.linalg.norm(h))
            Jn = float(np.linalg.norm(J, ord=2))
            local_remainder = .5*s.curvature*en**2
            tube_remainder = .5*s.curvature*s.radius**2
            budget = Jn*s.radius+s.correction_bound+s.jump_bound+tube_remainder
            observed_next = float(np.linalg.norm(yn-xn))
            finite_numbers = [en,rn,cn,hn,Jn,local_remainder,tube_remainder,budget,observed_next]
            if not np.all(np.isfinite(finite_numbers)):
                raise ValueError(f"stage {j}: arithmetic overflow in bound evaluation")
            checks = dict(stage=j, radius=float(s.radius), next_radius=float(s.next_radius),
                          observed_error=en, observed_next_error=observed_next,
                          observed_remainder=rn, local_remainder_bound=local_remainder,
                          tube_remainder_bound=tube_remainder, jacobian_norm=Jn,
                          propagated_budget=budget, bounds_source=s.bounds_source)
            for label, actual, bound in [
                ("input tube",en,s.radius), ("output tube",observed_next,s.next_radius),
                ("correction",cn,s.correction_bound), ("jump",hn,s.jump_bound),
                ("curvature remainder",rn,local_remainder),
                ("radius budget",budget,s.next_radius)]:
                if actual > bound + _slack(actual,bound):
                    raise ValueError(f"stage {j}: {label} bound violated")
            guards_ok = True
            for g in s.guards:
                if not all(np.isfinite(v) and v >= 0 for v in (g.margin,g.lipschitz)):
                    raise ValueError(f"stage {j}: invalid oriented guard")
                if g.nominal_gap is not None and (
                        not np.isfinite(g.nominal_gap)
                        or abs(g.nominal_gap-g.margin) > _slack(g.nominal_gap,g.margin)):
                    raise ValueError(f"stage {j}: guard nominal gap mismatch")
                guards_ok &= g.margin-g.lipschitz*s.radius > _slack(g.margin,g.lipschitz*s.radius)
            checks["guard_stable"] = bool(s.same_trace and guards_ok)
            if s.guards and guards_ok and not s.same_trace:
                raise ValueError(f"stage {j}: supplied stable complete-trace guards contradict observed route change")
            checks["route_alternative"] = ("uniform complete-trace guards"
                                           if checks["guard_stable"] else "actual-state branch-jump bound")
            report["radii_checks"].append(checks)
            parsed.append((x,y,xn,yn,J,c,h,r))

        W = np.asarray(readouts, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] < 2 or W.shape[1] != parsed[-1][2].size or not np.all(np.isfinite(W)):
            raise ValueError("readouts must be finite candidate-by-terminal-state matrix with >=2 candidates")
        xfinal, yfinal = parsed[-1][2:4]
        reference_logits, candidate_logits = W@xfinal, W@yfinal
        if not np.all(np.isfinite(reference_logits)) or not np.all(np.isfinite(candidate_logits)):
            raise ValueError("nonfinite terminal logits")
        if reference_choice is None:
            reference_choice = int(np.argmax(reference_logits))
        if not isinstance(reference_choice, (int,np.integer)) or not 0 <= reference_choice < W.shape[0]:
            raise ValueError("reference choice out of candidate range")
        a = int(reference_choice)
        report["reference_choice"] = a
        report["candidate_count"] = int(W.shape[0])
        report["observed_checks"] = dict(reference_logits=reference_logits.tolist(),
                                          candidate_logits=candidate_logits.tolist(),
                                          actual_prediction_preserved=bool(np.all(
                                              candidate_logits[a] > np.delete(candidate_logits,a))))
        all_positive = True
        for b in range(W.shape[0]):
            if b == a:
                continue
            pullback = W[a]-W[b]
            D, R, actual_r = 0.0, 0.0, 0.0
            stage_terms = []
            for j in reversed(range(len(stages))):
                _,_,_,_,J,c,h,r = parsed[j]
                term = float(pullback@(c+h))
                D += term
                stage_terms.append(dict(stage=j, correction=float(pullback@c),
                                        jump=float(pullback@h), combined=term))
                actual_r += float(pullback@r)
                R += float(np.linalg.norm(pullback))*.5*stages[j].curvature*stages[j].radius**2
                pullback = pullback@J
            initial = float(pullback@(parsed[0][1]-parsed[0][0]))
            D += initial
            nominal = float(reference_logits[a]-reference_logits[b])
            actual_change = float((W[a]-W[b])@(yfinal-xfinal))
            lower = nominal+D-R
            if not np.all(np.isfinite([D,R,actual_r,initial,nominal,actual_change,lower])):
                raise ValueError("nonfinite signed propagation")
            if abs(actual_change-D-actual_r) > _slack(actual_change,D,actual_r)*len(stages):
                raise ValueError("signed telescoping identity mismatch")
            interval_deficit = max(0.0, abs(actual_change-D)-R)
            report["interval_comparisons_checked"] += 1
            report["max_interval_deficit"] = max(report["max_interval_deficit"], interval_deficit)
            if interval_deficit > _slack(actual_change,D,R)*len(stages):
                report["numerical_interval_violations"] += 1
                raise ValueError("signed remainder bound violated")
            passed = nominal > _slack(nominal) and lower > _slack(nominal,D,R)
            all_positive &= passed
            report["signed_terms"].append(D)
            report["remainder"].append(R)
            report["margins"].append(dict(competitor=b, reference_margin=nominal,
                                          signed_change=D, remainder_bound=R,
                                          certified_lower_margin=lower,
                                          actual_candidate_margin=nominal+actual_change,
                                          interval_deficit=interval_deficit,
                                          initial_term=initial,
                                          stage_terms=list(reversed(stage_terms)),
                                          passed=bool(passed)))
        observed_preserved = report["observed_checks"]["actual_prediction_preserved"]
        report["algebraic_margin_passed"] = bool(all_positive)
        report["accepted"] = bool(all_positive and observed_preserved)
        report["reason"] = ("strict reference prediction preserved under supplied hypotheses"
                            if report["accepted"] else
                            "observed prediction contradicts certificate" if all_positive else
                            "reference or certified candidate margin is not strictly positive")
    except (ValueError, TypeError, AttributeError, OverflowError, np.linalg.LinAlgError) as exc:
        report["accepted"] = False
        report["reason"] = str(exc)
    return report


def route_overlap(logits: np.ndarray, values: np.ndarray,
                  S: Sequence[int], T: Sequence[int]) -> dict:
    """Same-state restricted reads, scoring only their charged union.

    Inputs are array-backed accessors here: only union entries are inspected.
    This does not silently charge zero work for already materialized logits.
    Unselected entries may be NaN; malformed/nonfinite union entries raise.
    Duplicate indices have set semantics, with deterministic sorted ordering.
    """
    s, v = np.asarray(logits,dtype=np.float64), np.asarray(values,dtype=np.float64)
    if v.ndim == 1:
        v = v[:,None]
    if s.ndim != 1 or v.ndim != 2 or len(v) != len(s) or v.shape[1] == 0:
        raise ValueError("expected logits[n] and values[n,d]")
    def indices(raw):
        a = np.asarray(list(raw))
        if a.ndim != 1 or not len(a) or a.dtype.kind not in "iu" or np.any(a<0) or np.any(a>=len(s)):
            raise ValueError("selected sets must be nonempty valid integer indices")
        return np.unique(a)
    S, T = indices(S), indices(T)
    union = np.union1d(S,T)
    su, vu = s[union], v[union]
    if not np.all(np.isfinite(su)) or not np.all(np.isfinite(vu)):
        raise ValueError("nonfinite union logits or values")
    ms, mt = np.isin(union,S), np.isin(union,T)
    def weights(mask):
        peak = float(su[mask].max())
        w = np.zeros(len(union),dtype=np.float64)
        w[mask] = np.exp(su[mask]-peak)
        total = float(w.sum())
        return w/total, peak+float(np.log(total))
    ps, zs = weights(ms)
    pt, zt = weights(mt)
    both = ms & mt
    overlap = float(np.minimum(ps[both],pt[both]).sum())
    tv = max(0.0,min(1.0,1.0-overlap))
    # Normalized intersection probabilities avoid overflow in exp(s)/max(Zs,Zt).
    oS, oT = ps@vu, pt@vu
    V = float(np.linalg.norm(vu,axis=1).max())
    if not np.all(np.isfinite([*oS,*oT,V,2*V*tv,zs,zt])):
        raise ValueError("nonfinite union-read arithmetic")
    return dict(tv=tv, tv_direct=float(.5*np.abs(ps-pt).sum()), overlap_mass_fraction=overlap,
                output_S=oS.tolist(), output_T=oT.tolist(),
                output_difference=(oS-oT).tolist(), output_error=float(np.linalg.norm(oS-oT)),
                output_error_bound=2*V*tv, value_norm_bound=V,
                log_Z_S=zs, log_Z_T=zt, union=union.tolist(),
                keys_scored=int(len(union)), values_read=int(len(union)),
                selected_S=int(len(S)), selected_T=int(len(T)),
                attention_score_semantics="actual attention logits at one common state")
