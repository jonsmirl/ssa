"""Complete CPU float64 transformer and analytic routed-path certificate reference.

This is deliberately a small random model, not Qwen and not a fast decoder.
Its uniform curvature bounds are analytic; Jacobians alone use autograd. The
certificate is a real-arithmetic certificate evaluated in float64, not interval
verified floating-point arithmetic. Every key score and replay is charged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .routed_correction_certificate import Guard, StageEvidence, certify_path


DTYPE = torch.float64


def fnorm(x):
    return float(torch.linalg.vector_norm(x))


class ReferenceTransformer:
    """Pre-norm, causal multi-head attention, tanh MLP, final norm and LM head."""

    def __init__(self, seed=0, n=8, d=4, heads=2, layers=2, top_k=2, vocab=7):
        if d % heads or not 1 <= top_k <= n:
            raise ValueError("invalid dimensions/budget")
        self.n, self.d, self.heads, self.layers = n, d, heads, layers
        self.top_k, self.vocab, self.eps = top_k, vocab, 0.5
        rng = torch.Generator().manual_seed(seed)
        def weight(a, b):
            return torch.randn(a, b, generator=rng, dtype=DTYPE) * (0.2 / math.sqrt(a))
        self.weights = [tuple(weight(d, d) for _ in range(4)) +
                        (weight(d, 2*d), weight(2*d, d)) for _ in range(layers)]
        self.lm_head = weight(d, vocab)
        self.token_embedding = torch.randn(vocab, d, generator=rng, dtype=DTYPE)
        self.token_ids = torch.arange(n) % vocab
        self.position_embedding = .01*torch.sin(torch.arange(n,dtype=DTYPE)[:,None] *
                                               torch.arange(1,d+1,dtype=DTYPE)[None,:])
        self.input = self.token_embedding[self.token_ids] + self.position_embedding
        self.beta = 1 / math.sqrt(d // heads)

    def norm(self, x):
        return x / torch.sqrt(x.square().mean(-1, keepdim=True) + self.eps)

    def qkv(self, x, layer):
        y = self.norm(x)
        return tuple((y @ w).reshape(self.n, self.heads, -1).transpose(0, 1)
                     for w in self.weights[layer][:3])

    def scores(self, x, layer):
        q, k, _ = self.qkv(x, layer)
        return self.beta * (q @ k.transpose(-1, -2))

    def route(self, x, layer):
        """Insertion sort: record EVERY executed comparison, ties favor lower key.

        No hidden topk/queue/cell-assignment decision participates in this router.
        The sort scans all causal scores and is intentionally quadratic or worse.
        """
        scores = self.scores(x, layer).detach()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        trace = []
        for h in range(self.heads):
            for i in range(self.n):
                order = []
                for key in range(i + 1):
                    position = len(order)
                    for j, previous in enumerate(order):
                        gap = float(scores[h, i, key] - scores[h, i, previous])
                        wins = gap > 0  # previous key is smaller and wins equality.
                        trace.append((h, i, key, previous, wins, abs(gap)))
                        if wins:
                            position = j
                            break
                    order.insert(position, key)
                mask[h, i, order[:self.top_k]] = True
        return mask, trace

    def attention(self, x, layer, mask):
        q, k, v = self.qkv(x, layer)
        p = torch.softmax((self.beta * (q @ k.transpose(-1, -2))).masked_fill(~mask, -torch.inf), -1)
        return (p @ v).transpose(0, 1).reshape(self.n, self.d) @ self.weights[layer][3]

    def correction(self, x, layer, mask, strength):
        """Actual-state Jensen cell-mean tail read, outside the fixed smooth map.

        One omitted cell per query/head. This toy reference computes its summaries
        by scanning omitted keys, and charges that scan. No upper cap is claimed.
        """
        q, k, v = self.qkv(x, layer)
        score = self.beta * (q @ k.transpose(-1, -2))
        visible = torch.ones(self.n, self.n, dtype=torch.bool).tril()[None]
        omitted = visible & ~mask
        counts = omitted.sum(-1)
        key_mean = omitted.to(DTYPE) @ k / counts.clamp_min(1)[..., None]
        value_mean = omitted.to(DTYPE) @ v / counts.clamp_min(1)[..., None]
        log_tail = counts.clamp_min(1).log() + self.beta * (q * key_mean).sum(-1)
        log_tail = log_tail.masked_fill(counts == 0, -torch.inf)
        selected = score.masked_fill(~mask, -torch.inf)
        selected_value = torch.softmax(selected, -1) @ v
        share = torch.sigmoid(log_tail - torch.logsumexp(selected, -1))
        delta = share[..., None] * (value_mean - selected_value)
        return strength * delta.transpose(0, 1).reshape(self.n, self.d) @ self.weights[layer][3]

    def union_jump_bound(self, x, layer, first, second):
        """Same-state restricted-read TV, transported through all heads and Wo."""
        if torch.equal(first, second):
            return 0.0
        q, k, v = self.qkv(x, layer)
        score = self.beta * (q @ k.transpose(-1, -2))
        log_first = torch.logsumexp(score.masked_fill(~first, -torch.inf), -1)
        log_second = torch.logsumexp(score.masked_fill(~second, -torch.inf), -1)
        log_intersection = torch.logsumexp(score.masked_fill(~(first & second), -torch.inf), -1)
        tv = -torch.expm1(log_intersection-torch.maximum(log_first, log_second))
        value_norm = torch.linalg.vector_norm(v, dim=-1)[:, None, :]
        largest = value_norm.expand_as(score).masked_fill(~(first | second), 0).max(-1).values
        return fnorm(2*largest*tv) * fnorm(self.weights[layer][3])

    def stage(self, x, kind, layer, mask=None):
        if kind == "attention":
            return x + self.attention(x, layer, mask)
        if kind == "mlp":
            up, down = self.weights[layer][4:]
            return x + torch.tanh(self.norm(x) @ up) @ down
        return self.norm(x)

    def uniform_bounds(self, x, radius, kind, layer):
        """Whole Frobenius-ball L,H and all routing-gap Lipschitz bounds.

        Norm map: a²=max(min row norm-radius,0)²/d+eps;
        L_N<=1/a,H_N<=6/(sqrt(d)*a²). Softmax L<=1,H<=8.
        Products and compositions use operator-norm upper bounds from Frobenius
        weight norms. Normalized state norm <=sqrt(n*d) globally.
        """
        min_norm = float(torch.linalg.vector_norm(x, dim=-1).min())
        a = math.sqrt(max(min_norm-radius, 0)**2/self.d + self.eps)
        ln, hn = 1/a, 6/(math.sqrt(self.d)*a*a)
        if kind == "norm":
            return ln, hn, 0.0
        if kind == "mlp":
            up, down = map(fnorm, self.weights[layer][4:])
            return 1+down*up*ln, 2*down*up*up*ln*ln+down*up*hn, 0.0
        wq, wk, wv, wo = map(fnorm, self.weights[layer][:4])
        u = math.sqrt(self.n*self.d)
        ls, hs = 2*self.beta*wq*wk*u, 2*self.beta*wq*wk
        la = self.heads*wo*(ls*wv*u + math.sqrt(self.n)*wv)
        ha = self.heads*wo*((8*ls*ls+hs)*wv*u + 2*ls*wv)
        return 1+la*ln, ha*ln*ln+la*hn, 2*ls*ln

    def readouts(self):
        result = np.zeros((self.vocab, self.n*self.d))
        result[:, -self.d:] = self.lm_head.T.numpy()
        return result


def run_trial(seed=0, strength=0.01, **dimensions):
    start = time.perf_counter()
    model = ReferenceTransformer(seed=seed, **dimensions)
    x, y = model.input.clone(), model.input.clone()
    radius = 0.0
    evidence, observations = [], []
    work = dict(reference_stages=0, candidate_stages=0, forced_branch_replays=0,
                jacobian_evaluations=0, jacobian_output_rows=0, guard_evaluations=0,
                routing_scores=0, causal_routing_scores=0, attention_key_reads=0, union_key_reads=0,
                dense_attention_score_entries=0,
                tail_summary_key_reads=0, bound_evaluations=0)
    schedule = [(kind, layer) for layer in range(model.layers)
                for kind in ("attention", "mlp")] + [("norm", 0)]
    for kind, layer in schedule:
        if kind == "attention":
            route_x, trace_x = model.route(x, layer)
            route_y, trace_y = model.route(y, layer)
            same_trace = [t[:5] for t in trace_x] == [t[:5] for t in trace_y]
            work["routing_scores"] += 2*model.heads*model.n**2
            work["causal_routing_scores"] += 2*model.heads*model.n*(model.n+1)//2
            # Two routers, three stage evaluations, Jacobian forward, correction,
            # plus a union-read evaluation only when the two masks differ.
            work["dense_attention_score_entries"] += (7+int(not torch.equal(route_x,route_y)))*model.heads*model.n**2
            work["guard_evaluations"] += len(trace_x)
            work["attention_key_reads"] += int(route_x.sum()+route_y.sum()+route_x.sum())
            work["union_key_reads"] += int((route_x | route_y).sum())
            visible = model.heads*model.n*(model.n+1)//2
            work["tail_summary_key_reads"] += visible-int(route_y.sum())
        else:
            route_x = route_y = None
            trace_x, same_trace = [], True
        f = lambda flat: model.stage(flat.reshape_as(x), kind, layer, route_x).reshape(-1)
        jacobian = torch.autograd.functional.jacobian(f, x.reshape(-1), vectorize=True)
        reference_next = model.stage(x, kind, layer, route_x)
        reference_at_candidate = model.stage(y, kind, layer, route_x)
        candidate_branch = model.stage(y, kind, layer, route_y)
        correction = (model.correction(y, layer, route_y, strength) if kind == "attention"
                      else torch.zeros_like(y))
        candidate_next = candidate_branch + correction
        jump = candidate_branch-reference_at_candidate
        uniform_lipschitz, curvature, gap_lipschitz = model.uniform_bounds(x, radius, kind, layer)
        jnorm = float(torch.linalg.matrix_norm(jacobian, ord=2))
        c = fnorm(correction)
        b = model.union_jump_bound(y, layer, route_x, route_y) if kind == "attention" else 0.0
        next_radius = jnorm*radius+c+b+curvature/2*radius*radius
        if not math.isfinite(next_radius):
            return dict(seed=seed,strength=strength,accepted=False,reason="nonfinite uniform tube",
                        work=work,seconds=time.perf_counter()-start)
        guards = tuple(Guard(t[5], gap_lipschitz) for t in trace_x)
        to_np = lambda a: a.detach().numpy().reshape(-1)
        stage = StageEvidence(reference=to_np(x),candidate=to_np(y),
            reference_next=to_np(reference_next),candidate_next=to_np(candidate_next),
            jacobian=jacobian.detach().numpy(),correction=to_np(correction),jump=to_np(jump),
            reference_at_candidate=to_np(reference_at_candidate),
            candidate_branch_at_candidate=to_np(candidate_branch),radius=radius,next_radius=next_radius,
            correction_bound=c,jump_bound=b,curvature=curvature,guards=guards,
            same_trace=same_trace,bounds_source="analytic RMSNorm/product/softmax/tanh Frobenius-ball bounds")
        evidence.append(stage)
        remainder = reference_at_candidate-reference_next-(jacobian @ (y-x).reshape(-1)).reshape_as(x)
        observations.append(dict(kind=kind,layer=layer,radius=radius,next_radius=next_radius,
            actual_error=fnorm(y-x),curvature=curvature,jacobian_norm=jnorm,correction_norm=c,
            uniform_lipschitz=uniform_lipschitz,guard_lipschitz=gap_lipschitz,
            jump_norm=fnorm(jump),union_jump_bound=b,same_trace=same_trace,comparison_count=len(guards),
            trace_guarded=all(t[5]>gap_lipschitz*radius for t in trace_x),
            taylor_remainder=fnorm(remainder),taylor_bound=curvature/2*radius*radius))
        for key in ("reference_stages","candidate_stages","forced_branch_replays",
                    "jacobian_evaluations","bound_evaluations"):
            work[key] += 1
        work["jacobian_output_rows"] += model.n*model.d
        x,y,radius = reference_next,candidate_next,next_radius
    # Matrix implementations materialize masked future scores too. These counts
    # separate useful causal reads from actual dense score-matrix entries; they
    # are not a FLOP count for backward-mode Jacobian construction.
    work["jacobian_scalar_entries"] = len(schedule)*(model.n*model.d)**2
    result = certify_path(evidence,model.readouts())
    logits_x,logits_y = model.readouts() @ x.reshape(-1).numpy(), model.readouts() @ y.reshape(-1).numpy()
    result.update(seed=seed,strength=strength,dimensions={k:getattr(model,k) for k in
        ("n","d","heads","layers","top_k","vocab")},
        reference_prediction=int(logits_x.argmax()),candidate_prediction=int(logits_y.argmax()),
        token_ids=model.token_ids.tolist(),weight_scale=0.2,rms_epsilon=model.eps,
        prediction_preserved=bool(logits_x.argmax()==logits_y.argmax()),
        max_logit_change=float(np.max(np.abs(logits_x-logits_y))),stages=observations,work=work,
        decision="accept_correction" if result["accepted"] else "reference_fallback",
        deployed_logits=(logits_y if result["accepted"] else logits_x).tolist(),
        interval_max_deficit=max([0.0]+[abs(m["actual_candidate_margin"]-m["reference_margin"]-
            m["signed_change"])-m["remainder_bound"] for m in result["margins"]]),
        interval_violations=sum(abs(m["actual_candidate_margin"]-m["reference_margin"]-
            m["signed_change"]) > m["remainder_bound"]+1e-12 for m in result["margins"]),
        taylor_max_deficit=max([0.0]+[s["taylor_remainder"]-s["taylor_bound"] for s in observations]),
        tube_max_deficit=max([0.0]+[s["actual_error"]-s["radius"] for s in observations]),
        seconds=time.perf_counter()-start)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seeds",default="0,1,2,3")
    parser.add_argument("--strengths",default="0,0.0001,0.01,0.25,1,4,64")
    parser.add_argument("--out",default="runs/routed_certificate_reference.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    for seed in map(int,args.seeds.split(",")):
        for strength in map(float,args.strengths.split(",")):
            row = run_trial(seed,strength)
            rows.append(row)
            print(json.dumps({k:row.get(k) for k in
                ("seed","strength","accepted","prediction_preserved","reason","seconds")}),flush=True)
    artifact = dict(scope="Random complete causal two-layer multihead transformer with token/position embeddings; weight scale .2 and RMS epsilon .5 are toy bound-friendly settings, not Qwen. Full-vocabulary reference prediction, not accuracy. Analytic real-arithmetic bounds evaluated in float64, not rounding certified. Dense all-score diagnostic; no runtime/subquadratic claim.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        core_sha256=hashlib.sha256(Path(__file__).with_name("routed_correction_certificate.py").read_bytes()).hexdigest(),
        torch_version=torch.__version__,device="cpu",dtype="float64",
        command="python -m ssa.routed_certificate_experiment " + shlex.join(sys.argv[1:]),rows=rows,
        summary=dict(trials=len(rows),accepted=sum(r["accepted"] for r in rows),
            algebraic_margin_passed=sum(r.get("algebraic_margin_passed",False) for r in rows),
            endpoint_oracle_vetoes=sum(r.get("algebraic_margin_passed",False) and not r["accepted"] for r in rows),
            nonzero_accepted=sum(r["accepted"] and r["strength"]!=0 for r in rows),
            observed_prediction_changes=sum(not r.get("prediction_preserved",True) for r in rows),
            interval_max_deficit=max(r.get("interval_max_deficit",0) for r in rows),
            interval_violations=sum(r.get("interval_violations",0) for r in rows),
            taylor_max_deficit=max(r.get("taylor_max_deficit",0) for r in rows),
            tube_max_deficit=max(r.get("tube_max_deficit",0) for r in rows),
            accepted_prediction_violations=sum(r["accepted"] and not r.get("prediction_preserved",False) for r in rows)))
    path.write_text(json.dumps(artifact,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
