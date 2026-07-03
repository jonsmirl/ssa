"""
Fast-weight / zero-attention memory — the OTHER corner of the trilemma (compression, not selection).

The P0–P7 programs measured the SELECTION corner (route to the few keys that matter, at read time).
This is the compression corner: a fixed- (or growing-) size state written at inference time — the
Mamba/DeltaNet/Titans family, what SubQ's CTO calls "zero attention". These are small, EXACT reference
implementations (d ≤ 128 states, no training) built to be measured against machine-checked predictions,
not to be a competitive SSM.

Theory anchors (machine-checked, sorry-free, in Substrate/Inference/*.lean — cited, not shipped here):
  - the READ rule sets the capacity class: `softmax_capacity` (RetrievalMarginRecognition) proves a
    softmax read recovers n < e^{β(1−ε)} patterns (exponential); a linear read o = S q is rank-d capped.
  - the WRITE rule is coherence control: `capacity_search_tension` (SearchTradeoff) — separation buys
    capacity; the delta rule (erase-before-write) maintains the near-orthogonality that keeps you on the
    exponential curve. A contracted read never searches, so the search penalty vanishes for this corner.
  - `forgetting_requires_shared_support` / `tag_resolves_conflict` (ContinualLearning): disjoint-support
    (owned/slotted, or episodically TAGGED) writes preserve old memories exactly; one map cannot store
    two values for one key — a tag resolves it.
  - `fold_not_hopfield` (Hopfield) + `detectability_is_a_fold` (Detectability): a fixed-dimension
    settling memory cannot track a regime shift; length-robustness across shifts forces the state to GROW.

Honest scope: reference implementations for measurement, not performance kernels; synthetic keys/values;
recall is the standard associative-memory decode (argmax over stored value embeddings).

Run the experiments:  python3 -m ssa.fastweight_capacity | fastweight_recall | fastweight_shift
"""
from __future__ import annotations
import torch

DEV = "cpu"                                                        # d≤128 mechanism probes — CPU is plenty


def _unit(x, dim=-1, eps=1e-9):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class FastWeightMemory:
    """A single associative state S (d×d), read by contraction o = S q (linear) — the zero-attention read.
    Also keeps the written (k,v) list for a `read_softmax` COMPARISON only (the linear read never sees it).

    rule: how a (k,v) pair is written into S —
      'additive'    : S += v kᵀ                          (vanilla linear attention; interference accumulates)
      'delta'       : S += β (v − S k) kᵀ                (DeltaNet: erase the old association for k, then write)
      'gated_delta' : S ← decay·(S + β (v − S k) kᵀ)     (Gated DeltaNet: a forgetting gate on the state)
    """

    def __init__(self, d, d_v=None, rule="additive", beta=1.0, decay=0.99, keep_kv=True, device=DEV):
        self.d, self.d_v = d, d_v or d                            # key dim, value dim (may differ: episodic tags)
        self.rule, self.beta, self.decay = rule, beta, decay
        self.S = torch.zeros(self.d_v, self.d, device=device)     # read o = S q̂  ∈ R^{d_v}
        self.keep_kv = keep_kv
        self.K, self.V = [], []                                   # for the softmax-read comparison ONLY
        self.device = device

    def write(self, k, v):
        k = _unit(k.to(self.device).float())
        v = v.to(self.device).float()
        if self.rule == "additive":
            self.S = self.S + torch.outer(v, k)
        elif self.rule == "delta":
            self.S = self.S + self.beta * torch.outer(v - self.S @ k, k)
        elif self.rule == "gated_delta":
            self.S = self.decay * (self.S + self.beta * torch.outer(v - self.S @ k, k))
        else:
            raise ValueError(self.rule)
        if self.keep_kv:
            self.K.append(k); self.V.append(v)

    def read_linear(self, q):
        """The zero-attention read: contract the state. o = S q̂. Never touches the stored (k,v) list."""
        return self.S @ _unit(q.to(self.device).float())

    def read_softmax(self, q, beta=8.0):
        """The attention read over the SAME stored pairs — for the read-rule capacity comparison (P1).
        `beta` here is the READ temperature (softmax sharpness), independent of the delta WRITE step above."""
        if not self.K:
            return torch.zeros(self.d_v, device=self.device)
        K = torch.stack(self.K); V = torch.stack(self.V)          # (m,d)
        w = torch.softmax(beta * (K @ _unit(q.to(self.device).float())), dim=0)
        return w @ V

    def state_floats(self):
        return self.d * self.d


def surprise(k, running_mean):
    """Write-time salience = residual of the (unit) key from the running key mean (the leverage idea from
    cascade_router._extract_outliers). Large ⇒ the key does not fit the established geometry."""
    return float((_unit(k.float()) - running_mean).norm())


class SlotMemory:
    """A partitioned memory: keys route to slots, each slot an independent FastWeightMemory (disjoint
    support ⇒ writes to one slot cannot corrupt another — `forgetting_requires_shared_support`). With
    `birth_threshold` set, a write whose surprise exceeds it BIRTHS a new slot (bounded by max_slots) —
    the growing state a fold forces (`fold_not_hopfield`). Read routes the query to its nearest slot."""

    def __init__(self, d, rule="additive", beta=1.0, decay=0.99, n_slots=1,
                 birth_threshold=None, max_slots=32, device=DEV):
        self.d, self.rule, self.beta, self.decay = d, rule, beta, decay
        self.birth_threshold, self.max_slots, self.device = birth_threshold, max_slots, device
        self.mems = [FastWeightMemory(d, rule=rule, beta=beta, decay=decay, keep_kv=False, device=device)
                     for _ in range(max(1, n_slots))]
        self.centroids = [None] * len(self.mems)                  # running key mean per slot
        self.counts = [0] * len(self.mems)

    def _route(self, kk):
        dists = [float((kk - c).norm()) if c is not None else float("inf") for c in self.centroids]
        j = int(torch.tensor(dists).argmin())
        return j, dists[j]

    def write(self, k, v):
        kk = _unit(k.to(self.device).float())
        j, dmin = self._route(kk)
        if (self.birth_threshold is not None and dmin > self.birth_threshold
                and len(self.mems) < self.max_slots):            # surprise ⇒ birth a new slot
            self.mems.append(FastWeightMemory(self.d, rule=self.rule, beta=self.beta, decay=self.decay,
                                              keep_kv=False, device=self.device))
            self.centroids.append(kk.clone()); self.counts.append(0)
            j = len(self.mems) - 1
        else:
            c = self.centroids[j]
            self.centroids[j] = kk.clone() if c is None else (c * self.counts[j] + kk) / (self.counts[j] + 1)
        self.counts[j] += 1
        self.mems[j].write(k, v)

    def read(self, q):
        qq = _unit(q.to(self.device).float())
        j, _ = self._route(qq)
        return self.mems[j].read_linear(q)

    def n_slots(self):
        return len(self.mems)

    def state_floats(self):
        return len(self.mems) * self.d * self.d


# -- shared measurement helpers (used by all P8 experiment modules) ---------------------------------

def random_keys(m, d, coherence=None, g=None):
    """m unit key vectors. coherence=None ⇒ i.i.d. random (ε ≈ √(2 log m / d)); a float ⇒ correlated
    keys sharing a common direction so pairwise inner products are ~coherence (P2's write-rule stress)."""
    g = g or torch.Generator(device=DEV)
    base = _unit(torch.randn(m, d, generator=g, device=DEV))
    if coherence is None:
        return base
    shared = _unit(torch.randn(d, generator=g, device=DEV))
    a = float(coherence) ** 0.5
    return _unit(a * shared + (1 - a) * base)                     # tunable common component


def decode_recall(reads, values, targets):
    """Standard associative-memory recall: predicted token = argmax_i ⟨read, value_i⟩ over stored values."""
    reads = _unit(torch.stack(reads))                            # (q, d)
    V = _unit(values)                                            # (m, d)
    pred = (reads @ V.T).argmax(1)
    return float((pred == torch.as_tensor(targets, device=pred.device)).float().mean())
