"""
P9 task suite — MQAR variants that span the P8 regimes, tokenized for the trained micro-LM.

- single-hop load sweep: reuse `ssa_demo.MQAR` directly (the load m = `n_pairs`).
- `MQARSalient`: single-hop, but the queried pair's key is a reserved SALIENT marker token — so the target
  is identifiable AT WRITE TIME (a learned write gate CAN learn to keep marked pairs). The write-salient
  arm of the read-vs-write-time relevance test (P8-P3).
- `MQAR2Hop`: a chained retrieval in a UNIFIED key/value vocab (so a value can BE a key): pair (k1→k2) and
  pair (k2→v2); query k1, target v2. The composition-law arm (P8-P6).

All emit `(ids, tgt)` with `tgt = -100` except query positions (masked cross-entropy), matching
`ssa_demo.MQAR` so `p9_microlm.train_model`/`recall` drive them unchanged.
"""
from __future__ import annotations
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MQARSalient:
    """Single-hop MQAR where a reserved SET of marker keys {0..K-1} are the queried needles — the keep-worthy
    pairs are thus identifiable AT WRITE TIME (a learned write gate can keep exactly them, discarding the
    distractors, even at overload m>dh). Each sequence plants K salient pairs (distinct markers) among
    n_pairs-K distractors and queries ALL K markers — multi-query, so the gradient is dense enough to grok
    (the same recipe MQAR needs). The write-salient regime; contrast: stock MQAR is read-salient.

    K (n_markers) should be ≤ the memory's state dim dh so the gate CAN fit the kept set — the whole point:
    at overload the gate keeps the K≤dh markers and drops the distractors, where a keep-everything write
    (no gate) overflows. `n_pairs ≥ K`."""

    def __init__(self, n_keys=256, n_vals=256, n_markers=8, seed=0):
        self.NK, self.NV = n_keys, n_vals
        self.K = n_markers                                             # reserved salient keys 0..K-1
        self.SEP = n_keys + n_vals
        self.PAD = n_keys + n_vals + 1
        self.vocab = n_keys + n_vals + 2
        self.MARK = 0                                                  # (kept for reference; markers are 0..K-1)
        self.rng = np.random.default_rng(seed)

    def batch(self, bs, n_pairs, n_queries=1, needle_dist=None, device=DEV):
        assert n_pairs >= self.K, (n_pairs, self.K)
        L = 2 * n_pairs + 1 + self.K                                   # K query slots (multi-query)
        ids = np.full((bs, L), self.PAD, np.int64)
        tgt = np.full((bs, L), -100, np.int64)
        for b in range(bs):
            markers = self.rng.permutation(self.K)                     # the K salient keys (0..K-1)
            distract = self.rng.permutation(np.arange(self.K, self.NK))[:n_pairs - self.K]
            keys = np.concatenate([markers, distract])
            vals = self.rng.integers(0, self.NV, n_pairs) + self.NK
            order = self.rng.permutation(n_pairs)                      # shuffle salient among distractors
            keys, vals = keys[order], vals[order]
            ids[b, 0:2 * n_pairs:2] = keys
            ids[b, 1:2 * n_pairs:2] = vals
            ids[b, 2 * n_pairs] = self.SEP
            for qi in range(self.K):                                   # query every marker key
                j = int(np.where(keys == qi)[0][0])
                pos = 2 * n_pairs + 1 + qi
                ids[b, pos] = qi
                tgt[b, pos] = vals[j]
        return torch.tensor(ids, device=device), torch.tensor(tgt, device=device)


class MQAR2Hop:
    """Two-hop chained recall in a UNIFIED vocab (a value can BE a key). Each of C chains contributes two
    pairs — (k1→k2) and (k2→v2), with v1==k2 the shared token — among distractor pairs; the query presents
    each chain head k1 and the target is v2 (requires traversing k1→k2→v2). ALL C heads are queried per
    sequence (multi-query, so the gradient is dense enough to grok — single-query composition does not).
    Load = total #pairs = 2·C + #distractors, so `n_pairs ≥ 2·C` and `n_tokens ≥ 2·n_pairs + 2`."""

    def __init__(self, n_tokens=192, n_chains=4, seed=0):
        self.V = n_tokens
        self.C = n_chains
        self.SEP = n_tokens
        self.PAD = n_tokens + 1
        self.vocab = n_tokens + 2
        self.rng = np.random.default_rng(seed)

    def batch(self, bs, n_pairs, n_queries=1, needle_dist=None, device=DEV):
        assert n_pairs >= 2 * self.C and self.V >= 2 * n_pairs + 2, (n_pairs, self.C, self.V)
        L = 2 * n_pairs + 1 + self.C                                           # C query slots (multi-query)
        ids = np.full((bs, L), self.PAD, np.int64)
        tgt = np.full((bs, L), -100, np.int64)
        for b in range(bs):
            toks = self.rng.permutation(self.V); i = 0
            pairs, heads, finals = [], [], []
            for _ in range(self.C):
                k1, k2, v2 = toks[i], toks[i + 1], toks[i + 2]; i += 3         # one chain's 3 distinct tokens
                pairs += [(k1, k2), (k2, v2)]                                  # (k1→k2), (k2→v2); v1==k2
                heads.append(k1); finals.append(v2)
            nd = n_pairs - 2 * self.C                                          # remaining are distractor pairs
            dk = toks[i:i + nd]; dv = toks[i + nd:i + 2 * nd]
            pairs += list(zip(dk.tolist(), dv.tolist()))
            self.rng.shuffle(pairs)                                            # chains at random positions
            for j, (kk, vv) in enumerate(pairs):
                ids[b, 2 * j] = kk; ids[b, 2 * j + 1] = vv
            ids[b, 2 * n_pairs] = self.SEP
            for qi in range(self.C):
                pos = 2 * n_pairs + 1 + qi
                ids[b, pos] = heads[qi]; tgt[b, pos] = finals[qi]              # query head, target 2-hop end
        return torch.tensor(ids, device=device), torch.tensor(tgt, device=device)
