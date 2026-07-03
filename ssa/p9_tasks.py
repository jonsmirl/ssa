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
    """Single-hop MQAR where the queried pair uses a reserved SALIENT key (token 0), always present and always
    the query. Distractor pairs use ordinary keys. Because the target pair is marked at write time, a learned
    write gate can keep it even at overload — the write-salient regime."""

    def __init__(self, n_keys=64, n_vals=64, seed=0):
        self.NK, self.NV = n_keys, n_vals
        self.SEP = n_keys + n_vals
        self.PAD = n_keys + n_vals + 1
        self.vocab = n_keys + n_vals + 2
        self.MARK = 0                                                   # the reserved salient key
        self.rng = np.random.default_rng(seed)

    def batch(self, bs, n_pairs, n_queries=1, needle_dist=None, device=DEV):
        L = 2 * n_pairs + 1 + 1
        ids = np.full((bs, L), self.PAD, np.int64)
        tgt = np.full((bs, L), -100, np.int64)
        for b in range(bs):
            keys = self.rng.permutation(np.arange(1, self.NK))[:n_pairs - 1]    # distractor keys (≠ MARK)
            keys = np.concatenate([[self.MARK], keys])                          # the salient pair first...
            vals = self.rng.integers(0, self.NV, n_pairs) + self.NK
            order = self.rng.permutation(n_pairs)                               # ...then shuffle its position
            keys, vals = keys[order], vals[order]
            ids[b, 0:2 * n_pairs:2] = keys
            ids[b, 1:2 * n_pairs:2] = vals
            ids[b, 2 * n_pairs] = self.SEP
            j = int(np.where(keys == self.MARK)[0][0])                          # query the salient pair
            ids[b, 2 * n_pairs + 1] = self.MARK
            tgt[b, 2 * n_pairs + 1] = vals[j]
        return torch.tensor(ids, device=device), torch.tensor(tgt, device=device)


class MQAR2Hop:
    """Two-hop chained recall in a UNIFIED vocab. Two pairs form the chain — (k1→k2) and (k2→v2) — among
    `n_pairs-2` distractor pairs; the query presents k1 and the target is v2 (requires traversing k1→k2→v2).
    `n_pairs ≥ 2`, `n_tokens ≥ 2·n_pairs + 2`."""

    def __init__(self, n_tokens=96, seed=0):
        self.V = n_tokens
        self.SEP = n_tokens
        self.PAD = n_tokens + 1
        self.vocab = n_tokens + 2
        self.rng = np.random.default_rng(seed)

    def batch(self, bs, n_pairs, n_queries=1, needle_dist=None, device=DEV):
        assert n_pairs >= 2 and self.V >= 2 * n_pairs + 2, (n_pairs, self.V)
        L = 2 * n_pairs + 1 + 1
        ids = np.full((bs, L), self.PAD, np.int64)
        tgt = np.full((bs, L), -100, np.int64)
        for b in range(bs):
            toks = self.rng.permutation(self.V)
            k1, k2, v2 = toks[0], toks[1], toks[2]                              # the chain tokens (distinct)
            pairs = [(k1, k2), (k2, v2)]                                        # v1 == k2 (the shared token)
            dk = toks[3:3 + (n_pairs - 2)]                                      # distractor keys (distinct)
            dv = toks[3 + (n_pairs - 2):3 + 2 * (n_pairs - 2)]                  # distractor values (distinct)
            pairs += list(zip(dk.tolist(), dv.tolist()))
            self.rng.shuffle(pairs)                                             # chain pairs at random positions
            for j, (kk, vv) in enumerate(pairs):
                ids[b, 2 * j] = kk; ids[b, 2 * j + 1] = vv
            ids[b, 2 * n_pairs] = self.SEP
            ids[b, 2 * n_pairs + 1] = k1                                        # query the chain head
            tgt[b, 2 * n_pairs + 1] = v2                                        # 2-hop target
        return torch.tensor(ids, device=device), torch.tensor(tgt, device=device)
