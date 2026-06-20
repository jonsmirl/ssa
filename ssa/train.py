"""
Retrieval-margin demonstrator — the trained-keys crux (experiment (i)).

The synthetic harness (experiments.py) showed a sublinear LSH selector is lossless WHEN keys are
separated. The open question it could not settle: does *training* actually drive a model's keys
toward that geometry?  Here we train a small encoder on a content-addressable retrieval task and
measure, on HELD-OUT items at LARGER scale than training, whether its learned keys let the LSH /
centroid selector stay lossless at sublinear cost — compared against an untrained (random) encoder.

Task: N items have raw features x_i; a query is a *corrupted* cue (40% of features masked + noise).
The encoder f_theta maps features -> unit keys; retrieval is softmax(beta * q.K^T); the encoder is
trained (cross-entropy) to retrieve the right item under corruption. Keys for the probe are f(x) on
items the encoder never saw in training.

Run: python3 -m ssa.train
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import CentroidSelector, LSHSelector, coherence

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


class Encoder(nn.Module):
    """Small MLP: raw features -> L2-normalized key/query (unit sphere)."""
    def __init__(self, d_raw, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_raw, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)


def corrupt(x, mask_frac=0.4, noise=0.1, gen=None):
    """Imperfect cue: zero a random `mask_frac` of feature dims, add gaussian noise."""
    m = (torch.rand(x.shape, device=x.device, generator=gen) > mask_frac).float()
    return x * m + noise * torch.randn(x.shape, device=x.device, generator=gen)


def train_encoder(X_train, d, beta=20.0, pool=256, batch=128, steps=3000, lr=1e-3,
                  prox_weight=0.0):
    """Train the retrieval encoder. With prox_weight>0, ALSO co-train for selector-friendliness:
    pull the corrupted cue toward its own clean key in angle (1 - cos(q, k_target)). This is the one
    term (i) showed was missing — it makes the cue an angular near-neighbor of the key, so the
    *unchanged* known LSH selector can find it. Retrieval CE keeps distinct keys apart (no collapse)."""
    enc = Encoder(X_train.shape[1], d).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    Ntr = len(X_train)
    for step in range(steps):
        idx = torch.randint(0, Ntr, (pool,), device=DEV)          # a retrieval pool
        Xp = X_train[idx]
        K = enc(Xp)                                                # (pool, d) keys
        qi = torch.randint(0, pool, (batch,), device=DEV)          # which items to query
        q = enc(corrupt(Xp[qi]))                                   # corrupted cues -> queries
        logits = beta * (q @ K.t())                                # (batch, pool)
        loss = F.cross_entropy(logits, qi)
        if prox_weight > 0.0:
            prox = (1.0 - (q * K[qi]).sum(-1)).mean()             # 1 - cos(cue, its clean key)
            loss = loss + prox_weight * prox
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 750 == 0 or step == steps - 1:
            acc = (logits.argmax(1) == qi).float().mean().item()
            print(f"    step {step:5d}  loss {loss.item():.3f}  train-pool acc {acc:.3f}")
    return enc


@torch.no_grad()
def encode_keys(enc, X):
    return enc(X).cpu().numpy().astype(np.float32)


@torch.no_grad()
def probe(enc, X_test, label, d, beta=20.0, k=64, trials=400, mask_frac=0.4, noise=0.1, seed=0):
    """On UNSEEN items at pool size N=len(X_test): exact retrieval acc, and LSH/centroid selector
    recall + cost + end-to-end acc. Reports key coherence too."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    K = encode_keys(enc, X_test)                                   # (N, d) unit keys
    n = len(K)
    eps = coherence(K)
    sels = [CentroidSelector(seed=seed).build(K), LSHSelector(L=12, bits=11, seed=seed).build(K)]
    exact_hits = 0
    s = {se.name: dict(rec=0, cost=0, e2e=0) for se in sels}
    for _ in range(trials):
        t = int(rng.integers(n))
        q = enc(corrupt(X_test[t:t+1], mask_frac, noise, gen))[0].cpu().numpy().astype(np.float32)
        scores = K @ q
        exact_hits += int(np.argmax(scores) == t)                 # dense retrieval correct?
        for se in sels:
            cand, cost = se.select(q, beta, k)
            st = s[se.name]; st["cost"] += cost
            if cand.size:
                st["rec"] += int(t in set(cand.tolist()))
                # end-to-end: retrieve = argmax over the selected candidates only
                st["e2e"] += int(cand[np.argmax(K[cand] @ q)] == t)
    print(f"  [{label}]  N={n}  key-coherence eps={eps:.3f}  exact dense acc={exact_hits/trials:.3f}")
    print(f"    {'selector':>9} {'sel-recall':>11} {'keys scored':>12} {'frac n':>8} {'end-to-end acc':>15}")
    out = {}
    for se in sels:
        st = s[se.name]
        print(f"    {se.name:>9} {st['rec']/trials:>11.3f} {st['cost']/trials:>12.0f} "
              f"{st['cost']/trials/n:>8.3f} {st['e2e']/trials:>15.3f}")
        out[se.name] = (st['rec']/trials, st['cost']/trials/n, st['e2e']/trials)
    return exact_hits/trials, eps, out


def main():
    print("=" * 80)
    print("TRAINED-KEYS CRUX: does training drive keys to be selector-friendly? (held-out items)")
    print("=" * 80)
    d_raw, d = 64, 128
    N_total = 24000
    X = torch.randn(N_total, d_raw, device=DEV)
    X = X / X.norm(dim=-1, keepdim=True)
    X_train, X_test = X[:12000], X[12000:]                         # disjoint items

    print("\n[1] untrained (random-init) encoder — baseline:")
    enc0 = Encoder(d_raw, d).to(DEV)
    for N in (2000, 10000):
        probe(enc0, X_test[:N], "untrained", d)

    print("\n[2] training the encoder on retrieval (under corrupted cues)...")
    enc = train_encoder(X_train, d)

    print("\n[3] trained encoder on UNSEEN items, scaling N beyond the training pool (256):")
    rows = []
    for N in (2000, 10000, 12000):
        acc, eps, out = probe(enc, X_test[:N], "trained, corrupted cue", d)
        rows.append((N, acc, out))

    print("\n[4] control — SAME trained encoder, NEAR-CLEAN cue (mask 0, noise 0.02):")
    print("    (isolates whether post-hoc LSH fails because of training or because of cue quality)")
    _, _, clean_out = probe(enc, X_test[:10000], "trained, near-clean cue", d,
                            mask_frac=0.0, noise=0.02)

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    a0 = rows[0][2]["lsh"]; aN = rows[-1][2]["lsh"]
    print(f"  trained LSH selector @ N={rows[0][0]}: recall {a0[0]:.3f}, cost {a0[1]*100:.1f}% of n, "
          f"end-to-end acc {a0[2]:.3f}")
    print(f"  trained LSH selector @ N={rows[-1][0]} (unseen, larger): recall {aN[0]:.3f}, "
          f"cost {aN[1]*100:.1f}% of n, end-to-end acc {aN[2]:.3f}")
    print("  -> if trained recall/acc stay high at sublinear cost on UNSEEN items where the untrained")
    print("     baseline is poor, training produced the selector-friendly geometry the theory needs.")


if __name__ == "__main__":
    main()
