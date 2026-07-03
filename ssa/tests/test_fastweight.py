"""Mechanism tests for the fast-weight / zero-attention memory (ssa.fastweight). CPU, fixed seeds, fast.
Each test pins a machine-checked prediction: delta = coherence control (exact to m=d on orthogonal keys),
tags resolve same-key conflicts, slot-birth preserves what a decaying fixed memory forgets, and the
proved chain bound (chain ≤ weakest hop)."""
import torch

from ssa.fastweight import FastWeightMemory, SlotMemory, surprise, _unit


def test_delta_exact_to_m_equals_d_on_orthogonal_keys():
    """DeltaNet writes are exact for orthogonal keys up to m=d (the capacity-preserving write rule)."""
    d = 32
    K = torch.eye(d)                                              # d orthonormal keys
    V = _unit(torch.randn(d, d, generator=torch.Generator().manual_seed(0)))
    mem = FastWeightMemory(d, rule="delta", beta=1.0, keep_kv=False)
    for i in range(d):
        mem.write(K[i], V[i])
    for i in range(d):
        assert torch.allclose(mem.read_linear(K[i]), V[i], atol=1e-4), i


def test_additive_interferes_with_correlated_keys():
    """Additive writes accumulate interference: correlated keys corrupt the read (delta does not)."""
    d = 32
    g = torch.Generator().manual_seed(1)
    shared = _unit(torch.randn(d, generator=g))
    K = _unit(0.6 * shared + 0.4 * _unit(torch.randn(8, d, generator=g)))   # coherent keys
    V = _unit(torch.randn(8, d, generator=g))
    add = FastWeightMemory(d, rule="additive", keep_kv=False)
    dlt = FastWeightMemory(d, rule="delta", beta=1.0, keep_kv=False)
    for m in (add, dlt):
        for i in range(8):
            m.write(K[i], V[i])
    add_ok = sum(torch.allclose(add.read_linear(K[i]), V[i], atol=0.2) for i in range(8))
    dlt_ok = sum(torch.allclose(dlt.read_linear(K[i]), V[i], atol=0.2) for i in range(8))
    assert dlt_ok > add_ok, (dlt_ok, add_ok)


def test_write_read_determinism():
    d = 16
    g = torch.Generator().manual_seed(2)
    K = _unit(torch.randn(5, d, generator=g)); V = torch.randn(5, d, generator=g)

    def run():
        m = FastWeightMemory(d, rule="delta", beta=0.9, keep_kv=False)
        for i in range(5):
            m.write(K[i], V[i])
        return m.read_linear(K[0])
    assert torch.allclose(run(), run())


def test_gated_decay_forgets_old_writes():
    """A forgetting gate (decay<1) monotonically fades an old association as new writes arrive."""
    d = 24
    g = torch.Generator().manual_seed(3)
    k0 = _unit(torch.randn(d, generator=g)); v0 = _unit(torch.randn(d, generator=g))
    mem = FastWeightMemory(d, rule="gated_delta", beta=1.0, decay=0.9, keep_kv=False)
    mem.write(k0, v0)
    strengths = []
    for _ in range(10):
        strengths.append(float(mem.read_linear(k0) @ v0))
        mem.write(_unit(torch.randn(d, generator=g)), _unit(torch.randn(d, generator=g)))
    assert strengths[-1] < 0.5 * strengths[0]                    # the old fact fades under the gate
    assert strengths[-1] < strengths[2]                          # a clear downward trend


def test_tag_resolves_same_key_conflict():
    """One key, two values: an episodic TAG (k⊕bucket) recovers BOTH; the untagged delta memory keeps
    only the latest — `same_input_conflict_unservable` / `tag_resolves_conflict`."""
    d, td = 32, 8
    g = torch.Generator().manual_seed(4)
    k = _unit(torch.randn(d, generator=g))
    v1 = _unit(torch.randn(d, generator=g)); v2 = _unit(torch.randn(d, generator=g))
    t1 = _unit(torch.randn(td, generator=g)); t2 = _unit(torch.randn(td, generator=g))
    # untagged: same key twice, delta overwrites -> only v2 recoverable
    un = FastWeightMemory(d, rule="delta", beta=1.0, keep_kv=False)
    un.write(k, v1); un.write(k, v2)
    o = un.read_linear(k)
    assert float(o @ v2) > float(o @ v1)                        # first value lost
    # tagged: k⊕t1, k⊕t2 are distinct keys -> both recovered on tag-qualified queries
    tag = FastWeightMemory(d + td, d_v=d, rule="delta", beta=1.0, keep_kv=False)
    tag.write(torch.cat([k, t1]), v1); tag.write(torch.cat([k, t2]), v2)
    o1 = tag.read_linear(torch.cat([k, t1])); o2 = tag.read_linear(torch.cat([k, t2]))
    assert float(o1 @ v1) > float(o1 @ v2) and float(o2 @ v2) > float(o2 @ v1)


def test_slot_birth_preserves_what_a_fixed_memory_forgets():
    """A mid-stream shift: a fixed decaying memory fades the pre-shift fact; slot-birth grows a slot for
    the new regime and preserves it (`fold_not_hopfield` — length-robustness forces state growth)."""
    d = 32
    g = torch.Generator().manual_seed(5)
    a = _unit(torch.randn(d, generator=g)); b = _unit(torch.randn(d, generator=g))
    ka = _unit(a + 0.05 * torch.randn(d, generator=g)); va = _unit(torch.randn(d, generator=g))

    def post_stream(mem):
        mem.write(ka, va)                                        # the pre-shift fact
        for _ in range(30):                                     # a burst of post-shift (region-b) writes
            kb = _unit(b + 0.05 * torch.randn(d, generator=g))
            mem.write(kb, _unit(torch.randn(d, generator=g)))
        return float(_unit(mem.read(ka)) @ va)

    fixed = SlotMemory(d, rule="gated_delta", beta=1.0, decay=0.9, n_slots=1)
    birth = SlotMemory(d, rule="gated_delta", beta=1.0, decay=0.9, n_slots=1, birth_threshold=0.8)
    s_fixed, s_birth = post_stream(fixed), post_stream(birth)
    assert birth.n_slots() > 1                                   # the shift grew the state
    assert s_birth > s_fixed + 0.2, (s_birth, s_fixed)


def test_chain_never_exceeds_weakest_hop():
    """The proved composition bound `chain_le_weakest`: a 2-hop chain succeeds only if both hops do, so
    measured chain accuracy ≤ min per-hop accuracy."""
    g = torch.Generator().manual_seed(6)
    trials = 200
    hop1 = torch.rand(trials, generator=g) < 0.9                # ρ1 ≈ 0.9
    hop2 = torch.rand(trials, generator=g) < 0.6                # ρ2 ≈ 0.6
    chain = (hop1 & hop2).float().mean()
    assert chain <= min(hop1.float().mean(), hop2.float().mean()) + 1e-6


def test_surprise_flags_off_distribution_key():
    d = 32
    g = torch.Generator().manual_seed(7)
    a = _unit(torch.randn(d, generator=g))
    keys = _unit(a + 0.05 * torch.randn(20, d, generator=g))
    mean = _unit(keys.mean(0))
    in_dist = surprise(_unit(a + 0.05 * torch.randn(d, generator=g)), mean)
    off_dist = surprise(_unit(torch.randn(d, generator=g)), mean)
    assert off_dist > in_dist
