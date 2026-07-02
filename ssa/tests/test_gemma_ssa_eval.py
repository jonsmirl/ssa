"""
Mechanics tests for the NIAH eval harness (ssa/gemma_ssa_eval.py): prompt assembly, depth control,
and exact-match scoring. No model/tokenizer — these lock in the harness logic so the GPU run is
trusted. The real-model dry-run lives in `python -m ssa.gemma_ssa_eval`.
"""
from ssa.gemma_ssa_eval import (make_niah_text, score_continuation, QUESTION,
                                make_two_hop_text, score_word, Q_CHAIN, Q_HOP2)


def test_needle_present_and_question_last():
    text, gold = make_niah_text(42424, 0.5, 20)
    assert "secret access code is 42424" in text
    assert text.rstrip().endswith("the secret access code is")
    assert gold == "42424"


def test_depth_controls_needle_position():
    early, _ = make_niah_text(11111, 0.1, 40)
    late, _ = make_niah_text(11111, 0.9, 40)
    be = early.split(QUESTION)[0]
    bl = late.split(QUESTION)[0]
    pe = be.index("11111") / len(be)
    pl = bl.index("11111") / len(bl)
    assert pe < 0.3 < 0.7 < pl, (pe, pl)


def test_score_exact_match_only():
    assert score_continuation(" the secret access code is 42424.", "42424")
    assert score_continuation("42424", "42424")
    assert score_continuation("code is 42424, confirmed", "42424")
    assert not score_continuation(" ... 424240 ...", "42424")   # trailing digit
    assert not score_continuation(" ... 142424 ...", "42424")   # leading digit
    assert not score_continuation(" no number here at all", "42424")


def test_distinct_codes_do_not_false_hit():
    text, gold = make_niah_text(73219, 0.5, 10)
    # a different generated code must not score as a hit
    assert not score_continuation("the code is 84102", gold)
    assert score_continuation("the code is 73219", gold)


def test_two_hop_both_needles_present_and_no_leak():
    text, gold = make_two_hop_text("cobalt", 55123, 0.2, 0.6, 20)
    assert "vault keyword is cobalt" in text
    assert "keyword is cobalt holds the number 55123" in text
    assert gold == "55123"
    q = text.split("Question:")[-1]                                    # the chain question leaks neither hop
    assert "cobalt" not in q and "55123" not in q
    assert text.rstrip().endswith("the number is")


def test_two_hop_depth_order_and_clamp():
    kw_first, _ = make_two_hop_text("marble", 12345, 0.2, 0.8, 40)     # needle1 (keyword) before needle2
    body = kw_first.split("Question:")[0]
    assert body.index("keyword is marble.") < body.index("holds the number")
    kw_late, _ = make_two_hop_text("marble", 12345, 0.8, 0.2, 40)      # reversed causal order
    body2 = kw_late.split("Question:")[0]
    assert body2.index("holds the number") < body2.index("keyword is marble.")
    make_two_hop_text("marble", 12345, -1.0, 2.0, 10)                  # out-of-range depths must not raise


def test_hop2_question_reveals_keyword_only():
    text, gold = make_two_hop_text("juniper", 98765, 0.3, 0.6, 15, question=Q_HOP2)
    assert text.rstrip().endswith("juniper hold? Answer: the number is")
    assert gold == "98765"


def test_score_word_boundary():
    assert score_word("the keyword is walnut, I think", "walnut")
    assert score_word("WALNUT", "walnut")                              # case-insensitive
    assert not score_word("walnuts are tasty", "walnut")              # word boundary
    assert not score_word("nothing relevant", "walnut")
