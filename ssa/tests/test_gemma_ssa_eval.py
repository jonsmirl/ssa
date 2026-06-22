"""
Mechanics tests for the NIAH eval harness (ssa/gemma_ssa_eval.py): prompt assembly, depth control,
and exact-match scoring. No model/tokenizer — these lock in the harness logic so the GPU run is
trusted. The real-model dry-run lives in `python -m ssa.gemma_ssa_eval`.
"""
from ssa.gemma_ssa_eval import make_niah_text, score_continuation, QUESTION


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
