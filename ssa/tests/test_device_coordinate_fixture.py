"""CPU tests of deterministic natural-text growth-fixture assembly."""
import hashlib

import pytest

from ssa.device_coordinate_fixture import article_stream, build_stream


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_article_stream_skips_both_previous_articles_without_repetition():
    rows = [" = A = ", "one", " = Ise @-@ class battleship = ", "excluded",
            " = Second Battle of Naktong Bulge = ", "excluded", " = B = ", "two"]
    articles = list(article_stream(rows))
    assert [a["title"] for a in articles] == ["A", "B"]
    assert [(a["row_start"], a["row_end_exclusive"]) for a in articles] == [(0, 2), (6, 8)]


def test_stream_truncates_natural_prefix_and_records_source_segments():
    rows = [" = A = ", "one", " = B = ", "two"]
    first = " = A = \none"
    full = first + "\n\n = B = \ntwo"
    ids, meta = build_stream(rows, CharacterTokenizer(), len(first) + 7)
    reconstructed = "".join(map(chr, ids))
    assert reconstructed == full[:len(ids)]
    assert meta["used_text_prefix_sha256"] == hashlib.sha256(reconstructed.encode()).hexdigest()
    assert meta["articles"][1]["character_start"] == len(first) + 2
    assert meta["articles"][0]["truncated_by_token_limit"] is False
    assert meta["articles"][1]["truncated_by_token_limit"] is True


def test_duplicate_title_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        list(article_stream([" = A = ", "x", " = A = ", "y"]))


@pytest.mark.parametrize("n", [0, 100])
def test_invalid_or_insufficient_length_rejected(n):
    with pytest.raises(ValueError):
        build_stream([" = A = ", "x"], CharacterTokenizer(), n)
