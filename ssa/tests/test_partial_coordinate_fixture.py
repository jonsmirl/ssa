"""CPU checks for named-document and scoped post-RoPE fixture capture."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ssa.partial_coordinate_fixture import capture_geometry, compact_kv, named_articles


def test_named_article_boundaries_exclude_next_title():
    rows = ["", " = A = \n", "body", " == Subsection == ", "more", " = B = \n", "last"]
    a, b = named_articles(rows, ("A", "B"))
    assert (a["row_start"], a["row_end_exclusive"]) == (1, 5)
    assert "Subsection" in a["text"] and " = B = " not in a["text"]
    assert b["row_end_exclusive"] == 7


@pytest.mark.parametrize("rows", [[" = A = "], [" = A = ", " = A = ", " = B = "]])
def test_named_articles_fail_on_missing_or_duplicate(rows):
    with pytest.raises(ValueError):
        named_articles(rows, ("A", "B"))


def test_compact_kv_verifies_gqa_grouping():
    x = torch.randn(1, 2, 3, 64)
    repeated = x.repeat_interleave(7, dim=1)
    assert torch.equal(compact_kv(repeated), x)
    repeated[:, 1, 0, 0] += 1
    with pytest.raises(ValueError, match="GQA"):
        compact_kv(repeated)


def test_capture_shape_head_mapping_and_restoration():
    original = F.scaled_dot_product_attention
    q = torch.randn(1, 14, 3, 64)
    k, v = torch.randn(1, 2, 3, 64), torch.randn(1, 2, 3, 64)
    with capture_geometry(3, [1, 2], layers=(0,)) as result:
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert F.scaled_dot_product_attention is original
    assert result["calls"] == 1
    assert result["arrays"]["Q_0"].shape == (4, 2, 64)
    np.testing.assert_array_equal(result["arrays"]["Q_0"], q[0, [0, 3, 7, 10]][:, [1, 2]].numpy())
    np.testing.assert_array_equal(result["arrays"]["K_0"], k[0].numpy())


def test_capture_restores_after_failure():
    original = F.scaled_dot_product_attention
    with pytest.raises(ValueError):
        with capture_geometry(3, [1]):
            F.scaled_dot_product_attention(torch.zeros(1, 2, 3, 64), torch.zeros(1, 2, 3, 64), torch.zeros(1, 2, 3, 64))
    assert F.scaled_dot_product_attention is original
