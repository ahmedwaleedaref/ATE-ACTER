from __future__ import annotations

import pytest

from src.eval.nobi import decode_nobi, encode_nobi, resolve_nested_spans
from src.eval.spans import decode


def test_nobi_encodes_and_decodes_nested_multi_token_term():
    tokens = ["Patient", "has", "a", "left", "bundle", "branch", "block", "."]
    outer = [(3, 7)]
    nested = [(4, 6)]

    labels = encode_nobi(tokens, outer, nested)

    assert labels == ["O", "O", "O", "B", "BN", "IN", "I", "O"]
    assert set(decode_nobi(tokens, labels)) == set(outer + nested)


def test_nobi_supports_single_token_nested_term():
    tokens = ["congestive", "heart", "failure"]
    labels = encode_nobi(tokens, [(0, 3)], [(1, 2)])

    assert labels == ["B", "BN", "I"]
    assert set(decode_nobi(tokens, labels)) == {(0, 3), (1, 2)}


def test_public_decode_dispatches_to_nobi_without_changing_bio():
    tokens = ["congestive", "heart", "failure"]
    labels = ["B", "BN", "I"]

    assert set(decode(tokens, labels, "nobi")) == {(0, 3), (1, 2)}
    assert decode(["heart", "failure"], ["B", "I"], "bio") == [(0, 2)]


def test_nobi_ignores_nested_span_outside_outer_span():
    assert encode_nobi(["a", "b"], [(0, 1)], [(1, 2)]) == ["B", "O"]


def test_nobi_preserves_outer_b_when_nested_term_shares_outer_start():
    assert encode_nobi(["heart", "failure"], [(0, 2)], [(0, 1)]) == ["B", "I"]


def test_nobi_resolves_overlaps_left_to_right_and_longest_first():
    outer = [(0, 8)]
    nested = [(1, 3), (2, 5), (5, 7), (5, 8), (0, 2)]

    assert resolve_nested_spans(["x"] * 8, outer, nested) == [(1, 3), (5, 8)]


def test_nobi_rejects_unknown_labels():
    with pytest.raises(ValueError):
        decode_nobi(["a"], ["X"])


def test_nobi_decoder_ignores_dangling_model_predictions():
    tokens = ["a", "b", "c", "d"]

    assert decode_nobi(tokens, ["IN", "BN", "I", "O"]) == []