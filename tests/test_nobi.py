from __future__ import annotations

import pytest

from src.eval.nobi import decode_nobi, encode_nobi


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


def test_nobi_rejects_nested_span_outside_outer_span():
    with pytest.raises(AssertionError):
        encode_nobi(["a", "b"], [(0, 1)], [(1, 2)])


def test_nobi_rejects_unknown_labels():
    with pytest.raises(ValueError):
        decode_nobi(["a"], ["X"])