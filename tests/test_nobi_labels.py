from __future__ import annotations

import pytest

from src.data.nobi_labels import (
    build_nobi_labels,
    find_nested_spans,
    find_term_matches,
)


def test_find_term_matches_keeps_overlapping_token_matches():
    tokens = ["congestive", "heart", "failure"]
    terms = {("congestive", "heart", "failure"), ("heart", "failure"), ("failure",)}

    assert find_term_matches(tokens, terms) == [
        (0, 3, ("congestive", "heart", "failure")),
        (1, 3, ("heart", "failure")),
        (2, 3, ("failure",)),
    ]


def test_build_nobi_labels_uses_only_terms_inside_bio_outer_span():
    tokens = ["Patient", "has", "left", "bundle", "branch", "block", "."]
    bio_labels = ["O", "O", "B", "I", "I", "I", "O"]
    terms = {("left", "bundle", "branch", "block"), ("bundle", "branch")}

    assert build_nobi_labels(tokens, bio_labels, terms) == [
        "O", "O", "B", "BN", "IN", "I", "O"
    ]


def test_unannotated_term_occurrence_does_not_create_nested_label():
    tokens = ["heart", "failure", "appears"]
    bio_labels = ["O", "O", "O"]
    terms = {("heart", "failure"), ("failure",)}

    assert find_nested_spans(tokens, bio_labels, terms) == ([], [])


def test_term_matching_is_case_insensitive():
    tokens = ["Congestive", "Heart", "Failure"]
    bio_labels = ["B", "I", "I"]
    terms = {("heart", "failure")}

    assert build_nobi_labels(tokens, bio_labels, terms) == ["B", "BN", "IN"]


def test_same_start_nested_term_is_ignored():
    tokens = ["heart", "failure"]
    bio_labels = ["B", "I"]
    terms = {("heart",), ("heart", "failure")}

    assert find_nested_spans(tokens, bio_labels, terms) == ([(0, 2)], [])