"""The hand-computed fixture: the one oracle in this suite not produced by the
implementation it checks.

# Expected values derived by hand from the metric definitions, not from running
# this code. Every other test in this suite compares the implementation against
# itself (round-trips) or against a number the implementation produced. A bug
# shared by encode and decode -- both treating `end` as inclusive, for instance
# -- passes all of those. This fixture is the only external oracle.
#
# If this test fails, the code is wrong until proven otherwise. Do not edit the
# expected values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.scorers import score_exact_spans, score_list
from src.eval.spans import decode
from src.eval.surface import spans_to_unique_list

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hand_computed.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_hand_computed_fixture():
    fixture = _load_fixture()
    file_id = fixture["file_id"]
    sentences = fixture["sentences"]
    expected = fixture["expected"]

    gold_flat: set[tuple[str, int, int, int]] = set()
    pred_flat: set[tuple[str, int, int, int]] = set()
    gold_sentences: list[tuple[list[str], list[tuple[int, int]]]] = []
    pred_sentences: list[tuple[list[str], list[tuple[int, int]]]] = []

    for sent_idx, sentence in enumerate(sentences):
        tokens = sentence["tokens"]

        gold_spans = decode(tokens, sentence["gold"], "bio")
        for start, end in gold_spans:
            gold_flat.add((file_id, sent_idx, start, end))
        gold_sentences.append((tokens, gold_spans))

        pred_spans = decode(tokens, sentence["pred"], "bio")
        for start, end in pred_spans:
            pred_flat.add((file_id, sent_idx, start, end))
        pred_sentences.append((tokens, pred_spans))

    # decode() reproduces exactly the hand-computed span sets.
    expected_gold_flat = {(file_id, s, st, en) for s, st, en in expected["gold_spans"]}
    expected_pred_flat = {(file_id, s, st, en) for s, st, en in expected["pred_spans"]}
    assert gold_flat == expected_gold_flat
    assert pred_flat == expected_pred_flat

    # Exact-span metric.
    exact_span = score_exact_spans(pred_flat, gold_flat)
    assert exact_span == pytest.approx(
        (
            expected["exact_span"]["precision"],
            expected["exact_span"]["recall"],
            expected["exact_span"]["f1"],
        ),
        abs=1e-9,
    )

    # Unique-list metric. Gold has 4 spans but 3 types -- "corruption" occurs in
    # both sentence 0 and sentence 2 and collapses to one entry; that
    # deduplication is exactly what's being observed here.
    gold_terms, gold_n_spans, gold_n_unique = spans_to_unique_list(gold_sentences)
    pred_terms, pred_n_spans, pred_n_unique = spans_to_unique_list(pred_sentences)

    assert gold_terms == set(expected["gold_terms"])
    assert gold_n_spans == expected["gold_n_spans"]
    assert gold_n_unique == len(expected["gold_terms"])

    assert pred_terms == set(expected["pred_terms"])
    assert pred_n_spans == expected["pred_n_spans"]
    assert pred_n_unique == len(expected["pred_terms"])

    unique_list = score_list(pred_terms, gold_terms)
    assert unique_list == pytest.approx(
        (
            expected["unique_list"]["precision"],
            expected["unique_list"]["recall"],
            expected["unique_list"]["f1"],
        ),
        abs=1e-9,
    )
