"""Tests for src.eval.scorers.score_list and the write_term_list / gold-loader
round trip.
"""

from __future__ import annotations

import pytest

from src.eval.run_eval import _gold_key_path
from src.eval.scorers import load_gold_list_into_set, score_list
from src.eval.surface import write_term_list
from src.statistics.loading import load_config


def test_identity_write_then_reload_matches_gold(tmp_path):
    """Write a gold key out through write_term_list, read it back through
    load_gold_list_into_set, and score it against the original gold set. Must
    be exactly (1.0, 1.0, 1.0) -- the gold loader (TSV, first column) and the
    prediction-side writer (one term per line) normalise the same way even
    though the two paths are otherwise unrelated code."""
    cfg = load_config()
    key_path = _gold_key_path(cfg, "corp", "ann")
    gold_set = load_gold_list_into_set(str(key_path))

    out_path = tmp_path / "roundtrip_terms.tsv"
    write_term_list(gold_set, str(out_path))
    reloaded = load_gold_list_into_set(str(out_path))

    assert score_list(reloaded, gold_set) == (1.0, 1.0, 1.0)


def test_reloaded_terms_carry_no_whitespace(tmp_path):
    """write_term_list emits one term per line, no
    extra column); load_gold_list_into_set must hand back bare terms with no
    newline, tab, or leading/trailing whitespace welded on. This is the
    assertion that would have caught the writer/loader normalisation
    mismatch this test file's identity test alone did not."""
    out_path = tmp_path / "small_terms.tsv"
    write_term_list({"heart failure", "diagnosis", "stent"}, str(out_path))

    reloaded = load_gold_list_into_set(str(out_path))

    assert reloaded == {"heart failure", "diagnosis", "stent"}
    for term in reloaded:
        assert "\n" not in term
        assert "\t" not in term
        assert term == term.strip()


def test_score_list_empty_prediction_returns_zero():
    gold = {"heart failure", "diagnosis"}
    assert score_list(set(), gold) == (0.0, 0.0, 0.0)


def test_score_list_zero_overlap_returns_zero():
    predicted = {"foo", "bar"}
    gold = {"baz", "qux"}
    assert score_list(predicted, gold) == (0.0, 0.0, 0.0)


def test_score_list_uppercase_prediction_raises():
    predicted = {"Heart Failure"}
    gold = {"heart failure"}
    with pytest.raises(AssertionError):
        score_list(predicted, gold)
