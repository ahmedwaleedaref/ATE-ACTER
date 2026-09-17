"""Tests for src.data.rare_terms.

Deliberately independent of any tokenizer: align_weights operates on
word_ids lists the same way align_labels does.
"""

from __future__ import annotations
import math
import pytest

from src.data.rare_terms import (
    BASELINE_WEIGHT,
    align_weights,
    count_term_frequencies,
    term_weight,
    token_weights_for_sentence,
)
from src.stats.loading import load_config



# Scope: corp + wind only, never equi or htfl

def test_frequency_scope_matches_configured_train_domains():
    """Whatever configs/data.json calls train_domains is what gets counted --
    not a hardcoded list here."""
    data_cfg = load_config()
    assert data_cfg.train_domains == ("corp", "wind"), (
        "this test's assumptions (and the leakage checks below) are pinned to "
        "the adopted split; re-check them if train_domains ever changes"
    )


def test_htfl_only_term_is_absent():
    """'heart failure' is htfl's own example term (Data_stats.md, nested-term
    table) and does not occur in corp or wind. If it ever showed up here, the
    loader pulled in a domain it should not have."""
    data_cfg = load_config()
    freqs = count_term_frequencies(
        data_cfg.train_domains,
        filter_max_tokens={"corp": None, "wind": 2},
        data_cfg=data_cfg,
    )
    assert "heart failure" not in freqs


def test_equi_terms_do_not_inflate_counts():
    
    data_cfg = load_config()
    correct = count_term_frequencies(
        ("corp", "wind"), filter_max_tokens={"corp": None, "wind": 2}, data_cfg=data_cfg)
    with_equi = count_term_frequencies(
        ("corp", "equi", "wind"), filter_max_tokens={"corp": None, "equi": None, "wind": 2},
        data_cfg=data_cfg)
    assert sum(with_equi.values()) > sum(correct.values())



# Weight formulas

def test_hapax_binary_formula():
    assert term_weight(1, formula="hapax_binary", hapax_weight=2.0) == 2.0
    assert term_weight(2, formula="hapax_binary", hapax_weight=2.0) == BASELINE_WEIGHT
    assert term_weight(350, formula="hapax_binary", hapax_weight=2.0) == BASELINE_WEIGHT


def test_inverse_sqrt_freq_formula():
    assert term_weight(1, formula="inverse_sqrt_freq") == pytest.approx(1.0)
    assert term_weight(4, formula="inverse_sqrt_freq") == pytest.approx(0.5)
    assert term_weight(350, formula="inverse_sqrt_freq") == pytest.approx(1 / math.sqrt(350))


def test_unknown_formula_rejected():
    with pytest.raises(ValueError):
        term_weight(1, formula="not_a_real_formula")



# Token-level weights from one sentence

def test_all_tokens_in_one_span_share_the_same_weight():
    tokens = ["myocyte", "hypertrophy", "was", "observed"]
    labels = ["B", "I", "O", "O"]
    freqs = {"myocyte hypertrophy": 1}
    weights = token_weights_for_sentence(
        tokens, labels, freqs, formula="hapax_binary", hapax_weight=3.0)
    assert weights == [3.0, 3.0, BASELINE_WEIGHT, BASELINE_WEIGHT]


def test_term_missing_from_table_falls_back_to_baseline():
    tokens = ["novel", "term"]
    labels = ["B", "I"]
    weights = token_weights_for_sentence(tokens, labels, {}, formula="inverse_sqrt_freq")
    assert weights == [BASELINE_WEIGHT, BASELINE_WEIGHT]



# Subword alignment -- hand-crafted word_ids, same style as
# test_alignment_gate.py::test_align_labels_on_a_hand_checked_sentence

def test_align_weights_on_a_hand_checked_sentence():
    """'self-employed' tokenizes to several wordpieces; word_ids below is a
    plausible fast-tokenizer output for
    ["The", "self-employed", "non", "-", "governmental", "body", "."]
    where 'self-employed' splits into 3 pieces and everything else into 1,
    plus a [CLS]/[SEP] special at each end (word_id None)."""
    token_weights = [1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # one weight per dataset token
    word_ids = [None, 0, 1, 1, 1, 2, 3, 4, 5, 6, None]

    aligned = align_weights(word_ids, token_weights)

    assert len(aligned) == len(word_ids)
    first_positions = [i for i, w in enumerate(word_ids)
                       if w is not None and (i == 0 or word_ids[i - 1] != w)]
    assert len(first_positions) == len(token_weights), "one weight per dataset token"
    assert [aligned[i] for i in first_positions] == token_weights
    assert all(aligned[i] == 0.0 for i in range(len(word_ids)) if i not in first_positions)


def test_align_weights_matches_align_labels_first_subword_positions():
    """align_weights and align_labels must agree on WHICH position is a
    token's first subword -- this is the property the T6 gate cares about,
    now checked for weights too."""
    from src.data.align import LABEL2ID, align_labels

    word_ids = [None, 0, 1, 1, 1, 2, 3, 4, 5, 6, None]
    labels = ["O", "B", "B", "I", "I", "I", "O"]
    token_weights = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]

    aligned_labels = align_labels(word_ids, labels, LABEL2ID)
    aligned_weights = align_weights(word_ids, token_weights)

    ignored_by_labels = [i for i, v in enumerate(aligned_labels) if v == -100]
    ignored_by_weights = [i for i, v in enumerate(aligned_weights) if v == 0.0]
    assert ignored_by_labels == ignored_by_weights
