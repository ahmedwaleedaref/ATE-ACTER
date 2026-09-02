"""Tests for src.eval.spans: decode()/encode() round-trip and IOB2 invariants.

Moved from src/eval/spans.py, where test_decode() ran as a module-level side
effect (a four-domain corpus pass) on every import.
"""

from __future__ import annotations

from src.eval.spans import decode, encode
from src.stats.loading import Document, load_domain

# Three sentence-initial "I" labels exist in the gold data, all cases of a term
# split across a spurious sentence boundary (wind_en_01 x2, htfl_en_171 x1).
# decode() drops them, so the re-encoded sequence differs at exactly these three
# positions. Measured over all four English domains, 222,281 tokens.
EXPECTED_MISMATCHES = 3


def test_decode():
    """
    this function will laod every domain , then for eac domain we will each document
    by loader we will get each token , label list for each sentance in document .

    """
    number_of_mismathces = 0
    Domains : list[str] = ["corp", "equi", "wind","htfl"]
    for domain in Domains :
        Documents : list[Document] = load_domain(domain)
        for doc in Documents :
            #each doc has list of sentences
            for sentence in doc.sentences :
                token = sentence[0]
                label = sentence[1]
                if( encode(token , decode(token,label,"bio"),"bio") != label ) :
                    number_of_mismathces += 1
    assert number_of_mismathces == EXPECTED_MISMATCHES


def test_decode_structural_invariants():
    """IOB2 structural invariants that hold for every sentence in the corpus,
    checked over all four English domains. These localise a decode failure
    faster than a label diff:

      * len(decode(tokens, labels, "bio")) == labels.count("B")
      * sum(e - s for s, e in spans) == labels.count("B") + labels.count("I")
        - dangling, where dangling is the number of "I" labels at index 0 or
        following an "O" (the cases decode() drops -- see decode()'s docstring).
    """
    for domain in ["corp", "equi", "wind", "htfl"]:
        for doc in load_domain(domain):
            for tokens, labels in doc.sentences:
                spans = decode(tokens, labels, "bio")

                assert len(spans) == labels.count("B")

                dangling = sum(
                    1 for i, label in enumerate(labels)
                    if label == "I" and (i == 0 or labels[i - 1] == "O")
                )
                assert (
                    sum(e - s for s, e in spans)
                    == labels.count("B") + labels.count("I") - dangling
                )
