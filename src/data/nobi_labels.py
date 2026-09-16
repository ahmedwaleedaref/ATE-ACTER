"""Build NOBI labels from ACTER BIO labels and tokenized gold terms.

The sequential ACTER files provide the annotated outer spans. The tokenized
unique-term list supplies candidate nested terms. A candidate is promoted to a
nested span only when its complete token sequence occurs strictly inside an
outer BIO span in the same sentence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.eval.nobi import Span, encode_nobi, resolve_nested_spans
from src.eval.spans import decode

Term = tuple[str, ...]
TermMatch = tuple[int, int, Term]


def tokenized_terms_path(data_cfg, domain: str) -> Path:
    directory = (
        data_cfg.data_root / data_cfg.language / domain / "annotated"
        / "annotations" / "unique_annotation_lists"
    )
    matches = sorted(directory.glob("*_tokenised_terms.tsv"))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one terms-only tokenized list in {directory}, found {matches}"
        )
    return matches[0]


def load_domain_terms(data_cfg, domain: str) -> set[Term]:
    """Load the terms-only tokenized key used to derive NOBI labels."""
    return read_tokenized_terms(tokenized_terms_path(data_cfg, domain))


def read_tokenized_terms(path: str | Path) -> set[Term]:
    """Read a tokenized ACTER unique-term TSV into lowercase token tuples."""
    terms: set[Term] = set()
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        term_text = raw_line.split("\t", 1)[0].strip().lower()
        term = tuple(term_text.split())
        if not term:
            raise AssertionError(f"empty term at {path}:{line_number}")
        terms.add(term)
    return terms


def find_term_matches(tokens: list[str], terms: Iterable[Term]) -> list[TermMatch]:
    """Find exact, overlapping term matches within one tokenized sentence."""
    lowered_tokens = [token.lower() for token in tokens]
    unique_terms = sorted(set(terms), key=lambda term: (len(term), term))
    matches: list[TermMatch] = []

    for term in unique_terms:
        term_length = len(term)
        for start in range(len(lowered_tokens) - term_length + 1):
            end = start + term_length
            if tuple(lowered_tokens[start:end]) == term:
                matches.append((start, end, term))

    return sorted(matches, key=lambda match: (match[0], match[1], match[2]))


def find_nested_spans(
    tokens: list[str],
    bio_labels: list[str],
    terms: Iterable[Term],
) -> tuple[list[Span], list[Span]]:
    """Return ``(outer_spans, nested_spans)`` for one sentence.

    Matches outside an annotated BIO span are ignored. A match beginning at an
    outer span's first token is rejected because one NOBI label cannot express
    both ``B`` and ``BN`` at the same position.
    """
    outer_spans = decode(tokens, bio_labels, "bio")
    nested_candidates: list[Span] = []

    for start, end, _term in find_term_matches(tokens, terms):
        containing = [
            (outer_start, outer_end)
            for outer_start, outer_end in outer_spans
            if outer_start <= start and end <= outer_end
        ]
        if not containing:
            continue

        nested_candidates.append((start, end))

    return outer_spans, resolve_nested_spans(tokens, outer_spans, nested_candidates)


def build_nobi_labels(
    tokens: list[str],
    bio_labels: list[str],
    terms: Iterable[Term],
) -> list[str]:
    """Convert one BIO-labeled sentence into labels for the NOBI model."""
    outer_spans, nested_spans = find_nested_spans(tokens, bio_labels, terms)
    return encode_nobi(tokens, outer_spans, nested_spans)