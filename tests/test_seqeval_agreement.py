"""Independent verification of score_exact_spans against seqeval.

seqeval needs typed tags (bare B/I/O are ambiguous about entity type), so every
label is mapped B -> "B-TERM", I -> "I-TERM", O -> "O" before being handed to
it. It is called with mode='strict', scheme=IOB2 EXPLICITLY: seqeval's default
is lenient conlleval semantics, where an "I" following "O" opens a new entity --
a different policy from decode(), which drops it (see decode()'s docstring and
configs/eval.yaml's dangling_i_policy). Getting the scheme flag wrong produces a
small constant disagreement (~0.001-0.01) that looks like a rounding error but
isn't -- it's the wrong policy being scored. That is why this test asserts
agreement to 6 decimal places rather than approximate equality: an exact
scoring function and a well-configured seqeval call should never differ beyond
float noise.
"""

from __future__ import annotations

import random

from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from src.eval.scorers import score_exact_spans
from src.eval.spans import decode, encode
from src.statistics.loading import load_domain

_DOMAINS = ("corp", "equi", "wind", "htfl")
_SEED = 1234
_DROP_FRACTION = 0.10
_SHIFT_FRACTION = 0.10


def _typed(labels: list[str]) -> list[str]:
    """Bare B/I/O -> seqeval's typed IOB2 tags."""
    return ["O" if label == "O" else f"{label}-TERM" for label in labels]


def _collect_sentences(domains):
    """One (file_id, sent_idx, tokens, gold_spans) record per sentence, gold
    spans from decode() over the gold labels -- the same "gold" that
    score_exact_spans is checked against elsewhere, so this test isolates
    scorer agreement rather than gold-representation differences."""
    records = []
    for domain in domains:
        for doc in load_domain(domain):
            for sent_idx, (tokens, labels) in enumerate(doc.sentences):
                gold_spans = decode(tokens, labels, "bio")
                records.append((doc.file_id, sent_idx, tokens, gold_spans))
    return records


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _perturb(records, seed: int, drop_fraction: float, shift_fraction: float):
    """Deterministically drop ~drop_fraction of gold spans and shift the end
    boundary of another ~shift_fraction by one token, so the score is not
    trivially 1.0. Returns a dict {(file_id, sent_idx): perturbed_spans}.

    Adjacent gold spans (no O token between them, e.g. two consecutive terms)
    are common in this corpus. encode() paints spans in list order with no
    overlap check, so extending one span's end into a surviving neighbour's
    start would have that neighbour's own "B" silently overwrite the
    extension -- the label sequence (and so seqeval) would then disagree with
    the literal (start, end) tuple this function recorded. Drops are applied
    first (they only ever free up room), then each shift is checked against
    the sentence's current survivors and skipped if it would collide.

    Overlapping spans are excluded by construction, not merely avoided as an
    optimisation: BIO cannot represent an overlap (each token carries exactly
    one label), so encode() has no way to encode one and this test has no way
    to check score_exact_spans/seqeval agreement on one. A reader should not
    infer from this test passing that overlapping-span behaviour is handled
    or even defined -- it isn't exercised here at all.
    """
    rng = random.Random(seed)

    # One entry per individual gold span occurrence, addressed by its
    # (file_id, sent_idx) sentence and its position within that sentence.
    span_refs = [
        (file_id, sent_idx, i)
        for file_id, sent_idx, _tokens, gold_spans in records
        for i in range(len(gold_spans))
    ]
    order = list(range(len(span_refs)))
    rng.shuffle(order)

    n_drop = round(drop_fraction * len(span_refs))
    n_shift = round(shift_fraction * len(span_refs))
    drop_idx = set(order[:n_drop])
    shift_idx = set(order[n_drop:n_drop + n_shift])

    tokens_by_key = {(fid, si): tokens for fid, si, tokens, _spans in records}
    kept = {(fid, si): list(spans) for fid, si, _tokens, spans in records}

    for ref_idx, (file_id, sent_idx, span_i) in enumerate(span_refs):
        if ref_idx in drop_idx:
            kept[(file_id, sent_idx)][span_i] = None

    for ref_idx, (file_id, sent_idx, span_i) in enumerate(span_refs):
        if ref_idx not in shift_idx:
            continue
        key = (file_id, sent_idx)
        span = kept[key][span_i]
        if span is None:
            continue
        start, end = span
        new_span = (start, end + 1)
        n_tokens = len(tokens_by_key[key])
        others = [s for j, s in enumerate(kept[key]) if j != span_i and s is not None]
        if end + 1 <= n_tokens and not any(_overlaps(new_span, other) for other in others):
            kept[key][span_i] = new_span
        # else: no valid one-token shift for this span -- left unchanged

    return {key: [s for s in spans if s is not None] for key, spans in kept.items()}


def test_score_exact_spans_agrees_with_seqeval_on_perturbed_prediction():
    records = _collect_sentences(_DOMAINS)
    perturbed_by_key = _perturb(records, _SEED, _DROP_FRACTION, _SHIFT_FRACTION)

    gold_flat: set[tuple[str, int, int, int]] = set()
    pred_flat: set[tuple[str, int, int, int]] = set()
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    for file_id, sent_idx, tokens, gold_spans in records:
        pred_spans = perturbed_by_key[(file_id, sent_idx)]

        for start, end in gold_spans:
            gold_flat.add((file_id, sent_idx, start, end))
        for start, end in pred_spans:
            pred_flat.add((file_id, sent_idx, start, end))

        y_true.append(_typed(encode(tokens, gold_spans, "bio")))
        y_pred.append(_typed(encode(tokens, pred_spans, "bio")))

    ours = score_exact_spans(pred_flat, gold_flat)

    theirs = (
        precision_score(y_true, y_pred, mode="strict", scheme=IOB2),
        recall_score(y_true, y_pred, mode="strict", scheme=IOB2),
        f1_score(y_true, y_pred, mode="strict", scheme=IOB2),
    )

    assert tuple(round(x, 6) for x in ours) == tuple(round(x, 6) for x in theirs)
