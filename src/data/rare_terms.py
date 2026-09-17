"""Rare-term weighting: term frequency over the training domains only, and the
weight formulas that turn that frequency into a per-term loss weight.

Frequency is counted from whatever ``configs/data.json`` names as
``train_domains`` (currently ``corp`` + ``wind``), read from ``DataConfig``
rather than hardcoded, so a future change to the split is picked up here too.
``equi`` (dev) and ``htfl`` (test) are never loaded by this module -- counting
either would leak evaluation-domain information into a training-time signal.

"""

from __future__ import annotations

import math
from collections import Counter

from src.eval.spans import decode
from src.stats.loading import DataConfig, Document, load_config, load_domain

IGNORE_WEIGHT = 0.0    # subword positions with no first-subword weight of their own
BASELINE_WEIGHT = 1.0  # "O" tokens, and any term the frequency table doesn't cover


def _term_string(tokens: list[str], start: int, end: int) -> str:
    """Same construction as ``src.eval.surface.spans_to_unique_list``, so a
    term counted here is the identical string to the one scored there."""
    return " ".join(tokens[start:end]).lower()


def count_term_frequencies(
    train_domains: tuple[str, ...],
    *,
    filter_max_tokens: dict[str, int | None],
    data_cfg: DataConfig | None = None,
) -> dict[str, int]:
    """Term -> occurrence count, over exactly the sentences ``build_examples``
    will train on.

    ``filter_max_tokens`` maps domain -> the same ``<= N tokens`` threshold
    ``TrainConfig.filter_for(domain, "train")`` returns for that domain
    (``None`` for no filtering).
    """
    data_cfg = data_cfg or load_config()
    counts: Counter[str] = Counter()

    for domain in train_domains:
        threshold = filter_max_tokens.get(domain)
        documents: list[Document] = load_domain(domain, data_cfg)
        for doc in documents:
            for tokens, labels in doc.sentences:
                if threshold is not None and len(tokens) <= threshold:
                    continue
                for start, end in decode(tokens, labels, "bio"):
                    counts[_term_string(tokens, start, end)] += 1

    return dict(counts)


def term_weight(freq: int, *, formula: str, hapax_weight: float = 2.0) -> float:
    """One term's loss weight from its training-domain frequency.

    formula:
      "hapax_binary"      -- hapax_weight if freq == 1, else BASELINE_WEIGHT
      "inverse_sqrt_freq" -- 1/sqrt(freq): a hapax (freq=1) weighs
                             BASELINE_WEIGHT and every other term weighs less
    """
    if formula == "hapax_binary":
        return hapax_weight if freq == 1 else BASELINE_WEIGHT
    if formula == "inverse_sqrt_freq":
        return BASELINE_WEIGHT / math.sqrt(freq)
    raise ValueError(f"unknown formula: {formula!r}")


def token_weights_for_sentence(
    tokens: list[str],
    labels: list[str],
    term_frequencies: dict[str, int],
    *,
    formula: str,
    hapax_weight: float = 2.0,
) -> list[float]:
    """One weight per DATASET token (pre-subword), via the same ``decode()``
    spans used everywhere else. Every token inside one span shares that
    span's weight; "O" tokens get ``BASELINE_WEIGHT``.
    """
    weights = [BASELINE_WEIGHT] * len(tokens)
    for start, end in decode(tokens, labels, "bio"):
        freq = term_frequencies.get(_term_string(tokens, start, end))
        w = (term_weight(freq, formula=formula, hapax_weight=hapax_weight)
             if freq is not None else BASELINE_WEIGHT)
        for i in range(start, end):
            weights[i] = w
    return weights


def align_weights(word_ids: list[int | None], token_weights: list[float]) -> list[float]:
    """Mirrors ``src.data.align.align_labels`` exactly, for floats instead of
    label ids. The first subword of each dataset token carries its weight.
    """
    prev: int | None = None
    out: list[float] = []
    for word_id in word_ids:
        if word_id is None:
            out.append(IGNORE_WEIGHT)
        elif word_id != prev:
            out.append(token_weights[word_id])
        else:
            out.append(IGNORE_WEIGHT)
        prev = word_id
    return out
