"""
T4 -- C-Value baseline: candidate generation -> C-Value -> threshold -> term list: one lowercased term per line

No training or gold labels are used. C-Value uses frequency/nesting over raw text only.

Locked decisions (see docs/cvalue_baseline.md):

  - reference_corpus = "target_only" -- htfl frequencies come only from its
    own annotated text (45,507 whitespace tokens). It has no
    unannotated_texts/ directory, and other domains are not mixed in.
  - candidate generation = stopword-boundary n-grams, not POS noun phrases.
  - min_n = 2 -- unigrams always score 0 because log2(1) = 0, so they are
    excluded during generation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

Sentence = tuple[list[str], list[str]]  # (tokens, labels); labels unused here


def _check(condition: bool, message: str) -> None:
    """Assertion that cannot be disabled with -O; matches loading.py."""
    if not condition:
        raise AssertionError(message)


# Stopwords -- boundary filter

# Closed-class words and contraction fragments. A candidate is rejected only
# when its FIRST or LAST token is a stopword; interior stopwords are allowed.
# Override via configs/cvalue.json -> "stopwords_path".
DEFAULT_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "and", "or", "but", "nor", "so", "yet", "if", "than", "as",
    "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "over", "under", "again",
    "further", "once", "out", "off", "not", "only", "own", "same",
    "i", "me", "my", "we", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they",
    "them", "their", "theirs", "this", "that", "these", "those",
    "who", "whom", "which", "what", "whose",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "there", "here", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "too", "very", "just", "also",
    "'s", "'re", "'ve", "'d", "'ll", "'m", "n't",
})

_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Reject numeric/statistical boundary tokens 
_NUMERIC_ONLY_RE = re.compile(r"^[+\-]?[\d.,%±]+$", re.UNICODE)


def load_stopwords(path: Path | None) -> frozenset[str]:
    """Load custom stopwords from one-word-per-line file, or use defaults."""
    if path is None:
        return DEFAULT_STOPWORDS
    path = Path(path)
    _check(path.is_file(), f"stopwords_path not found: {path}")
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    _check(len(words) > 0, f"stopwords file is empty: {path}")
    return frozenset(words)


def _is_boundary_ok(token: str, stopwords: frozenset[str]) -> bool:
    """Check that a candidate boundary is not punctuation, numeric, or stopword."""
    lowered = token.lower()
    if _PUNCT_ONLY_RE.match(token):
        return False
    if _NUMERIC_ONLY_RE.match(token):
        return False
    if lowered in stopwords:
        return False
    return True


# Candidate generation

@dataclass
class Candidate:
    tokens: tuple[str, ...]   # lowercased, in order
    freq: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


def generate_candidates(
    sentences: list[Sentence],
    *,
    min_n: int,
    max_n: int,
    stopwords: frozenset[str],
) -> dict[str, Candidate]:
    """Generate n-grams within sentences using boundary filtering.

    Windows never cross sentence boundaries. Candidates are lowercased for
    case-insensitive deduplication and frequency counting.
    """
    _check(1 <= min_n <= max_n, f"need 1 <= min_n <= max_n, got min_n={min_n} max_n={max_n}")

    candidates: dict[str, Candidate] = {}
    for tokens, _labels in sentences:
        n_tokens = len(tokens)
        for n in range(min_n, max_n + 1):
            if n > n_tokens:
                continue
            for start in range(0, n_tokens - n + 1):
                window = tokens[start:start + n]
                if not _is_boundary_ok(window[0], stopwords):
                    continue
                if not _is_boundary_ok(window[-1], stopwords):
                    continue
                lowered = tuple(t.lower() for t in window)
                key = " ".join(lowered)
                entry = candidates.get(key)
                if entry is None:
                    candidates[key] = Candidate(tokens=lowered, freq=1)
                else:
                    entry.freq += 1
    return candidates


def apply_min_frequency(candidates: dict[str, Candidate], min_frequency: int) -> dict[str, Candidate]:
    """Remove candidates occurring fewer than min_frequency times before nesting."""
    _check(min_frequency >= 1, f"min_frequency must be >= 1, got {min_frequency}")
    return {key: c for key, c in candidates.items() if c.freq >= min_frequency}


# Nesting -- type-level containment

def compute_nesting(candidates: dict[str, Candidate]) -> dict[str, set[str]]:
    """Find distinct longer candidate types containing each candidate."""
    nested_of: dict[str, set[str]] = {}
    # Group by length so shorter sub-sequences can be checked efficiently.
    by_length: dict[int, list[Candidate]] = {}
    for c in candidates.values():
        by_length.setdefault(c.length, []).append(c)

    for b in candidates.values():
        for sub_len in range(1, b.length):
            for start in range(0, b.length - sub_len + 1):
                sub_key = " ".join(b.tokens[start:start + sub_len])
                if sub_key in candidates and sub_key != b.text:
                    nested_of.setdefault(sub_key, set()).add(b.text)
    return nested_of


# C-Value

def compute_cvalue(
    candidates: dict[str, Candidate],
    nesting: dict[str, set[str]],
) -> dict[str, float]:
    """
    c_value(a) = log2(|a|) * f(a)                                     if T_a is empty
    c_value(a) = log2(|a|) * (f(a) - (1 / |T_a|) * sum_{b in T_a} f(b)) otherwise

    |a| = token length, f(a) = frequency, T_a = distinct longer candidates
    containing a. Unigrams score 0 because log2(1) = 0.
    """
    scores: dict[str, float] = {}
    for key, c in candidates.items():
        log_len = math.log2(c.length)
        t_a = nesting.get(key)
        if not t_a:
            scores[key] = log_len * c.freq
        else:
            nested_freq_sum = sum(candidates[b].freq for b in t_a)
            scores[key] = log_len * (c.freq - nested_freq_sum / len(t_a))
    return scores


# Threshold + term-list output

def threshold_terms(scores: dict[str, float], threshold: float) -> list[str]:
    """Keep scores >= threshold and return sorted, deduplicated terms."""
    kept = [key for key, score in scores.items() if score >= threshold]
    return sorted(set(kept))


def write_term_list(terms: list[str], path: Path) -> None:
    """Write sorted, lowercase, deduplicated terms: one per UTF-8 line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = sorted({t.lower() for t in terms})
    _check(
        all(t == t.strip() and t for t in deduped),
        "write_term_list: got an empty or whitespace-padded term",
    )
    path.write_text("\n".join(deduped) + ("\n" if deduped else ""), encoding="utf-8")