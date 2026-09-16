"""Measure NOBI gold-label ceilings over the ACTER corpus.

This is an evaluation-only measurement. It derives NOBI labels from the
existing BIO annotations and tokenized gold term lists, then scores the terms
decoded from those labels. Existing BIO ceiling outputs are untouched.
"""

from __future__ import annotations

from pathlib import Path

from src.data.nobi_labels import build_nobi_labels, load_domain_terms
from src.eval.run_eval import _gold_key_path
from src.eval.scorers import load_gold_list_into_set, score_list
from src.eval.spans import decode
from src.eval.surface import spans_to_unique_list
from src.stats.loading import load_config, load_domain

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT = _REPO_ROOT / "results" / "nobi_ceilings.md"


def compute_nobi_ceiling(domain: str) -> dict:
    cfg = load_config()
    terms = load_domain_terms(cfg, domain)
    sentences = []
    n_nested_tokens = 0
    n_nested_spans = 0

    for document in load_domain(domain, cfg):
        for tokens, bio_labels in document.sentences:
            nobi_labels = build_nobi_labels(tokens, bio_labels, terms)
            n_nested_tokens += sum(label in {"BN", "IN"} for label in nobi_labels)
            n_nested_spans += sum(label == "BN" for label in nobi_labels)
            sentences.append((tokens, decode(tokens, nobi_labels, "nobi")))

    predicted, n_spans, n_unique = spans_to_unique_list(sentences)
    scores = {}
    for key in ("ann", "nes"):
        gold_path = _gold_key_path(cfg, domain, key)
        gold = load_gold_list_into_set(str(gold_path))
        precision, recall, f1 = score_list(predicted, gold)
        scores[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_gold": len(gold),
        }

    return {
        "domain": domain,
        "n_spans": n_spans,
        "n_unique": n_unique,
        "n_nested_tokens": n_nested_tokens,
        "n_nested_spans": n_nested_spans,
        "scores": scores,
    }


def render(rows: list[dict]) -> str:
    lines = [
        "# NOBI list-metric ceilings",
        "",
        "Gold NOBI labels are derived from ACTER BIO spans plus the tokenized "
        "terms-only key. Same-start nested terms are ignored; overlapping "
        "nested terms use left-to-right, longest-match-first resolution.",
        "",
        "| domain | key | precision | recall | f1 | spans | unique types | nested spans | nested tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for key in ("ann", "nes"):
            score = row["scores"][key]
            lines.append(
                f"| {row['domain']} | {key.upper()} | {score['precision']:.4f} | "
                f"{score['recall']:.4f} | {score['f1']:.4f} | {row['n_spans']} | "
                f"{row['n_unique']} | {row['n_nested_spans']} | {row['n_nested_tokens']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [compute_nobi_ceiling(domain) for domain in load_config().domains]
    report = render(rows)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"wrote {_OUT.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()