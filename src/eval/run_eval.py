"""List-metric ceilings: max_recall / max_precision per domain.

max_recall and max_precision are the scores obtained by decoding the GOLD
labels into a term list and scoring that list against the gold unique key.
They are ceilings -- no tagger using this annotation scheme can exceed them --
and they exist only in the list metric (spans_to_unique_list / score_list),
never in the span metric.

Run:  python -m src.eval.run_eval        (or: python src/eval/run_eval.py)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.scorers import load_gold_list_into_set, score_list
from src.eval.surface import generate_unique_list
from src.stats.loading import DataConfig, load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOMAINS = ("corp", "equi", "wind", "htfl")
_DEFAULT_OUT = _REPO_ROOT / "results" / "ceilings.md"

# Gold unique-term-list keys sit beside near-identically named non-tokenised
# variants that split hyphens and internal punctuation differently (see
# surface.spans_to_unique_list). Only the TOKENISED variant matches decode()'s
# space-joined surface form. ANN = all terms, NES = named entities excluded.
_KEY_SUFFIXES = {
    "ANN": "",
    "NES": "_nes",
}


def _gold_key_path(cfg: DataConfig, domain: str, key: str) -> Path:
    """Tokenised gold unique-term-list path for one domain/key, built from the
    domain name and the config -- never a hardcoded single path."""
    suffix = _KEY_SUFFIXES[key]
    return (
        cfg.data_root / cfg.language / domain / "annotated" / "annotations"
        / "unique_annotation_lists"
        / f"{domain}_{cfg.language}_tokenised_terms{suffix}.tsv"
    )


def compute_ceilings(domain: str) -> dict:
    """
    Decode gold labels for one domain, collapse to a unique term list, and score
    against both gold keys.

    Returns per-key precision/recall/F1 plus the span and type counts.
    """
    cfg = load_config()
    predicted_unique, n_spans, n_unique = generate_unique_list(domain)

    result = {"domain": domain, "n_spans": n_spans, "n_unique": n_unique}
    for key in _KEY_SUFFIXES:
        key_path = _gold_key_path(cfg, domain, key)
        assert key_path.is_file(), f"gold key not found for {domain}/{key}: {key_path}"
        gold_set = load_gold_list_into_set(str(key_path))
        precision, recall, f1 = score_list(predicted_unique, gold_set)
        result[key] = {"precision": precision, "recall": recall, "f1": f1}

    return result


def _render_table(rows: list[dict]) -> str:
    header = "| domain | key | precision | recall | f1 | n_spans | n_unique |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['key']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['n_spans']} | {row['n_unique']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute list-metric ceilings (max_recall / max_precision) per domain."
    )
    parser.add_argument("--domains", nargs="+", default=list(_DEFAULT_DOMAINS))
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    rows = []
    for domain in args.domains:
        ceilings = compute_ceilings(domain)
        for key in _KEY_SUFFIXES:
            scores = ceilings[key]
            rows.append({
                "domain": domain,
                "key": key,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "n_spans": ceilings["n_spans"],
                "n_unique": ceilings["n_unique"],
            })

    table = _render_table(rows)
    print(table)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
