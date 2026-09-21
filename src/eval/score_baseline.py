"""Score a term list prediction (e.g. the T4 C-Value baseline)
through the unique-list F1 harness.

This is a first look at a number, not a result: the frequency-corpus decision
behind any given prediction file is unrecorded (see the header this script
writes), and the score will change once that decision is made explicit.

Unique-list F1 only. C-Value ranks candidate types, not occurrences, and
produces no spans -- score_exact_spans does not apply here and no spans are
constructed from its output.

This script does not modify the system that produced --pred and does not
modify the harness it scores against (decode, encode, spans_to_unique_list,
generate_unique_list, generate_flatten_spans, score_list, score_exact_spans,
load_gold_list_into_set, write_term_list are all reused as-is). If the
prediction file is malformed, that is reported, not
repaired.

Run:  python -m src.eval.score_baseline --pred results/T4_cvalue_baseline/htfl_cvalue_terms.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.eval.run_eval import _gold_key_path
from src.eval.scorers import load_gold_list_into_set, score_list
from src.statistics.loading import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOMAIN = "htfl"
_DEFAULT_OUT = _REPO_ROOT / "results" / "baseline_cvalue.md"

# Tokenised gold keys only (see _gold_key_path / run_eval.py) -- the
# non-tokenised variants split hyphens and internal punctuation differently.
_KEYS = ("ann", "nes")

# BIO-tagger ceilings for htfl (results/T3_eval_harness/ceilings.md, without_named_entities,
# tokenised keys). For orientation only -- C-Value is not a tagger, produces
# no spans, and is not bound by them. Do not present the baseline as a
# fraction of these.
_HTFL_CEILINGS = {
    "ann": {"max_recall": 0.9316, "max_precision": 0.8887},
    "nes": {"max_recall": 0.8549, "max_precision": 0.8911},
}

_UNTOKENISED_PATTERN = re.compile(r"\S['-]\S")


def _validate_prediction_file(path: Path) -> tuple[list[str], int, int]:
    """Check the RAW prediction file -- one term
    per line, lowercased, deduplicated, UTF-8, no header, no index column --
    without repairing anything. load_gold_list_into_set() is not used here:
    it strips whitespace on load, which would silently absorb exactly the
    defects this function exists to catch.

    Returns (failures, n_lines, n_untokenised_looking). ``failures`` is empty
    when the three hard checks (whitespace/tab, casing, duplicates) pass; the
    apostrophe/hyphen count is informational only (see its comment below) and
    never appears in ``failures``.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    failures: list[str] = []

    whitespace_bad = [line for line in lines if line != line.strip() or "\t" in line]
    if whitespace_bad:
        failures.append(
            f"{len(whitespace_bad)} line(s) contain a tab, newline, or "
            f"leading/trailing whitespace, e.g. {whitespace_bad[0]!r}"
        )

    not_lower = [line for line in lines if line != line.lower()]
    if not_lower:
        failures.append(
            f"{len(not_lower)} line(s) are not their own lowercase form, "
            f"e.g. {not_lower[0]!r}"
        )

    n_unique = len(set(lines))
    if n_unique != len(lines):
        failures.append(
            f"{len(lines) - n_unique} duplicate line(s): {len(lines)} lines, "
            f"{n_unique} unique"
        )

    # Informational, not a failure condition: "public prosecutor's office" is
    # not tokenised ("public prosecutor 's office" is), and a non-tokenised
    # prediction silently scores near zero on affected terms rather than
    # raising. But hyphenated compounds ("30-day", "6-minute") are expected
    # even in correctly tokenised text, so a nonzero count alone does not mean
    # the file is bad -- it is reported for a human to look at, not asserted on.
    n_untokenised_looking = sum(1 for line in lines if _UNTOKENISED_PATTERN.search(line))

    return failures, len(lines), n_untokenised_looking


def _render_report(domain: str, pred_path: Path, rows: list[dict], n_untokenised_looking: int) -> str:
    lines = [
        f"# C-Value baseline: unique-list F1 on {domain}",
        "",
        "This is a first look at a number, not a result.",
        "",
        "- metric: unique-list F1 (TermEval 2020 protocol; see configs/eval.yaml)",
        f"- domain: {domain}",
        f"- prediction file: {pred_path}",
        "- keys scored (one predicted list, both keys, tokenised gold): "
        + ", ".join(key.upper() for key in _KEYS),
        f"- entries in the prediction that look non-tokenised (apostrophe/hyphen, "
        f"no surrounding space; informational, not a failure): {n_untokenised_looking}",
        "",
        "**Frequency-corpus decision: UNRECORDED.** Which corpus C-Value counted "
        "term frequencies over -- annotated text only, or including the "
        "unannotated portion as reference material (docs/Tasks.md, T4) -- is not "
        "recorded for this prediction file. Both are legitimate and are not "
        "comparable to each other. This is stated as open, not guessed at.",
        "",
    ]

    if domain == "htfl":
        c = _HTFL_CEILINGS
        lines += [
            "**BIO-tagger ceilings, for orientation only:** on htfl, max_recall "
            f"is {c['ann']['max_recall']} (ANN) / {c['nes']['max_recall']} (NES); "
            f"max_precision is {c['ann']['max_precision']} (ANN) / "
            f"{c['nes']['max_precision']} (NES) (results/T3_eval_harness/ceilings.md). These are "
            "the caps for a BIO sequence tagger under this annotation scheme. "
            "C-Value is not a tagger, produces no spans, and is not bound by "
            "them -- listed here for context, not as a denominator for the "
            "score below.",
            "",
        ]
    else:
        lines += [
            f"**BIO-tagger ceilings** are not reproduced here for {domain} -- "
            "see results/T3_eval_harness/ceilings.md.",
            "",
        ]

    lines += [
        "| key | precision | recall | f1 | n_predicted | n_gold |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['key']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['n_predicted']} | {row['n_gold']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a term list prediction against unique-list F1."
    )
    parser.add_argument("--pred", required=True, help="path to a term list file: one lowercased term per line")
    parser.add_argument("--domain", default=_DEFAULT_DOMAIN)
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    pred_path = Path(args.pred)
    assert pred_path.is_file(), f"prediction file not found: {pred_path}"

    failures, n_lines, n_untokenised_looking = _validate_prediction_file(pred_path)

    print(f"Validating {pred_path} ({n_lines} lines)...")
    print(
        f"  entries that look non-tokenised (apostrophe/hyphen, no surrounding "
        f"space): {n_untokenised_looking} (informational)"
    )
    if failures:
        print("FAILED format validation:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("  format checks: OK")

    # Loaded through the harness only after the raw file has already passed
    # validation above -- load_gold_list_into_set() strips whitespace, which
    # would otherwise mask exactly what was just checked for.
    predicted = load_gold_list_into_set(str(pred_path))

    cfg = load_config()
    rows = []
    for key in _KEYS:
        key_path = _gold_key_path(cfg, args.domain, key)
        assert key_path.is_file(), f"gold key not found for {args.domain}/{key}: {key_path}"
        gold_set = load_gold_list_into_set(str(key_path))
        precision, recall, f1 = score_list(predicted, gold_set)
        rows.append({
            "key": key.upper(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_predicted": len(predicted),
            "n_gold": len(gold_set),
        })

    output = _render_report(args.domain, pred_path, rows, n_untokenised_looking)
    print()
    print(output)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
