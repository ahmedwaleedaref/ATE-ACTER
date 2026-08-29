"""T2 item 1 -- sentence-length statistic over the annotated portion, with the
inventory-ratio check that guards it.

Sentence length = number of tokens in the sentence block. Punctuation and
numbers are separate tokens and are counted. Nothing is filtered.

Percentile definition (implemented here, never imported):

    Sort ascending. Nearest-rank. For an integer percentile ``p`` in
    ``[0, 100]`` over ``n`` values, the 1-based rank is ``ceil(p/100 * n)``
    and the 0-based index is ``rank - 1``, clamped to ``[0, n - 1]``. Integer
    arithmetic (``(p*n + 99)//100 - 1``) so exact boundaries such as p90 of
    10 values are not thrown off by float rounding.

Integrity check (blocking): inventory ratio -- paired-file tokens /
whole-domain-corpus tokens, same unit on both sides. Raises if wind's ratio
reaches 0.30 (the signature of having loaded the unannotated wind corpus).

Outputs, written to results/data_stats/ only after the check passes:
  s01_lengths.md    -- the rendered tables (byte-identical to what is printed)
  s01_lengths.json  -- every raw number, plus config for provenance

Run:  python -m src.stats.s01_lengths        (or: python src/stats/s01_lengths.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.stats.loading import (  # noqa: E402
    DataConfig,
    inventory_ratios,
    load_config,
    load_domain,
)

PERCENTILES = (50, 90, 95, 99)
_OUT_DIR = _REPO_ROOT / "results" / "data_stats"

PERCENTILE_DEFINITION = (
    "sort ascending, nearest-rank; 0-based index = ceil(p/100 * n) - 1, "
    "clamped to [0, n-1]; computed with integer arithmetic"
)


# --------------------------------------------------------------------------- #
# Percentile (hand-rolled, per the spec)
# --------------------------------------------------------------------------- #
def percentile(sorted_values, p: int) -> int:
    """Nearest-rank percentile of an already-ascending-sorted sequence.

    ``p`` is an integer in [0, 100]. 1-based rank = ceil(p/100 * n); the
    0-based index is that minus one, clamped to [0, n-1]. Integer arithmetic
    avoids float rounding at exact boundaries.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of an empty sequence")
    if not (0 <= p <= 100):
        raise ValueError(f"percentile p out of range: {p}")
    idx = (p * n + 99) // 100 - 1          # ceil(p*n/100) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]


# --------------------------------------------------------------------------- #
# Sentence-length summary
# --------------------------------------------------------------------------- #
def summarise_lengths(lengths) -> dict:
    # No empty guard: an empty domain is a loader failure, and s[0] / percentile
    # both raise here rather than returning a fake None.
    s = sorted(lengths)
    summary = {
        "n_sentences": len(s),
        "n_tokens": sum(s),
        "min": s[0],
        "max": s[-1],
        "n_sentences_le_2_tokens": sum(1 for x in s if x <= 2),
    }
    for p in PERCENTILES:
        summary[f"p{p}"] = percentile(s, p)
    return summary


def collect(cfg: DataConfig):
    """Load every in-scope domain and summarise sentence lengths.

    Returns ``(per_domain, pooled, documents)``. Prints one inventory-count
    line per domain before any number is computed.
    """
    per_domain: dict = {}
    documents: dict = {}
    all_lengths: list[int] = []

    for domain in cfg.domains:
        docs = load_domain(domain, cfg)
        print(f"[inventory] domain={domain} n_files={len(docs)}")
        documents[domain] = docs

        lengths = [len(tokens) for doc in docs for tokens, _labels in doc.sentences]
        all_lengths += lengths
        summary = summarise_lengths(lengths)
        summary["n_documents"] = len(docs)
        per_domain[domain] = summary

    pooled = summarise_lengths(all_lengths)
    pooled["n_documents"] = sum(per_domain[d]["n_documents"] for d in cfg.domains)
    return per_domain, pooled, documents


# --------------------------------------------------------------------------- #
# Rendering -- one markdown string, printed and written verbatim
# --------------------------------------------------------------------------- #
def _stats_rows(cfg, per_domain, pooled) -> list:
    lines = [
        ("| domain | n_docs | n_sentences | n_tokens | min | p50 | p90 | p95 "
         "| p99 | max | sents <=2 tok (n) | sents <=2 tok (%) |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (*cfg.domains, "pooled"):
        s = pooled if name == "pooled" else per_domain[name]
        n = s["n_sentences"]
        pct = 100.0 * s["n_sentences_le_2_tokens"] / n
        lines.append(
            f"| {name} | {s['n_documents']} | {n} | {s['n_tokens']} | "
            f"{s['min']} | {s['p50']} | {s['p90']} | {s['p95']} | {s['p99']} | "
            f"{s['max']} | {s['n_sentences_le_2_tokens']} | {pct:.2f}% (N={n}) |"
        )
    return lines


def render(cfg, per_domain, pooled, ratios) -> str:
    lines = [
        "# s01 -- sentence length distribution",
        "",
        "Scope: English, annotated portion only, sequential IOB annotations, "
        "`without_named_entities` labels. Sentence length = number of tokens; "
        "punctuation and numbers count; nothing is filtered.",
        "",
        f"Percentiles: {PERCENTILE_DEFINITION}. Percentage columns are of that "
        "row's `n_sentences` (`N`).",
        "",
        *_stats_rows(cfg, per_domain, pooled),
        "",
        "## Integrity check -- inventory ratio",
        "",
        "`paired-file tokens / whole-domain-corpus tokens`, both whitespace-split "
        "(annotated `texts_tokenised/` + `unannotated_texts/`). Bound: wind "
        "`< 0.30` (`data_layout.md` section 1: ~52k annotated of ~314k). No "
        "bound on the others -- htfl is annotated in full, ratio 1.0 by "
        "construction.",
        "",
        "| domain | paired tokens | corpus tokens | ratio | bound |",
        "|---|---:|---:|---:|---|",
    ]
    for d in cfg.domains:
        paired, corpus, ratio = ratios[d]
        lines.append(
            f"| {d} | {paired} | {corpus} | {ratio:.4f} | "
            f"{'< 0.30' if d == 'wind' else '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(cfg, per_domain, pooled, ratios) -> dict:
    return {
        "generated_by": "src/stats/s01_lengths.py",
        "config": {
            "data_root": str(cfg.data_root),
            "language": cfg.language,
            "train_domains": list(cfg.train_domains),
            "test_domain": cfg.test_domain,
            "sequential_scheme": cfg.sequential_scheme,
            "ne_variant": cfg.ne_variant,
        },
        "domains": {d: per_domain[d] for d in cfg.domains},
        "pooled": pooled,
        "inventory_ratio": [
            {
                "domain": d,
                "paired_tokens": ratios[d][0],
                "corpus_tokens": ratios[d][1],
                "ratio": ratios[d][2],
            }
            for d in cfg.domains
        ],
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="T2 item 1: sentence-length statistic.")
    parser.add_argument("--config", type=Path, default=None,
                        help="path to the data config JSON (default: configs/data.json)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    per_domain, pooled, documents = collect(cfg)     # prints inventory counts

    ratios = inventory_ratios(cfg, documents)        # raises on the wind bound

    report = render(cfg, per_domain, pooled, ratios)
    print()
    print(report)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "s01_lengths.md").write_text(report + "\n", encoding="utf-8")
    (_OUT_DIR / "s01_lengths.json").write_text(
        json.dumps(build_payload(cfg, per_domain, pooled, ratios),
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[s01] wrote {_OUT_DIR / 's01_lengths.md'}")
    print(f"[s01] wrote {_OUT_DIR / 's01_lengths.json'}")


if __name__ == "__main__":
    main()
