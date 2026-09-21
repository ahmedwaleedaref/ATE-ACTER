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

import yaml

from src.eval.scorers import load_gold_list_into_set, score_list
from src.eval.surface import generate_unique_list
from src.statistics.loading import DataConfig, load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOMAINS = ("corp", "equi", "wind", "htfl")
_DEFAULT_OUT = _REPO_ROOT / "results" / "T3_eval_harness" / "ceilings.md"
_DEFAULT_EVAL_CONFIG = _REPO_ROOT / "configs" / "eval.yaml"

# configs/eval.yaml records the decisions this run reports (headline/secondary
# metric, scheme, label variant, which keys, the dangling-I policy) so they are
# greppable, not folklore. The gold key filename suffix per key name is a path
# detail, not a decision, so it stays here rather than in the config.
_KEY_SUFFIX_BY_NAME = {"ann": "", "nes": "_nes"}


def load_eval_config(path=None) -> dict:
    """Load configs/eval.yaml -- the locked evaluation decisions."""
    path = Path(path) if path is not None else _DEFAULT_EVAL_CONFIG
    assert path.is_file(), f"eval config not found: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _gold_key_path(cfg: DataConfig, domain: str, key: str) -> Path:
    """Tokenised gold unique-term-list path for one domain/key, built from the
    domain name and the config -- never a hardcoded single path."""
    suffix = _KEY_SUFFIX_BY_NAME[key]
    return (
        cfg.data_root / cfg.language / domain / "annotated" / "annotations"
        / "unique_annotation_lists"
        / f"{domain}_{cfg.language}_tokenised_terms{suffix}.tsv"
    )


def compute_ceilings(domain: str, eval_config: dict) -> dict:
    """
    Decode gold labels for one domain, collapse to a unique term list, and score
    against both gold keys named in ``eval_config['keys']``.

    Returns per-key precision/recall/F1 plus the span and type counts.
    """
    cfg = load_config()
    predicted_unique, n_spans, n_unique = generate_unique_list(domain)

    result = {"domain": domain, "n_spans": n_spans, "n_unique": n_unique}
    for key in eval_config["keys"]:
        key_path = _gold_key_path(cfg, domain, key)
        assert key_path.is_file(), f"gold key not found for {domain}/{key}: {key_path}"
        gold_set = load_gold_list_into_set(str(key_path))
        precision, recall, f1 = score_list(predicted_unique, gold_set)
        result[key] = {"precision": precision, "recall": recall, "f1": f1}

    return result


def _render_header(eval_config: dict, cfg: DataConfig) -> str:
    """Preamble recording which key each row used, the label directory, the
    scheme, and the dangling-I policy -- a ceiling reported without its key is
    meaningless."""
    lines = [
        "# List-metric ceilings (max_recall / max_precision)",
        "",
        f"- headline metric: {eval_config['headline_metric']}",
        f"- secondary metric: {eval_config['secondary_metric']}",
        f"- scheme: {eval_config['scheme']}",
        f"- labels: {cfg.ne_variant}",
        f"- keys scored (one predicted list, both keys): {', '.join(eval_config['keys'])}",
        f"- dangling-I policy: {eval_config['dangling_i_policy']} "
        f"(seqeval mode={eval_config['seqeval_mode']!r}, scheme={eval_config['seqeval_scheme']!r})",
        "",
    ]
    return "\n".join(lines)


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

    eval_config = load_eval_config()
    data_cfg = load_config()

    rows = []
    for domain in args.domains:
        ceilings = compute_ceilings(domain, eval_config)
        for key in eval_config["keys"]:
            scores = ceilings[key]
            rows.append({
                "domain": domain,
                "key": key.upper(),
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "n_spans": ceilings["n_spans"],
                "n_unique": ceilings["n_unique"],
            })

    header = _render_header(eval_config, data_cfg)
    table = _render_table(rows)
    output = header + "\n" + table
    print(output)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
