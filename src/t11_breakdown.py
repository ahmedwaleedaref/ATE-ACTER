"""T11 error analysis: one run -> one markdown report.

Generated scaffolding, same rule as ``src/aggregate.py``. It never re-scores and
never touches a model: it reads the predicted term list ``run_train.py --dump-terms``
wrote, recomputes the gold-side facts through the hand-written harness
(``load_gold_list_into_set``, ``decode``, ``load_domain``), and does arithmetic.

Every breakdown is a set operation between the predicted list and a bucket of the
gold key, so no checkpoint and no GPU are needed.

    python -m src.t11_breakdown --run results/runs/t11/deberta_lr1e-05_e5/seed_42.json
    python -m src.t11_breakdown --terms <list.txt> --label "C-Value" --out <path.md>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.eval.run_eval import _gold_key_path, load_eval_config
from src.eval.scorers import load_gold_list_into_set
from src.eval.spans import decode
from src.stats.loading import load_config, load_domain

_REPO_ROOT = Path(__file__).resolve().parents[1]

# results/ceilings.md, ANN rows. Measured, copied verbatim.
CEILING = {"equi": 0.9523, "htfl": 0.9096}

# EXPERIMENTS.md E05, the cell this run reproduces. Quoted for the drift check,
# never overwritten.
E05 = {"equi_mean": 0.5590, "equi_std": 0.0086, "htfl_mean": 0.5784, "htfl_std": 0.0228}

LENGTH_BUCKETS = ("1", "2", "3", "4+")


def bucket_of(term: str) -> str:
    n = len(term.split())
    return str(n) if n <= 3 else "4+"


def gold_facts(domain: str, key: str) -> dict:
    """Everything the gold side supplies. No model involved.

    ``occurrences`` counts DECODED GOLD maximal spans, which is the (a) column of
    Data_stats.md section 7.3 -- the definition the 47.7% hapax figure uses. A gold
    term with 0 occurrences is one the scheme never decodes as its own span.
    """
    cfg = load_config()
    gold_key = load_gold_list_into_set(str(_gold_key_path(cfg, domain, key)))

    occurrences: Counter = Counter()
    for doc in load_domain(domain):
        for tokens, labels in doc.sentences:
            for start, end in decode(tokens, labels, "bio"):
                occurrences[" ".join(tokens[start:end]).lower()] += 1

    return {"key": gold_key, "occurrences": occurrences, "decoded": set(occurrences)}


def train_length_profile(data_cfg) -> tuple[Counter, int]:
    """Decoded gold span lengths over the training domains, occurrence-weighted.

    Occurrences, not types: this is what the model is shown. The htfl side of the
    comparison is type-weighted, because the metric is. The two units are stated
    rather than silently mixed.
    """
    spans: Counter = Counter()
    total = 0
    for domain in data_cfg.train_domains:
        for doc in load_domain(domain):
            for tokens, labels in doc.sentences:
                for start, end in decode(tokens, labels, "bio"):
                    spans[bucket_of(" ".join(tokens[start:end]))] += 1
                    total += 1
    return spans, total


def pct(n: int, d: int) -> str:
    return f"{n / d:.4f}" if d else "--"


def build_report(pred: set[str], gold: dict, run: dict | None, label: str,
                 data_cfg) -> str:
    key, occ, decoded = gold["key"], gold["occurrences"], gold["decoded"]
    reachable = key & decoded
    unreachable = key - decoded
    hit = key & pred

    L = []
    L.append(f"# T11 — error analysis, {label}")
    L.append("")
    L.append("htfl, ANN key. Every breakdown is a set operation between the predicted")
    L.append("term list and a bucket of the gold key; no checkpoint, no GPU.")
    L.append("")

    # ---- headline -----------------------------------------------------------
    if run is not None:
        t = run["test"]
        L.append("| | this run | E05 cell (5-seed mean) |")
        L.append("|---|--:|--:|")
        L.append(f"| equi, best epoch {run['best_epoch']} | {run['best_equi_f1']:.4f} | "
                 f"{E05['equi_mean']:.4f} +/- {E05['equi_std']:.4f} |")
        L.append(f"| htfl list F1 | {t['list_ann_f1']:.4f} | "
                 f"{E05['htfl_mean']:.4f} +/- {E05['htfl_std']:.4f} |")
        L.append("")
        L.append(f"htfl P {t['list_ann_p']:.4f} / R {t['list_ann_r']:.4f}, "
                 f"{t['n_pred_spans']:,} spans -> {t['n_pred_types']:,} types. "
                 f"Ceiling {CEILING['htfl']:.4f}, so {t['list_ann_f1'] / CEILING['htfl']:.3f} of it.")
        L.append("")
        L.append("A gap against E05 is the nondeterminism E04 measured (`cudnn_deterministic`")
        L.append("false, a repeated seed diverging from epoch 2). E05 is not overwritten.")
    else:
        L.append(f"No run JSON: list F1, span F1 and the invalid-tag counts are unavailable.")
        L.append(f"Predicted {len(pred):,} types against a gold key of {len(key):,}.")
    L.append("")

    # ---- 1. recall by length ------------------------------------------------
    L.append("## 1. Recall by term length")
    L.append("")
    L.append("`ceiling` is what decoding GOLD BIO recovers in this bucket. It is not a")
    L.append("hard cap on a model: a tagger that segments differently can emit a term")
    L.append("gold only ever marks as nested, so `/ ceiling` may exceed 1.0.")
    L.append("")
    L.append("| length | gold | ceiling | recall | / ceiling |")
    L.append("|---|--:|--:|--:|--:|")
    for b in LENGTH_BUCKETS:
        g = {t for t in key if bucket_of(t) == b}
        c = len(g & decoded) / len(g)
        r = len(g & pred) / len(g)
        L.append(f"| {b} | {len(g):,} | {c:.4f} | {r:.4f} | "
                 f"{r / c if c else float('nan'):.3f} |")
    L.append(f"| **all** | **{len(key):,}** | **{len(reachable) / len(key):.4f}** | "
             f"**{len(hit) / len(key):.4f}** | "
             f"**{(len(hit) / len(key)) / (len(reachable) / len(key)):.3f}** |")
    L.append("")
    L.append("**Prediction (Data_stats.md 8.2, logged contrarian):** multi-word recall")
    L.append("exceeds single-word via head-position transfer. The standard ATE finding is")
    L.append("the reverse. Resolve on the `/ ceiling` column -- the raw column is tilted")
    L.append("toward the prediction by roughly 6 points of scheme ceiling.")
    L.append("")
    L.append("VERDICT: ")
    L.append("")

    # ---- 2. recall by frequency ---------------------------------------------
    L.append("## 2. Recall by gold frequency")
    L.append("")
    L.append("Frequency = decoded gold-BIO maximal-span occurrences (Data_stats.md 7.3,")
    L.append("column (a)). The 0 bucket is the terms the scheme never decodes as their own")
    L.append("span -- recall there is not bounded by 0, and is the direct test of")
    L.append("data_layout.md 5.1's claim that no BIO tagger can emit them.")
    L.append("")
    L.append("| frequency | gold | share of key | recall |")
    L.append("|---|--:|--:|--:|")
    buckets = {
        "0 (never a maximal span)": unreachable,
        "1 (singleton)": {t for t in reachable if occ[t] == 1},
        ">= 2": {t for t in reachable if occ[t] >= 2},
    }
    for name, g in buckets.items():
        L.append(f"| {name} | {len(g):,} | {len(g) / len(key):.3f} | "
                 f"{len(g & pred) / len(g) if g else float('nan'):.4f} |")
    assert sum(len(g) for g in buckets.values()) == len(key), "buckets do not partition the key"
    L.append("")
    L.append("VERDICT: ")
    L.append("")

    # ---- 3. span vs list ----------------------------------------------------
    L.append("## 3. Exact-span F1 against unique-list F1")
    L.append("")
    if run is not None and "span_f1" in run["test"]:
        t = run["test"]
        L.append("| metric | P | R | F1 | ceiling |")
        L.append("|---|--:|--:|--:|--:|")
        L.append(f"| unique-list (headline) | {t['list_ann_p']:.4f} | {t['list_ann_r']:.4f} | "
                 f"{t['list_ann_f1']:.4f} | {CEILING['htfl']:.4f} |")
        L.append(f"| exact-span (diagnostic) | {t['span_p']:.4f} | {t['span_r']:.4f} | "
                 f"{t['span_f1']:.4f} | 1.0 |")
        L.append("")
        L.append(f"Gap: {t['span_f1'] - t['list_ann_f1']:+.4f}. Span F1 has no ANN/NES split --")
        L.append("gold spans come from the labels, and the key distinction exists only in the")
        L.append("list metric. Span space carries no scheme loss, so its ceiling is 1.0.")
        L.append("")
        L.append("**Prediction (T3 spec):** span high with list low means frequent terms found")
        L.append("and rare types missed -- the expected shape at 47.7% hapax. The two landing")
        L.append("close together would mean uniform performance across the frequency")
        L.append("distribution, which would be surprising.")
        L.append("")
        L.append("VERDICT: ")
    else:
        L.append("Unavailable -- needs a run JSON carrying `span_f1`.")
    L.append("")

    # ---- 4. invalid tags ----------------------------------------------------
    L.append("## 4. Invalid tag sequences")
    L.append("")
    if run is not None and "n_i_after_o" in run["test"]:
        t = run["test"]
        total = t["n_i_sentence_initial"] + t["n_i_after_o"]
        L.append("| pattern | count |")
        L.append("|---|--:|")
        L.append(f"| `I` opening a sentence | {t['n_i_sentence_initial']:,} |")
        L.append(f"| `I` after `O` | {t['n_i_after_o']:,} |")
        L.append(f"| **total dangling runs** | **{total:,}** |")
        L.append("")
        L.append(f"Against {t['n_pred_spans']:,} predicted spans. Counted as events: a run of")
        L.append("`I` with no `B` scores once. Gold htfl is (1, 0), so these are the model's")
        L.append("alone. The dangling-I policy already drops them in `decode`, so this is what")
        L.append("was being discarded silently.")
        L.append("")
        L.append("**Decides whether a CRF is worth considering at all.**")
        L.append("")
        L.append("VERDICT: ")
    else:
        L.append("Unavailable -- needs a run JSON carrying `n_i_after_o`.")
    L.append("")

    # ---- 5. long terms in training ------------------------------------------
    L.append("## 5. Long terms in training")
    L.append("")
    L.append("Seed-invariant -- a property of the corpus, identical in every run's report.")
    L.append("")
    train_spans, train_total = train_length_profile(data_cfg)
    L.append(f"| length | train occurrences ({'+'.join(data_cfg.train_domains)}) | share | "
             f"htfl key types | share |")
    L.append("|---|--:|--:|--:|--:|")
    for b in LENGTH_BUCKETS:
        g = {t for t in key if bucket_of(t) == b}
        L.append(f"| {b} | {train_spans[b]:,} | {train_spans[b] / train_total:.3f} | "
                 f"{len(g):,} | {len(g) / len(key):.3f} |")
    L.append("")
    L.append("Units differ and are not mixed: training is occurrence-weighted, because that")
    L.append("is what the model is shown; htfl is type-weighted, because the metric is.")
    L.append("")
    L.append("VERDICT: ")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="T11 error analysis for one run.")
    ap.add_argument("--run", help="results/runs/<group>/seed_<n>.json; its term list is "
                                  "read from the test_term_list pointer")
    ap.add_argument("--terms", help="a term list directly, when there is no run JSON")
    ap.add_argument("--label", default=None, help="title for the report")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.run or args.terms, "need --run or --terms"

    run = None
    if args.run:
        run_path = Path(args.run)
        run = json.loads(run_path.read_text(encoding="utf-8"))
        assert "test_term_list" in run, (
            f"{run_path} has no test_term_list -- it was produced without --dump-terms")
        terms_path = _REPO_ROOT / run["test_term_list"]["path"]
        label = args.label or (f"{run['config']['model_name']} "
                               f"lr{run['config']['learning_rate']:g} "
                               f"e{run['config']['num_epochs']} seed {run['seed']}")
        out = Path(args.out) if args.out else run_path.parent / f"breakdown_seed_{run['seed']}.md"
    else:
        terms_path = Path(args.terms)
        label = args.label or terms_path.stem
        out = Path(args.out) if args.out else terms_path.with_suffix(".breakdown.md")

    pred = load_gold_list_into_set(str(terms_path))
    if run is not None:
        assert len(pred) == run["test"]["n_pred_types"], (
            f"term list has {len(pred)} entries, run JSON reports "
            f"{run['test']['n_pred_types']} predicted types")

    data_cfg = load_config()
    eval_cfg = load_eval_config()
    gold = gold_facts(data_cfg.test_domain, eval_cfg["keys"][0])

    report = build_report(pred, gold, run, label, data_cfg)
    out = out if out.is_absolute() else _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    try:
        shown = out.relative_to(_REPO_ROOT)
    except ValueError:      # --out pointed outside the repo
        shown = out
    print(f"wrote {shown}")


if __name__ == "__main__":
    main()
