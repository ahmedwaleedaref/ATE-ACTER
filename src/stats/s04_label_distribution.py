"""T2 item 6 -- label distribution and class imbalance, in dataset-token space.

Reuses loading.load_domain, no tokenizer. Part A is a fixed invariant of ACTER
v1.5: a dataloader that later reports a different positive rate has broken its
label alignment. Part B splits sentences short (<=2 tok) / prose (>=3 tok) to
decide whether wind's gold terms live in its table-cell fragments.
Run: python -m src.stats.s04_label_distribution
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.stats.loading import load_config, load_domain  # noqa: E402

_OUT_DIR = _REPO_ROOT / "results" / "data_stats"


def _dist(B, I, O, n_sent, n_zero, n_docs):
    t = B + I + O
    return {"n_documents": n_docs, "n_sentences": n_sent, "n_tokens": t,
            "B": B, "I": I, "O": O,
            "pct_B": 100 * B / t, "pct_I": 100 * I / t, "pct_O": 100 * O / t,
            "positive_rate": (B + I) / t, "mean_term_occurrence_len": (B + I) / B,
            "n_zero_positive_sentences": n_zero,
            "pct_zero_positive_sentences": 100 * n_zero / n_sent}


def _bucket(n_sent, n_tok, b, i, o, dom):
    return {"n_sentences": n_sent,
            "pct_of_domain_sentences": 100 * n_sent / dom["n_sentences"],
            "n_tokens": n_tok, "pct_of_domain_tokens": 100 * n_tok / dom["n_tokens"],
            "B": b, "I": i, "O": o, "positive_rate": (b + i) / (b + i + o),
            "pct_of_domain_B": 100 * b / dom["B"]}


def collect(cfg):
    domains = list(cfg.domains)
    per, buckets = {}, {}
    tB = tI = tO = tsent = tzero = tdocs = 0
    wind_short, wind_labels = Counter(), {}

    for d in domains:
        docs = load_domain(d, cfg)
        B = I = O = nsent = nzero = 0
        s = [0, 0, 0, 0, 0]                       # n_sent, n_tok, B, I, O  (short)
        p = [0, 0, 0, 0, 0]                       # ...                     (prose)
        for doc in docs:
            for tokens, labels in doc.sentences:
                nsent += 1
                cb, ci, co = labels.count("B"), labels.count("I"), labels.count("O")
                B += cb; I += ci; O += co
                nzero += cb + ci == 0
                bk = s if len(tokens) <= 2 else p
                bk[0] += 1; bk[1] += len(tokens); bk[2] += cb; bk[3] += ci; bk[4] += co
                if d == "wind" and len(tokens) <= 2:
                    wind_short[tuple(tokens)] += 1
                    wind_labels[tuple(tokens)] = labels
        per[d] = _dist(B, I, O, nsent, nzero, len(docs))
        buckets[d] = {"short": _bucket(*s, per[d]), "prose": _bucket(*p, per[d])}
        tB += B; tI += I; tO += O; tsent += nsent; tzero += nzero; tdocs += len(docs)

    per["pooled"] = _dist(tB, tI, tO, tsent, tzero, tdocs)
    top = [(list(t), wind_labels[t], c) for t, c in wind_short.most_common(20)]
    return domains, per, buckets, top


def build_markdown(domains, per, buckets, top):
    L = ["# s04 -- label distribution and class imbalance", "",
         "Dataset-token space, no tokenizer. B / I / O counted separately -- B is "
         "the term-*occurrence* count (item 4 needs it). **Part A is a fixed "
         "invariant of ACTER v1.5, not a modelling signal**: a different positive "
         "rate downstream means broken label alignment.", "",
         "## Part A -- per domain and pooled", "",
         "B/I/O `%` is of the row's `n_tokens`; zero-positive `%` is of its "
         "`n_sentences`.", "",
         "| domain | n_docs | n_sent | n_tokens | B | I | O | pos. rate | "
         "mean occ. len | zero-pos sents |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for d in (*domains, "pooled"):
        p = per[d]
        L.append(f"| {d} | {p['n_documents']} | {p['n_sentences']} | {p['n_tokens']} "
                 f"| {p['B']} ({p['pct_B']:.1f}%) | {p['I']} ({p['pct_I']:.1f}%) | "
                 f"{p['O']} ({p['pct_O']:.1f}%) | {p['positive_rate']:.4f} | "
                 f"{p['mean_term_occurrence_len']:.2f} | "
                 f"{p['n_zero_positive_sentences']} ({p['pct_zero_positive_sentences']:.1f}%) |")

    L += ["", "## Part B -- short (<=2 tokens) vs prose (>=3 tokens)", "",
          "| domain | bucket | n_sent | % dom sent | n_tok | % dom tok | B | I | O | "
          "pos. rate | B % of domain B |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for d in domains:
        for k in ("short", "prose"):
            b = buckets[d][k]
            L.append(f"| {d} | {k} | {b['n_sentences']} | "
                     f"{b['pct_of_domain_sentences']:.1f}% | {b['n_tokens']} | "
                     f"{b['pct_of_domain_tokens']:.1f}% | {b['B']} | {b['I']} | "
                     f"{b['O']} | {b['positive_rate']:.4f} | {b['pct_of_domain_B']:.1f}% |")

    ws = buckets["wind"]["short"]["pct_of_domain_B"]
    verdict = ("under ~5% -- structural noise, filterable at training time" if ws < 10
               else "20%+ -- wind terminology lives in table cells; dropping them "
               "discards real training signal and every wind result carries that caveat"
               if ws >= 20 else "between the thresholds -- a judgement call, see the sample")
    L += ["", f"**Decision (wind).** The short bucket holds **{ws:.1f}%** of wind's "
          f"total B count: {verdict}. This statistic measures; it does not filter.", "",
          "## 20 most frequent wind short-bucket sentences", "",
          "| count | tokens | labels |", "|--:|---|---|"]
    for t, lab, c in top:
        cell = " ".join(t).replace("|", "\\|")
        L.append(f"| {c} | {cell} | {' '.join(lab)} |")
    L.append("")
    return "\n".join(L)


def main():
    cfg = load_config()
    domains, per, buckets, top = collect(cfg)

    report = build_markdown(domains, per, buckets, top)
    print()
    print(report)

    payload = {"generated_by": "src/stats/s04_label_distribution.py",
               "note": ("fixed invariant of ACTER v1.5; a different positive rate "
                        "downstream means broken label alignment"),
               "per_domain": per, "short_prose_split": buckets,
               "wind_short_bucket_top_sentences":
                   [{"count": c, "tokens": t, "labels": lab} for t, lab, c in top]}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "s04_label_distribution.md").write_text(report + "\n", encoding="utf-8")
    (_OUT_DIR / "s04_label_distribution.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[s04] wrote {_OUT_DIR / 's04_label_distribution.md'}")
    print(f"[s04] wrote {_OUT_DIR / 's04_label_distribution.json'}")


if __name__ == "__main__":
    main()
