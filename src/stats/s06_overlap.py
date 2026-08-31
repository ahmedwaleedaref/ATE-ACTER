"""T2 item 5 -- term-set overlap between the training domains and htfl.

Three overlaps denominated in htfl gold terms: type (shared list entry), text
(sequence in the training annotated stream), head (shared final token). Terms-only
keys; a terms+NE table follows. Reuses loading.load_domain + s05_term_stats.
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
from src.stats.s05_term_stats import ngram_counts, read_terms, unique_lists_dir  # noqa: E402

_OUT_DIR = _REPO_ROOT / "results" / "data_stats"
_LEN = ("1", "2", "3", "4+")


def load_gold(cfg, domain, suffix):
    path, = unique_lists_dir(cfg, domain).glob(f"*_tokenised_terms{suffix}.tsv")
    return read_terms(path)


def ann_stream(docs):
    """One lowercased token list per sentence, from the annotation files."""
    return [[t.lower() for t in tok] for doc in docs for tok, _ in doc.sentences]


def bucket(term):
    k = len(term.split())
    return str(k) if k <= 3 else "4+"


def pct(n, d):
    return f"{100 * n / d:.1f}% (N={d})"


def overlaps(cfg, train, test, suffix, dom_stream, combined_stream, occ):
    htfl = load_gold(cfg, test, suffix)
    n = len(htfl)
    dom_gold = {d: set(load_gold(cfg, d, suffix)) for d in train}
    train_gold = set().union(*dom_gold.values())
    heads = {g.split()[-1] for g in train_gold}
    lengths = sorted({len(t.split()) for t in htfl})
    train_ng = ngram_counts(combined_stream, lengths)
    dom_ng = {d: ngram_counts(dom_stream[d], lengths) for d in train}

    type_hits = [t for t in htfl if t in train_gold]
    text_hits = [t for t in htfl if tuple(t.split()) in train_ng]
    text_not_gold = [t for t in text_hits if t not in train_gold]
    mw = [t for t in htfl if len(t.split()) >= 2]
    head_hits = [t for t in mw if t.split()[-1] in heads]
    head_not_type = [t for t in head_hits if t not in train_gold]

    by_len = {}
    for b in _LEN:
        denom = sum(1 for t in htfl if bucket(t) == b)
        by_len[b] = (sum(1 for t in type_hits if bucket(t) == b), denom)

    per_domain = {}
    for d in train:
        per_domain[d] = {
            "type_n": sum(1 for t in htfl if t in dom_gold[d]),
            "text_n": sum(1 for t in htfl if tuple(t.split()) in dom_ng[d]),
        }

    total_occ = sum(occ.get(t, 0) for t in htfl)
    type_occ = sum(occ.get(t, 0) for t in type_hits)
    top = sorted(htfl, key=lambda t: occ.get(t, 0), reverse=True)[:20]
    top_rows = [(t, occ.get(t, 0), t in train_gold,
                 tuple(t.split()) in train_ng,
                 len(t.split()) >= 2 and t.split()[-1] in heads) for t in top]

    return {
        "n_htfl": n,
        "type": {"n": len(type_hits), "by_length": by_len},
        "text": {"n": len(text_hits), "not_in_gold_n": len(text_not_gold)},
        "head": {"n_mw": len(mw), "n": len(head_hits),
                 "not_type_n": len(head_not_type)},
        "per_domain": per_domain,
        "occ_weighted": {"total": total_occ, "in_type": type_occ},
        "top20": top_rows,
    }


def collect(cfg):
    train = list(cfg.train_domains)
    test = cfg.test_domain
    dom_stream = {d: ann_stream(load_domain(d, cfg)) for d in train}
    combined_stream = [s for d in train for s in dom_stream[d]]

    # htfl (a) occurrence counts -- decode gold BIO spans inline: B opens a
    # span, I continues it (or opens one), anything else ends it.
    occ = Counter()
    for doc in load_domain(test, cfg):
        for tokens, labels in doc.sentences:
            cur = []
            for tok, lab in zip(tokens, labels):
                if lab == "B":
                    if cur:
                        occ[" ".join(cur).lower()] += 1
                    cur = [tok]
                elif lab == "I":
                    cur.append(tok)
                else:
                    if cur:
                        occ[" ".join(cur).lower()] += 1
                    cur = []
            if cur:
                occ[" ".join(cur).lower()] += 1

    res = {k: overlaps(cfg, train, test, sfx, dom_stream, combined_stream, occ)
           for sfx, k in (("", "terms_only"), ("_nes", "terms_nes"))}
    return train, test, res


def build_markdown(train, res):
    o = res["terms_only"]
    n = o["n_htfl"]
    t, h, ow = o["text"], o["head"], o["occ_weighted"]
    L = ["# s06 -- term-set overlap, training domains vs htfl", "",
         f"Terms-only keys (`without_named_entities`), N = {n} htfl gold terms; a terms+NE table follows. Training side = corp + equi + wind gold lists / annotated streams combined. Token-sequence matching, lowercased. Every `%` carries its `N`.", "",
         "## 1. Type overlap -- htfl terms that are also training gold entries", "",
         "| measure | count | pct |", "|---|--:|--:|",
         f"| htfl ∩ training gold | {o['type']['n']} | {pct(o['type']['n'], n)} |", "",
         "### by htfl term length", "",
         "| length | overlap | htfl terms | pct |", "|---|--:|--:|--:|"]
    for b in _LEN:
        hit, den = o["type"]["by_length"][b]
        L.append(f"| {b} | {hit} | {den} | {pct(hit, den)} |")

    L += ["", "## 2. Text overlap -- htfl term sequence occurs in the training annotated token stream (any label)", "",
          "| measure | count | pct |", "|---|--:|--:|",
          f"| htfl term sequence seen in training text | {t['n']} | {pct(t['n'], n)} |",
          f"| ... of those, not in any training gold list | {t['not_in_gold_n']} | {pct(t['not_in_gold_n'], n)} |"]

    L += ["", "## 3. Head overlap -- htfl term final token = a training gold term final token", "",
          f"Restricted to htfl terms of ≥2 tokens (N = {h['n_mw']}).", "",
          "| measure | count | pct |", "|---|--:|--:|",
          f"| shared final token | {h['n']} | {pct(h['n'], h['n_mw'])} |",
          f"| ... not already in type overlap | {h['not_type_n']} | {pct(h['not_type_n'], h['n_mw'])} |"]

    L += ["", "## 4. Per training domain alone, vs htfl", "",
          "| domain | type overlap | pct | text overlap | pct |", "|---|--:|--:|--:|--:|"]
    for d in train:
        pd = o["per_domain"][d]
        L.append(f"| {d} | {pd['type_n']} | {pct(pd['type_n'], n)} | {pd['text_n']} | {pct(pd['text_n'], n)} |")

    L += ["", "## 5. 20 most frequent htfl terms by (a), with overlap flags", "",
          "| term | (a) | type | text | head |", "|---|--:|:-:|:-:|:-:|"]
    flag = {True: "yes", False: "—"}
    for term, a, ty, tx, hd in o["top20"]:
        L.append(f"| `{term}` | {a} | {flag[ty]} | {flag[tx]} | {flag[hd]} |")

    L += ["", "## 6. Occurrence-weighted type overlap", "",
          "htfl annotated term occurrences (s05 count (a)) belonging to terms present in the training gold lists.", "",
          "| set | occurrences | pct |", "|---|--:|--:|",
          f"| all htfl gold terms | {ow['total']} | {pct(ow['total'], ow['total'])} |",
          f"| in training gold types | {ow['in_type']} | {pct(ow['in_type'], ow['total'])} |"]

    ne, m = res["terms_nes"], res["terms_nes"]["n_htfl"]
    hn = ne["head"]["n_mw"]
    L += ["", "## terms+NE key (both sides) -- comparison only", "",
          f"htfl gold terms+NE: N = {m}. Named entities recur across domains and inflate overlap; keys are not mixed across the two sides.", "",
          "| measure | count | pct |", "|---|--:|--:|",
          f"| type overlap | {ne['type']['n']} | {pct(ne['type']['n'], m)} |",
          f"| text overlap | {ne['text']['n']} | {pct(ne['text']['n'], m)} |",
          f"| ... in text but not in training gold | {ne['text']['not_in_gold_n']} | {pct(ne['text']['not_in_gold_n'], m)} |",
          f"| head overlap (N_mw = {hn}) | {ne['head']['n']} | {pct(ne['head']['n'], hn)} |",
          f"| ... not already in type overlap | {ne['head']['not_type_n']} | {pct(ne['head']['not_type_n'], hn)} |", ""]
    return "\n".join(L)


def main():
    cfg = load_config()
    train, _, res = collect(cfg)
    report = build_markdown(train, res)
    print(report)
    payload = {"generated_by": "src/stats/s06_overlap.py",
               "note": ("overlap denominated in htfl gold terms; terms-only "
                        "keys, with a separate terms+NE comparison"),
               "terms_only": res["terms_only"], "terms_nes": res["terms_nes"]}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "s06_overlap.md").write_text(report + "\n", encoding="utf-8")
    (_OUT_DIR / "s06_overlap.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[s06] wrote {_OUT_DIR / 's06_overlap.md'}")
    print(f"[s06] wrote {_OUT_DIR / 's06_overlap.json'}")


if __name__ == "__main__":
    main()
