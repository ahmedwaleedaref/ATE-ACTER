"""T2 items 3 & 4 -- term length and term frequency over the gold unique lists.

Completeness statistics and inputs to T4 (C-Value). Tokenised gold lists only.
Reuses loading.load_domain and s01_lengths.percentile.
Run: python -m src.statistics.s04_term_stats
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.statistics.loading import load_config, load_domain  # noqa: E402
from src.statistics.s01_lengths import percentile  # noqa: E402

_OUT_DIR = _REPO_ROOT / "results" / "T2_data_stats"
_BUCKETS = ("1", "2", "3", "4", "5", "6", "7", "8+")
_KEYS = (("terms", "terms"), ("terms_nes", "terms+NE"))


def unique_lists_dir(cfg, domain):
    return (cfg.data_root / cfg.language / domain / "annotated"
            / "annotations" / "unique_annotation_lists")


def read_terms(path):
    """TSV, two columns: term then category. Drop the category column."""
    return [ln.split("\t")[0] for ln in path.read_text("utf-8").splitlines()]


def length_stats(terms):
    lens = sorted(len(t.split()) for t in terms)
    n = len(lens)
    cnt = Counter(min(x, 8) for x in lens)
    return {
        "n_entries": n,
        "length_dist": [(b, cnt[i + 1], 100 * cnt[i + 1] / n)
                        for i, b in enumerate(_BUCKETS)],
        "cum_pct_le_4": 100 * sum(cnt[k] for k in (1, 2, 3, 4)) / n,
        "p50": percentile(lens, 50), "p95": percentile(lens, 95),
        "p99": percentile(lens, 99), "max": lens[-1],
        "longest_10": sorted(terms, key=lambda t: len(t.split()), reverse=True)[:10],
        "n_with_uppercase": sum(1 for t in terms if t != t.lower()),
        "n_dup_after_lower": n - len({t.lower() for t in terms}),
    }


def ngram_counts(lines, lengths):
    c = Counter()
    for toks in lines:
        for k in lengths:
            for i in range(len(toks) - k + 1):
                c[tuple(toks[i:i + k])] += 1
    return c


def freq_summary(values):
    v = sorted(values)
    hapax = sum(1 for x in v if x == 1)
    return {"hapax": hapax, "hapax_pct": 100 * hapax / len(v),
            "p50": percentile(v, 50), "p90": percentile(v, 90),
            "p99": percentile(v, 99), "max": v[-1]}


def collect(cfg):
    domains = list(cfg.domains)
    part_a, part_b = {}, {}
    for d in domains:
        udir = unique_lists_dir(cfg, d)
        only_path, = udir.glob("*_tokenised_terms.tsv")
        nes_path, = udir.glob("*_tokenised_terms_nes.tsv")
        gold = read_terms(only_path)
        part_a[d] = {"terms": length_stats(gold),
                     "terms_nes": length_stats(read_terms(nes_path))}

        docs = load_domain(d, cfg)
        # (a) decode gold BIO spans: B opens a span, I continues it, anything
        # else ends it; an I with no open span opens one too and is counted.
        spans, i_no_b = Counter(), 0
        for doc in docs:
            for tokens, labels in doc.sentences:
                cur = []
                for tok, lab in zip(tokens, labels):
                    if lab == "B":
                        if cur:
                            spans[" ".join(cur).lower()] += 1
                        cur = [tok]
                    elif lab == "I":
                        if not cur:
                            i_no_b += 1
                        cur.append(tok)
                    else:
                        if cur:
                            spans[" ".join(cur).lower()] += 1
                        cur = []
                if cur:
                    spans[" ".join(cur).lower()] += 1
        ann_lines = [[t.lower() for t in tokens]
                     for doc in docs for tokens, _ in doc.sentences]
        whole = []
        for sub in ("annotated/texts_tokenised", "unannotated_texts"):
            dpath = cfg.data_root / cfg.language / d / sub
            if dpath.is_dir():
                for f in sorted(dpath.glob("*.txt")):
                    for ln in f.read_text(encoding="utf-8").splitlines():
                        whole.append([t.lower() for t in ln.split()])

        lengths = sorted({len(t.split()) for t in gold})
        bc = ngram_counts(ann_lines, lengths)
        cc = ngram_counts(whole, lengths)
        rows = [(t, spans.get(t, 0), bc.get(tuple(t.split()), 0),
                 cc.get(tuple(t.split()), 0)) for t in gold]
        _, av, bv, cv = zip(*rows)
        part_b[d] = {
            "n_terms": len(gold), "i_without_preceding_b": i_no_b,
            "total_a": sum(av), "total_b": sum(bv), "total_c": sum(cv),
            "b_over_a": sum(bv) / sum(av),
            "a": freq_summary(av), "b": freq_summary(bv), "c": freq_summary(cv),
            "top10_by_a": sorted(rows, key=lambda r: r[1], reverse=True)[:10],
        }
    return domains, part_a, part_b


def build_markdown(domains, part_a, part_b):
    L = ["# s04 -- term length and term frequency (gold unique lists)", "",
         "Tokenised gold lists only; the non-tokenised `*_terms.tsv` / `*_terms_nes.tsv` variant exists in every domain and is ignored. Length = whitespace tokens; percentiles s01 nearest-rank; every `%` is of the row's `N`.", "",
         "## Part A -- term length", "",
         "`terms` / `terms+NE` = gold key excluding / including named entities. Bucket cells `count (pct)`. `upper` = entries with an uppercase char (§4 says the lists are lowercased); `dup` = entries minus distinct lowercased forms.", "",
         "| domain | key | N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ | cum ≤4 | p50 | "
         "p95 | p99 | max | upper | dup |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for d in domains:
        for key, label in _KEYS:
            s = part_a[d][key]
            cells = " | ".join(f"{c} ({p:.1f}%)" for _b, c, p in s["length_dist"])
            L.append(
                f"| {d} | {label} | {s['n_entries']} | {cells} | {s['cum_pct_le_4']:.1f}% "
                f"| {s['p50']} | {s['p95']} | {s['p99']} | {s['max']} "
                f"| {s['n_with_uppercase']} | {s['n_dup_after_lower']} |")
    L += ["", "### 10 longest terms per domain and key", ""]
    for d in domains:
        for key, label in _KEYS:
            ts = part_a[d][key]["longest_10"]
            L.append(f"- **{d} / {label}**: "
                     + "; ".join(f"`{t}` ({len(t.split())})" for t in ts))
    L += ["", "## Part B -- term frequency (terms-only key)", "",
          "(a) decoded gold-BIO span occurrences (tokens space-joined, lowercased). (b) the term's token sequence in the annotated token stream, any label. (c) same over the whole corpus (`texts_tokenised/` + `unannotated_texts/`, htfl has none). Every gold term counted, zeros included.", ""]
    for d in domains:
        pb = part_b[d]
        srows = [("total occurrences", pb["total_a"], pb["total_b"], pb["total_c"]),
                 ("hapax (freq = 1)", pb["a"]["hapax"], pb["b"]["hapax"], pb["c"]["hapax"]),
                 (f"hapax % (N={pb['n_terms']})", *(f"{pb[x]['hapax_pct']:.1f}%" for x in "abc"))]
        srows += [(f"{q} freq", pb["a"][q], pb["b"][q], pb["c"][q]) for q in ("p50", "p90", "p99", "max")]
        L += [f"### {d} (N={pb['n_terms']} terms)", "",
              f"`I` with no preceding `B` during decode: {pb['i_without_preceding_b']}. "
              f"total (b) / total (a) = {pb['total_b']} / {pb['total_a']} = {pb['b_over_a']:.2f}.", "",
              "| stat | (a) annotated occ | (b) surface annotated | (c) surface corpus |",
              "|---|--:|--:|--:|"]
        L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in srows]
        L += ["", "10 most frequent terms by (a):", "",
              "| term | (a) | (b) | (c) |", "|---|--:|--:|--:|"]
        L += [f"| `{t}` | {a} | {b} | {c} |" for t, a, b, c in pb["top10_by_a"]]
        L.append("")
    return "\n".join(L)


def main():
    cfg = load_config()
    print("# file inventory -- unique_annotation_lists/\n")
    for d in cfg.domains:
        udir = unique_lists_dir(cfg, d)
        print(f"[{d}] {udir}")
        for f in sorted(udir.glob("*")):
            n = len(f.read_text(encoding="utf-8").splitlines())
            print(f"    {f.name:<40s} {n:>6d} lines")
    print()
    domains, part_a, part_b = collect(cfg)
    report = build_markdown(domains, part_a, part_b)
    print(report)
    payload = {"generated_by": "src/statistics/s04_term_stats.py",
               "note": ("tokenised gold lists only; the non-tokenised variant "
                        "exists in every domain and is ignored"),
               "part_a_term_length": part_a, "part_b_term_frequency": part_b}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "s04_term_stats.md").write_text(report + "\n", encoding="utf-8")
    (_OUT_DIR / "s04_term_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[s04] wrote {_OUT_DIR / 's04_term_stats.md'}")
    print(f"[s04] wrote {_OUT_DIR / 's04_term_stats.json'}")


if __name__ == "__main__":
    main()
