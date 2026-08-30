"""T2 item 1b -- wordpiece length statistic, four tokenizers, over the same
annotated portion as s01. Reuses loading.load_domain and s01_lengths.percentile.
Needs transformers, sentencepiece, protobuf; first run downloads vocab files
only. Run: python -m src.stats.s02_wordpieces
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.stats.loading import load_config, load_domain  # noqa: E402
from src.stats.s01_lengths import percentile  # noqa: E402

_OUT_DIR = _REPO_ROOT / "results" / "data_stats"
PERCENTILES = (50, 90, 95, 99)

# name, HF repo, pinned revision (main-branch commit), from_pretrained kwargs.
# roberta / xlm-r use byte-level BPE, which folds a leading space into the
# token; add_prefix_space=True makes the first word of a sentence tokenize the
# same as the rest under is_split_into_words=True.
TOKENIZERS = [
    ("bert-base-cased", "google-bert/bert-base-cased",
     "cd5ef92a9fb2f889e972770a36d4ed042daf221e", {}),
    ("roberta-base", "FacebookAI/roberta-base",
     "e2da8e2f811d1448a5b465c236feacd80ffbac7b", {"add_prefix_space": True}),
    ("deberta-v3-base", "microsoft/deberta-v3-base",
     "8ccc9b6f36199bec6961081d44eb72fb3f7353f3", {}),
    ("xlm-roberta-base", "FacebookAI/xlm-roberta-base",
     "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", {"add_prefix_space": True}),
]


def _row(wp, sub_in, sub_out, n512, n256) -> dict:
    w = sorted(wp)
    row = {
        "n_sentences": len(w),
        "wp_min": w[0],
        "wp_max": w[-1],
        "infl_inside_span": sum(sub_in) / len(sub_in),          # label B or I
        "infl_outside_span": sum(sub_out) / len(sub_out),       # label O
        "infl_overall": (sum(sub_in) + sum(sub_out)) / (len(sub_in) + len(sub_out)),
        "n_over_512": n512,
        "n_over_256": n256,
    }
    for p in PERCENTILES:
        row[f"wp_p{p}"] = percentile(w, p)
    return row


def _md_row(label, r) -> str:
    return (f"| {label} | {r['n_sentences']} | {r['wp_min']} | {r['wp_p50']} | "
            f"{r['wp_p90']} | {r['wp_p95']} | {r['wp_p99']} | {r['wp_max']} | "
            f"{r['infl_inside_span']:.3f} | {r['infl_outside_span']:.3f} | "
            f"{r['infl_overall']:.3f} | {r['n_over_512']} | {r['n_over_256']} |")


def collect(cfg):
    corpus = {d: load_domain(d, cfg) for d in cfg.domains}
    domains = list(cfg.domains)

    per = {}            # (tokenizer_name, domain) -> _row(...)
    over512 = {}        # tokenizer_name -> [(file_id, sentence_index, n_wordpieces)]
    fragmenters = {}    # tokenizer_name -> [(token, subwords, occurrences, label)]

    for name, repo, revision, kwargs in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True, **kwargs)
        assert tok.is_fast, f"{name}: word_ids() needs the fast tokenizer"

        over512[name] = []
        frag = {}                       # token -> [max_subwords, occurrences, Counter(labels)]
        pool_wp, pool_in, pool_out = [], [], []

        for domain in domains:
            wp, sub_in, sub_out = [], [], []
            n512 = n256 = 0
            for doc in corpus[domain]:
                for idx, (tokens, labels) in enumerate(doc.sentences):
                    # token list in, not a joined string, so wordpiece splits
                    # land on the dataset's token boundaries. No truncation: it
                    # would cap long sentences at the model limit and hide the
                    # distribution this measures.
                    enc = tok(tokens, is_split_into_words=True)
                    n = len(enc["input_ids"])       # includes the special tokens
                    wp.append(n)
                    n512 += n > 512
                    n256 += n > 256
                    if n > 512:
                        over512[name].append((doc.file_id, idx, n))

                    # word_ids(): one entry per wordpiece position -- None for a
                    # special token, else the index of the input token it came
                    # from. Counting them gives subwords per input token.
                    counts = Counter(w for w in enc.word_ids() if w is not None)
                    for ti, c in counts.items():
                        (sub_in if labels[ti] in ("B", "I") else sub_out).append(c)
                        f = frag.setdefault(tokens[ti], [c, 0, Counter()])
                        f[0] = max(f[0], c)
                        f[1] += 1
                        f[2][labels[ti]] += 1

            per[(name, domain)] = _row(wp, sub_in, sub_out, n512, n256)
            pool_wp += wp
            pool_in += sub_in
            pool_out += sub_out

        per[(name, "all")] = _row(
            pool_wp, pool_in, pool_out, len(over512[name]),
            sum(per[(name, d)]["n_over_256"] for d in domains),
        )
        top = sorted(frag.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)[:20]
        fragmenters[name] = [(t, v[0], v[1], v[2].most_common(1)[0][0]) for t, v in top]

    return domains, per, over512, fragmenters


def build_markdown(domains, per, over512, fragmenters, test_domain) -> str:
    L = [
        "# s02 -- wordpiece length distribution",
        "",
        "Four tokenizers over the same annotated portion as s01 (English, "
        "sequential IOB, `without_named_entities`). Each sentence tokenized with "
        "`is_split_into_words=True` and no truncation; the wordpiece count "
        "includes special tokens. `infl` = mean subwords per input token, split "
        "by whether the label is inside a span (`B`/`I`) or outside (`O`).",
        "",
        "## Summary",
        "",
        "| tokenizer | htfl p99 | corpus max | inside-span inflation |",
        "|---|---:|---:|---:|",
    ]
    for name, *_ in TOKENIZERS:
        h, a = per[(name, test_domain)], per[(name, "all")]
        L.append(f"| {name} | {h['wp_p99']} | {a['wp_max']} | {a['infl_inside_span']:.3f} |")
    L.append("")

    header = ("| domain | n_sent | min | p50 | p90 | p95 | p99 | max | infl B/I "
              "| infl O | infl all | >512 | >256 |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for name, repo, revision, _kw in TOKENIZERS:
        L += [f"## {name}", "", f"`{repo}` @ `{revision}`", "", header, sep]
        L += [_md_row(d, per[(name, d)]) for d in (*domains, "all")]
        L.append("")
        if over512[name]:
            L.append(f"Sentences over 512 wordpieces ({len(over512[name])}):")
            L += [f"- `{fid}` sentence {idx}: {n}" for fid, idx, n in over512[name]]
            L.append("")
        L += ["20 dataset tokens with the most subwords:", "",
              "| token | subwords | occurrences | label |", "|---|---:|---:|---|"]
        L += [f"| `{t}` | {sw} | {occ} | {lab} |" for t, sw, occ, lab in fragmenters[name]]
        L.append("")
    return "\n".join(L)


def main() -> None:
    cfg = load_config()
    domains, per, over512, fragmenters = collect(cfg)

    report = build_markdown(domains, per, over512, fragmenters, cfg.test_domain)
    print()
    print(report)

    payload = {
        "generated_by": "src/stats/s02_wordpieces.py",
        "tokenizers": [{"name": n, "repo": r, "revision": rev}
                       for n, r, rev, _ in TOKENIZERS],
        "per_tokenizer_domain": {
            f"{name}/{d}": per[(name, d)]
            for name, *_ in TOKENIZERS for d in (*domains, "all")
        },
        "sentences_over_512": {
            name: [{"file_id": f, "sentence_index": i, "n_wordpieces": n}
                   for f, i, n in over512[name]]
            for name, *_ in TOKENIZERS
        },
        "top_fragmenting_tokens": {
            name: [{"token": t, "subwords": sw, "occurrences": o, "label": lab}
                   for t, sw, o, lab in fragmenters[name]]
            for name, *_ in TOKENIZERS
        },
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "s02_wordpieces.md").write_text(report + "\n", encoding="utf-8")
    (_OUT_DIR / "s02_wordpieces.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[s02] wrote {_OUT_DIR / 's02_wordpieces.md'}")
    print(f"[s02] wrote {_OUT_DIR / 's02_wordpieces.json'}")


if __name__ == "__main__":
    main()
