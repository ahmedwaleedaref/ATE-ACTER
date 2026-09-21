# s01 -- sentence length distribution

Scope: English, annotated portion only, sequential IOB annotations, `without_named_entities` labels. Sentence length = number of tokens; punctuation and numbers count; nothing is filtered.

Percentiles: sort ascending, nearest-rank; 0-based index = ceil(p/100 * n) - 1, clamped to [0, n-1]; computed with integer arithmetic. Percentage columns are of that row's `n_sentences` (`N`).

| domain | n_docs | n_sentences | n_tokens | min | p50 | p90 | p95 | p99 | max | sents <=2 tok (n) | sents <=2 tok (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corp | 12 | 2002 | 50845 | 1 | 22 | 49 | 61 | 98 | 281 | 228 | 11.39% (N=2002) |
| equi | 34 | 3090 | 58203 | 1 | 18 | 33 | 39 | 52 | 95 | 255 | 8.25% (N=3090) |
| wind | 5 | 6638 | 57766 | 1 | 1 | 27 | 36 | 62 | 258 | 4048 | 60.98% (N=6638) |
| htfl | 190 | 2432 | 55467 | 1 | 20 | 43 | 54 | 78 | 171 | 338 | 13.90% (N=2432) |
| pooled | 241 | 14162 | 222281 | 1 | 13 | 36 | 46 | 72 | 281 | 4869 | 34.38% (N=14162) |

## Integrity check -- inventory ratio

`paired-file tokens / whole-domain-corpus tokens`, both whitespace-split (annotated `texts_tokenised/` + `unannotated_texts/`). Bound: wind `< 0.30` (`data_layout.md` section 1: ~52k annotated of ~314k). No bound on the others -- htfl is annotated in full, ratio 1.0 by construction.

| domain | paired tokens | corpus tokens | ratio | bound |
|---|---:|---:|---:|---|
| corp | 50845 | 181923 | 0.2795 | - |
| equi | 58203 | 109387 | 0.5321 | - |
| wind | 57766 | 319887 | 0.1806 | < 0.30 |
| htfl | 55467 | 55467 | 1.0000 | - |

