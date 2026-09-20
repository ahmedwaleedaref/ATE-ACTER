# T11 — error analysis, microsoft/deberta-v3-base lr1e-05 e6 seed 42

htfl, ANN key. Every breakdown is a set operation between the predicted
term list and a bucket of the gold key; no checkpoint, no GPU.

| | this run | E05 cell (5-seed mean) |
|---|--:|--:|
| equi, best epoch 5 | 0.5870 | 0.5590 +/- 0.0086 |
| htfl list F1 | 0.6096 | 0.5784 +/- 0.0228 |

htfl P 0.5791 / R 0.6434, 7,444 spans -> 2,599 types. Ceiling 0.9096, so 0.670 of it.

A gap against E05 is the nondeterminism E04 measured (`cudnn_deterministic`
false, a repeated seed diverging from epoch 2). E05 is not overwritten.

## 1. Recall by term length

`ceiling` is what decoding GOLD BIO recovers in this bucket. It is not a
hard cap on a model: a tagger that segments differently can emit a term
gold only ever marks as nested, so `/ ceiling` may exceed 1.0.

| length | gold | ceiling | recall | / ceiling |
|---|--:|--:|--:|--:|
| 1 | 1,029 | 0.8980 | 0.7026 | 0.782 |
| 2 | 754 | 0.9549 | 0.6485 | 0.679 |
| 3 | 366 | 0.9508 | 0.6284 | 0.661 |
| 4+ | 190 | 0.9842 | 0.3316 | 0.337 |
| **all** | **2,339** | **0.9316** | **0.6434** | **0.691** |

**Prediction (Data_stats.md 8.2, logged contrarian):** multi-word recall
exceeds single-word via head-position transfer. The standard ATE finding is
the reverse. Resolve on the `/ ceiling` column -- the raw column is tilted
toward the prediction by roughly 6 points of scheme ceiling.

VERDICT: 

## 2. Recall by gold frequency

Frequency = decoded gold-BIO maximal-span occurrences (Data_stats.md 7.3,
column (a)). The 0 bucket is the terms the scheme never decodes as their own
span -- recall there is not bounded by 0, and is the direct test of
data_layout.md 5.1's claim that no BIO tagger can emit them.

| frequency | gold | share of key | recall |
|---|--:|--:|--:|
| 0 (never a maximal span) | 160 | 0.068 | 0.2062 |
| 1 (singleton) | 1,115 | 0.477 | 0.5767 |
| >= 2 | 1,064 | 0.455 | 0.7791 |

VERDICT: 

## 3. Exact-span F1 against unique-list F1

| metric | P | R | F1 | ceiling |
|---|--:|--:|--:|--:|
| unique-list (headline) | 0.5791 | 0.6434 | 0.6096 | 0.9096 |
| exact-span (diagnostic) | 0.7018 | 0.5421 | 0.6117 | 1.0 |

Gap: +0.0022. Span F1 has no ANN/NES split --
gold spans come from the labels, and the key distinction exists only in the
list metric. Span space carries no scheme loss, so its ceiling is 1.0.

**Prediction (T3 spec):** span high with list low means frequent terms found
and rare types missed -- the expected shape at 47.7% hapax. The two landing
close together would mean uniform performance across the frequency
distribution, which would be surprising.

VERDICT: 

## 4. Invalid tag sequences

| pattern | count |
|---|--:|
| `I` opening a sentence | 0 |
| `I` after `O` | 152 |
| **total dangling runs** | **152** |

Against 7,444 predicted spans. Counted as events: a run of
`I` with no `B` scores once. Gold htfl is (1, 0), so these are the model's
alone. The dangling-I policy already drops them in `decode`, so this is what
was being discarded silently.

**Decides whether a CRF is worth considering at all.**

VERDICT: 

## 5. Long terms in training

Seed-invariant -- a property of the corpus, identical in every run's report.

| length | train occurrences (corp+wind) | share | htfl key types | share |
|---|--:|--:|--:|--:|
| 1 | 4,880 | 0.529 | 1,029 | 0.440 |
| 2 | 3,316 | 0.359 | 754 | 0.322 |
| 3 | 889 | 0.096 | 366 | 0.156 |
| 4+ | 148 | 0.016 | 190 | 0.081 |

Units differ and are not mixed: training is occurrence-weighted, because that
is what the model is shown; htfl is type-weighted, because the metric is.

VERDICT: 

