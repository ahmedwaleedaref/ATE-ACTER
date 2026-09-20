# T11 — error analysis, deberta ctx32, 6 epochs, 5 seeds

htfl, ANN key, 5 seeds. Every cell is mean +/- std (ddof=1) across
seeds, the standard everything else in this project uses. Per-seed statistic
is best epoch on equi. Breakdowns are set operations between each seed's
predicted term list and a bucket of the gold key.

| | mean +/- std | ceiling | / ceiling |
|---|--:|--:|--:|
| equi (dev) | 0.5866 +/- 0.0031 | 0.9523 | 0.616 |
| htfl list F1 | 0.6013 +/- 0.0058 | 0.9096 | 0.661 |

htfl P 0.5815 +/- 0.0045, R 0.6227 +/- 0.0165, 2505 +/- 84 types.

## 1. Recall by term length

| length | gold | ceiling | recall | / ceiling |
|---|--:|--:|--:|--:|
| 1 | 1,029 | 0.8980 | 0.6696 +/- 0.0232 | 0.746 +/- 0.026 |
| 2 | 754 | 0.9549 | 0.6310 +/- 0.0169 | 0.661 +/- 0.018 |
| 3 | 366 | 0.9508 | 0.6224 +/- 0.0228 | 0.655 +/- 0.024 |
| 4+ | 190 | 0.9842 | 0.3368 +/- 0.0283 | 0.342 +/- 0.029 |

single-word 0.6696 +/- 0.0232 against multi-word 0.5860 +/- 0.0187.

**Prediction (Data_stats.md 8.2, logged contrarian):** multi-word recall
exceeds single-word via head-position transfer.

VERDICT: 

## 2. Recall by gold frequency

| frequency | gold | share of key | recall |
|---|--:|--:|--:|
| 0 (never a maximal span) | 160 | 0.068 | 0.2100 +/- 0.0511 |
| 1 (singleton) | 1,115 | 0.477 | 0.5552 +/- 0.0206 |
| >= 2 | 1,064 | 0.455 | 0.7556 +/- 0.0197 |

VERDICT: 

## 3. Exact-span F1 against unique-list F1

| metric | P | R | F1 | ceiling |
|---|--:|--:|--:|--:|
| unique-list (headline) | 0.5815 +/- 0.0045 | 0.6227 +/- 0.0165 | 0.6013 +/- 0.0058 | 0.9096 |
| exact-span (diagnostic) | 0.7035 +/- 0.0030 | 0.5159 +/- 0.0184 | 0.5951 +/- 0.0119 | 1.0 |

Gap (span - list), paired per seed: -0.0062 +/- 0.0068.

**Prediction (T3 spec):** span high with list low.

VERDICT: 

## 4. Invalid tag sequences

| pattern | count |
|---|--:|
| `I` opening a sentence | 0.0 +/- 0.0 |
| `I` after `O` | 161.8 +/- 23.6 |
| predicted spans | 7067 +/- 264 |

Dangling runs per predicted span: 0.0230 +/- 0.0040. Gold htfl is (1, 0), so
these are the model's alone. **Decides whether a CRF is worth considering.**

VERDICT: 

## 5. Long terms in training

Seed-invariant -- a property of the corpus.

| length | train occurrences (corp+wind) | share | htfl key types | share |
|---|--:|--:|--:|--:|
| 1 | 4,880 | 0.529 | 1,029 | 0.440 |
| 2 | 3,316 | 0.359 | 754 | 0.322 |
| 3 | 889 | 0.096 | 366 | 0.156 |
| 4+ | 148 | 0.016 | 190 | 0.081 |

Units differ and are not mixed: training is occurrence-weighted, htfl is
type-weighted, because the metric is.

VERDICT: 

## Per seed

| seed | best epoch | equi | htfl list F1 | span F1 | types | I-after-O |
|---|--:|--:|--:|--:|--:|--:|
| 42 | 5 | 0.5870 | 0.6096 | 0.6117 | 2599 | 152 |
| 43 | 5 | 0.5892 | 0.5933 | 0.5798 | 2370 | 196 |
| 44 | 5 | 0.5893 | 0.6010 | 0.6008 | 2526 | 165 |
| 45 | 5 | 0.5859 | 0.6006 | 0.5911 | 2506 | 131 |
| 46 | 6 | 0.5817 | 0.6020 | 0.5919 | 2525 | 165 |

