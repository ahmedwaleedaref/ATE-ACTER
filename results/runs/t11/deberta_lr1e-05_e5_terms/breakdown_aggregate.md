# T11 — error analysis, deberta-v3-base lr1e-05 e5, 5 seeds

htfl, ANN key, 5 seeds. Every cell is mean +/- std (ddof=1) across
seeds, the standard everything else in this project uses. Per-seed statistic
is best epoch on equi. Breakdowns are set operations between each seed's
predicted term list and a bucket of the gold key.

| | mean +/- std | ceiling | / ceiling |
|---|--:|--:|--:|
| equi (dev) | 0.5592 +/- 0.0090 | 0.9523 | 0.587 |
| htfl list F1 | 0.5782 +/- 0.0230 | 0.9096 | 0.636 |

htfl P 0.5515 +/- 0.0363, R 0.6093 +/- 0.0278, 2594 +/- 225 types.

## 1. Recall by term length

| length | gold | ceiling | recall | / ceiling |
|---|--:|--:|--:|--:|
| 1 | 1,029 | 0.8980 | 0.6503 +/- 0.0273 | 0.724 +/- 0.030 |
| 2 | 754 | 0.9549 | 0.6467 +/- 0.0371 | 0.677 +/- 0.039 |
| 3 | 366 | 0.9508 | 0.5973 +/- 0.0332 | 0.628 +/- 0.035 |
| 4+ | 190 | 0.9842 | 0.2621 +/- 0.0365 | 0.266 +/- 0.037 |

single-word 0.6503 +/- 0.0273 against multi-word 0.5771 +/- 0.0324.

**Prediction (Data_stats.md 8.2, logged contrarian):** multi-word recall
exceeds single-word via head-position transfer.

VERDICT: 

## 2. Recall by gold frequency

| frequency | gold | share of key | recall |
|---|--:|--:|--:|
| 0 (never a maximal span) | 160 | 0.068 | 0.2288 +/- 0.0175 |
| 1 (singleton) | 1,115 | 0.477 | 0.5440 +/- 0.0363 |
| >= 2 | 1,064 | 0.455 | 0.7350 +/- 0.0250 |

VERDICT: 

## 3. Exact-span F1 against unique-list F1

| metric | P | R | F1 | ceiling |
|---|--:|--:|--:|--:|
| unique-list (headline) | 0.5515 +/- 0.0363 | 0.6093 +/- 0.0278 | 0.5782 +/- 0.0230 | 0.9096 |
| exact-span (diagnostic) | 0.6729 +/- 0.0354 | 0.5017 +/- 0.0218 | 0.5746 +/- 0.0242 | 1.0 |

Gap (span - list), paired per seed: -0.0036 +/- 0.0041.

**Prediction (T3 spec):** span high with list low.

VERDICT: 

## 4. Invalid tag sequences

| pattern | count |
|---|--:|
| `I` opening a sentence | 0.0 +/- 0.0 |
| `I` after `O` | 185.2 +/- 29.3 |
| predicted spans | 7193 +/- 335 |

Dangling runs per predicted span: 0.0258 +/- 0.0045. Gold htfl is (1, 0), so
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
| 42 | 4 | 0.5673 | 0.6046 | 0.6074 | 2702 | 150 |
| 43 | 3 | 0.5442 | 0.5783 | 0.5728 | 2347 | 186 |
| 44 | 1 | 0.5581 | 0.5652 | 0.5592 | 2548 | 229 |
| 45 | 1 | 0.5627 | 0.5474 | 0.5454 | 2922 | 191 |
| 46 | 5 | 0.5639 | 0.5953 | 0.5879 | 2452 | 170 |

