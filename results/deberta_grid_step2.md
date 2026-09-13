# Hyperparameter grid, step 2 — paired comparison (microsoft/deberta-v3-base)

Reference cell: **LR 1e-05, 5 epochs** — highest equi mean (0.5590). Compared against the other 3.

Paired t on five per-seed differences, df = 4. The same seeds ran in every cell with identical head init and shuffle order, so the seed's own strength cancels in the difference. Unpaired t shown alongside for contrast only — it discards the pairing and is not the test.

Bands: t < 2.132 not detected, 2.132–2.776 suggestive, > 2.776 take seriously (two-tailed p = 0.10, 0.05).

| cell | mean | gap to ref | s_d | **t paired** | t unpaired | signs | reading |
|---|--:|--:|--:|--:|--:|--:|---|
| **LR 1e-05 / 5ep (ref)** | **0.5590** | — | — | — | — | — | — |
| LR 1e-05 / 3ep | 0.5589 | +0.0000 | 0.0062 | **0.01** | 0.01 | 2+/3− | no difference detected |
| LR 2e-05 / 5ep | 0.5523 | +0.0067 | 0.0088 | **1.71** | 1.52 | 4+/1− | no difference detected |
| LR 2e-05 / 3ep | 0.5504 | +0.0086 | 0.0085 | **2.26** | 1.88 | 4+/1− | suggestive |

## Per-seed differences (reference minus cell)

| cell | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---|---|---|---|---|
| LR 1e-05 / 3ep | +0.0092 | -0.0038 | -0.0051 | -0.0039 | +0.0038 |
| LR 2e-05 / 5ep | +0.0160 | -0.0056 | +0.0052 | +0.0142 | +0.0036 |
| LR 2e-05 / 3ep | +0.0187 | -0.0015 | +0.0039 | +0.0159 | +0.0058 |

## Multiple comparisons

3 tests were run against one reference. Bonferroni at df = 4 would demand t > 3.961 rather than 2.776. The raw t is reported above and the count is stated here rather than a correction being applied silently; judge the family accordingly.

## Tie set

Not distinguishable from the reference at n = 5: LR 1e-05/3ep, LR 2e-05/5ep.

Not the same as identical — this experiment cannot separate them. Cheapest config in the tie set including the reference: **LR 1e-05, 3 epochs**.

