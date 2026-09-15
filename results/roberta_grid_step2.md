# Hyperparameter grid, step 2 — paired comparison (FacebookAI/roberta-base)

Reference cell: **LR 2e-05, 5 epochs** — highest equi mean (0.5085). Compared against the other 5.

Paired t on five per-seed differences, df = 4. The same seeds ran in every cell with identical head init and shuffle order, so the seed's own strength cancels in the difference. Unpaired t shown alongside for contrast only — it discards the pairing and is not the test.

Bands: t < 2.132 not detected, 2.132–2.776 suggestive, > 2.776 take seriously (two-tailed p = 0.10, 0.05).

| cell | mean | gap to ref | s_d | **t paired** | t unpaired | signs | reading |
|---|--:|--:|--:|--:|--:|--:|---|
| **LR 2e-05 / 5ep (ref)** | **0.5085** | — | — | — | — | — | — |
| LR 3e-05 / 5ep | 0.5063 | +0.0022 | 0.0126 | **0.38** | 0.15 | 2+/3− | no difference detected |
| LR 1e-05 / 5ep | 0.5040 | +0.0045 | 0.0131 | **0.77** | 0.37 | 3+/2− | no difference detected |
| LR 3e-05 / 3ep | 0.5032 | +0.0053 | 0.0142 | **0.83** | 0.40 | 3+/2− | no difference detected |
| LR 1e-05 / 3ep | 0.5011 | +0.0074 | 0.0171 | **0.96** | 0.63 | 3+/2− | no difference detected |
| LR 2e-05 / 3ep | 0.5008 | +0.0077 | 0.0150 | **1.15** | 0.62 | 3+/2− | no difference detected |

## Per-seed differences (reference minus cell)

| cell | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---|---|---|---|---|
| LR 3e-05 / 5ep | +0.0071 | -0.0077 | -0.0016 | +0.0216 | -0.0087 |
| LR 1e-05 / 5ep | -0.0054 | +0.0243 | +0.0089 | +0.0035 | -0.0087 |
| LR 3e-05 / 3ep | +0.0122 | -0.0113 | +0.0217 | +0.0117 | -0.0079 |
| LR 1e-05 / 3ep | +0.0026 | +0.0195 | +0.0304 | -0.0088 | -0.0069 |
| LR 2e-05 / 3ep | +0.0101 | -0.0019 | +0.0297 | +0.0105 | -0.0099 |

## Multiple comparisons

5 tests were run against one reference. Bonferroni at df = 4 would demand t > 4.604 rather than 2.776. The raw t is reported above and the count is stated here rather than a correction being applied silently; judge the family accordingly.

## Tie set

Not distinguishable from the reference at n = 5: LR 3e-05/5ep, LR 1e-05/5ep, LR 3e-05/3ep, LR 1e-05/3ep, LR 2e-05/3ep.

Not the same as identical — this experiment cannot separate them. Tiebreak is highest mean on the selection domain, which takes **LR 2e-05, 5 epochs**.

