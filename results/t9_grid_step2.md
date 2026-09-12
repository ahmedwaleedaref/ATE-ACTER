# T9 — hyperparameter grid, step 2 (paired comparison)

Reference cell: **LR 3e-05, 5 epochs** — highest equi mean (0.4811). Compared against the other five.

Paired t on five per-seed differences, df = 4. The same seeds ran in every cell with identical head init and shuffle order, so the seed's own strength cancels in the difference. Unpaired t shown alongside for contrast only — it discards the pairing and is not the test.

Bands: t < 2.132 not detected, 2.132–2.776 suggestive, > 2.776 take seriously (two-tailed p = 0.10, 0.05).

| cell | mean | gap to ref | s_d | **t paired** | t unpaired | signs | reading |
|---|--:|--:|--:|--:|--:|--:|---|
| **LR 3e-05 / 5ep (ref)** | **0.4811** | — | — | — | — | — | — |
| LR 2e-05 / 5ep | 0.4744 | +0.0067 | 0.0082 | **1.82** | 0.51 | 5+/0− (p=0.0625) | no difference detected |
| LR 5e-05 / 5ep | 0.4710 | +0.0101 | 0.0106 | **2.13** | 0.73 | 4+/1− | suggestive |
| LR 3e-05 / 3ep | 0.4657 | +0.0154 | 0.0140 | **2.47** | 1.16 | 4+/1− | suggestive |
| LR 5e-05 / 3ep | 0.4652 | +0.0159 | 0.0117 | **3.03** | 1.22 | 5+/0− (p=0.0625) | take seriously |
| LR 2e-05 / 3ep | 0.4637 | +0.0175 | 0.0097 | **4.04** | 1.33 | 5+/0− (p=0.0625) | take seriously |

## Per-seed differences (reference minus cell)

| cell | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 |
|---|---|---|---|---|---|
| LR 2e-05 / 5ep | +0.0060 | +0.0007 | +0.0008 | +0.0051 | +0.0208 |
| LR 5e-05 / 5ep | +0.0052 | -0.0047 | +0.0237 | +0.0143 | +0.0120 |
| LR 3e-05 / 3ep | +0.0188 | +0.0074 | -0.0022 | +0.0351 | +0.0181 |
| LR 5e-05 / 3ep | +0.0067 | +0.0162 | +0.0261 | +0.0286 | +0.0018 |
| LR 2e-05 / 3ep | +0.0168 | +0.0095 | +0.0069 | +0.0298 | +0.0243 |

## Multiple comparisons

Five tests were run against one reference. Bonferroni at df = 4 would demand t > 4.604 rather than 2.776. The raw t is reported above and the count is stated here rather than a correction being applied silently; judge the family accordingly.

## Tie set

Not distinguishable from the reference at n = 5: LR 2e-05/5ep.

Not the same as identical — this experiment cannot separate them. Cheapest config in the tie set including the reference: **LR 3e-05, 5 epochs**.

