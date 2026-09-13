# Hyperparameter grid, step 1 (bert-base-cased)

Each cell is `mean ± std` of five best-epoch equi ANN F1 scores, ddof=1. Measurement only: nothing here is a comparison.

| | epochs 3 | epochs 5 |
|---|---|---|
| **LR 2e-05** | 0.4637 ± 0.0233 | 0.4744 ± 0.0232 |
| **LR 3e-05** | 0.4657 ± 0.0236 | 0.4811 ± 0.0179 (E02) |
| **LR 5e-05** | 0.4652 ± 0.0228 | 0.4710 ± 0.0250 |

## Per-cell detail

| LR | epochs | equi mean ± std | htfl mean ± std | best epochs | collapsed |
|---|---|---|---|---|---|
| 2e-05 | 3 | 0.4637 ± 0.0233 | 0.5235 ± 0.0109 | 2, 3, 2, 3, 2 | — |
| 2e-05 | 5 | 0.4744 ± 0.0232 | 0.5277 ± 0.0140 | 4, 3, 3, 3, 2 | — |
| 3e-05 | 3 | 0.4657 ± 0.0236 | 0.5194 ± 0.0073 | 3, 3, 2, 3, 2 | — |
| 3e-05 | 5 | 0.4811 ± 0.0179 | 0.5278 ± 0.0156 | 4, 3, 3, 5, 3 | — |
| 5e-05 | 3 | 0.4652 ± 0.0228 | 0.5177 ± 0.0137 | 3, 3, 3, 3, 2 | — |
| 5e-05 | 5 | 0.4710 ± 0.0250 | 0.5310 ± 0.0137 | 4, 3, 4, 5, 3 | — |

Selection happens on equi. htfl is recorded per run and must not enter the comparison in step 2.

