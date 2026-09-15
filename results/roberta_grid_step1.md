# Hyperparameter grid, step 1 (FacebookAI/roberta-base)

Each cell is `mean ± std` of five best-epoch equi ANN F1 scores, ddof=1. Measurement only: nothing here is a comparison.

| | epochs 3 | epochs 5 |
|---|---|---|
| **LR 1e-05** | 0.5011 ± 0.0185 | 0.5040 ± 0.0202 |
| **LR 2e-05** | 0.5008 ± 0.0210 | 0.5085 ± 0.0184 |
| **LR 3e-05** | 0.5032 ± 0.0229 | 0.5063 ± 0.0266 (E06) |

## Per-cell detail

| LR | epochs | equi mean ± std | htfl mean ± std | best epochs | collapsed |
|---|---|---|---|---|---|
| 1e-05 | 3 | 0.5011 ± 0.0185 | 0.5399 ± 0.0151 | 3, 1, 1, 2, 3 | — |
| 1e-05 | 5 | 0.5040 ± 0.0202 | 0.5445 ± 0.0101 | 3, 1, 1, 1, 2 | — |
| 2e-05 | 3 | 0.5008 ± 0.0210 | 0.5527 ± 0.0080 | 3, 1, 2, 2, 2 | — |
| 2e-05 | 5 | 0.5085 ± 0.0184 | 0.5525 ± 0.0136 | 3, 1, 1, 4, 2 | — |
| 3e-05 | 3 | 0.5032 ± 0.0229 | 0.5454 ± 0.0313 | 3, 1, 1, 2, 2 | — |
| 3e-05 | 5 | 0.5063 ± 0.0266 | 0.5632 ± 0.0124 | 5, 1, 1, 4, 2 | — |

Selection happens on equi. htfl is recorded per run and must not enter the comparison in step 2.

