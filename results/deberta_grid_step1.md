# Hyperparameter grid, step 1 (microsoft/deberta-v3-base)

Each cell is `mean ± std` of five best-epoch equi ANN F1 scores, ddof=1. Measurement only: nothing here is a comparison.

| | epochs 3 | epochs 5 |
|---|---|---|
| **LR 1e-05** | 0.5589 ± 0.0071 | 0.5590 ± 0.0086 |
| **LR 2e-05** | 0.5504 ± 0.0054 | 0.5523 ± 0.0048 |

## Per-cell detail

| LR | epochs | equi mean ± std | htfl mean ± std | best epochs | collapsed |
|---|---|---|---|---|---|
| 1e-05 | 3 | 0.5589 ± 0.0071 | 0.5801 ± 0.0160 | 3, 3, 3, 1, 3 | — |
| 1e-05 | 5 | 0.5590 ± 0.0086 | 0.5784 ± 0.0228 | 4, 3, 1, 1, 5 | — |
| 2e-05 | 3 | 0.5504 ± 0.0054 | 0.5870 ± 0.0133 | 3, 1, 1, 3, 3 | — |
| 2e-05 | 5 | 0.5523 ± 0.0048 | 0.5902 ± 0.0126 | 1, 4, 1, 5, 2 | — |

Selection happens on equi. htfl is recorded per run and must not enter the comparison in step 2.

