# DeBERTa — selected configuration

**`microsoft/deberta-v3-base`, LR 1e-5, 5 epochs.** Effective batch 16, warmup
10%, AdamW weight decay 0.01, max_grad_norm 1.0 — everything except LR and
epochs as E01. Selected on **equi**, ANN unique-list F1, per-seed statistic =
best epoch on equi. Full record in `docs/EXPERIMENTS.md` E05.

## Selection number

| | mean (5 seeds) | std (ddof=1) | ceiling | / ceiling |
|---|--:|--:|--:|--:|
| **equi (dev)** | **0.5590** | 0.0086 | 0.9523 | 0.587 |
| htfl (test) | 0.5784 | 0.0228 | 0.9096 | 0.636 |

0 of 20 grid runs collapsed.

## Why this cell

Highest equi mean of the four measured cells, and the tiebreak is highest mean
on the selection domain.

| cell | equi mean | t paired vs ref | reading |
|---|--:|--:|---|
| **1e-5 / 5ep (selected)** | **0.5590** | — | — |
| 1e-5 / 3ep | 0.5589 | 0.01 | not detected — tie set |
| 2e-5 / 5ep | 0.5523 | 1.71 | not detected — tie set |
| 2e-5 / 3ep | 0.5504 | 2.26 | suggestive |

Nothing clears Bonferroni for 3 tests (t > 3.961), so strictly the grid
separates no cell from the reference. The tie set is `{1e-5, 3}` and
`{2e-5, 5}`; the rule takes the highest mean, which is this cell.

**The epoch axis is flat.** `{1e-5, 3}` vs `{1e-5, 5}` differ by 0.0001 at
t = 0.01 — the flattest comparison in the project. 3 epochs costs 286 s per run
against 440 s, so the same score is available at 35% less compute if compute
ever becomes the binding constraint.

## Reference run

**Seed 42**, best epoch 4 — highest on equi within the cell.

| | equi (dev) | htfl (test) |
|---|--:|--:|
| F1 | 0.5657 | 0.6039 |
| P / R | 0.5341 / 0.6012 | 0.5628 / 0.6516 |

It is also the highest htfl run in the entire grid (0.6039 of 25 runs). That
was not a selection criterion — it tops equi within this cell, and the htfl
figure is reported after the fact. **0.6039 is not this config's score**: it is
the max of five on dev and inherits that bias. The config's htfl figure is the
five-seed mean, 0.5784 ± 0.0228.

## Caveats carried forward

**Test cannot see the LR axis.** All three paired comparisons on htfl come back
t = 0.89–1.09, not detected. The spread of cell means on htfl (0.0118) is
smaller than the typical within-cell seed std (0.0162). equi ranks the cells;
htfl cannot.

**Environment.** These runs are Colab/Kaggle T4, torch 2.10–2.11, Python 3.13 —
against bert's local RTX 3050, torch 2.14.0, Python 3.14.4. `transformers` is
5.16.1 throughout. Any bert-vs-deberta comparison spans two environments.
