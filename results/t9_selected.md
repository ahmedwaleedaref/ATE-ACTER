# T9 — selected configuration

**LR 3e-5, 5 epochs**, `bert-base-cased`. Batch effective 16, warmup 10%,
AdamW weight decay 0.01, max_grad_norm 1.0 — all as E01. This is what
`configs/train.json` already holds; no file change was needed.

Selected on **equi**, ANN unique-list F1, per-seed statistic = best epoch on
equi. Full record in `EXPERIMENTS.md` E03.

## Selection number

| | mean (5 seeds) | std (ddof=1) | ceiling | / ceiling |
|---|--:|--:|--:|--:|
| **equi (dev)** | **0.4811** | 0.0179 | 0.9523 | 0.505 |
| htfl (test) | 0.5278 | 0.0156 | 0.9096 | 0.580 |

0 of 30 runs collapsed.

## Why this cell

Paired t against the other five cells, df = 4, five per-seed differences:

| cell | gap | t paired | reading |
|---|--:|--:|---|
| 2e-5 / 5ep | +0.0067 | 1.82 | not detected — **tie set** |
| 5e-5 / 5ep | +0.0101 | 2.13 | suggestive |
| 3e-5 / 3ep | +0.0154 | 2.47 | suggestive |
| 5e-5 / 3ep | +0.0159 | 3.03 | take seriously |
| 2e-5 / 3ep | +0.0175 | 4.04 | take seriously |

The tie set is {2e-5, 5} alone. It costs the same 5 epochs, so "take the
cheaper" does not discriminate and the higher mean holds.

Nothing clears Bonferroni for five tests (t > 4.604 at df = 4). The weight is
on uniform direction — all five cells below the reference — and on three cells
having all five per-seed differences of one sign, p = 0.0625 each.

Losers included: `t9_grid_step1.md`, `t9_grid_step2.md`, and the 30 run JSONs
under `runs/` and `runs/t9/`.

## Reference run for T11 — not part of the selection

T9 selects hyperparameters. A seed is noise, not a hyperparameter, so the run
below is **not** part of the selected config; it is the single concrete model
carried into T11's breakdowns, where one is needed.

`20260912-061545_bert-base-cased_lr3e-05_e5_seed42`, best epoch 4:

| | equi (dev) | htfl (test) |
|---|--:|--:|
| F1 | 0.5081 | 0.5441 |
| P / R | 0.5241 / 0.4930 | 0.5611 / 0.5280 |

Seed 42 is the highest of the five on equi, which is what selecting a run is
allowed to use. **Its htfl 0.5441 is a selected number and is not this
config's score** — it is the max of five on dev and inherits that bias. The
config's htfl figure is the five-seed mean, 0.5278 ± 0.0156.

If the config is re-run, the config survives and these particular numbers do
not.
