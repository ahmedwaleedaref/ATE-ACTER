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

## What T9 hands downstream

The **hyperparameters**, nothing else. T10 holds LR 3e-5 and 5 epochs fixed
while it sweeps encoders; whichever encoder wins there is the selected model,
and it need not be `bert-base-cased`. T11's breakdowns run on that, over all
five seeds. No model is selected by this task.

## Highest-dev seed in this cell

Recorded because it was asked for. It has no standing downstream — T9's output
is the config, and this is one of its five samples.

`20260912-061545_bert-base-cased_lr3e-05_e5_seed42`, best epoch 4:

| | equi (dev) | htfl (test) |
|---|--:|--:|
| F1 | 0.5081 | 0.5441 |
| P / R | 0.5241 / 0.4930 | 0.5611 / 0.5280 |

Seed 42 is the highest of the five on equi. **Its htfl 0.5441 is a selected
number and is not this config's score** — it is the max of five on dev and
inherits that bias. The config's htfl figure is the five-seed mean,
0.5278 ± 0.0156.

Nothing was checkpointed, so this model's weights no longer exist. The row
above is what remains of it.
