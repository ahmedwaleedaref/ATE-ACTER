# RoBERTa — selected configuration

**`FacebookAI/roberta-base`, LR 3e-5, 5 epochs** — bert's config from T9.
Effective batch 16, warmup 10%, AdamW weight decay 0.01, everything else as
E01. `configs/train.json` unchanged; the only CLI override is `--model`.
Local RTX 3050, torch 2.14.0+cu130, Python 3.14.4 — same stack as bert, so the
roberta-vs-bert comparison carries no environment confound.

| | mean (5 seeds) | std (ddof=1) | ceiling | / ceiling |
|---|--:|--:|--:|--:|
| **equi (dev)** | **0.5063** | 0.0266 | 0.9523 | 0.532 |
| htfl (test) | 0.5632 | 0.0124 | 0.9096 | 0.619 |

Recorded in `EXPERIMENTS.md` E06 (this cell) and E07 (the LR grid around it).

## How this cell was chosen, precisely

E07 tuned LR over {1e-5, 2e-5, 3e-5} × epochs {3, 5}, 30 runs. **equi could not
separate any cell from any other** — five paired comparisons, every t between
0.38 and 1.15, the spread of cell means (0.0077) a third of the typical
within-cell std (0.0213). All six cells are in the tie set.

`{3e-5, 5}` is therefore a legitimate pick on equi: it is inside the tie set.
It is **not** what the standing tiebreak returns. "Highest mean on the selection
domain" takes `{2e-5, 5}` at 0.5085 — ahead by 0.0022 at t = 0.38, which is
noise. This cell was chosen instead for continuity with bert's config; the
difference between the two is not measurable here either way.

## Reference run — seed 46

| | equi (dev) | htfl (test) |
|---|--:|--:|
| F1 | 0.5276 | 0.5710 |
| best epoch | 2 | — |

**Seed 46 is rank 2 on both domains, not rank 1 on either.** Seed 44 tops equi
(0.5328, but rank 5 on htfl); seed 42 tops htfl (0.5755, rank 5 on equi). The
highest-dev rule used for bert (T9) and deberta (E05) would take seed 44. Seed
46 was chosen as the balanced run, and that is a different rule — recorded here
so a later reader does not assume the usual one.

Its htfl 0.5710 is not this config's score. The config's htfl figure is the
five-seed mean, **0.5632 ± 0.0124**.

## Against bert, same five seeds, same machine

| | mean gap | t | signs | reading |
|---|--:|--:|--:|---|
| equi | +0.0252 | 1.40 | 4+/1− | not detected |
| htfl | +0.0354 | **8.26** | **5+/0−** | take seriously |

roberta beats bert on test about as decisively as this project measures
anything, and dev cannot see it.

## Open, not closed

E07 established that equi at n = 5 has no power over roberta's hyperparameters.
The LR-stability trend the grid hints at — equi std 0.0185 / 0.0210 / 0.0229
across 1e-5 / 2e-5 / 3e-5 at 3 epochs — does not hold at 5 epochs
(0.0202 / 0.0184 / 0.0266), and at df = 4 those stds are not separable anyway.
Left as a question rather than a finding. Resolving it needs more seeds or a
different dev domain, and was judged not worth the compute against fractional
F1 gains.
