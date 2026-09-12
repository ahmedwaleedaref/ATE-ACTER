# Experiments

One entry per run, **written before the run**. The prediction and its mechanism
go in first; the result is filled in after. Runs that went nowhere stay in the
file — they are what make the ones that worked credible.

Smoke and debug runs are not experiments. They live in `results/runs/` with a
`reason` field saying so, and are not logged here.

Every htfl evaluation is also appended to `results/test_evaluations.log`.
Selection happens on equi, always.

---

## E01 — T7, first end-to-end run

**Purpose:** prove the pipeline runs end to end — raw files → tokenised batches
→ model → first-subword logits → argmax → `recover_token_labels` → `decode` →
term list → `score_list`. A good score is not the goal.

**Config** (`configs/train.json`, seed 42):

| | |
|---|---|
| encoder | `bert-base-cased` |
| LR / schedule | 3e-5, linear, 10% warmup = 143 of 1,435 optimizer steps |
| batch | effective 16 = per-device 8 × grad-accum 2 |
| epochs | 5 |
| optimizer | AdamW, weight decay 0.01, bias and LayerNorm excluded |
| grad clipping | max_grad_norm 1.0 |
| max_length | 256, first-subword labelling, continuations −100 |
| train | corp + wind, wind ≤2-token sentences dropped — 4,592 sentences |
| dev | equi, 3,090 sentences (unfiltered) |
| test | htfl, 2,432 sentences (unfiltered) |

**Reference points:**

| | htfl unique-list F1 (ANN) |
|---|--:|
| BIO ceiling — `results/ceilings.md` | **0.9097** (max_P 0.8887, max_R 0.9316) |
| TermEval 2020 winner | 0.467 |
| C-Value baseline — `results/baseline_cvalue.md` | 0.2003 (P 0.1941, R 0.2069) |
| equi (dev) ceiling, for the dev number | 0.9523 |

**What the data says to expect**, so the prediction is a prediction and not a guess:

- `Data_stats.md` §9.2 — training prior ≈ 0.14 against htfl's 0.2604. A model
  fits its training prior, so **recall suppressed on htfl, precision roughly
  unaffected** is the predicted shape.
- §9.3 — type overlap between the training gold and htfl is **10 terms, 0.4%**
  (and lower under the adopted split). Essentially nothing here can be lexical
  memorisation.
- §8.2 — 23% of htfl's multi-word terms share a final token with a training
  term. That head-position transfer is the channel that does exist.
- Training loss will plateau well above zero (`data_layout.md` §5.2b). Expected,
  not a bug. **[False premise — see Result. §5.2b is corrected.]**

**Predicted (htfl unique-list F1, ANN key):**
 i predict around 0.4 to 0.5 f1 the winner team was using bert with no seq labeling we are using bert with seq labeling i expect f1 score to be around i do not suspect any gradien vanshing , exploding problem or even low postive rate which main problem with bert i do not thing there will be any here . still there is no reall
 hyper parameter choicing so who knows or reall analysis if f1 is so low
**Predicted (equi unique-list F1, ANN key):**
same range as htfl .

**Mechanism:**

**Result:** run `20260909-044422_bert-base-cased_lr3e-05_e5_seed42`, 1435/1435
optimizer steps, 315 s on an RTX 3050.

| | F1 | P | R | / ceiling |
|---|--:|--:|--:|--:|
| **htfl (test), ANN** | **0.5200** | 0.6001 | 0.4587 | 0.572 of 0.9097 |
| htfl (test), NES | 0.5060 | 0.6147 | 0.4300 | — |
| equi (dev), ANN, epoch 5 | 0.4887 | 0.5569 | 0.4354 | 0.513 of 0.9523 |
| equi (dev), ANN, best epoch (4) | 0.5081 | 0.5241 | 0.4930 | 0.534 of 0.9523 |

htfl: 5,201 predicted spans → 1,788 types, against a gold key of 2,339.

Per-epoch, equi ANN F1: 0.4607 / 0.4471 / 0.4622 / 0.5081 / 0.4887.
Train loss: 0.2438 / 0.0475 / 0.0196 / 0.0084 / 0.0043.

**On the loss.** The pre-run expectation that it would plateau well above zero
was wrong, and so was the premise behind it. `data_layout.md` §5.2b and
`Data_stats.md` §7.2 are corrected: gold-term occurrences that are not their
own maximal span are nested inside longer terms and still carry positive
labels. The supervision is not contradictory, so nothing predicted a floor.
Near-zero loss for a 110M-parameter model fitting 4,592 sentences over 1,435
steps is ordinary and needs no explanation. It raises no memorisation question
either: htfl shares 0.4% of its term types with the training keys, so a 0.52
score cannot be retrieval — there is almost nothing to retrieve.

**Reading:**

---

## E02 — T8, seed variance (5 seeds, one config)

**Purpose:** measure the seed std of the E01 config, so every later comparison
has a noise floor to be read against. Not a tuning run — the config is E01's,
unchanged in every field.

**Entry written after the runs.** This file's protocol is prediction-first and
this entry breaks it: no prediction was recorded in advance. Nothing below is a
confirmed expectation. Flagged rather than backfilled, because a prediction
written after the result is the failure mode this file exists to prevent.

**Config** (`configs/train.json`, commit `fa0e9255`): identical to E01. Seeds
42, 43, 44, 45, 46, fixed before the first run. 1,435/1,435 optimizer steps in
all five. 1,495 s total on an RTX 3050.

**Per-seed statistic — fixed here, binding on T9 and T10:** each seed
contributes its **best epoch on equi**, ANN unique-list F1. htfl is evaluated
once per run, on those weights, never per epoch.

**Result:**

| seed | best epoch | equi ANN F1 | htfl ANN F1 | htfl P | htfl R | htfl types |
|---|--:|--:|--:|--:|--:|--:|
| 42 | 4 | 0.5081 | 0.5441 | 0.5611 | 0.5280 | 2,201 |
| 43 | 3 | 0.4882 | 0.5356 | 0.5951 | 0.4870 | 1,914 |
| 44 | 3 | 0.4755 | 0.5029 | 0.5901 | 0.4382 | 1,737 |
| 45 | 5 | 0.4729 | 0.5241 | 0.5802 | 0.4780 | 1,927 |
| 46 | 3 | 0.4609 | 0.5322 | 0.5904 | 0.4844 | 1,919 |

| | mean | std (ddof=1) | ceiling | / ceiling |
|---|--:|--:|--:|--:|
| **equi (dev), ANN** | **0.4811** | **0.0179** | 0.9523 | 0.505 |
| **htfl (test), ANN** | **0.5278** | **0.0156** | 0.9096 | 0.580 |
| htfl (test), NES | 0.5154 | 0.0168 | 0.8726 | 0.591 |

Collapse: **0 of 5**, consistent with the 13–26% positive rate rather than
NER's 1–2%. Encoder sha256 identical across all five (`963a5823bad227c2…`),
asserted by `src/aggregate.py`. Full report: `results/t8_seed_variance.md`.

**What the std makes detectable.** Against a config with the same std,
`SE(gap) = 0.0179·sqrt(2/5) = 0.0113`:

| t | gap on equi | reading |
|---|--:|---|
| < 1.58 | < 0.0179 | no difference detected at n=5 |
| > 3.16 | > 0.0358 | too large to be seed luck |

**Per-epoch equi ANN F1** (best in bold):

| seed | e1 | e2 | e3 | e4 | e5 |
|---|--:|--:|--:|--:|--:|
| 42 | 0.4607 | 0.4471 | 0.4622 | **0.5081** | 0.4887 |
| 43 | 0.4609 | 0.4710 | **0.4882** | 0.4501 | 0.4688 |
| 44 | 0.3482 | 0.4753 | **0.4755** | 0.4692 | 0.4596 |
| 45 | 0.4400 | 0.4246 | 0.4627 | 0.4491 | **0.4729** |
| 46 | 0.3919 | 0.4103 | **0.4609** | 0.4323 | 0.4222 |

Train loss is near-identical across seeds — e1 0.2302–0.2590, e5 0.0041–0.0045.

**Six things this shows.**

1. **Part of the 0.0179 is epoch noise, not seed noise.** Within a single seed,
   consecutive epochs swing by more than the between-seed std: seed 42 gains
   0.0459 from e3 to e4, seed 46 loses 0.0387 from e3 to e5. A max over five
   noisy epochs picks the top of that jitter, so the spread of the per-seed
   statistic carries both sources. The number is still the right one to gate
   T9 — the rule is identical in every cell — but it is not a pure measure of
   what the seed does.

2. **The curves are not monotonic and epoch 5 is rarely best** (3, 3, 3, 4, 5).
   Seed 46 peaks at e3 and is 0.0387 lower by e5. 10 epochs in the T9 grid will
   likely be wasted compute; the useful question on that axis is whether 3 beats
   5, not whether 10 beats 5.

3. **Train loss does not discriminate between these five models.** All five fit
   the training set equally well — e5 loss 0.0041–0.0045, e3 loss 0.0181–0.0202
   — and still land 0.0472 apart on equi (that is the range of the five; the std
   is 0.0179). Train loss measures fit to corp+wind, equi F1 measures
   generalisation to a domain never seen, so the two are free to disagree: the
   variance is in what each seed generalises to, not in how well any run
   optimised.

   It is **not** in the scoring path. `decode()` and `spans_to_unique_list()`
   are deterministic functions of the predicted labels — identical labels give
   identical output — so they cannot contribute variance. An earlier draft of
   this entry claimed the spread lived "after argmax"; that was wrong, and the
   commit message on 8eaac16 repeats the error.

   Separating "different fit" from "same fit, different decoding" needs **equi
   loss** logged per seed beside equi F1: tight equi loss with scattered equi F1
   would point at the label→span→type projection amplifying small logit
   differences near decision boundaries. `evaluate()` returns no loss today, so
   T8 cannot answer it.

4. **htfl scores above equi on every seed**, and higher against its own ceiling
   (0.580 vs 0.505). equi is the harder domain despite being the dev set.

5. **Dev and test ranks cross.** Seed 45 sits below seed 44 on equi (0.4729 vs
   0.4755) and above it on htfl (0.5241 vs 0.5029). Within noise at n=5, but on
   record before T9 selects on equi nine times.

6. **Precision exceeds recall on htfl in all five** (P 0.56–0.60, R 0.44–0.53),
   which is the shape E01 predicted from the training prior ≈0.14 against
   htfl's 0.2604. Seed 42 is the outlier that predicts most (2,201 types,
   R 0.5280) and takes the top htfl score with it.

**Known defect in the records.** Seeds 43–46 carry `"dirty": true` while seed 42
carries `false`, all at commit `fa0e9255`. The code never changed: `git_state()`
uses `git status --porcelain`, which counts the untracked `seed_42.json` the
previous run had just written. Every run after the first in a multi-run sequence
marks itself dirty because of its predecessor's output. Provenance here is
intact — one commit, five runs — but the flag is not trustworthy as written.

**Reading:**

---

<!--
Footnote on rounding: ceiling F1 recomputed from full-precision P and R is
0.9096 (htfl) and 0.9523 (equi); Tasks_week2.md quotes 0.9097 and 0.9524,
which come from rounding P and R to 4 dp before combining. Same measurement,
1 in the fourth decimal.
-->
