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

<!--
Footnote on rounding: ceiling F1 recomputed from full-precision P and R is
0.9096 (htfl) and 0.9523 (equi); Tasks_week2.md quotes 0.9097 and 0.9524,
which come from rounding P and R to 4 dp before combining. Same measurement,
1 in the fourth decimal.
-->
