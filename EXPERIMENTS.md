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

## E03 — T9, hyperparameter pass (bert-base-cased)

**Written before the runs.** Step 1 is measurement only; the comparison in
step 2 does not begin until every cell in the table below is filled.

**Purpose:** choose the LR and epoch count everything after this is run at.
Not to find a good score — to find the config T10's encoder sweep and T11's
breakdowns are built on, and to know whether the choice is even resolvable at
n=5.

**Grid:** LR ∈ {2e-5, 3e-5, 5e-5} × epochs ∈ {3, 5} = 6 cells, 5 seeds each
(42–46), 30 runs. Batch fixed at effective 16, warmup fixed at 10%, everything
else as E01. **10 epochs was dropped** — E02's curves peak at epoch 3 in three
seeds of five, and seed 46 ends 0.0387 below its own epoch-3 score.

**{3e-5, 5} is E02 and is not recomputed.** Its five runs already exist at
commit `fa0e9255`. That leaves **25 new runs**. Reusing it is only legitimate
because the per-seed statistic, the seeds and the config are identical — the
same reason `src/aggregate.py` asserts on all three.

**Per-seed statistic:** best epoch on equi, ANN unique-list F1 — fixed in T8,
identical in every cell. Max-over-epochs is biased upward; the bias is harmless
only while every cell carries it equally, so no cell may deviate.

### Step 1 — mean and std per cell

Measurement only. No cell is called better than another in this step.

| | epochs 3 | epochs 5 |
|---|---|---|
| **LR 2e-5** | — | — |
| **LR 3e-5** | — | **0.4811 ± 0.0179** (E02) |
| **LR 5e-5** | — | — |

Each cell is `mean ± std` of five best-epoch equi ANN F1 scores, std with
ddof=1. **htfl is scored on every run and recorded per cell**, deliberately:
nothing is checkpointed, so a score skipped now cannot be recovered without
retraining that cell. Recording it is not selecting on it — the 3 × 2 table
above and the comparison in step 2 are both equi only.

Per-cell detail to be recorded alongside: the five raw equi numbers, the five
best epochs, the collapse count, and the encoder hash.

**A cell containing a collapse gets no t-statistic in step 2.** Its std is
dominated by one point. Record the mean and std including the zero, plus the
collapse rate, and treat the rate as the cell's headline property.

### Step 2 — comparison (not started until step 1 is complete)

**The test is paired, on the seed.** An earlier draft of this entry specified
the unpaired `t = |mean_A − mean_B| / sqrt(s_A²/5 + s_B²/5)`. That was wrong
here and is recorded as wrong rather than quietly replaced: every cell runs the
same five seeds, and because `set_seed(n)` precedes `from_pretrained`, seed n
has an identical classifier-head init and an identical shuffle order in every
cell — verified, `first_batch_indices` for seed 42 is the same ten indices in
all six. Same subject, two treatments. The unpaired formula throws that away
and charges the between-seed spread to its own uncertainty twice, so on
{2e-5,5} vs {3e-5,5} it returns t = 0.51 where the paired test returns 1.82 on
the same data — a denominator 3.6× too large.

Per pair of cells: take the five per-seed differences d = B − A, discard the
raw scores, and run a one-sample t against zero.

    mean(d) / (s_d / sqrt(5))    df = 4

| t (df = 4) | p two-tailed | reading |
|---|--:|---|
| < 2.132 | > 0.10 | no difference detected at n=5 |
| 2.132 – 2.776 | 0.05 – 0.10 | suggestive; add seeds |
| > 2.776 | < 0.05 | take seriously |

These are t-distribution points. T8's 1.58/3.16 were the unpaired 1-std and
2-std gaps rewritten and do not transfer.

**Sign test alongside, because df = 4 cannot verify normality.** Five
differences sharing a sign has probability 2·(1/2)⁵ = 0.0625 under the null —
the strongest distribution-free statement available at this n. Reported for
every pair.

**Scope: the highest-mean cell against each of the other five.** Five tests,
not fifteen. Bonferroni at df = 4 would demand t > 4.604; the raw t is reported
and the count stated rather than a correction applied silently.

Cells not separated from the reference form the tie set, and the cheapest
config in it wins. **"No cell separates, take the cheapest" is a legitimate
result of T9**, not a failure of it — it means this experiment cannot see a
difference at n = 5, which is not the same as there being none.

**Selection is on equi in both steps.** htfl is present for all 30 runs and
must not enter the comparison — not to rank cells, not to break a tie, not as a
sanity check on the winner before it is chosen. It is recorded so that after
the choice is made, the grid can be asked whether equi-selection tracked htfl
rank at all, which is the question E02 raised when seeds 44 and 45 crossed
between domains. Asking it before the choice is fitting the test set.

This adds 25 tuning looks to `results/test_evaluations.log`, each tagged with
its cell. That cost is accepted here in exchange for the rank data being
recoverable at all.

**Predicted (which cell wins, and by how much):**
it hard to tell but it is between LR (3e-5 , 3) or (LR 2e-5 , 5) i think that the decay of the learning rate here play a role maybe we want to invistegate more about differen mechansims not .
this comes from why epc 3 wins in E02 .
**Predicted (does anything clear t = 1.58 at all):**
i guass there will be t < 1.58 aka identical runs 
**Mechanism:** the prediction above already contains one — *"i think that the
decay of the learning rate here play a role"* — and it is the one that
survived. Written out, because E02's "epoch 3 wins" supports two incompatible
mechanisms and T9 can tell them apart:

| mechanism | predicts |
|---|---|
| **Overfitting.** Past epoch 3 the model has learnt what it can and begins memorising, so the extra passes cost accuracy. | 3-epoch cells beat 5-epoch cells |
| **Schedule.** LR decays to zero across the whole run, so epoch 3 of a 5-epoch run is a different state from epoch 3 of a 3-epoch run. The peak epoch index does not transfer between cells. | 5-epoch cells can still win |

Same observation, opposite consequences. Whichever way T9 falls, one of these
is refuted — which is the point of writing them down before the runs rather
than explaining the result afterwards.

**Result:** 30 runs, 0 collapses, commit `05a6c041` (25 new) and `fa0e9255`
(E02's 5). Full tables in `results/t9_grid_step1.md` and
`results/t9_grid_step2.md`.

Step 1, equi mean ± std:

| | epochs 3 | epochs 5 |
|---|---|---|
| **LR 2e-5** | 0.4637 ± 0.0233 | 0.4744 ± 0.0232 |
| **LR 3e-5** | 0.4657 ± 0.0236 | **0.4811 ± 0.0179** (E02) |
| **LR 5e-5** | 0.4652 ± 0.0228 | 0.4710 ± 0.0250 |

Step 2, paired against the highest-mean cell {3e-5, 5}:

| cell | gap | s_d | t paired | t unpaired | signs | reading |
|---|--:|--:|--:|--:|--:|---|
| 2e-5 / 5ep | +0.0067 | 0.0082 | 1.82 | 0.51 | 5+/0− | not detected |
| 5e-5 / 5ep | +0.0101 | 0.0106 | 2.13 | 0.73 | 4+/1− | suggestive |
| 3e-5 / 3ep | +0.0154 | 0.0140 | 2.47 | 1.16 | 4+/1− | suggestive |
| 5e-5 / 3ep | +0.0159 | 0.0117 | 3.03 | 1.22 | 5+/0− | take seriously |
| 2e-5 / 3ep | +0.0175 | 0.0097 | 4.04 | 1.33 | 5+/0− | take seriously |

**Selected: LR 3e-5, 5 epochs.** Tie set is {2e-5, 5} alone; it costs the same
5 epochs, so "take the cheaper" does not discriminate and the higher mean holds.

**Four things this shows.**

1. **The epoch axis carries the result; the LR axis barely moves anything.** All
   three 3-epoch cells separate from the reference (t = 2.47, 3.03, 4.04) and
   land within 0.0020 of each other (0.4637 / 0.4652 / 0.4657) against stds of
   ~0.023 — at 3 epochs the learning rate does essentially nothing. At 5 epochs
   only 5e-5 is even suggestive.

2. **Nothing survives Bonferroni.** Five tests against one reference needs
   t > 4.604 at df = 4; the largest is 4.04. Under strict family-wise control
   at 0.05, T9 establishes nothing. What carries weight instead is the
   uniformity: all five cells fall below the reference, and three have all five
   per-seed differences of one sign (p = 0.0625 each, distribution-free).

3. **"3 epochs" and "5 epochs" are not the same schedule, and the axis is
   confounded.** The LR decays linearly to zero over the whole run, so a
   3-epoch run decays roughly 1.7× faster and is at LR ≈ 0 by its epoch 3,
   where a 5-epoch run still has ~44% of peak left. The 3-epoch cells peak at
   epoch 2–3 and the 5-epoch cells at 3–4, which is what that predicts. So the
   finding is "longer with a gentler decay beats shorter with a steeper decay",
   not "more passes over the data helps". Separating the two needs a fixed
   schedule length with a varying stop point — T12's territory.

4. **The paired correction decided the outcome.** Every unpaired t is 2.5–3×
   smaller than its paired counterpart and none reaches 1.58. Run as E03
   originally specified, all six cells would have been declared a tie and T9
   would have reported that its grid was unresolvable at n = 5. That
   conclusion would have been an artefact of the test.

**htfl, looked at only now that the choice is made** (E03 step 2's stated
condition). Both domains rank every 5-epoch cell above every 3-epoch cell, so
equi-selection tracked htfl on the axis that mattered. They disagree only
inside the 5-epoch group — htfl prefers 5e-5 (0.5310) where equi prefers 3e-5
(0.5278 htfl) — and that is the group where equi could not separate the cells
anyway. The E02 rank-crossing worry did not materialise where it counted.

### Selected configuration and its test number

**LR 3e-5, 5 epochs. Seed 42** — highest on equi (dev), which is what selection
is allowed to use.

| | equi (dev) | htfl (test) |
|---|--:|--:|
| F1 | **0.5081** | **0.5441** |
| P / R | 0.5241 / 0.4930 | 0.5611 / 0.5280 |
| / ceiling | 0.534 of 0.9523 | 0.598 of 0.9096 |

Run `20260912-061545_bert-base-cased_lr3e-05_e5_seed42`, best epoch 4, 6,174
predicted spans → 2,201 types against a gold key of 2,339.

**0.5441 is a selected number and must not be reported as the config's score.**
It is the max of five on dev, and its htfl score comes along with that
selection, so it carries the max's upward bias. The defensible headline for
this config is the five-seed mean, **htfl 0.5278 ± 0.0156**; 0.5441 is what one
particular seed did, useful for the breakdowns in T11 where a single concrete
model is needed.

Two observations about this seed. It is **also the highest htfl of the five**
(rank 1 of 5) — but at n = 5 that happens 20% of the time by chance, so it is
not evidence that dev-selection reliably finds the best test seed. And it is
the only seed whose equi precision and recall are close (0.5241 / 0.4930);
the other four sit at P ≈ 0.55 against R ≈ 0.40–0.43. Seed 42 predicts far more
(2,201 types where the others give 1,737–1,927), which buys recall and is where
its lead comes from.

Against E01: the same seed at the same config reported htfl 0.5200 there and
0.5441 here. The model is identical — only the per-seed statistic changed, from
final epoch to best epoch on equi.

**Reading:** the two predictions failed in different ways, and the difference
is worth keeping.

**The winner was missed, narrowly.** {3e-5, 3} and {2e-5, 5} were called; it was
{3e-5, 5}. But {2e-5, 5} turned out to be the sole member of the tie set — the
one config T9 genuinely could not separate from the winner.

**"t < 1.58, aka identical runs" was exactly right about the test as
specified.** Every unpaired t landed below 1.58 — 0.51, 0.73, 1.16, 1.22, 1.33,
not one exception. Had step 2 run as this entry originally described, that
prediction would have been a clean hit on all five pairs, and T9 would have
concluded its grid was unresolvable at n = 5.

Under the corrected paired test three cells separate. So the prediction was
correct about the instrument and wrong about the world. That is a different
failure from being wrong about both, and the distinction only survives because
the wrong test is still recorded here beside the right one.

**The mechanism prediction was a hit.** The schedule row of the table above is
what the data supports: every 5-epoch cell above every 3-epoch cell on both
domains, 3-epoch cells peaking at epoch 2–3 and 5-epoch cells at 3–4. The
overfitting mechanism is refuted. It also means T9's epoch axis measured
duration and schedule steepness together and cannot separate them — T12 can.

---

## E04 — T10, deberta-v3-base at the T9 config

**Run artifacts were lost.** The five runs happened on a Colab T4 and the JSONs
were never downloaded before the session ended. The numbers below are
transcribed from the run console output and are real measurements; they are
**not** backed by `results/runs/` files, and `src/aggregate.py` cannot read
this cell. Nothing here was reconstructed or inferred — the full 64-character
`encoder_weight_hash` is simply unavailable, since the console prints only its
first 16, which is why no run JSON was written by hand to stand in for them.

**Purpose:** the first cell of T10's encoder sweep. bert-base-cased's
hyperparameters from T9 are held fixed; only the encoder changes.

**Config:** `microsoft/deberta-v3-base`, LR 3e-5, 5 epochs, effective batch 16,
warmup 10% — identical to E02/E03's selected cell. Seeds 42–46.
1,435/1,435 optimizer steps, 4,592 train / 3,090 dev / 2,432 test, matching
local exactly. Encoder sha256 `eef90bafbd38f054…`, identical across all five.

**Environment differs from E01–E03 and this is a confound to carry forward:**
Tesla T4 on Colab, Python 3.13, Colab's torch — against Python 3.14.4 and
torch 2.14.0 locally. `transformers==5.16.1` and `tokenizers==0.23.1` matched.
seqeval would not build there, so 46 of 47 tests ran; `score_exact_spans` is
unverified against seqeval on that machine, though verified locally.

**Result:**

| seed | best epoch | equi ANN F1 | htfl ANN F1 | htfl P | htfl R | htfl NES | types |
|---|--:|--:|--:|--:|--:|--:|--:|
| 42 | 2 | 0.5402 | 0.5879 | 0.5996 | 0.5767 | 0.5710 | 2,250 |
| 43 | 1 | 0.5455 | 0.5811 | 0.5969 | 0.5661 | 0.5672 | 2,218 |
| 44 | 1 | 0.5451 | 0.5554 | 0.5742 | 0.5378 | 0.5384 | 2,191 |
| 45 | 5 | 0.5460 | 0.6004 | 0.6125 | 0.5887 | 0.5824 | 2,248 |
| 46 | 5 | 0.5545 | 0.6002 | 0.6183 | 0.5832 | 0.5808 | 2,206 |

| | mean | std (ddof=1) | ceiling | / ceiling |
|---|--:|--:|--:|--:|
| **equi (dev), ANN** | **0.5463** | **0.0052** | 0.9523 | 0.574 |
| **htfl (test), ANN** | **0.5850** | 0.0185 | 0.9096 | 0.643 |

0 of 5 collapsed.

**Against bert-base-cased (E03's selected cell), paired on the same five seeds:**

| seed | 42 | 43 | 44 | 45 | 46 |
|---|--:|--:|--:|--:|--:|
| equi difference | +0.0321 | +0.0573 | +0.0696 | +0.0731 | +0.0936 |

mean **+0.0651**, s_d 0.0226, **t = 6.44** (df = 4; 2.776 is p<0.05, Bonferroni
for 5 tests is 4.604), signs 5+/0−, sign test p = 0.0625. DeBERTa also carries
**a third of BERT's seed variance** on equi: 0.0052 against 0.0179.

**Per-epoch equi ANN F1** (best in bold):

| seed | e1 | e2 | e3 | e4 | e5 |
|---|--:|--:|--:|--:|--:|
| 42 | 0.5377 | **0.5402** | 0.5389 | 0.5263 | 0.5282 |
| 43 | **0.5455** | 0.4785 | 0.5309 | 0.5389 | 0.5259 |
| 44 | **0.5451** | 0.4802 | 0.5180 | 0.5241 | 0.5219 |
| 45 | 0.5081 | 0.4642 | 0.4808 | 0.5455 | **0.5460** |
| 46 | 0.5438 | 0.5407 | 0.5498 | 0.5425 | **0.5545** |

**Four readings.**

1. **3e-5 is probably too high for this encoder.** Seeds 43, 44 and 45 dip hard
   at epoch 2, exactly where LR peaks — and the dip is a precision/recall
   trade, not a general collapse: seed 43 goes P 0.5412 → 0.6349 while
   R 0.5497 → 0.3839 and predicted types fall 1,164 → 693. The model turns
   conservative under high LR, then recovers as the schedule decays. 3e-5 was
   selected in T9 **on BERT**. This is what E05 tests.

2. **The top of the curve is flat.** Best epoch scatters across 2, 1, 1, 5, 5
   while the peak *value* has std 0.0052. Epochs are not where the leverage is
   for DeBERTa, whatever they were worth for BERT.

3. **Train loss again fails to predict the good epoch.** Seeds 45 and 46 reach
   their best equi at train loss 0.0076 and 0.0074; seeds 43 and 44 reach
   theirs at 0.2576 and 0.2996. Same score either way. Same finding as E02
   reading 3, now on a second encoder.

4. **The same seed is not bit-reproducible on a T4.** Seed 43 was run twice in
   different sessions: epoch 1 agreed exactly (0.5455) and the later epochs
   diverged (e2 0.4793 vs 0.4785, e3 0.5338 vs 0.5309), while best epoch and
   htfl were identical because the peak was epoch 1 in both. `cudnn_deterministic`
   is false, so nondeterminism accumulates with steps. Worth knowing before any
   claim rests on a single late-epoch number.

**Reading:**

---

## E05 — deberta-v3-base learning-rate grid

**Entry written after the runs.** Like E04 this breaks the file's
prediction-first protocol; no prediction was recorded in advance. Flagged
rather than backfilled.

**Purpose:** E04 ran deberta-v3-base at 3e-5, the LR T9 selected *on BERT*, and
its per-epoch curves peaked at the end of warmup then dipped hard at epoch 2 —
a precision/recall trade, not a collapse — which is what too high an LR looks
like. This grid tests lower ones.

**Grid:** LR ∈ {1e-5, 2e-5} × epochs ∈ {3, 5}, seeds 42–46, 20 runs. 3e-5 is
**not** a cell: `{3e-5,5}` is E04, whose run JSONs were lost, and `{3e-5,3}`
was never run. E04's numbers are cited below for context and are not part of
any table `src/aggregate.py` produces.

**Environment, and it is not uniform.** `{1e-5,3}` ran on Colab (torch
2.11.0+cu128); the other three on Kaggle (torch 2.10.0+cu128). Both Tesla T4,
Python 3.13, `transformers==5.16.1` and `tokenizers==0.23.1` throughout. E01–E03
are local: RTX 3050, torch 2.14.0+cu130, Python 3.14.4. Any bert-vs-deberta
comparison therefore spans environments, and one cell of this grid differs from
its three neighbours.

### Step 1 — measurement

| equi | epochs 3 | epochs 5 |
|---|---|---|
| **LR 1e-5** | 0.5589 ± 0.0071 | **0.5590 ± 0.0086** |
| **LR 2e-5** | 0.5504 ± 0.0054 | 0.5523 ± 0.0048 |

| cell | equi mean ± std | htfl mean ± std | best epochs |
|---|--:|--:|---|
| 1e-5 / 3ep | 0.5589 ± 0.0071 | 0.5801 ± 0.0160 | 1, 3, 3, 3, 3 |
| 1e-5 / 5ep | 0.5590 ± 0.0086 | 0.5784 ± 0.0228 | 1, 1, 3, 4, 5 |
| 2e-5 / 3ep | 0.5504 ± 0.0054 | 0.5870 ± 0.0133 | 1, 1, 3, 3, 3 |
| 2e-5 / 5ep | 0.5523 ± 0.0048 | 0.5902 ± 0.0126 | 1, 1, 2, 4, 5 |
| *3e-5 / 5ep (E04)* | *0.5463* | *0.5850* | *1, 1, 2, 5, 5* |

0 of 20 collapsed.

### Step 2 — paired, reference `{1e-5, 5}`

| cell | gap | s_d | t paired | signs | reading |
|---|--:|--:|--:|--:|---|
| 1e-5 / 3ep | +0.0000 | 0.0062 | 0.01 | 2+/3− | not detected |
| 2e-5 / 5ep | +0.0067 | 0.0088 | 1.71 | 4+/1− | not detected |
| 2e-5 / 3ep | +0.0086 | 0.0085 | 2.26 | 4+/1− | suggestive |

Bonferroni for 3 tests at df = 4 needs t > 3.961. Nothing reaches it.

**Selected: LR 1e-5, 5 epochs** — `results/deberta_selected.md`. Reference run
seed 42, best epoch 4, equi 0.5657, htfl 0.6039.

**The tiebreak rule was changed during this experiment, after the results were
visible.** It was "take the cheaper config in the tie set"; it is now "highest
mean on the selection domain". Recorded here because the timing matters to how
much the selection is worth. The change does not alter T9's outcome — that tie
set's only member cost the same 5 epochs as its reference, so cost never broke
the tie there — and it does not alter this one either: `{1e-5,5}` is the
highest equi mean of the four cells, so both rules and the plain ranking agree.
The rule that *would* have differed is one ranking on htfl, which would take
`{2e-5,5}`, and that is selection on the test set.

### Five readings

1. **The LR hypothesis holds on dev.** 1e-5 beats E04's 3e-5 by +0.0127 paired,
   t = 3.55, 5+/0− signs, and beats 2e-5 with 4+/1− at t = 1.71–2.26. The
   epoch-2 dip in E04 did point at a real problem.

2. **The epoch axis is flat — the flattest result in this project.**
   `{1e-5,3}` and `{1e-5,5}` differ by **0.0001** at t = 0.01. Against 286 s
   and 440 s per run, the same score is available for 35% less compute.

3. **htfl cannot see the LR axis at all.** Paired against the best-htfl cell,
   every comparison returns t = 0.89–1.09, not detected. The spread of the four
   cell means on htfl is 0.0118; the typical within-cell seed std is 0.0162.
   The differences are smaller than the noise, so the apparent htfl ordering —
   which runs opposite to equi's — is not established and must not be reported
   as an inversion.

4. **htfl is roughly twice as seed-noisy as equi here.** In the selected cell,
   0.0228 against 0.0086; across cells, 0.0126–0.0228 against 0.0048–0.0086.
   One config spans 0.0565 on htfl across its five seeds. The best single htfl
   run in the grid (0.6039) sits in the cell with the **lowest** htfl mean,
   which is what a max over five noisy draws does.

5. **Nothing survives Bonferroni**, so strictly this grid separates no cell
   from the reference. What carries weight is direction and consistency: 1e-5
   above 2e-5 in 4 of 5 seeds at both epoch counts, and above E04's 3e-5 in
   5 of 5.

**Reading:**

---

## E06 — T10, roberta-base at the T9 config

**Written before the runs.** E04 and E05 were both written afterwards; this one
is not. The predictions below are the author's, stated before any seed ran, and
are recorded verbatim.

**Purpose:** the second cell of T10's encoder sweep. bert-base-cased's
hyperparameters from T9 are held fixed; only the encoder changes. Same design
as E04, which did this for deberta-v3-base.

**Config:** `FacebookAI/roberta-base`, LR 3e-5, 5 epochs, effective batch 16,
warmup 10%, AdamW weight decay 0.01 — `configs/train.json` unchanged, so the
only CLI override is `--model`. Seeds 42–46.

**Environment: local, and this matters.** RTX 3050, torch 2.14.0+cu130, Python
3.14.4 — the same machine and stack as E01–E03. roberta-base fits in 4 GB
(2.64 G allocated, 3.06 G reserved, 0.62 G headroom, measured). So **E06 is
directly comparable to bert's E03 with no environment confound**, which E04 and
E05 cannot claim: those ran on T4s at torch 2.10–2.11. A roberta-vs-bert
comparison is therefore cleaner than any deberta-vs-bert comparison so far.

**Two things that could have broken and did not:** roberta's byte-level BPE
needs `add_prefix_space=True` with `is_split_into_words=True`, already handled
in `get_tokenizer` — without it the failure presents as "roberta is bad at this
task" rather than as an error. And its checkpoint is fp32, so the AdamW-on-fp16
NaN that killed deberta-v3-base does not apply.

**Reference points** (E03's selected cell, same config, same machine):

| | bert-base-cased |
|---|--:|
| equi (dev), ANN | 0.4811 ± 0.0179 |
| htfl (test), ANN | 0.5278 ± 0.0156 |
| equi ceiling | 0.9523 |
| htfl ceiling | 0.9096 |

For scale, deberta-v3-base at this same config (E04) reached equi 0.5463 and
htfl 0.5850, +0.0651 over bert on equi.

**Predicted (equi and htfl against bert):**
> "Benchmark target: RoBERTa should outperform BERT due to its pretraining on
> larger datasets."

**Predicted (seed stability):**
> "Fine-tuning stability is expected to align with BERT's baseline."

Read as: equi std near bert's **0.0179**, rather than deberta's much tighter
0.0052.

**Mechanism:**
> "RoBERTa shares virtually the same underlying architecture as BERT."

More pretraining data on an architecture that is otherwise the same predicts a
better starting representation and so a higher score, with the optimisation
behaviour — and therefore the seed spread — unchanged. This is falsifiable in a
way the score alone is not: if roberta wins but its seed std also collapses
toward deberta's 0.0052, the "same architecture, more data" account is
incomplete, because something changed how the fine-tuning behaves and not just
where it starts.

**Result:** 5 runs, 0 collapsed, local RTX 3050 / torch 2.14.0+cu130 / Python
3.14.4 — same stack as E01–E03. Encoder sha256 `d941b5d11bc60f3c…`, identical
across all five. 1,435/1,435 optimizer steps each.

| seed | best epoch | equi ANN F1 | htfl ANN F1 | htfl P | htfl R |
|---|--:|--:|--:|--:|--:|
| 42 | 5 | 0.4749 | 0.5755 | 0.6188 | 0.5378 |
| 43 | 1 | 0.5150 | 0.5561 | 0.5617 | 0.5507 |
| 44 | 1 | 0.5328 | 0.5451 | 0.5682 | 0.5237 |
| 45 | 4 | 0.4814 | 0.5681 | 0.6350 | 0.5139 |
| 46 | 2 | 0.5276 | 0.5710 | 0.6075 | 0.5387 |

| encoder | equi mean ± std | / ceiling | htfl mean ± std | / ceiling |
|---|--:|--:|--:|--:|
| bert-base-cased (E03) | 0.4811 ± 0.0179 | 0.505 | 0.5278 ± 0.0156 | 0.580 |
| **roberta-base** | **0.5063 ± 0.0266** | 0.532 | **0.5632 ± 0.0124** | 0.619 |
| deberta-v3-base (E04)* | 0.5463 ± 0.0052 | 0.574 | 0.5850 ± 0.0185 | 0.643 |

\* E04 is T4 / torch 2.11, transcript numbers, no artifacts. roberta and bert
share a machine and a stack, so only that pair is confound-free.

**Paired against bert, same five seeds:**

| | per-seed differences | mean | s_d | t | signs | reading |
|---|---|--:|--:|--:|--:|---|
| equi | −0.0332 +0.0268 +0.0573 +0.0085 +0.0667 | +0.0252 | 0.0402 | **1.40** | 4+/1− | not detected |
| htfl | +0.0314 +0.0205 +0.0421 +0.0439 +0.0389 | +0.0354 | 0.0096 | **8.26** | 5+/0− | take seriously |

t = 8.26 is the largest in this project and clears Bonferroni for the encoder
family (2 tests against bert, df = 4, needs t > 3.495) several times over.

**Reading:** the two predictions came apart, and so did the two domains.

**The score prediction holds on test and is not established on dev.** htfl
+0.0354 at t = 8.26 with all five seeds positive and s_d of only 0.0096 — the
per-seed gaps span 0.0205 to 0.0439, remarkably tight. equi +0.0252 at t = 1.40
is not detectable, because the per-seed differences swing from −0.0332 to
+0.0667: s_d there is 0.0402, four times htfl's.

**The stability prediction is refuted, and inverted between domains.** Predicted
was equi std near bert's 0.0179; measured is **0.0266**, 49% wider. On htfl it
goes the other way — 0.0124 against bert's 0.0156, *tighter*. roberta is more
seed-sensitive than bert on dev and less on test, which no reading of "more
pretraining data, same architecture" predicts.

**The mechanism's second half is what fails.** "Same architecture, more data"
predicts a better starting representation with fine-tuning behaviour unchanged.
Behaviour did change: best epochs are `1, 1, 2, 4, 5` against bert's
`3, 3, 3, 4, 5` — roberta peaks earlier and far more scattered. Three of five
seeds peak by epoch 2, where bert has none before epoch 3. Whatever the extra
pretraining bought, it did not leave the optimisation trajectory alone.

**Consequence for T10, and it is uncomfortable.** Selection happens on equi.
On equi, roberta and bert are a tie at n = 5. On htfl, roberta beats bert about
as decisively as this project can measure anything. An encoder sweep that
selects on dev would call this pair indistinguishable and would be wrong by
0.0354 on the number that matters. Note this is the mirror of E05, where equi
separated the LR cells and htfl could not resolve them at all — the two domains
have now each been the only one able to see a real effect, in different
experiments.

---

## E07 — roberta-base hyperparameter grid

**Written before the runs.** Second entry in a row to manage it.

**Purpose:** E06 found roberta beating bert decisively on htfl (+0.0354,
t = 8.26) while equi could not separate them (t = 1.40) — because roberta's
per-seed differences on equi swing from −0.0332 to +0.0667, giving an equi std
of 0.0266 against bert's 0.0179. 3e-5 is bert's LR, selected in T9 **on bert**.
This grid asks whether another LR is steadier on equi.

**Grid:** LR ∈ {1e-5, 2e-5, 3e-5} × epochs ∈ {3, 5} — the same axes bert got in
E03, so the two encoders are tuned alike. Seeds 42–46. `{3e-5, 5}` is E06 and
is **reused, not recomputed**: it ran on this machine at this stack and has
artifacts, unlike deberta's E04. That leaves **5 cells × 5 seeds = 25 runs**,
about 95 minutes locally.

**Environment: local throughout.** RTX 3050, torch 2.14.0+cu130, Python 3.14.4 —
same as E01–E03 and E06. No cell of this grid crosses a machine boundary, which
neither E05 nor the bert-vs-deberta comparison can say.

**Per-seed statistic:** best epoch on equi, ANN unique-list F1 — fixed in T8,
identical in every cell.

**Selection stays on equi.** The prediction below is about dev and test
agreeing, which is an empirical claim about the grid, not a selection rule:
htfl is recorded per run and read only after a cell is chosen on equi.

**Reference points:**

| | equi | htfl |
|---|--:|--:|
| roberta at 3e-5/5 (E06) | 0.5063 ± 0.0266 | 0.5632 ± 0.0124 |
| bert at 3e-5/5 (E03) | 0.4811 ± 0.0179 | 0.5278 ± 0.0156 |
| deberta at 1e-5/5 (E05) | 0.5590 ± 0.0086 | 0.5784 ± 0.0228 |

**Predicted:**
> "i predict that we will find some stable LR, epoch that have good dev and
> test score together."

Two claims, separable:

1. **Stability** — some cell has an equi std materially below E06's 0.0266,
   nearer bert's 0.0179 or deberta's 0.0086.
2. **Agreement** — that same cell is also strong on htfl, rather than the two
   domains picking different winners as they did in E05 (equi ranked the LR
   axis, htfl could not) and in E06 (htfl separated the encoders, equi could
   not).

Claim 2 is the harder one and the more interesting if it lands. In every grid
so far, the domain that could resolve an effect was the *only* one that could.

**Mechanism:** *(left blank before the runs — no mechanism was offered for why
a particular LR should be steadier on equi.)*

**Result:** 25 new runs plus E06's 5, 30 total, 0 collapsed. Local RTX 3050 /
torch 2.14.0+cu130 / Python 3.14.4 throughout — no cell crosses a machine
boundary. Encoder sha256 `d941b5d11bc60f3c…` identical across all 30.

| equi | epochs 3 | epochs 5 |
|---|---|---|
| **LR 1e-5** | 0.5011 ± 0.0185 | 0.5040 ± 0.0202 |
| **LR 2e-5** | 0.5008 ± 0.0210 | **0.5085 ± 0.0184** |
| **LR 3e-5** | 0.5032 ± 0.0229 | 0.5063 ± 0.0266 (E06) |

| cell | equi mean ± std | htfl mean ± std | best epochs |
|---|--:|--:|---|
| 1e-5 / 3ep | 0.5011 ± 0.0185 | 0.5399 ± 0.0151 | 1, 1, 2, 3, 3 |
| 1e-5 / 5ep | 0.5040 ± 0.0202 | 0.5445 ± 0.0101 | 1, 1, 1, 2, 3 |
| 2e-5 / 3ep | 0.5008 ± 0.0210 | 0.5527 ± 0.0080 | 1, 2, 2, 2, 3 |
| 2e-5 / 5ep | 0.5085 ± 0.0184 | 0.5525 ± 0.0136 | 1, 1, 2, 3, 4 |
| 3e-5 / 3ep | 0.5032 ± 0.0229 | 0.5454 ± 0.0313 | 1, 1, 2, 2, 3 |
| 3e-5 / 5ep (E06) | 0.5063 ± 0.0266 | **0.5632 ± 0.0124** | 1, 1, 2, 4, 5 |

**Step 2, paired, reference `{2e-5, 5}` (highest equi mean):**

| cell | gap | s_d | t | signs | reading |
|---|--:|--:|--:|--:|---|
| 3e-5 / 5ep | +0.0022 | 0.0126 | 0.38 | 2+/3− | not detected |
| 1e-5 / 5ep | +0.0045 | 0.0131 | 0.77 | 3+/2− | not detected |
| 3e-5 / 3ep | +0.0053 | 0.0142 | 0.83 | 3+/2− | not detected |
| 1e-5 / 3ep | +0.0074 | 0.0171 | 0.96 | 3+/2− | not detected |
| 2e-5 / 3ep | +0.0077 | 0.0150 | 1.15 | 3+/2− | not detected |

**Every cell is in the tie set.** Selected by the tiebreak: LR 2e-5, 5 epochs.

**The same grid read on htfl**, reference `{3e-5, 5}` — recorded after the
selection above, not used for it:

| cell | gap | s_d | t | signs | reading |
|---|--:|--:|--:|--:|---|
| 2e-5 / 3ep | +0.0104 | 0.0184 | 1.27 | 4+/1− | not detected |
| **2e-5 / 5ep** | +0.0107 | 0.0068 | **3.54** | **5+/0−** | **take seriously** |
| 3e-5 / 3ep | +0.0178 | 0.0227 | 1.75 | 3+/2− | not detected |
| **1e-5 / 5ep** | +0.0187 | 0.0107 | **3.89** | **5+/0−** | **take seriously** |
| **1e-5 / 3ep** | +0.0232 | 0.0139 | **3.72** | **5+/0−** | **take seriously** |

**Reading:** claim 1 is hollow, claim 2 is refuted, and the reason is the same.

**equi cannot resolve this grid at all.** Six cells, five comparisons, every t
between 0.38 and 1.15. The spread of the cell means is 0.0077 against a typical
within-cell std of 0.0213 — a ratio of **0.36**. The differences are a third of
the noise. Selecting a roberta config on equi is selecting on noise.

**htfl can.** Spread 0.0232 against a typical std of 0.0151, ratio **1.54**, and
three of five comparisons land at t = 3.54–3.89 with all five seeds agreeing in
sign. (None clears Bonferroni for 5 tests, t > 4.604, so family-wise nothing is
established; three independent 5+/0− sign patterns at p = 0.0625 each is what
carries the weight.)

**The two domains disagree, and this time the disagreement is established.**
equi's pick is `{2e-5, 5}`; htfl's is `{3e-5, 5}` — E06's config, bert's LR,
the one this grid was run to improve on. And htfl does not merely prefer its
own: it says equi's pick is **worse by 0.0107 at t = 3.54 with 5+/0− signs**.
In E05 and E06 only one domain could see anything at a time; here they see
opposite things, and the one that can see is not the one selection uses.

**Claim 1, stability, is technically satisfied and means little.** `{2e-5, 5}`
does have the lowest equi std, 0.0184 against E06's 0.0266, a 31% reduction.
But at n = 5 a std estimate has df = 4 and the true σ sits plausibly in
[0.6σ̂, 2.1σ̂] (T8), so 0.0184 and 0.0266 are not separable; and stability in a
measurement that cannot distinguish any cell from any other is not a property
worth selecting on.

**Claim 2, agreement, is refuted.** No cell is good on both, because "good on
equi" is not a thing this grid can identify. The cell strongest on htfl is the
one already in hand from E06.

**What this does not license.** The response to "dev cannot tune roberta" is
not to tune on htfl — that converts the test set into a second dev set and
there is no third domain to check it against. It is a finding about the
measurement: equi, at n = 5, has no power over roberta's hyperparameters. The
options are more seeds, a different dev domain, or accepting that roberta's
config is untunable here and keeping E06's.

---

<!--
Footnote on rounding: ceiling F1 recomputed from full-precision P and R is
0.9096 (htfl) and 0.9523 (equi); Tasks_week2.md quotes 0.9097 and 0.9524,
which come from rounding P and R to 4 dp before combining. Same measurement,
1 in the fourth decimal.
-->
