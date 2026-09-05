# Tasks — Week 2

Goal: a BERT-family BIO tagger, tuned as far as fine-tuning alone takes it,
with every number reported against its seed variance and its ceiling.

Week 1 artifacts this depends on: the loader (`src/stats/loading.py`), the
harness (`src/eval/`), the ceilings (`data_layout.md` §5.5), and the alignment
invariant (`Data_stats.md` §6.1).

### The model workstream — one chain, six checkpoints

T6 through T11 are **not separable tasks**. They are stages of a single pipeline,
each with a gate that must pass before the next begins. They are numbered so
progress is checkable, not so they can be parallelised or reordered.

```
T6 ──► T7 ──► T8 ──► T9 ──┬──► T10 ──► T11
gate   runs   noise   HP   │    encoders  breakdowns
                           └──► T12  (warmup probe, independent branch)
```

| Stage | What it settles |
|---|---|
| T6 — Dataloader + alignment gate | labels land where they should |
| T7 — First end-to-end run | the pipeline runs at all |
| T8 — Seed variance | what difference is detectable |
| T9 — Hyperparameter pass (BERT) | the config everything else is run at |
| T10 — Encoder sweep | which encoder |
| T11 — Breakdowns | where it fails, and the two predictions |
| T12 — Warmup probe | branches after T9; runs alongside T10 |

**The binding order is T8 before T9 before T10.** Without the seed std the
hyperparameter grid cannot be read; without fixed epochs from T9 the encoder
sweep has two free axes.

### Independent — start now

| Task | Depends on |
|---|---|
| T13 — Nested-term count | — |
| T14 — Tran et al. metric unit | — |

Neither needs a model. **T13 gates week 3**, so it should not wait on the
workstream above.

---

## Experiment discipline

**`EXPERIMENTS.md`, one entry per run, written before the run.**

```
## E07 — DeBERTa-v3 vs BERT, LR tuned per encoder
Predicted: DeBERTa wins by >1σ. Mechanism — it fragments htfl terms at
1.382 vs BERT's 2.043 (data_layout.md §8.1), and first-subword labelling
means less fragmentation is a stronger first piece.
Result: [filled after]
```

Running many comparisons and reporting the ones that worked is the failure mode
this prevents. Log the runs that went nowhere — they are what make the ones that
worked credible.

**`results/test_evaluations.log`** — every htfl evaluation: date, config, seed,
number, and why it was run. Selection happens on `equi`, always. A visible count
with no retroactive story.

**Report format, every table:** mean ± std over 5 seeds, and the score as a
fraction of that domain's ceiling F1.

| domain | ceiling F1 |
|---|--:|
| equi (dev) | 0.9524 |
| htfl (test) | 0.9097 |

**~4.3 points of any dev-test gap is ceiling difference, not generalisation.**
Report both raw and ceiling-normalised or the gap will be misread.

---

## T6 — Dataloader + alignment gate

**Input:** `load_domain(domain, cfg) -> list[Document]`, with
`Document.sentences` as parallel `(tokens, labels)`.

**Build:**

- `tok(tokens, is_split_into_words=True)` on the pre-split list. Never a
  re-joined string — ACTER's tokenisation is inconsistent about internal
  punctuation (`non - governmental` is three tokens, `self-employed` is one) and
  re-tokenizing shifts every label after the first hyphenated word.
- `word_ids()` for the subword→token map. Never string matching.
- Label on the **first** subword; every continuation `-100`. Special tokens
  `-100`.
- `label2id = {"O": 0, "B": 1, "I": 2}`, and set `id2label` on the model config
  so checkpoints are self-describing.
- `DataCollatorForTokenClassification` for dynamic per-batch padding.
- wind ≤2-token filter, **training split only**, as a config flag. It must not
  apply to equi or htfl.

**The gate — before any training:**

Positive rate over batched tensors, ignoring `-100`, must reproduce exactly:

| domain | expected |
|---|--:|
| htfl | 0.2604 |
| wind (post-filter) | 0.1562 |
| corp | 0.1262 |
| equi | 0.1822 |

**Run the gate with truncation disabled.** `max_length=256` truncates 8
sentences corpus-wide and truncated tokens lose their labels, so exact equality
fails for a reason that is not a bug. The gate tests alignment, not length
policy. Then enable truncation and log separately how many positive labels it
drops — that number goes in the results, not the assertion.

**Done:** four rates reproduced exactly in a test; truncation loss logged; the
filter flag verified off for equi and htfl.

**Risk:** this is the one bug the harness cannot catch. Misalignment happens
upstream of `decode`, nothing crashes, and the model trains to a plausible
score against shifted labels.

---

## T7 — First end-to-end run

Purpose is **not** a good score. It is proving raw files → tokenised batches →
model → first-subword logits → argmax → labels → `decode()` → term list →
`score_list()` runs end to end.

**Config:** `bert-base-cased`, LR 3e-5, batch 16, 5 epochs, AdamW, weight decay
0.01, linear schedule with 10% warmup, one seed.

**Inference path must reuse the T3 harness.** Do not reimplement decode or
scoring in the training script. If the number is wrong, there should be exactly
one place it can be wrong.

**Write the training loop by hand.** ~40 lines, and it is where the LR schedule
and warmup live — both of which T12 tests. `Trainer` hides them.

**Expect:** training loss plateaus well above zero. That is `data_layout.md`
§5.2b — the same string carries different labels depending on whether it sits
inside a longer term. Not a bug, and not a reason to keep training.

**Done:** one dev number and one test number, produced end to end, logged with
full config and seed.

---

## T8 — Seed variance

Same config as T7, **5 seeds**, `transformers.set_seed(n)`.

Randomness has three sources: the classifier head (768 × 3 + 3 = 2,307
parameters, the only randomly initialised weights), data shuffle order, and
dropout masks. Seeding weights alone stabilises nothing.

**Report:** mean ± std on **equi**. Also record the std on htfl — informative,
never used for selection.

**This number gates every later comparison.** Interpreting differences between
5-seed means:

| difference | reading |
|---|---|
| < 1 s | not established |
| 1–2 s | suggestive; add seeds before claiming |
| > 2 s | take seriously |

**If a seed collapses** — flat, majority-class output — do not discard it and do
not restart. **Train that seed longer at the same LR.** Two accounts of
fine-tuning instability exist: the head corrupting pretrained weights early, and
the run simply being stuck in optimization. They predict different fixes. If the
seed recovers with more steps, the second fits this setup; if it stays flat, the
first does. One extra run on a seed already in hand.

Collapse is less likely here than in typical NER: the positive rate is 13–26%,
not 1–2%.

**Done:** a std, written prominently in `results/`, because every later table is
read against it.

---

## T9 — Hyperparameter pass, BERT only

Grid: LR ∈ {2e-5, 3e-5, 5e-5} × epochs ∈ {3, 5, 10}. Batch fixed at 16. Warmup
fixed at 10% (T12 tests it separately — a third axis makes 27 configs).

5 seeds per cell. **Select on equi.** Anything inside the T8 std is a tie — take
the cheaper config and say so in the writeup.

**Also settle here: argmax or a tuned decision threshold.**
`data_layout.md` §9.2 — the training prior is ~0.14 under the adopted split
against htfl's 0.2604, so a threshold tuned on training-domain data inherits the
wrong prior. **Argmax plus a stated limitation is the defensible default.** If
tuning, tune on equi and record the value. Tuning on htfl is fitting the test
set.

**Done:** a selected config with its selection number on equi; the threshold
decision recorded; the full grid in `results/`, losers included.

---

## T10 — Encoder sweep

**Tune LR per encoder** — 3 values each, selected on equi, epochs and batch
fixed from T9. Carrying BERT's LR to every encoder biases the comparison toward
BERT and forces a caveat that undercuts the result.

3 encoders × 3 LRs × 5 seeds = 45 runs.

| encoder | note |
|---|---|
| `bert-base-cased` | the ATE reference point |
| `deberta-v3-base` | 128k SentencePiece vocab; fragments htfl terms at 1.382 |
| `roberta-base` | **needs `add_prefix_space=True`** |

`AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)`. RoBERTa's
BPE treats a leading space as part of the token; with `is_split_into_words=True`
and no flag it either raises or silently tokenizes unlike its pretraining. That
failure presents as "RoBERTa is bad at this task."

XLM-R only if the T5 comparison needs it — `data_layout.md` §8.1 has it worst on
htfl terms.

**Do not use BioBERT, SciBERT or PubMedBERT.** Biomedical pretraining is
selection using knowledge of the test domain and destroys the cross-domain
claim. A general-purpose encoder with a bigger vocabulary does not.

**Prediction, logged before running:** DeBERTa-v3 wins, because it fragments
htfl terms at 1.382 against BERT's 2.043 and first-subword labelling makes the
first piece carry the classification. If it does not win, the fragmentation
hypothesis is weaker than §8.1 suggests — also a finding.

**Done:** a table of mean ± std per encoder, one selected model, the prediction
resolved either way in `EXPERIMENTS.md`.

---

## T11 — Breakdowns on the selected model

On htfl, selected model, all 5 seeds.

- **Recall by term length** — 1, 2, 3, 4+ tokens
- **Recall by gold frequency** — hapax vs. the rest (htfl hapax is 47.7%)
- **Exact-span F1 alongside unique-list F1**, and the gap

**Two predictions to resolve:**

`Data_stats.md` §8.2 predicts multi-word recall exceeds single-word recall via
head-position transfer. **Logged as contrarian** — the standard ATE finding is
the reverse, and htfl's single-word terms are also its most frequent. Either
outcome is reportable; the field-matching one is expected.

The T3 spec predicts span F1 high with list F1 low — frequent terms found, rare
types missed, the expected shape at 47.7% hapax. The two landing close together
would mean uniform performance across the frequency distribution, which would be
surprising.

**Done:** both breakdown tables; both predictions resolved in writing; this is
the input to week 3's NOBI experiment.

---

## T12 — Warmup probe

Separate from T9's grid, deliberately.

Warmup 0% vs 10%, same 5 seeds, everything else fixed at T9's selection.

**Prediction:** 0% warmup shows visibly **higher seed variance**, not merely a
lower mean. The mechanism is that the randomly initialised head produces large
noisy gradients that flow into the pretrained encoder before the head computes
anything meaningful; warmup holds the LR near zero until it does.

Ten runs. The variance claim is the sharp part — "warmup helps the mean" is
inherited, "warmup reduces variance" is a specific prediction this setup can
test.

---

## T13 — Nested-term count

Deferred from T2 (`data_layout.md` §7). No model needed.

Gold terms that occur **only** as a contiguous token subsequence of a longer
gold term and never as a maximal span. **Split by 1-token (NOBI-addressable) vs
≥2-token (not addressed by NOBI)** — without the split, week 3's experiment has
no predicted effect size.

Not the same as "substring of another gold term": a term nested in one place and
standalone in another is fully reachable, and character-substring matching gives
false hits (`art` in `heart failure`). Token-sequence containment, not string
containment.

This decomposes htfl's measured `max_recall` gap (0.9316 ANN) into its
nested-term and discontinuous-fragment components.

---

## T14 — Tran et al. metric unit

Still open from `Data_stats.md` §9.4. TermEval 2020 is settled — unique-list,
since ACTER 1.2 shipped no span annotations. Tran et al. (2024) do sequence
labelling and could report either.

Also pull their **per-length recall** numbers if reported — that is the direct
comparison for T11.

---

## Not this week

NOBI implementation · document-level or cross-sentence context · multilingual
training · architecture changes. `DEPTH_DEBT.md`.