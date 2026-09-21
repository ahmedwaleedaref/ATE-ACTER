# Tasks

Planning only. Data facts live in `docs/data_layout.md`, measured statistics in
`docs/Data_stats.md`, and one entry per run in `EXPERIMENTS.md`.

Tasks, not days. A task is done when its **definition of done** is met.

A task is the unit of planning; an **experiment** is the unit that produces runs.
They are not one-to-one — T10 spans four experiments, and several tasks produce
no runs at all. Each task names the experiments it owns; experiments never name
tasks back.

| Task | Phase | Experiments | Depends on | Status |
|---|---|---|---|---|
| T1 — Repo, data, layout analysis | 1 | — | — | **done** |
| T2 — Data statistics | 1 | — | T1 | **done** |
| T3 — Evaluation harness | 1 | — | T1 | **done** |
| T4 — C-Value baseline | 1 | — | — | **done**; 3 decisions unrecorded |
| T6 — Dataloader + alignment gate | 2 | — | T2, T3 | **done** |
| T7 — First end-to-end run | 2 | E01 | T6 | **done** |
| T8 — Seed variance | 2 | E02 | T7 | **done** |
| T9 — Hyperparameter pass (BERT) | 2 | E03 | T8 | **done** |
| T10 — Encoder sweep | 2 | E04–E07 | T9 | **done** |
| T11 — Breakdowns on the selected model | 3 | E08 | T10 | **done** |
| T12 — Warmup probe | 2 | — | T9 | not started |
| T13 — Nested-term count | 2 | — | — | not started |
| T14 — Tran et al. metric unit | 2 | — | — | not started |
| T15 — Document-level context (FLERT) | 3 | E09 | T11 | **done** |
| T16 — Weight decay | 3 | — | — | not started |
---

# Phase 1 — data, harness, baseline

## T1 — Repo, data, layout analysis · Ahmed · DONE

### Description

Set the repo up and read ACTER by hand before any code exists: what the
directory layout is, what the annotation files contain, which variants ship,
and which of them this project uses. Write it down.

### Experiments and new concepts

None. No runs, no method — this is reading the dataset and recording what is
there.

### Produced

```
docs/data_layout.md    what ACTER is, section by section
CLAUDE.md              the project rules
configs/data.json      data root, language, domains, split, label variant
.gitignore             data/raw/ excluded — ACTER is CC BY-NC-SA
requirements.txt       every dependency pinned
data/raw/ACTER         cloned at tag v1.5, gitignored
```

### Decisions locked

| Decision | Value |
|---|---|
| Annotation scheme | IOB, not IO |
| Training labels | `without_named_entities` |
| Scoring | one output list, scored against **both** keys |
| Document | one source text file |
| Sentence boundary | blank line in the sequential annotation file |
| Casing | lowercase at deduplication |
| Lemmatisation | none, at any stage |
| Language | English |
| Train / validate / test | corruption + wind energy / dressage / heart failure |

The split is the standard ACTER cross-domain setting (Lang et al. 2021; Tran et
al. 2022, 2024), adopted for comparability with that line of work. Consequences
in `docs/Data_stats.md` §8.5.

Two ceilings were identified here and measured later in T3: nested terms cap
recall, discontinuous-term fragments cap precision (`data_layout.md` §5).

---

## T2 — Data statistics · Ahmed · DONE

### Description

Compute the statistics listed in `data_layout.md` §7, over the **annotated
portion only**. Scripts, not a notebook — these get re-run. The risk is
computing over unannotated files, which produces plausible numbers rather than
an error.

Six statistics, one line each:

- `loading.py` — inventory ratio, paired tokens over whole-corpus tokens with the
  unit cancelling; wind 0.1806 proves the 84%-unannotated corpus was not loaded.
- `s01_lengths` — sentence length in dataset tokens; wind is bimodal, 61% of its
  sentences are ≤2 tokens and 3,499 are a single period, which became the
  ≤2-token training filter.
- `s02_wordpieces` — wordpieces per sentence across four tokenizers, split by
  whether the token sits inside a gold span; **this is where 256 comes from**, and
  it also measured htfl terms fragmenting at 2.043 pieces under BERT against 1.382
  under DeBERTa — the encoder lever T10 later pulled.
- `s03_label_distribution` — B/I/O counts in dataset-token space; 13–26% positive,
  so no loss weighting, and the four rates became T6's alignment invariant.
- `s04_term_stats` — term length and frequency over the gold keys; ≥95% of terms
  are ≤4 tokens and hapax is 42–48%, which caps any frequency-based method near
  70% recall.
- `s05_overlap` — term-set overlap between the training domains and htfl; 4 shared
  types, 0.2%, so no reported score can be lexical memorisation.

### Experiments and new concepts

No experiments — no model exists yet. No new method: every item is measurement.

### Produced

```
src/statistics/loading.py                 the shared corpus loader
src/statistics/s01_lengths.py             sentence length
src/statistics/s02_wordpieces.py          wordpiece length, four tokenizers
src/statistics/s03_label_distribution.py  B/I/O counts
src/statistics/s04_term_stats.py          term length and frequency
src/statistics/s05_overlap.py             term-set overlap, train vs htfl
tests/test_loading.py                     parser, loader wiring, inventory ratio
docs/Data_stats.md                        the write-up

results/T2_data_stats/
  s01_lengths.md              s01_lengths.json
  s02_wordpieces.md           s02_wordpieces.json
  s03_label_distribution.md   s03_label_distribution.json
  s04_term_stats.md           s04_term_stats.json
  s05_overlap.md              s05_overlap.json
```

Every script reads the corpus only through `loading.py`, so a loader bug is a
bug in every statistic at once, in the same direction, with no crash — which is
why the loader is tested first.

**The number this task produced is `max_length` = 256.**

### Decisions locked

| Decision | Value |
|---|---|
| `max_length` | **256** subword pieces, truncation only, dynamic per-batch padding |
| wind filtering | drop sentences of ≤2 dataset tokens, training split only |
| Class imbalance | no loss weighting and no weighted sampler at 13–26% positive |
| T4 candidate cap | 4 tokens |
| T6 alignment invariant | corp 0.1262, equi 0.1822, wind 0.1562 post-filter, htfl 0.2604 |
| Statistics scope | annotated portion only, every time |

Wind's annotated word count is 57,766, not 314k. One §7 item, T2-7 (nested-term
count), is **deferred to T13** alongside the NOBI ceiling comparison.

---

## T3 — Evaluation harness · Ahmed · depends on T1 · DONE

### Description

Build the scorer by hand — not delegated, not generated. Two metrics over one
decode path, plus the tests that keep them honest.

**Unique-list F1** — decoded spans collapsed to a deduplicated lowercased set and
compared against the gold unique list. **The headline metric**: it is TermEval
2020's protocol, and every published number this project compares against is in
that unit.

**Exact-span F1** — decoded spans compared positionally against gold spans,
micro-averaged, **both boundaries matching exactly**; `assist device` predicted
where `ventricular assist device` is gold scores zero. Secondary and diagnostic.

Both consume the output of the same `decode()`. The expensive part is shared and
the second metric is one extra aggregation.

**Why both.** They are not one measurement at two granularities. Exact-span F1 is
occurrence-weighted and unique-list F1 is type-weighted — `heart failure` is 350
units of span F1 and exactly one list entry, and at htfl's 47.7% hapax those
diverge. And the ceilings exist **only** in the list metric: in span space there
is no representational loss at all. So exact-span F1 measures tagger quality
uncontaminated by the annotation scheme, unique-list F1 measures tagger quality
plus what the scheme costs, and the gap between them is a diagnostic.

**Naming.** "Span-level F1" is ambiguous — it is also used for token-level F1,
which scores each token independently and awards partial credit. Use
`exact_span` in the config and "exact-span F1, micro-averaged" in the writeup.

### Experiments and new concepts

No experiments — no model exists yet. **Three concepts enter the project here**,
and every later task uses them:

- **The two-metric decomposition** — type-weighted against occurrence-weighted,
  and the gap as a diagnostic rather than a discrepancy to explain away.
- **The ceilings.** `max_recall` and `max_precision` are what decoding the *gold*
  labels scores against the gold key. They bound the list metric before any model
  exists, and no reported number is readable without them.
- **The dangling-`I` policy.** A sentence-initial `I`, or an `I` after `O`, opens
  no span and is dropped — matching seqeval under `mode='strict', scheme=IOB2`.
  Gold barely exercises it; model output is unconstrained, so it is load-bearing
  there.

**One prediction logged here, resolved in E08 — refuted.** Span F1 high with list
F1 low was predicted, meaning frequent terms found and rare types missed. Measured
at n=5 the gap is −0.0036 ± 0.0041, indistinguishable from zero.

### Produced

```
src/eval/spans.py           encode, decode, count_invalid_tags
src/eval/surface.py         spans_to_unique_list, generate_unique_list,
                            generate_flatten_spans, write_term_list
src/eval/scorers.py         score_list, score_exact_spans, load_gold_list_into_set
src/eval/run_eval.py        compute_ceilings
src/eval/score_baseline.py  score an external term list through the same path
configs/eval.yaml           headline metric, scheme, keys, dangling-I policy
docs/Eval_harness_doc.md    the write-up
results/T3_eval_harness/ceilings.md         max_recall / max_precision, four domains, both keys
```

**Six tests, four specified and two beyond the spec.** The distinction matters:
most of them compare the code against itself.

1. **Identity** — gold list scored as its own prediction. Exactly 1.0. Catches
   casing, dedup and normalisation errors.
2. **Span round-trip** — gold spans → `encode` → `decode` → scored against the
   same spans. Exactly 1.0, a real assertion: span space has no representational
   loss, so any deviation is an `encode`/`decode` bug.
3. **List round-trip** — the same path scored against the gold unique list. Does
   **not** return 1.0; it returns the ceilings. A measurement, not an assertion.
4. **Hand-computed fixture** — the only external oracle, and the reason this task
   was not delegated. Adapted from `corp_en_01_seq_terms.tsv`, deliberately not
   verbatim: one label changed (`life` `O` → `I`) to include a two-token span.
   Expected values derived by hand from the metric definitions. The same
   prediction gives **recall 0.5 by span and 0.667 by type** — the two-metric
   divergence, at a scale that can be checked by counting.
5. **Label round-trip** (beyond spec) — `encode(decode(gold)) == gold` as exact
   sequence equality, asserting exactly **3** mismatches (`data_layout.md` §3).
   Stronger than the span round-trip, which is a set comparison and survives a
   pair of symmetric bugs; equality names the failing token instead of returning
   0.998.
6. **seqeval agreement** (beyond spec) — `score_exact_spans` cross-checked
   against seqeval on a perturbed prediction, agreeing to 6 dp under
   `mode='strict', scheme=IOB2`. The perturbation excludes overlapping spans by
   construction, so **overlap behaviour is untested** and no reader should infer
   otherwise.

Pinned span counts: corp 4,180 / equi 8,662 / wind 5,053 / htfl 9,636.

### Decisions locked

| Decision | Value |
|---|---|
| Headline metric | unique-list F1; exact-span F1 is diagnostic |
| Span convention | `(file_id, sent_idx, start, end)`, `end` **exclusive** |
| Dangling `I` | dropped, opens no span; matches seqeval strict IOB2 |
| `scheme` parameter | present from the start; only `bio` implemented, NOBI is a branch |
| Metric naming | `exact_span`, never "span-level" |
| htfl ANN ceiling | **0.9096** — the cap on every number this project reports |
| equi ANN ceiling | 0.9523 |

Findings recorded in `data_layout.md`: strict IOB2 and the dangling-`I` policy
(§3), the tokenised/non-tokenised key trap with measured counts (§4), and the
ceilings for all four domains (§5.5).

---

## T4 — C-Value baseline · khaled Ahmed · no dependencies · DONE

### Description

A no-training statistical reference point, giving a lower bound to improve on.
C-Value scores a candidate phrase by how often it occurs, how many words it
contains, and how often it appears nested inside longer phrases.

Candidate generation → C-Value scoring → threshold → term list, one lowercased
term per line. Runs from a config file.

**Scored on unique-list F1 only.** C-Value ranks candidate types, not
occurrences, so the exact-span metric does not apply to it.

It could start before T3 existed, tested against a hand-made 10-line gold file
and swapped onto the real scorer when the harness landed.

### Experiments and new concepts

No experiments — nothing is trained and there are no seeds. No new concept for
this project: C-Value is a published method (Frantzi et al.), used here as a
reference point rather than a contribution.

### Produced

```
src/models/cvalue.py        candidate generation, C-Value scoring, threshold
src/models/run_cvalue.py    the runner
configs/cvalue.json         paths and parameters
docs/cvalue_baseline.md     the write-up

results/T4_cvalue_baseline/
  htfl_cvalue_terms.txt       the predicted term list
  htfl_cvalue_run_info.json   config and provenance for that run
  baseline_cvalue.md          scored through src/eval/score_baseline.py
```

**htfl unique-list F1 0.2003 (ANN) / 0.2048 (NES).** 2,494 predicted terms
against 2,339 gold. Input passed format validation — no tabs, no uppercase, no
duplicates.

For orientation only: **no TermEval participant used C-Value**, so there is no
direct published comparison. The nearest comparable system is e-Terminology's
TSR (statistical, frequency threshold ≥2) at F1 20.1 incl NE / 21.4 excl NE on
the same test set — so 0.20 is an ordinary place for a no-training statistical
baseline to land. Not comparable digit-for-digit: those numbers are on ACTER
1.2's non-tokenised flat lists against a different gold.

**Why it cannot go much higher.** TermEval found recall lowest on hapax terms
across every system, and e-Terminology reached **0%** hapax recall because of its
frequency cut-off. At htfl's 47.7% hapax (`Data_stats.md` §7.3) a frequency-based
method is structurally capped well below the ceiling. That is a property of
C-Value, not a defect in this implementation.

### Decisions locked

| Decision | Value |
|---|---|
| Metric | unique-list F1 only; exact-span does not apply |
| Candidate length cap | 4 tokens, from `Data_stats.md` §7.1 |
| Role | lower bound for orientation, never a tuned system |

### Still unrecorded — the number is provisional until these are

1. **Frequency corpus.** Annotated portion only, or the unannotated texts as
   reference material too? Both legitimate, not comparable to each other. Inputs
   in `Data_stats.md` §4.5 and §7.3.
2. **Tokeniser.** 247 of 2,494 predicted terms (~10%) contain an apostrophe or
   hyphen with no surrounding space, while ACTER's tokenised key writes
   `public prosecutor 's office`. If candidate generation used its own tokeniser
   those terms cannot match regardless of ranking quality, and the failure is
   silent — a false positive and a false negative on the same term. Some will be
   genuine (`30-day`); the split is unknown.
3. **Threshold.** 2,494 predicted against 2,339 gold is close enough to suggest
   the cutoff was set to produce a plausible *count*. If it was tuned against
   htfl in any way the baseline is contaminated — htfl is the held-out test set.
   A threshold from the training domains, or a fixed C-Value score, is fine but
   must be stated.

---

## Phase 1 status

| Task | Status |
|---|---|
| T1 — Repo, data, layout | done |
| T2 — Data statistics | done; nested-term count deferred to week 3 |
| T3 — Evaluation harness | **done**, 30 tests |
| T4 — C-Value baseline | **done**; F1 ≈ 0.20; 3 decisions unrecorded |

**Nothing blocks phase 2.** The T4 decisions are khaled's to record and do not
gate model training; the harness is what phase 2 depends on and it is closed.

---

## Not this phase (phase 1)

NOBI implementation · multilingual or cross-lingual training · any model
training · architecture changes.

---

# Phase 2 — the model

Goal: a BERT-family BIO tagger, tuned as far as fine-tuning alone takes it,
with every number reported against its seed variance and its ceiling.

Week 1 artifacts this depends on: the loader (`src/statistics/loading.py`), the
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
| T8 — Seed variance | what difference is detectable — **s = 0.0179 on equi** (E02) |
| T9 — Hyperparameter pass (BERT) | the config everything else is run at — **LR 3e-5, 5 epochs** (E03) |
| T10 — Encoder sweep | which encoder — **deberta-v3-base**, roberta second (E04–E07) |
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

## T6 — Dataloader + alignment gate · Ahmed · DONE

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

**Done.** `src/data/align.py` (hand-written: `align_labels`,
`recover_token_labels`, `positive_rate`, plus `LABEL2ID`/`ID2LABEL`) and
`src/data/dataset.py` (tokenisation, filter, `ATEDataset`, collator,
length-grouped batching). Report in `results/t6_alignment.md`; split moved into
`configs/data.json`, run settings into `configs/train.json`. 38 tests pass.

**The gate changed shape.** Instead of asserting the four positive rates, it
asserts per-example exact sequence equality — `recover_token_labels(word_ids,
align_labels(word_ids, gold), …) == gold` — over all 14,162 sentences,
truncation disabled. It catches misalignment, a lost token and a length
mismatch, and names the failing sentence and token instead of reporting a
domain-level number that is slightly off. The rates are now reported, and the
reason is the finding below.

**Finding: 41 dataset tokens tokenize to zero wordpieces.** All in
`wind_en_01`, the Private Use Area characters U+F8EF and U+F8FA left by PDF
extraction, all gold `O`; corp, equi and htfl have none. They get no model
position, so their labels land nowhere and wind's rate is measured over 53,450
positions rather than 53,491 dataset tokens: **0.1563, not §6.1's 0.1562**. An
exact-equality assertion on the four rates would have failed on the first run
for a reason that is not a bug. corp 0.1262, equi 0.1822 and htfl 0.2604
reproduce exactly. Recorded in `Data_stats.md` §6.1; the count is a property of
the tokenizer, not of ACTER, so it will change at T10.

**Truncation costs nothing on dev or test.** 8 sentences exceed 256 subwords
corpus-wide (corp 2, wind 5, htfl 1 — matching `data_layout.md` §8.2), losing
346 tokens and 48 positive labels, **all of them in the training domains**.
equi loses nothing; htfl loses 2 tokens, both `O`. Truncated tokens are
predicted `O` by policy, which can only cost recall — and on the reported
metrics that cost is zero.

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

**Expect:** nothing in particular from the training loss. This task originally
said it would plateau well above zero, citing `data_layout.md` §5.2b — the same
string carrying different labels depending on whether it sits inside a longer
term. **That premise was false and is now corrected in §5.2b and
`Data_stats.md` §7.2:** a nested occurrence still carries a positive label, and
`B` versus `I` follows from sentence context. E01 reached loss 0.0043, which is
ordinary for a 110M-parameter model fitting 4,592 sentences, and needs no
explanation. It raises no memorisation question either — htfl shares 0.2% of
its term types with the training keys.

**Done:** one dev number and one test number, produced end to end, logged with
full config and seed.

---

## T8 — Seed variance · Ahmed · DONE

Same config as T7, **5 seeds**, `transformers.set_seed(n)`. Seeds fixed before
the first run: **42, 43, 44, 45, 46**. Written into `configs/train.json`, not
chosen as they go.

Randomness has three sources: the classifier head (768 × 3 + 3 = 2,307
parameters, the only randomly initialised weights), data shuffle order, and
dropout masks. Seeding weights alone stabilises nothing.

**Per-seed statistic — fixed here, used everywhere after.** Each seed
contributes its **best epoch on equi**, which is how selection works
downstream. Max-over-epochs is biased upward and its spread is not the spread
of final-epoch scores; the bias is harmless only if every cell in T9 and T10
uses the identical rule. Whichever is chosen, it is chosen once, in this task.

**Report:** mean ± std on **equi**, sample std (`ddof=1`). Also record the std
on htfl — informative, never used for selection.

### What the std is for

Every later comparison is two 5-seed means. The question is never "is this mean
bigger" — it is "is this gap larger than gaps produced by seed luck alone."

```
SE(gap) = sqrt( s_A^2 / 5 + s_B^2 / 5 )
t       = |mean_A - mean_B| / SE(gap)
```

Both configs contribute their own std; they are not assumed equal. Subtraction
adds variance, so the noise floor on a gap is larger than on either mean.

| t | reading | action |
|---|---|---|
| < 1.58 | no difference detected | tie — take the cheaper config, say so in the writeup |
| 1.58 – 3.16 | suggestive | add seeds; SE shrinks as √n, the gap does not move |
| > 3.16 | take seriously | scoped to equi, this metric, this n |

`t < 1.58` is **not** "the configs are identical." It is "this experiment
cannot see a difference at n=5." Never write the first sentence.

`t > 3.16` is **not** "better over these 5 seeds." The sample mean already said
that and needed no statistics. It is that the gap is too large to be seed luck,
so the configs plausibly differ underneath.

**Report the gap in F1 points alongside t.** t grows with √n, so a large enough
n makes a 0.002 gap significant and still worthless.

**Sanity check, not a second verdict:** if every run of A beats every run of B,
the mean is not being carried by one lucky seed. If t is large but the runs
interleave, look for an outlier before believing it.

σ̂ is itself noisy at n=5. df = 4 puts the true σ plausibly in
[0.6·σ̂, 2.1·σ̂]. The measuring stick has a measuring error, which is why the
middle band says add seeds rather than probably yes.

### If a seed collapses

Flat, majority-class output — every token `O`, zero terms decoded, F1 exactly
0.0000. Token accuracy stays high because `O` is 74–87% of labels, so accuracy
will not reveal it. Collapse is less likely here than in typical NER: the
positive rate is 13–26%, not 1–2%.

**Do not discard it.** The mean estimates what this config does under the seed
lottery, and collapse is an outcome of that lottery. Dropping it reports the
mean of the runs that worked — a different and flattering quantity, and the
exact failure mode `EXPERIMENTS.md` exists to prevent.

**Do not restart it.** Re-rolling until a seed behaves is selection on the
outcome.

**Train that same seed longer at the same LR.** Two accounts of fine-tuning
instability predict different outcomes: the run stuck in optimization should
recover with more steps; early corruption of the pretrained encoder by the
randomly initialised head's first gradients stays flat. One extra run on a seed
already in hand, and the answer feeds T12 — the corruption account is the
mechanism the warmup prediction rests on.

**With a collapse in the set, do not compute t.** The std is dominated by one
point and the distribution is not remotely normal. Report the five-seed mean
and std including the zero, the collapse rate, and the diagnostic outcome. A
1-in-5 collapse rate is a property of the config and matters more than its
mean.

**Done:** a std, written prominently in `results/`, because every later table is
read against it; five runs logged with full config and seed; the per-seed
statistic stated.

**Result (E02):** equi 0.4811 ± 0.0179, htfl 0.5278 ± 0.0156, 0 of 5 collapsed.
A gap on equi must clear 0.0179 F1 to be detected at n=5 and 0.0358 to be taken
seriously. Full report in `results/t8_seed_variance.md`; per-epoch curves, the
six readings and a known defect in the `dirty` flag are in `EXPERIMENTS.md` E02.

---

## T9 — Hyperparameter pass, BERT only · Ahmed · DONE

Grid: LR ∈ {2e-5, 3e-5, 5e-5} × epochs ∈ {3, 5}. Batch fixed at 16. Warmup
fixed at 10% (T12 tests it separately — a third axis makes 18 configs).

**10 epochs dropped.** E02's per-epoch curves peak at epoch 3 in three seeds of
five and at 4 and 5 in the others, and seed 46 is 0.0387 *below* its epoch-3
score by epoch 5. Nothing in that data suggests 10 epochs is worth the compute;
the live question on this axis is 3 against 5.

**{3e-5, 5} is already measured** — it is E02, the T8 cell. Five seeds at that
config exist in `results/runs/`. Do not recompute it; reuse those five numbers
as the grid cell, which leaves **5 cells × 5 seeds = 25 runs**.

5 seeds per cell, the same five in every cell. **Select on equi.**

**The comparison is paired on the seed**, not the unpaired two-sample test this
line used to specify. `set_seed(n)` runs before `from_pretrained`, so seed n
carries an identical head init and shuffle order into every cell — the cells
are the same five subjects under different treatments. Take the five per-seed
differences and run a one-sample t against zero, df = 4: below 2.132 no
difference detected, above 2.776 take it seriously. Report the sign test too;
five differences of one sign is p = 0.0625 and assumes no distribution.

The unpaired test is not merely weaker here, it is the wrong test: it charges
the between-seed spread to its own uncertainty twice, and on measured cells
returns t = 0.51 where the paired test returns 1.82. T8's 1.58/3.16 bands were
that formula's 1-std and 2-std gaps and do not carry over.

Ties go to the highest mean on the selection domain, said so in the writeup.
(This replaces an earlier "take the cheaper config" rule. It does not change
T9's recorded outcome: that tie set's only member cost the same 5 epochs as
its reference, so cost never broke the tie there.)

**Done:** a selected config with its selection number on equi; the full grid in
`results/`, losers included.

**Result (E03):** **LR 3e-5, 5 epochs**, equi 0.4811 ± 0.0179. Tie set is
{2e-5, 5} alone, same 5-epoch cost, so the higher mean holds. 30 runs, 0
collapses. `results/t9_selected.md`; grid in `results/t9_grid_step1.md` and
`results/t9_grid_step2.md`, losers included.

The epoch axis carried it — all three 3-epoch cells separate (paired t 2.47,
3.03, 4.04) and land within 0.0020 of each other, so LR does almost nothing at
3 epochs. That axis is confounded with schedule steepness: LR decays to zero
across the whole run, so "5 epochs" is also "gentler decay". T12 separates
them.

---

## T10 — Encoder sweep · Ahmed · DONE

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

XLM-R only if the T14 comparison needs it — `data_layout.md` §8.1 has it worst on
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

**Result (E04–E07).** Each encoder at its selected config, 5 seeds:

| encoder | config | equi mean ± std | htfl mean ± std |
|---|---|--:|--:|
| bert-base-cased | 3e-5 / 5ep | 0.4811 ± 0.0179 | 0.5278 ± 0.0156 |
| roberta-base | 3e-5 / 5ep | 0.5063 ± 0.0266 | 0.5632 ± 0.0124 |
| **deberta-v3-base** | 1e-5 / 5ep | **0.5590 ± 0.0086** | **0.5784 ± 0.0228** |

**The prediction holds: DeBERTa-v3 wins, on both domains.** So the
fragmentation account in `data_layout.md` §8.1 survives — deberta fragments
htfl terms at 1.382 against bert's 2.043, and first-subword labelling makes the
first piece carry the classification.

**Two caveats the table cannot show.** DeBERTa ran on Colab/Kaggle T4s at torch
2.10–2.11 while bert and roberta ran locally at 2.14.0, so only the
bert-vs-roberta pair is free of an environment confound. And LR was tuned per
encoder as this task requires (bert T9/E03, deberta E05, roberta E07) — but
roberta's grid came back with all six cells tied on equi, so its LR is not
tuned so much as untunable at n = 5.

**Selected model: `deberta-v3-base`, 1e-5 / 5 epochs, seed 42** —
`results/model_selection.md`, which also records the run chosen for each of the
three encoders. It wins both domains. Its E05 cell has artifacts under
`results/runs/deberta_grid/deberta_lr1e-05_e5/` — the cell lost with the Colab
session was E04's `{3e-5, 5}`, a different config.

The cost of that choice: deberta cannot train on this machine (2.94 GB of fp32
weights, gradients and AdamW moments against 3.65 GB usable, OOM at every batch
size), so T11's breakdowns require rented GPU time rather than a local re-run.

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

## Not this phase (phase 2)

NOBI implementation · document-level or cross-sentence context · multilingual
training · architecture changes.

---

# Phase 3 — error analysis, context, schedule

T11 and T12 are specified in phase 2 above and are not restated here. T11 is
done (E08); T12 is not started.

---

## T15 — Document-level context (FLERT) · Ahmed · DONE

FLERT-style: left and right context tokens around each sentence, loss and
prediction computed on the sentence tokens only. Context is attended
(`attention_mask` 1) and never scored (label `-100`).

**Measure the fill rate before implementing.** For every sentence, how many
context tokens are actually available at the chosen window, per domain, as a
distribution rather than a mean. Documents are wildly uneven — wind 11,553
tokens per document, corp 4,237, equi 1,712, htfl 292 — so the test domain has
by far the least context available while both training domains have plenty.
**That is a distribution shift in the feature being added**, not only a
missing-input problem, and it has to be named either way the result falls.

Report equi and htfl separately, never pooled.

Keep it inside the 512-token limit. No larger encoders.

**Done:** fill rate measured and recorded before any implementation; the gate
below passing; equi and htfl reported separately with the epoch axis controlled.

**The gate, and it is the whole correctness argument.** Context positions are
`-100`, so they leave the scored denominator untouched: the positive rate with
context on must equal the sentence-level rate **exactly** — corp 0.126211,
equi 0.182156, wind 0.156257, htfl 0.260371. If one context label reaches the
loss, that rate moves and nothing crashes. Plus per-example sequence equality on
the sliced recovery, as in T6.

**The sentence is budgeted against `max_length` first and context fills the
remainder.** Otherwise context pushes a sentence's own tail past the truncation
frontier, those tokens lose their labels, and recall drops for a reason that is
not the feature under test — a context run would look worse than the baseline
because of bookkeeping. Verified: the sentences that lose labels are the same
set with and without context, and equi and htfl lose none.

**Result (E09).** Window 32, 6 epochs, seeds 42–46. Three cells, because the
first comparison changed context *and* epochs:

| | equi | htfl |
|---|--:|--:|
| ctx 0, 5 ep (E08) | 0.5592 ± 0.0090 | 0.5782 ± 0.0230 |
| ctx 0, 6 ep (control) | 0.5592 ± 0.0056 | 0.5877 ± 0.0178 |
| **ctx 32, 6 ep** | **0.5866 ± 0.0031** | **0.6013 ± 0.0058** |

Context alone: equi **+0.0274, t 7.71, 5+/0−**; htfl +0.0136, t 1.88, 5+/0−.
Epochs alone: equi −0.0000, t 0.01. **Context helps on dev decisively and is not
established on test at n = 5** — which is what the fill rate predicted, htfl
being the domain with the least context on the page.

Seed variance collapsed: equi 0.0090 → 0.0031, htfl 0.0230 → 0.0058.

The logged prediction — that context would lift rare-term recall most — is
**refuted in both halves**: singletons +0.011 against frequent terms +0.021, and
the largest gain is on 4+ token terms (+0.075), where training is thinnest.

Records: `results/context_experiment.md`, `results/context_fill_rate.md`,
EXPERIMENTS.md E09.

**Open:** window 64 (to be run only if 32 helped — it does, on dev); the
within-htfl fill-rate correlation, which needs per-document predictions the
corpus-wide term dump cannot supply; and the control cell's artifacts, lost to a
Kaggle session and surviving only as console-transcribed headline numbers.

---

## T16 — Weight decay · not started

**Check the arithmetic before spending GPU time.** AdamW decouples the decay from
the gradient, so each step multiplies every decayed weight by `(1 - lr·wd)`.

At lr 1e-5 and wd 0.01 that is **1e-7 per step**. The linear schedule halves the
average learning rate over the run, so total shrinkage across 1,435 steps is
about `1435 × 1e-7 × 0.5 ≈ 7.2e-5` — **0.007%**.

A second framing agrees and does not depend on the run length: AdamW's update is
`lr·(m̂/(√v̂+ε) + wd·θ)`. The Adam term is normalised to order 1; the decay term
is `wd·θ ≈ 0.01 × 0.04 ≈ 4e-4` for typical encoder weight magnitudes. The decay
contributes roughly **0.04% of each update**.

**If that holds, weight decay at the conventional value is arithmetically inert
here and the finding is the calculation, not an experiment.** Verify the
reasoning first; only run the grid if the arithmetic says there is something to
see.

Note this is a property of *this* learning rate. At 5e-5 the per-step factor is
5× larger, so the conclusion does not transfer to the bert cells without redoing
the sum.

---

## Not this phase (phase 3)

Multilingual training and a multilingual encoder with context · NOBI
implementation (T13 is its prerequisite and is not started) · architecture
changes.
