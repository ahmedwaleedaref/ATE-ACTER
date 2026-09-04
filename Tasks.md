# Tasks — Week 1

Data facts live in `docs/data_layout.md`. This file is planning only.

Tasks, not days. A task is done when its **definition of done** is met, not when
a day ends.

| Task | Owner | Depends on | Status |
|---|---|---|---|
| T1 — Repo, data, layout analysis | Ahmed | — | **done** |
| T2 — Data statistics | Ahmed | T1 | **done** |
| T3 — Evaluation harness | Ahmed | T1, interface contract | **done** |
| T4 — C-Value baseline | khaled Ahmed | interface contract | **done**; 3 decisions unrecorded |
| T5 — Prior work and comparison table | Moamen Talaat | — | **done** |

T4 and T5 do **not** wait for T3. See the interface contract below — that is what
makes them parallel.

---

## Interface contract

Fix this before anyone writes code. It is the only thing that crosses a task
boundary.

**A term list file:**

- one term per line
- lowercased
- deduplicated
- UTF-8
- no header, no index column

Every system that produces terms writes this format. The scorer reads this format
and does not know or care what produced it.

Consequences:

- T4 can be built and tested today against a hand-written 10-line file, then
  plugged into the real scorer the moment T3 lands.
- The neural model in week 2 emits the same format, so it needs no scorer changes.
- Gold standard files are already in a compatible shape (one term per line,
  lowercased) — the label column is stripped on load.

**[T3] The loader strips trailing whitespace.** Gold TSVs have a second column,
so the newline lands there and `split("\t")[0]` is clean; a contract-format file
has one column, so without the strip the newline welds to the last term of every
line and the file scores near zero with no error. The strip lives in the loader,
never in the writer — a writer that emits a padding tab to compensate is not
producing contract format, and T4's output would not round-trip.

**Note:** the term list is the contract for *cross-task* exchange. T3 also
computes an exact-span metric internally (see below), which never crosses a task
boundary — T4 produces no spans, only a list.

---

## T1 — Repo, data, layout analysis · Ahmed · DONE

Repo created with pinned dependencies and `CLAUDE.md` rules. ACTER cloned at tag
`v1.5`, gitignored. Data inspected by hand and written up in
`docs/data_layout.md`.

**Decisions locked:**

| Decision | Value |
|---|---|
| Annotation scheme | IOB (not IO) — through week 2 |
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
in `Data_stats.md` §8.5.

**Findings that shape later tasks:** the recall ceiling from nested terms, the
precision ceiling from discontinuous-term fragments, and NOBI as the published
attack on the first (see `data_layout.md` §5).

Tag the repo: `git tag day-1`.

---

## T2 — Data statistics · Ahmed · DONE

Compute the statistics listed in `data_layout.md` §7, all over the **annotated
portion only**.

**Output:** `results/data_stats.md` plus the script that generated it, in `src/`.
Not a notebook — this gets re-run.

**The number that matters most:** term-set overlap between the training domains
and heart failure. It tells you how much of any future score is memorisation
rather than cross-domain generalisation, and it is the first thing a sharp reader
will ask about.

**Definition of done:** every §7 T2 item has a number; the wind-energy word count
is near 52k, not 314k; `data_stats.md` committed with the script.

**Risk:** computing over unannotated files. The word-count table in
`data_layout.md` §1 is the cross-check.

**Done.** Statistics computed by `src/stats/s01`, `s02`, `s04`–`s06` (loader +
sentence length, wordpieces, label distribution, term length, term frequency,
train↔htfl overlap); outputs in `results/data_stats/`, write-up in
`docs/Data_stats.md`. Wind annotated word count is 57,766 (inventory ratio
0.18), not 314k. One §7 item, T2-7 (nested-term count), is **deferred to week 3**
alongside the NOBI ceiling comparison — recorded in `Data_stats.md` §10.

---

## T3 — Evaluation harness · Ahmed · depends on T1 + contract

Written by hand. Not delegated, not generated.

### Two metrics, one decode path

**Unique-list F1** — decoded spans collapsed to a deduplicated lowercased set of
strings and compared against the gold unique list. **This is the headline
metric.** It is TermEval 2020's protocol, and every published number this project
compares against is in that unit.

**Exact-span F1** — decoded spans compared positionally against gold spans,
micro-averaged. A prediction counts only if **both boundaries match exactly**;
`assist device` predicted where `ventricular assist device` is gold scores zero.
Secondary and diagnostic.

Naming matters: "span-level F1" is ambiguous, since it is also used for
token-level F1, which scores each token's label independently and awards partial
credit for partial overlap. That is a different, more forgiving metric.
**Use `exact_span` in the config and "exact-span F1, micro-averaged" in the
writeup.**

**The headline is fixed now, before either number exists.** Choosing a metric
after seeing results is how a project talks itself into a favourable framing.

### Why both

They are not the same measurement at different granularity. They differ in two
ways that matter:

**Weighting.** Exact-span F1 is occurrence-weighted; unique-list F1 is
type-weighted. `heart failure` is 350 units of span F1 and exactly one list
entry. With htfl at 47.7% hapax (`Data_stats.md` §7.3), these will diverge.

**Ceilings.** `max_recall` and `max_precision` (`data_layout.md` §5.3) exist
**only in the list metric**. Scored against gold spans there is no
representational loss at all.

This gives a clean decomposition:

- **exact-span F1** — tagger quality, uncontaminated by the annotation scheme
- **unique-list F1** — tagger quality *plus* what the scheme costs
- **the gap** — partly the ceiling, partly type-vs-occurrence weighting

**Predicted, so the gap is tested rather than rationalised afterwards:** span F1
high with list F1 low means the model gets frequent terms right and misses rare
types — the expected outcome at 47.7% hapax. The two landing close together
would mean performance is uniform across the frequency distribution, which would
be surprising and worth investigating.

### Build

- `encode(tokens, spans, scheme)` — `scheme` is a parameter from the start, even
  though only `bio` is implemented now. Week 3 adds `nobi` as a branch, not a
  rewrite.
- `decode(tokens, labels, scheme)` → spans → surface strings
- `score_list(pred_list_path, gold_list_path)` → P, R, F1
- `score_spans(pred_spans, gold_spans)` → P, R, F1, exact-span, micro-averaged

Both scorers consume the output of the same `decode()`. The expensive part is
shared; the second metric is one extra aggregation.

### Four tests, permanent, in `tests/`

1. **Identity.** Gold term list fed to `score_list` as the prediction. **Must be
   exactly 1.0.** Catches casing, dedup, and normalisation errors.
2. **Span round-trip.** Gold spans → `encode` → `decode` → `score_spans` against
   the same gold spans. **Must be exactly 1.0.** This is a real assertion: there
   is no representational loss in span space, so any deviation is a bug in
   `encode`/`decode`, not a property of the dataset.
3. **List round-trip.** Gold spans → `encode` → `decode` → term list →
   `score_list` against the gold unique list. **Does not return 1.0.** It returns
   `max_recall` and `max_precision`. A measurement, not an assertion.
4. **Hand-computed fixture.** A small toy example — roughly 10 tokens, two or
   three gold spans including one nested and one adjacent pair — with P, R and F1
   worked out by hand for **both** metrics and asserted. This is the second
   oracle: it catches errors the corpus-level tests cannot, because it is the
   only test where the expected answer was derived independently of the code.

Tests 2 and 3 together separate the two failure modes the single round-trip
conflated: **2 fails → conversion bug. 2 passes, 3 below 1.0 → annotation-scheme
ceiling, which is a finding.**

### Definition of done

- Identity test returns exactly 1.0
- Span round-trip returns exactly 1.0
- List round-trip produces `max_recall` and `max_precision`, recorded in `results/`
- Hand-computed fixture passes for both metrics
- `score_list` runs against either gold key by path argument
- Config records which metric is the headline and that `exact_span` means
  exact-boundary match

### Risk

This is where the project fails silently. A wrong choice in detokenisation,
casing, dedup, or corpus portion produces a plausible F1 rather than an error,
and the error is systematic — it survives multi-seed reporting and leaves
ablation orderings intact. The metric also *steers*: it selects hyperparameters
and checkpoints, so fixing it later gives correct numbers for a badly selected
model.

Building both metrics also de-risks T5: whichever unit the comparison papers
report in, the number exists. The open question in `Data_stats.md` §9.4 drops
from blocking to "which column sits next to theirs."

### Done

`src/eval/`: `spans.py` (`encode`, `decode`), `surface.py`
(`spans_to_unique_list`, `generate_unique_list`, `write_term_list`),
`scorers.py` (`score_list`, `score_exact_spans`, `load_gold_list_into_set`,
`generate_flatten_spans`), `run_eval.py` (`compute_ceilings`),
`score_baseline.py`; `configs/eval.yaml`. **30 tests in `tests/`.**

Against the definition of done — all six met:

- Identity test — exactly 1.0, gold written through the contract writer and read
  back through the gold loader, so the two normalisation paths are compared
- Span round-trip — exactly 1.0, all four domains, set sizes pinned
  (corp 4,180 / equi 8,662 / wind 5,053 / htfl 9,636)
- List round-trip — `max_recall` / `max_precision` in `results/ceilings.md` and
  `data_layout.md` §5.5
- Hand-computed fixture — passes for both metrics
- `score_list` runs against either key by path argument
- `configs/eval.yaml` records the headline metric and `exact_span` semantics

**Two tests beyond the spec:**

- **Label round-trip.** `encode(decode(gold)) == gold` as exact sequence
  equality, asserting exactly **3** mismatches (`data_layout.md` §3). Stronger
  than the span round-trip, which is a set comparison and passes under a pair of
  symmetric bugs; equality also names the failing token instead of returning
  0.998.
- **seqeval agreement.** Independent cross-check of `score_exact_spans` on a
  perturbed prediction, agreeing to 6 dp under `mode='strict', scheme=IOB2`. The
  perturbation excludes overlapping spans by construction — BIO cannot represent
  them — so overlap behaviour is untested and no reader should infer otherwise.

**On the fixture.** Adapted from `corp_en_01_seq_terms.tsv`, not verbatim: one
label was changed (`life` `O` → `I`) to include a two-token span. Labels are
valid IOB2 but do not match the corpus file. Expected values were derived by hand
from the metric definitions. Same prediction gives **recall 0.5 by span and 0.667
by type** — the two-metric divergence this task is built on, at a scale that can
be checked by counting.

Every other test compares the code against itself or against a number the code
produced; a bug shared by `encode` and `decode` passes all of them. The fixture
is the only external oracle, which is why it was not delegated.

**Findings recorded in `data_layout.md`:** strict IOB2 with the dangling-`I`
policy (§3), the tokenised/non-tokenised key trap with measured counts (§4), and
the ceilings for all four domains (§5.5).

Tag the repo: `git tag day-3`.

---

## T4 — C-Value baseline · khaled Ahmed · depends on contract only

A no-training statistical reference point, giving a lower bound to improve on.

C-Value scores a candidate phrase using how often it occurs, how many words it
contains, and how often it appears nested inside longer phrases.

**Build:** candidate generation → C-Value scoring → threshold → term list in the
contract format.

**Can start immediately.** Test against a hand-made 10-line gold file; swap in the
real scorer when T3 lands.

**Scored on unique-list F1 only.** C-Value ranks candidate types, not
occurrences, so the exact-span metric does not apply to it.

**Open decision, to be made explicitly and recorded:** which corpus the frequency
counts come from. Using the unannotated portion as reference material is
legitimate — TermEval participants used training domains as reference material,
not only as labelled data — but scoring still happens on annotated text only.
Write down what was counted. Inputs to this decision are in `Data_stats.md` §4.5
and §7.3.

**Definition of done:** produces a valid term list for heart failure; the
frequency-corpus decision is recorded; runs from a config file.

### Scored, provisionally

htfl unique-list F1 **0.2003** (ANN key) / **0.2048** (NES key), via
`src/eval/score_baseline.py`. 2,494 predicted terms against 2,339 gold (ANN).
Input passed contract validation — no tabs, no uppercase, no duplicates.

For orientation only: **no TermEval participant used C-Value**, so there is no
direct published comparison. The nearest comparable system is e-Terminology's TSR
(statistical, frequency threshold ≥2), which scored F1 20.1 incl NE / 21.4 excl
NE on the same test set. So 0.20 is an ordinary place for a no-training
statistical baseline to land. Not comparable digit-for-digit — those numbers are
on ACTER 1.2's non-tokenised flat lists against a different gold (2,361 unique
terms + 224 NEs, against our 2,339 / 2,556).

**Three decisions to record before this number is usable:**

1. **Frequency corpus.** Annotated portion only, or including the unannotated
   texts as reference material? Both legitimate, not comparable to each other.
   Already in the definition of done above.
2. **Tokeniser.** 247 of 2,494 predicted terms (~10%) contain an apostrophe or
   hyphen with no surrounding space. ACTER's tokenised key writes
   `public prosecutor 's office`. If candidate generation used its own tokeniser,
   those terms cannot match regardless of ranking quality, and the failure is
   silent — a false positive and a false negative on the same term. Some will be
   genuine (`30-day`); the split is unknown.
3. **Threshold.** 2,494 predicted against 2,339 gold is close enough to suggest
   the cutoff was set to produce a plausible *count*. If it was tuned against
   htfl in any way the baseline is contaminated — htfl is the held-out test set.
   A threshold from the training domains, or a fixed C-Value score, is fine but
   must be stated.

Relevant to interpretation: TermEval found recall lowest for hapax terms across
every system, and e-Terminology reached **0%** hapax recall because of its
frequency cut-off. At 47.7% hapax in htfl (`Data_stats.md` §7.3), a
frequency-based method is structurally capped well below the ceiling. That is a
property of C-Value, not a defect in the implementation.

---

## T5 — Prior work and comparison table · Moamen Talaat · no dependencies

Fully independent. Nothing here needs code.

**Read and extract numbers from:**

1. Rigouts Terryn et al. (2020), *TermEval 2020* — the shared task overview.
   Pull the full results table: precision, recall, F1 per team, both keys.
2. Tran et al. (2024), *Can cross-domain term extraction benefit from
   cross-lingual transfer and nested term labeling?* (*Machine Learning*) —
   English heart-failure F1 on both keys (they call them ANN and NES), and the
   BIO-vs-NOBI recall delta.
3. Lang et al. (2021) — the origin of the train/validation/test split this
   project uses.

**Specific question to answer first** (`Data_stats.md` §9.4): **how does each
paper compute its F1** — over a deduplicated unique term list, or over spans? The
ACTER authors warn the two can lead to very different conclusions. T3 now builds
both metrics, so this no longer blocks, but every number in the comparison table
must be labelled with the unit it was measured in. Mixing them silently would
make the table wrong.

Also record whether each paper reports on the ANN key, the NES key, or both.

**Output:** `docs/prior_work.md` — a comparison table this project's results will
sit next to, plus one paragraph per system on its approach.

**Point to be explicit about:** ACTER 1.2, the version used in the 2020 shared
task, provided only flat lists of unique terms. Sequential span annotations
arrived in v1.5. The 2020 participants therefore could not have done BIO tagging.
Same test set, same metric, but **more supervision available to us** — this must
be stated in the writeup, not glossed over.

**Secondary question:** does Lang et al. or Tran et al. discuss `equi` being an
unrepresentative validation domain for this test set (`Data_stats.md` §8.5)? If
nobody has, it is worth a paragraph.

**Definition of done:** comparison table committed with all numbers sourced to a
specific table in a specific paper, each labelled with its metric unit and key.

### Starting point for item 1

TermEval 2020 English track, heart-failure test set, percentages
(Rigouts Terryn et al. 2020, Table 4). **Verify against the paper before use —
do not take these on trust from this file:**

| Rank | Team | Method | P | R | F1 incl NE | F1 excl NE |
|---|---|---|--:|--:|--:|--:|
| 1 | TALN-LS2N | BERT binary classification | 34.8 | 70.9 | 46.7 | 45.0 |
| 2 | RACAI | TextRank + TFIDF + embeddings | 42.4 | 40.3 | 41.3 | 39.3 |
| 3 | NYU | Termolator, chunking + TFIDF | 43.5 | 23.6 | 30.6 | 31.5 |
| 4 | e-Terminology | TSR filtering, statistical | 34.4 | 14.2 | 20.1 | 21.4 |
| 5 | NLPLab UQAM | BiLSTM + GloVe | 21.4 | 15.6 | 18.1 | 17.8 |

Two things this settles:

- **The unit.** These are unique-list scores. ACTER 1.2 shipped annotations only
  as flat lists of unique terms, so no participant could have reported a
  span-level number. Our headline metric is in the same unit. §9.4's question is
  answered for this paper; Tran et al. (2024) still needs checking.
- **The supervision asymmetry, confirmed from the source.** Sequential span
  annotations arrived in v1.5. Same test set, same metric, more supervision
  available to us — state it, do not gloss it.

Also useful for the writeup: English hapax terms were 43% of the gold with NEs
included, and recall was lowest for hapax terms across every system. Consistent
with our 47.7% (`Data_stats.md` §7.3) and with the span-vs-list divergence T3
predicts.

Our own ceilings (`data_layout.md` §5.5) belong in this table as a row — a
reader comparing an F1 against published work needs to know the cap.

---

## Week 1 status

| Task | Status |
|---|---|
| T1 — Repo, data, layout | done |
| T2 — Data statistics | done; nested-term count deferred to week 3 |
| T3 — Evaluation harness | **done**, 30 tests |
| T4 — C-Value baseline | **done**; F1 ≈ 0.20; 3 decisions unrecorded |
| T5 — Prior work | **done**; TermEval table sourced |

**Nothing blocks week 2.** The T4 decisions are khaled's to record and do not
gate model training; the harness is what week 2 depends on and it is closed.

---

## Not this week

NOBI implementation · multilingual or cross-lingual training · any model
training · architecture changes. Entries for these live in `DEPTH_DEBT.md`.