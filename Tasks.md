# Tasks — Week 1

Data facts live in `docs/data_layout.md`. This file is planning only.

Tasks, not days. A task is done when its **definition of done** is met, not when
a day ends.

| Task | Owner | Depends on | Status |
|---|---|---|---|
| T1 — Repo, data, layout analysis | Ahmed | — | **done** |
| T2 — Data statistics | Ahmed | T1 | **done** |
| T3 — Evaluation harness | Ahmed | T1, interface contract | next |
| T4 — C-Value baseline | khaled Ahmed | interface contract | can start now |
| T5 — Prior work and comparison table | Moamen Talaat | — | can start now |

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

---

## Not this week

NOBI implementation · multilingual or cross-lingual training · any model
training · architecture changes. Entries for these live in `DEPTH_DEBT.md`.