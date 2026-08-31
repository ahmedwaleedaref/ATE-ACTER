# Tasks — Week 1

Data facts live in `docs/data_layout.md`. This file is planning only.

Tasks, not days. A task is done when its **definition of done** is met, not when
a day ends.

| Task | Owner | Depends on | Status |
|---|---|---|---|
| T1 — Repo, data, layout analysis | Ahmed | — | **done** |
| T2 — Data statistics | Ahmed | T1 | **done** |
| T3 — Evaluation harness | Ahmed | T1, interface contract | |
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
| Train / test | corruption, dressage, wind energy / heart failure |

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

**Build:**

- `encode(tokens, spans, scheme)` — scheme is a parameter from the start, even
  though only `bio` is implemented now. Week 3 adds `nobi` as a branch, not a
  rewrite.
- `decode(tokens, labels, scheme)` → spans → surface strings
- normalise → lowercase → deduplicate → term list
- `score(pred_list_path, gold_list_path)` → precision, recall, F1

**Two tests, permanent, in `tests/`:**

1. **Identity** — gold term list fed to the scorer as the prediction. Must be
   exactly 1.0. Catches casing, dedup, normalisation errors.
2. **Round-trip** — gold spans encoded to BIO, run through the full decode path,
   scored. **Does not return 1.0 on this dataset** — it returns `max_recall` and
   `max_precision` (`data_layout.md` §5.3). It is a measurement, not an assertion.

**Definition of done:** identity test returns exactly 1.0; round-trip produces
both ceiling numbers, recorded in `results/`; scorer runs against either gold key
by path argument.

**Risk:** this is where the project fails silently. A wrong choice in
detokenisation, casing, dedup, or corpus portion produces a plausible F1 rather
than an error, and the error is systematic — it survives multi-seed reporting and
leaves ablation orderings intact. The metric also *steers*: it selects
hyperparameters and checkpoints, so fixing it later gives correct numbers for a
badly selected model.

---

## T4 — C-Value baseline · khaled Ahmed · depends on contract only

A no-training statistical reference point, giving a lower bound to improve on.

C-Value scores a candidate phrase using how often it occurs, how many words it
contains, and how often it appears nested inside longer phrases.

**Build:** candidate generation → C-Value scoring → threshold → term list in the
contract format.

**Can start immediately.** Test against a hand-made 10-line gold file; swap in the
real scorer when T3 lands.

**Open decision, to be made explicitly and recorded:** which corpus the frequency
counts come from. Using the unannotated portion as reference material is
legitimate — TermEval participants used training domains as reference material,
not only as labelled data — but scoring still happens on annotated text only.
Write down what was counted.

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

**Output:** `docs/prior_work.md` — a comparison table this project's results will
sit next to, plus one paragraph per system on its approach.

**Point to be explicit about:** ACTER 1.2, the version used in the 2020 shared
task, provided only flat lists of unique terms. Sequential span annotations
arrived in v1.5. The 2020 participants therefore could not have done BIO tagging.
Same test set, same metric, but **more supervision available to us** — this must
be stated in the writeup, not glossed over.

**Definition of done:** comparison table committed with all numbers sourced to a
specific table in a specific paper.

---

## Not this week

NOBI implementation · multilingual or cross-lingual training · any model
training · architecture changes. Entries for these live in `DEPTH_DEBT.md`.