# T3 — Evaluation harness

Written by hand. Closed with 30 passing tests, ceilings measured for all four
domains, and one external oracle.

Data facts live in `data_layout.md`. Task planning lives in `TASKS.md`. This file
is the record of what the harness is, why it is shaped this way, and what was
learned building it.

---

## 1. What was built

| Function | File | Signature |
|---|---|---|
| `decode` | `src/eval/spans.py` | `(tokens, labels, scheme) -> list[(start, end)]` |
| `encode` | `src/eval/spans.py` | `(tokens, spans, scheme) -> list[str]` |
| `spans_to_unique_list` | `src/eval/surface.py` | `(sentences) -> (set, n_spans, n_unique)` |
| `generate_unique_list` | `src/eval/surface.py` | `(domain) -> (set, n_spans, n_unique)` |
| `write_term_list` | `src/eval/surface.py` | `(terms, path)` |
| `load_gold_list_into_set` | `src/eval/scorers.py` | `(path) -> set[str]` |
| `score_list` | `src/eval/scorers.py` | `(pred, gold) -> (P, R, F1)` |
| `score_exact_spans` | `src/eval/scorers.py` | `(pred, gold) -> (P, R, F1)` |
| `generate_flatten_spans` | `src/eval/scorers.py` | `(domain) -> set[Span]` |
| `compute_ceilings` | `src/eval/run_eval.py` | `(domain, config) -> dict` |

Plus `src/eval/score_baseline.py`, `configs/eval.yaml`, and 30 tests in `tests/`.

`Span = tuple[str, int, int, int]` — `(file_id, sent_idx, start, end)`.

**Module boundaries.** `spans.py` knows nothing about strings or F1 — pure
label↔span conversion. `surface.py` owns the string path (join, lowercase,
dedup, write). `scorers.py` takes spans and strings, never corpus paths.
`run_eval.py` is the only file that knows about paths, domains, and config. The
split is not tidiness: if the string path could affect the span metric, the
two-metric decomposition collapses.

---

## 2. Design decisions settled during the work

**One sentence per `decode` call.** `tokens` and `labels` are flat parallel
lists for a single sentence. The caller loops and attaches `file_id` /
`sent_idx`. The sentence boundary becomes structural — a function that never
sees two sentences cannot emit a span crossing them. Passing list-of-lists would
put that guarantee back in a loop where it can be got wrong.

**`end` is exclusive.** `tokens[start:end]` is the term. Stated in the docstring
because a consistent inclusive/exclusive error survives every round-trip test.

**Spans are keyed on four fields.** Micro-averaged exact-span F1 over a
flattened token stream invites two bugs — per-sentence offsets colliding across
sentences, and spans crossing a blank-line boundary. The key makes both
inexpressible.

**Lowercase before dedup.** `Bent` (a name) and `bent` (a verb) are one entry in
the gold key. The reverse order leaves both in the set, and one is a guaranteed
false positive on every sentence-initial term in the corpus.

**`score_list` asserts, never normalises.** The contract says term lists arrive
lowercased and deduplicated, and §4 confirms zero uppercase entries in all eight
gold files. If the scorer lowercased defensively, a casing bug in `decode` would
be invisible in list F1 while still corrupting exact-span F1 — and the divergence
would read as the type-vs-occurrence finding the T3 spec already predicts.

**The two scorers stay duplicated.** `score_list` and `score_exact_spans` are
line-for-line identical with different types. They are deliberately not factored
into a shared helper: the moment they share an implementation, a change made for
one metric silently applies to the other, and the decomposition the project rests
on stops being independent.

**Whitespace stripping lives in the loader, not the writer.** Briefly
`write_term_list` emitted a padding tab (`"term\t\n"`) so terms would round-trip
past a loader missing `.strip()`. Two bugs cancelling — and the output was not
contract format, so T4's C-Value list and the week-2 model output would not have
loaded. The strip belongs in the loader, where it is inert on gold TSVs and
load-bearing on contract files.

**`scheme` is a parameter from the start.** Only `"bio"` is implemented;
`assert scheme == "bio"`. Week 3's NOBI is a branch inside `encode` / `decode`,
not a rewrite of every call site.

---

## 3. Findings

### 3.1 ACTER is strict IOB2

Measured over all four English domains, `without_named_entities`, 222,281
tokens: **zero** `I` labels follow an `O`. The IOB1 convention — where a term not
preceded by another term begins with `I` — is ruled out.

Three sentence-initial `I` labels exist, all terms split across a spurious
sentence boundary:

| file | fragment | full term |
|---|---|---|
| `wind_en_01` | `theory` | `BEM theory` |
| `wind_en_01` | `coefficient` | `rotor power coefficient` |
| `htfl_en_171` | `Arg83Gly` | `p . Arg83Gly` |

Segmentation artifacts, not evidence of IOB1.

**Policy: a dangling `I` opens no span and is dropped.** This matches seqeval
under `mode='strict', scheme=IOB2`. Gold barely exercises the branch; model
output is a free 3-way choice per token, so the policy is load-bearing there.

Consequence: `encode(decode(gold)) == gold` differs on exactly these three
tokens. A pinned assertion, not a bug.

### 3.2 The two gold key variants are different files

Each domain ships `<domain>_en_terms[_nes].tsv` and
`<domain>_en_tokenised_terms[_nes].tsv`. corp terms+NE: **1,173 non-tokenised vs
1,172 tokenised**, 12 entries unique to one and 11 to the other.

| non-tokenised | tokenised |
|---|---|
| `public prosecutor's office` | `public prosecutor 's office` |
| `non- governmental organisations` | `non - governmental organisations` |
| `expenditure budget line(s)` | `expenditure budget line ( s )` |

`decode` joins dataset tokens with spaces, so it can only produce the tokenised
form. Scoring against the other costs a false positive and a false negative on
the same term, raises no error, and lands on multi-word terms. **Assert the
expected entry count on load** — an identity test passes at 1.0 against either
file, so the count is the only tripwire.

### 3.3 The ceilings

Gold BIO decoded to a unique term list, scored against the gold unique key. No
model. Reproduced digit-for-digit by an independent implementation.

| domain | spans | types | ANN max_P | ANN max_R | NES max_P | NES max_R |
|---|--:|--:|--:|--:|--:|--:|
| corp | 4,180 | 904 | 0.9668 | 0.9438 | 0.9668 | 0.7457 |
| equi | 8,662 | 1,204 | 0.9294 | 0.9764 | 0.9294 | 0.7168 |
| wind | 5,053 | 1,072 | 0.9468 | 0.9295 | 0.9468 | 0.6638 |
| htfl | 9,636 | 2,452 | **0.8887** | **0.9316** | 0.8911 | 0.8549 |

**htfl ANN `max_recall` 0.9316 caps every number this project reports.**

**The NES recall column is not a scheme ceiling.** Labels are
`without_named_entities`, so the named entities in the NES key were never
markable. Roughly 20 points of the ANN→NES drop is that, not BIO. Reported alone,
wind's 0.6638 would read as "BIO costs a third of recall" — false. A ceiling
without its key is unreadable.

**htfl has the lowest `max_precision` and is the only domain where it differs
between keys.** Precision's denominator is the decoded list, which does not
change with the key, so a few decoded htfl terms are absent from ANN but present
in NES. Discontinuous-term fragments (§5.2) concentrated in clinical text.

**equi's shape is unlike the other three** — 0.9764 recall with 0.9294
precision. Almost everything in its key is reachable and almost nothing decoded
is spurious. Further evidence for the §8.5 concern about equi as validation
domain.

**Span/type ratios are 3.9–4.7**, not the ~2 predicted from the hapax rate
during planning. Hapax is over gold key *types*; span counts are *occurrences*
over annotated text. Different denominators — do not reason from one to the
other.

---

## 4. The tests

| Test | Asserts | Catches |
|---|---|---|
| Identity | exactly 1.0 | writer/loader normalisation mismatch |
| Label round-trip | exactly **3** mismatches | asymmetric encode/decode bugs, by token |
| Span round-trip | exactly 1.0, set sizes pinned | dropped or merged spans |
| List round-trip | a measurement, not 1.0 | produces the ceilings |
| seqeval agreement | 6 dp, `mode='strict', scheme=IOB2` | scheme and convention drift |
| Hand fixture | six numbers, both metrics | symmetric bugs — the external oracle |
| Empty / zero-overlap / uppercase | guards | division by zero, contract violations |
| IOB2 structural invariants | span count = `B` count; token sum | localises a decode failure fast |

Pinned set sizes: corp 4,180 / equi 8,662 / wind 5,053 / htfl 9,636.

### Why the label round-trip is stronger than the span round-trip

The span round-trip is a set comparison and passes under a pair of symmetric
bugs. `encode(decode(gold)) == gold` is exact sequence equality over 222,281
tokens against data the code did not produce, and it names the failing token
instead of returning 0.998.

### The fixture is the only external oracle

Every other test compares the code against itself (round-trips) or against a
number the code produced. A bug shared by `encode` and `decode` — both treating
`end` as inclusive, say — passes all of them.

Adapted from `corp_en_01_seq_terms.tsv`, **not verbatim**: one label was changed
(`life` `O` → `I`) to include a two-token span. The labels are valid IOB2 but do
not match the corpus file. Expected values were derived by hand from the metric
definitions.

Four sentences, a deliberately wrong prediction, and the result that matters:

```
exact-span:   P 0.4   R 0.5     F1 0.444
unique-list:  P 0.4   R 0.667   F1 0.5
```

Same prediction, **recall 0.5 by span and 0.667 by type** — because `corruption`
occurs twice in gold and collapses to one list entry. That is the two-metric
divergence the project rests on, at a scale that can be verified by counting.

### seqeval blind spot

The perturbation generator excludes overlapping spans by construction, since BIO
cannot represent them. Adjacent terms with no `O` between them are common in this
corpus, so overlap behaviour is **untested**. No reader should infer it is
handled.

---

## 5. Errors caught while building

Worth keeping — the failure modes are the point of the task.

**In the implementation:**

- No separator in the term join — `heart failure` became `heartfailure`. Every
  multi-word term wrong, no error raised, `max_recall` would have landed near the
  single-token proportion and looked like a real finding.
- `for label in labels: label = "O"` — rebinding a loop copy, not the list
  element. `encode` returned `""` everywhere `O` belonged. Would have failed the
  round trip in a way that pointed at `decode`.
- Sentence counts used as denominators in the first exact-span scorer.
  Precision inflated ~2×, able to exceed 1.0.
- Predictions compared against themselves — gold never entered the precision
  loop.
- `x.islower()` as the lowercase check. Returns `False` for `2020`, `%`, `hr` —
  all valid entries. `x == x.lower()` is the predicate.
- The non-tokenised gold key loaded on the first run.
- `score_list(gold, gold)` called an identity test. It is a tautology and returns
  1.0 for any loader.

**In my advice:**

- Predicted a span/type ratio near 2 from the hapax rate. Actual 3.9–4.7.
  Different denominators.
- Claimed the identity check would catch `file_id` / `sent_idx` collisions in the
  flattening. It cannot — both sides come from the same function, so collisions
  shrink both sets identically.
- Transcribed the fixture's `life` label as `I` when the corpus has `O`, then
  "corrected" a right answer (TP = 1) to a wrong one (TP = 2). Caught by Claude
  Code checking provenance against disk.

---

## 6. Configuration

`configs/eval.yaml`, read by `run_eval.py` and written into every results
header:

```yaml
headline_metric: unique_list_f1
secondary_metric: exact_span_f1
scheme: bio
labels: without_named_entities
keys: [ann, nes]
dangling_i_policy: drop
seqeval_mode: strict
seqeval_scheme: IOB2
```

The headline was fixed before either number existed. Choosing a metric after
seeing results is how a project talks itself into a favourable framing.

---

## 7. Outputs

- `results/ceilings.md` — 8 rows, header records key, label directory, scheme,
  dangling-`I` policy
- `results/baseline_cvalue.md` — C-Value scored through the harness, provisional

---

## 8. What is not done

- **NOBI** — week 3, a branch inside `encode` / `decode` via the `scheme`
  parameter
- **Overlapping-span behaviour** — untested, see §4
- **Nested-term count** — deferred T2 item; it decomposes htfl's `max_recall` gap
  into nested-term and discontinuous-fragment components