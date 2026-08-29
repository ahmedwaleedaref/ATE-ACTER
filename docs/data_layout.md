# ACTER v1.5 — Data Layout

Working notes on the dataset's structure, formats, and known constraints.
Everything marked **[CONFIRMED]** was verified by inspecting the files directly.
Everything marked **[OPEN]** still needs checking.

Dataset: ACTER v1.5, `github.com/AylaRT/ACTER`, pinned tag `v1.5`
License: CC BY-NC-SA 4.0 (non-commercial, share-alike)

---

## 1. Directory layout

**[CONFIRMED]**

```
<language>/<domain>/annotated/
├── texts/                          # raw text
├── texts_tokenised/                # whitespace-tokenised text
└── annotations/
    ├── sequential_annotations/
    │   ├── iob_annotations/
    │   │   ├── with_named_entities/
    │   │   └── without_named_entities/
    │   └── io_annotations/
    │       ├── with_named_entities/
    │       └── without_named_entities/
    └── unique_annotation_lists/
```

Languages: `en`, `fr`, `nl`
Domains: `corp` (corruption), `equi` (dressage), `htfl` (heart failure), `wind` (wind energy)

File IDs follow `<domain>_<lang>_<nn>`, e.g. `corp_en_01`, matching entries in `sources.txt`.

**Project scope:** English only. Train on `corp`, `equi`, `wind`. Test on `htfl`.

### Annotated vs. unannotated files

**[CONFIRMED]** Not every text file in the repo is annotated. The corpora contain
unannotated texts alongside the annotated ones, and files without annotations
simply have no corresponding file under `annotations/`.

**This is self-enforcing for the pipeline.** If the loader iterates over
*annotation* files and pairs each with its text, an unannotated text has nothing
to pair with and can never enter training or evaluation. Make the invariant
explicit anyway:

```python
assert seq_path.exists(), f"no annotation for {text_id}"
```

Cheap, and it turns a silent wrong-data bug into a loud crash.

**Where it does bite: corpus statistics.** The unannotated text is real text in
the repo, and computing term frequency or hapax proportions over everything found
on disk gives wrong numbers. All statistics must be over the annotated portion
only.

English word counts, whole corpus vs. annotated part (figures from the TermEval
2020 paper, ACTER 1.2 — v1.5 may differ slightly):

| Domain | Whole corpus | Annotated part |
|---|---|---|
| Corruption | 176,314 | 45,234 |
| Dressage | 102,654 | 51,470 |
| Wind energy | 314,618 | 51,911 |
| Heart failure | 45,788 | 45,788 |

Wind energy is the trap: ~84% of that corpus is unannotated. **If day-2 counts
come out near 314k, the wrong files were loaded.**

Two details:

- **Heart failure is annotated in full**, so the test domain has no unannotated
  files. The evaluation side is clean regardless.
- For corruption, annotations exist only in the *parallel* corpus; the
  comparable corruption corpus is entirely unannotated.

**Open decision, day 4:** C-Value is frequency-based, and better frequency
estimates come from more text. Using the unannotated portion as *reference
material* for the statistical baseline is legitimate — shared task participants
used training domains as reference material, not only as labelled data. Scoring
still happens on annotated text only. Decide this explicitly and record which
corpus the counts came from; do not let it happen by accident.

---

## 2. Text formats

**[CONFIRMED]**

- `texts/` — raw source text.
- `texts_tokenised/` — same content, whitespace-tokenised.
- Tokenisation is by whitespace; punctuation and numbers are separate tokens.
  One token = one word, number, or punctuation mark.
- **One sentence per line.**
- Document length: roughly 30–400 lines, averaging ~150.

**[OPEN]** Tokens per *sentence* (not lines per file) — this is the number that
determines `max_length`. To be measured.

---

## 3. Sequential annotations

**[CONFIRMED]**

Format: one token per line, TAB, then the label. TSV.

```
Corruption<TAB>B
?<TAB>O
<blank line>
Not<TAB>O
in<TAB>O
our<TAB>O
company<TAB>B
…<TAB>O
```

### Blank lines are sentence boundaries

`corp_en_01_seq_terms.tsv` contains **227 blank lines**, appearing exactly at
sentence ends. This is the CoNLL convention.

**The parser must treat a blank line as a sentence separator, not skip it as
whitespace.** Skipping merges sentences and lets predicted spans cross sentence
boundaries.

**[OPEN]** Verify that `wc -l` on the seq file minus blank lines equals
`wc -w` on the corresponding tokenised text. If they match, the seq file is the
same token stream flattened, and sentence structure can be recovered either from
the blank lines or by walking the tokenised text line by line.

### IO vs IOB — these are two different formats

| Format | Labels | Can it separate adjacent terms? |
|---|---|---|
| IO | `I`, `O` | **No** |
| IOB | `I`, `O`, `B` | Yes — `B` marks a term's first token |

`O` means **outside only**. There is no end-of-term marker; a term ends when the
next token is not `I`.

**Decision: use IOB.**

Reason: with IO, "diabetic patient" is `I I` whether it is one two-word term or
two separate one-word terms. That distinction is real in this dataset and IO
discards it.

### Encoding

Non-ASCII characters survive into the token stream (e.g. `…` U+2026). The README
documents prior cleanup — `İ` replaced with `I` to avoid lowercasing problems
(mainly `wind_en_01`), and certain characters removed because some transformers
handle them poorly.

---

## 4. Unique annotation lists

**[CONFIRMED]**

TSV — tab-separated, two columns: the term, and its category.

Four categories: Specific Term, Common Term, Out-of-Domain Term, Named Entity.
The model predicts term / not-term; it does not predict the category.

Counts for `corp_en`:

| File | Entries |
|---|---|
| `corp_en_tokenised_terms.tsv` | 926 |
| `corp_en_tokenised_terms_nes.tsv` | 1172 |

Difference: **246 named entities, ~21% of the larger key.**

This is the evaluation answer key.

**[CONFIRMED — decided]** TermEval 2020 did not pick one. All three term labels
were merged into a single binary notion of "term", and two gold standards were
maintained: terms only, and terms plus Named Entities. Every participating
system was scored against **both**. The authors state explicitly that labelling
Named Entities does not mean they are considered terms.

**Protocol adopted, matching the shared task:**

- **Train** on the `without_named_entities` labels.
- **Produce one output list.**
- **Score that same list twice** — against the terms-only key and against the
  terms-plus-NE key. Report both.

Observed effect in the shared task: the two scores differ by roughly one
percentage point on average, and not consistently in the same direction. This is
a low-stakes decision, not a blocking one.

### No lemmatisation

Inflected forms are kept as they appear — *device* and *devices* are separate
entries. **Do not lemmatise predictions.** Doing so would look like a modelling
failure but is a preprocessing bug.

### Casing

"Normalised but with original casing" refers to *character* normalisation, not
case-folding. The annotated text preserves casing, so capitalisation is available
to the model as a signal.

**[CONFIRMED]** The unique gold lists **are lowercased**. Only identical
lowercased forms are merged into one entry, and no lemmatisation is applied.

The dressage corpus illustrates both rules at once: *bent* (past tense of "to
bend") and *Bent* (a person's name) collapse into the single entry *bent*, while
other inflected forms of the verb get their own entries if they occur.

**Consequence:** the pipeline must lowercase at the deduplication step, or every
sentence-initial term becomes a miss.

---

## 5. Two hard ceilings on achievable score

This is the most important section. Both ceilings come from the annotation
scheme itself, exist before any model is trained, and bound every number the
project can report.

### 5.1 Nested terms → recall ceiling

BIO can only represent the **longest** span. If *myocyte hypertrophy* is
annotated as a term, a nested *myocyte* is invisible in the tag sequence.

If a term appears in the unique list but only ever occurs nested inside a longer
annotated term, **no BIO tagger can ever emit it.** Not a model weakness — a
representational limit.

### 5.2 Discontinuous terms → precision ceiling

From the ACTER README: annotations of *parts* of terms also receive a positive
(`I` or `B`) label, but those fragments are excluded from the unique lists.

Example: in "left and right ventricular assist devices", the term
*left ventricular assist devices* is interrupted. Sequential annotation can only
mark contiguous spans, so *left* receives a positive label — while *left* is
deliberately **not** an entry in the unique annotation list.

So decoding gold BIO tags produces strings that are correct at the token level
but count as **false positives** against the unique list.

### 5.3 Measuring both

Decode the **gold** BIO tags to a term list and compare against the gold unique
list:

```
max_recall    = |decode(gold_BIO) ∩ gold_unique| / |gold_unique|
max_precision = |decode(gold_BIO) ∩ gold_unique| / |decode(gold_BIO)|
```

Computed from the gold data alone — no model involved, but it does require the
harness's `decode()`, so this is measured on day 3 rather than during data
inspection.

**Consequence for the evaluation harness:** the round-trip test does **not**
return 1.0 on this dataset, and must not assert it. It is a measurement, not an
assertion. The identity test (gold unique list → scorer → F1) must still return
exactly 1.0.

Both ceilings go in the results table. Reported F1 is meaningless without them.

### 5.4 NOBI — a published attempt at the recall ceiling

**[CONFIRMED]** The recall ceiling in 5.1 is a known problem with published work
on it. Tran et al. (2024), *Can cross-domain term extraction benefit from
cross-lingual transfer and nested term labeling?* (*Machine Learning*), introduce
**NOBI**: an annotation scheme adding an extra encoding for nested single-word
terms. With NOBI plus an XLM-R classifier they report the best results on ACTER.

The ACTER authors describe the same limitation directly: *congestive heart
failure* contains the nested *heart failure*, but IOB captures only the longest
span, giving `B I I`. They also warn that scores computed from a candidate-term
list versus from sequential labels can lead to very different conclusions.

**Consequences for this project:**

- The comparison target is no longer TermEval 2020 alone. Tran et al. evaluate on
  the same heart-failure test set and the same two keys (they call them ANN =
  excluding NEs, NES = including NEs). **Read this paper before day 3.**
- Two findings of theirs are directly relevant:
  - NOBI was expected to help short one-word nested terms, but the NOBI-trained
    classifiers performed better than BIO ones on **multi-word** terms. The paper
    reports this without fully explaining it.
  - Under both schemes, the best models predicted terms up to four tokens
    (English, Dutch) or three (French). **The long-term ceiling persists under
    both.**
- Nested term extraction is where the subfield is moving: the Binder model,
  built for nested named entities, won all three tracks of the RuTermEval
  competition on nested terms.

**Experiment this suggests (week 3):**

1. Measure `max_recall` under BIO and under NOBI — ceilings only, no model.
2. Train under both schemes, measure actual F1.
3. Report how much of the ceiling gain converts into score gain.

A gap between the two — ceiling up 8 points, F1 up 2 — is itself a reportable
result, and it is the kind of decomposition this project is designed to produce.

---

## 6. Decisions locked

| Decision | Value |
|---|---|
| Annotation format | IOB, not IO |
| Document | one source text file (`corp_en_01`, etc.) |
| Sentence boundary | blank line in the sequential annotation file |
| Training unit | one sentence per example (baseline) |
| Language | English |
| Train domains | corruption, dressage, wind energy |
| Test domain | heart failure |
| Lemmatisation | none, at any stage |

Note on the training unit: sentence-level input means the model sees only the
current sentence. The first-occurrence locality hypothesis is a claim that
context from elsewhere in the document helps — so sentence-level is the
baseline, and going beyond it is the contribution.

---

## 7. Open data questions

Task planning lives in `TASKS.md`. Listed here are the questions about the data
itself that are still unanswered, tagged with the task that resolves each.

- **[T2]** Tokens per sentence distribution → sets `max_length`
- **[T2]** Line-count vs. token-count consistency between the sequential
  annotation files and `texts_tokenised`
- **[T2]** Term length distribution in the gold lists
- **[T2]** Term frequency distribution, especially the hapax proportion
- **[T2]** Term-set overlap between the training domains and heart failure
- **[T2]** Proportion of tokens carrying a positive label (class imbalance)
- **[T2]** Nested-term count: gold terms that are substrings of other gold terms
  (also the input to the NOBI ceiling comparison in 5.4)
- **[T3]** `max_recall` and `max_precision` ceilings (section 5.3) — these need
  the harness's `decode()`, and running them through it validates the conversion
  code at the same time
- **[T5]** Tran et al. (2024) English heart-failure F1 on both keys, and their
  BIO-vs-NOBI recall delta

All T2 statistics are computed over the **annotated portion only** — cross-check
against the word counts in section 1.

### Optional

- `diff` the `with_named_entities` and `without_named_entities` sequential
  folders. Low priority: training uses the without-NE labels, and the with-NE
  data is needed only as a unique list, not as tags.