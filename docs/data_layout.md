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
<language>/<domain>/
├── unannotated_texts/              # raw, NOT under annotated/  [T2]
└── annotated/
    ├── texts/                      # raw text
    ├── texts_tokenised/            # whitespace-tokenised text
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

**[CONFIRMED — T2]** `annotated/texts_tokenised/` contains *only* the annotated
files. The unannotated remainder lives in a sibling `unannotated_texts/`,
outside `annotated/` entirely, and is raw (not tokenised). htfl has no such
directory.

Languages: `en`, `fr`, `nl`
Domains: `corp` (corruption), `equi` (dressage), `htfl` (heart failure), `wind` (wind energy)

File IDs follow `<domain>_<lang>_<nn>`, e.g. `corp_en_01`, matching entries in `sources.txt`.

**Project scope:** English only. Train on `corp` + `wind`, validate on `equi`,
test on `htfl` — the standard ACTER cross-domain split (§6).

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

English word counts, whole corpus vs. annotated part. **These are `wc -w` over
the raw, non-tokenised `texts/` files** (figures from the TermEval 2020 paper,
ACTER 1.2):

| Domain | Whole corpus | Annotated part | v1.5 annotated, tokenised stream [T2] |
|---|---|---|---|
| Corruption | 176,314 | 45,234 | 50,845 |
| Dressage | 102,654 | 51,470 | 58,203 |
| Wind energy | 314,618 | 51,911 | 57,766 |
| Heart failure | 45,788 | 45,788 | 55,467 |

**[CONFIRMED — T2]** Raw `wc -w` on v1.5 reproduces the 1.2 figures to under
0.3%, so the annotated file set did not change between versions. T5 can place
this project's numbers next to the 2020 shared task without a corpus caveat.

The fourth column is 10–18% higher because the sequential annotation stream
splits punctuation into separate tokens. **The two units are not comparable** —
do not use the 1.2 figures as a tolerance check against a tokenised count.

Wind energy is the trap: ~84% of that corpus is unannotated. The check that
catches it is the **inventory ratio**: paired-file tokens over whole-domain
corpus tokens, both whitespace-split, so the unit cancels. Measured [T2]:
corp 0.2795, equi 0.5321, **wind 0.1806**, htfl 1.0000. A wind ratio near 1.0
means the unannotated corpus was loaded.

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

**[CONFIRMED — T2]** Tokens per sentence, whitespace, per domain:

| domain | n_sent | p50 | p95 | p99 | max | ≤2 tokens |
|---|--:|--:|--:|--:|--:|--:|
| corp | 2,002 | 22 | 61 | 98 | 281 | 11.4% |
| equi | 3,090 | 18 | 39 | 52 | 95 | 8.3% |
| wind | 6,638 | 1 | 36 | 62 | 258 | **61.0%** |
| htfl | 2,432 | 20 | 54 | 78 | 171 | 13.9% |

**This does not determine `max_length` on its own** — that budget is counted in
subword pieces, not whitespace tokens. See section 8.

**wind is bimodal.** 4,048 of its 6,638 "sentences" are ≤2 tokens, and 3,499 of
those are a single period — a segmentation artifact, not text. They hold 0.9%
of wind's term occurrences. **Decision: filter wind sentences of ≤2 tokens at
training time.** Under the adopted split wind is roughly half the training data,
so this filter is load-bearing rather than cosmetic — see `data_stats.md` §8.5.

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

**[CONFIRMED — T2]** Verified per file, string for string and in order:
**241 of 241 files** have a sequential-annotation token stream byte-identical to
their `texts_tokenised/` counterpart, with sentence blocks matching line
structure. Sentence boundaries can therefore be recovered either way.

This is also what guarantees decoded spans match the gold lists: both are joins
over the same token stream.

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

**[CONFIRMED — T2]** Every domain ships **both** a tokenised and a
non-tokenised variant of each key. **Use the tokenised variant only.** The
non-tokenised list splits differently on hyphens and internal punctuation, so
it cannot be matched against decoded spans. Pin the exact filename in the
interface contract; do not describe it.

Entry counts, all four domains [T2]:

| Domain | terms only | terms + NE | NEs |
|---|--:|--:|--:|
| `corp_en` | 926 | 1,172 | 246 (21%) |
| `equi_en` | 1,146 | 1,561 | 415 (27%) |
| `wind_en` | 1,092 | 1,529 | 437 (29%) |
| `htfl_en` | **2,339** | 2,556 | 217 (8%) |

This is the evaluation answer key. **htfl's key is roughly twice the size of
any training domain's** — the test set has more to find, and each term is worth
0.043% of recall.

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

**[CONFIRMED — T2]** Verified across all eight English files: **0 entries
contain an uppercase character, 0 duplicates after lowercasing.** The
lowercase-at-dedup assumption is safe.

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

### 5.2b Annotation is not exhaustive — measured [T2]

Related to both ceilings, and measured directly. For each gold term, compare
(a) decoded gold-BIO span occurrences against (b) occurrences of the same token
sequence anywhere in the annotated stream, regardless of label:

| domain | (a) tagged | (b) surface | (b)/(a) |
|---|--:|--:|--:|
| corp | 4,116 | 6,632 | 1.61 |
| equi | 8,546 | 12,606 | 1.48 |
| wind | 4,982 | 9,304 | **1.87** |
| htfl | 9,243 | 13,890 | 1.50 |

A third to a half of the surface occurrences of gold terms carry no positive
label. In wind, `wind` is tagged 57 times and appears 642; `rotor` 99 against
284. In htfl, `heart failure` is tagged 350 times and appears 530.

Two mechanisms, which these numbers do not separate: **nesting** (section 5.1 —
BIO marks only the longest span, so `wind` inside `wind turbine` is invisible)
and **genuinely unannotated occurrences**. The distinction matters — nesting is
a representational limit to report, non-exhaustive annotation is training
noise.

**Consequence for training:** the same string carries different labels in
different sentences, distinguished only by whether it sits inside a longer
term. Training loss will not approach zero, and that is not a bug.

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

**Consequence for the evaluation harness:** the *list* round-trip test does
**not** return 1.0 on this dataset, and must not assert it. It is a measurement,
not an assertion. The identity test (gold unique list → scorer → F1) must still
return exactly 1.0.

**The ceilings exist only in the list metric.** T3 computes two: unique-list F1
(the headline, TermEval's protocol) and exact-span F1 (decoded spans scored
positionally against gold spans, micro-averaged, exact boundaries required).
Scored in span space there is no representational loss, so the *span* round-trip
— gold spans → encode → decode → score against the same gold spans — **must be
exactly 1.0** and is a genuine assertion.

That separates two failure modes cleanly:

| span round-trip | list round-trip | meaning |
|---|---|---|
| = 1.0 | < 1.0 | annotation-scheme ceiling — a finding |
| < 1.0 | — | bug in `encode`/`decode` |

Both ceilings go in the results table. Reported list F1 is meaningless without
them.

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
  excluding NEs, NES = including NEs), and on the same train/validation/test
  split this project has adopted (§6). **Read this paper before day 3.**
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
| **Train domains** | **corruption + wind energy** |
| **Validation domain** | **dressage (`equi`)** |
| Test domain | heart failure |
| Lemmatisation | none, at any stage |
| Gold list variant | **tokenised** (`*_tokenised_terms*.tsv`), never the non-tokenised [T2] |
| `max_length` | **256** subword pieces, truncation only; dynamic per-batch padding [T2] |
| Encoder | `bert-base-cased` for week 2 [T2] |
| Subword labelling | label on **first** subword; continuations `-100`; `word_ids()` for the mapping [T2] |
| Decode output | join the **original dataset tokens** of a span; wordpieces never reach the output [T2] |
| wind filtering | drop wind sentences of ≤2 dataset tokens at training time [T2] |

**On the split.** Train `corp` + `wind`, validate `equi`, test `htfl` is the
standard ACTER cross-domain setting established by Lang et al. (2021) and
followed by Tran et al. (2022, 2024), who adopt it explicitly to allow direct
comparison with prior benchmark approaches. Adopted here for the same reason: an
in-domain random split would make this project's numbers incomparable to the
modern line of work on this dataset. Consequences and the known limitation of
using `equi` as the validation domain are recorded in `data_stats.md` §8.5.

Note on the training unit: sentence-level input means the model sees only the
current sentence. The first-occurrence locality hypothesis is a claim that
context from elsewhere in the document helps — so sentence-level is the
baseline, and going beyond it is the contribution.

---

## 7. Open data questions

Task planning lives in `TASKS.md`. Listed here are the questions about the data
itself that are still unanswered, tagged with the task that resolves each.

**Resolved by T2** — numbers and reasoning in `results/data_stats.md`:

- ~~Tokens per sentence distribution~~ → section 2; `max_length` set in section 8
- ~~Line/token consistency vs `texts_tokenised`~~ → section 3, 241/241 identical
- ~~Term length distribution~~ → ≥95% of terms are ≤4 tokens in every domain and
  key; T4's candidate cap is 4
- ~~Term frequency and hapax proportion~~ → hapax 42–48% over the annotated
  portion, ~30% using the unannotated reference material (htfl 46%, with no
  reference material available). **Caps any frequency-based method near 70%
  recall.**
- ~~Term-set overlap, training vs heart failure~~ → section 9
- ~~Proportion of tokens carrying a positive label~~ → section 9

**Still open:**

- **[T2, deferred to week 3]** Nested-term count: gold terms that occur only as
  a contiguous token subsequence of a longer gold term and never as a maximal
  span. Split by 1-token (NOBI-addressable) vs ≥2-token (not addressed by
  NOBI) — without that split the section 5.4 experiment has no predicted effect
  size. Note this is not the same as "substring of another gold term": a term
  nested in one place and standalone in another is fully reachable, and
  character-substring matching gives false hits (`art` in `heart failure`).
- **[T3]** `max_recall` and `max_precision` ceilings (section 5.3) — these need
  the harness's `decode()`, and running them through it validates the conversion
  code at the same time. Section 5.2b is the closest existing measurement.
- **[T5]** Tran et al. (2024) English heart-failure F1 on both keys, and their
  BIO-vs-NOBI recall delta
- **[T5, no longer blocking]** How the comparison papers compute their F1 —
  deduplicated unique-list F1, or span/token-level. T3 now builds both metrics,
  so this determines which column the comparison sits in rather than gating the
  harness. Every number in the comparison table must still be labelled with its
  unit. See `data_stats.md` §9.4.

All T2 statistics were computed over the **annotated portion only**, verified by
the inventory ratio in section 1.

### Optional

- `diff` the `with_named_entities` and `without_named_entities` sequential
  folders. Low priority: training uses the without-NE labels, and the with-NE
  data is needed only as a unique list, not as tags.

---

## 8. Subword tokenization — measured [T2]

The dataset's token stream is not the model's. BERT's WordPiece splits each
dataset token into pieces from a fixed 30k vocabulary, so one dataset token
occupies several model positions.

### 8.1 Inflation is modest overall, large on terms

Subwords per dataset token, four tokenizers, measured with
`tok(tokens, is_split_into_words=True)` on the pre-split list, no truncation:

| tokenizer | overall | htfl inside span (B/I) | htfl outside (O) | ratio |
|---|--:|--:|--:|--:|
| `bert-base-cased` | 1.257 | **2.043** | 1.215 | **1.68×** |
| `roberta-base` | 1.179 | 1.722 | 1.141 | 1.51× |
| `deberta-v3-base` | 1.121 | **1.382** | 1.102 | 1.25× |
| `xlm-roberta-base` | 1.404 | 2.031 | 1.367 | 1.49× |

Overall inflation is 1.12–1.40, not the ~1.8 assumed during planning. But
**terms fragment far harder than the text around them**: under BERT a htfl gold
term token becomes 2.04 wordpieces against 1.22 for ordinary text. Clinical
vocabulary is not in a 30k general-English vocabulary.

The inside-span figures here are **htfl-only**. `data_stats.md` §5 carries a
pooled-across-domains version of the same column (BERT 1.636); the two are
different measurements and must not be quoted interchangeably.

wind is the exception — B/I 1.443 against O 1.299, a gap of only 1.11×,
consistent with its terms being short table-cell nouns.

### 8.2 `max_length` = 256

Sentences exceeding the threshold, of 14,162:

| tokenizer | >256 | >512 | htfl >256 |
|---|--:|--:|--:|
| `bert-base-cased` | 8 | 1 | 1 |
| `roberta-base` | 5 | 0 | 0 |
| `deberta-v3-base` | 6 | 1 | 0 |
| `xlm-roberta-base` | 6 | 0 | 0 |

htfl p99 is 106–120 across all four, so 256 leaves double the headroom on the
test domain and costs a quarter of 512's attention compute. Truncation loses at
most one sentence and no term type; it is not a ceiling worth reporting.

The single >512 case is a wind artifact whose four tokenizers disagree by 2.7×
(BERT 1,074, DeBERTa 1,061, XLM-R 488, RoBERTa 399) — a long unbroken character
sequence, not prose. Excluding it, BERT's corpus max is 325.

### 8.3 Encoder choice

**Decision: `bert-base-cased` for week 2.** It is the ATE reference point and
keeps results comparable.

**Logged as the first lever if htfl recall underperforms:** DeBERTa-v3
fragments htfl terms at 1.382 against BERT's 2.043 — a third fewer pieces per
term, the largest gap in the table and located exactly where the task is
hardest. It is a general-purpose encoder with a 128k vocabulary, **not** a
biomedical one, so it introduces no leakage into the cross-domain claim.
BioBERT, SciBERT and PubMedBERT must not be used for that reason.

XLM-R is both the longest and the worst on htfl terms — relevant to T5, since
Tran et al. (2024) worked under a heavier length budget than this project.

### 8.4 Label alignment

One gold label per dataset token. After tokenization that token spans several
positions, so:

- the **first** subword of each dataset token carries its label
- every other subword is `-100`, which the loss ignores — no prediction, no
  gradient, no penalty
- at inference the prediction is read from the **first** subword; the rest are
  discarded unread

The mapping comes from `word_ids()` on the fast tokenizer, never from string
matching. This requires passing the pre-split token list with
`is_split_into_words=True`. Joining tokens into a string and re-tokenizing
shifts every label after the first hyphenated word, because BERT's
pre-tokenizer splits internal punctuation while ACTER's tokenisation is
inconsistent about it (`non - governmental` is three tokens, `self-employed`
is one).

**Wordpieces never appear on the output path.** A decoded term string is built
by joining the *original dataset tokens* of a span. Reassembling
`self-employed` from `self`, `-`, `employed` yields `self - employed`, which
matches no gold entry — a silent false negative plus a false positive, landing
hardest on the fragmented domain vocabulary the task exists to extract. The
model says *which* tokens form a term; it is never asked what they say.

---

## 9. Label distribution and domain overlap — measured [T2]

### 9.1 Positive rates are an alignment invariant

| domain | n_tokens | B | I | O | pos. rate | mean occ. len |
|---|--:|--:|--:|--:|--:|--:|
| corp | 50,845 | 4,180 (8.2%) | 2,239 (4.4%) | 44,426 (87.4%) | 0.1262 | 1.54 |
| equi | 58,203 | 8,662 (14.9%) | 1,940 (3.3%) | 47,601 (81.8%) | 0.1822 | 1.22 |
| wind | 57,766 | 5,053 (8.7%) | 3,354 (5.8%) | 49,359 (85.4%) | 0.1455 | 1.66 |
| htfl | 55,467 | 9,636 (17.4%) | 4,806 (8.7%) | 41,025 (74.0%) | **0.2604** | 1.50 |

The imbalance needs no handling — 13–26% positive requires no loss weighting.
The value of these numbers is as a **fixed reference constant**. First-subword
labelling scores exactly one position per dataset token, so the positive rate
computed over the batched tensors (ignoring `-100`) must reproduce this table
exactly. If it does not, label alignment is broken — and nothing crashes when
that happens.

**These are pre-filter figures.** wind's 0.1455 counts the ≤2-token sentences
that §2 drops at training time; the filtered dataloader will produce
**0.1562** for wind. Assert **htfl 0.2604, wind 0.1562 (post-filter), corp
0.1262, equi 0.1822** — see `data_stats.md` §6.1. Asserting wind's pre-filter
value would fire a false alarm on the first run.

### 9.2 htfl has a higher term density than any training domain

26% positive tokens against a training prior near **0.14** under the adopted
split — corp at 0.1262 and wind at 0.1562 post-filter, with equi (the densest
of the three candidate domains, at 0.1822) moved to validation. A model fits the
label prior of its training distribution, so this is a distribution shift in the
prior itself, independent of any lexical question, and the split makes it wider
rather than narrower.

**Predicted effect: recall suppressed on htfl, precision unaffected.**

Consequence for T3: a decode threshold tuned on training-domain data inherits
the wrong prior. Either use argmax and report the shift as a known limitation,
or tune on the validation domain and state the value used. Tuning on htfl is
fitting the test set.

### 9.3 The domains are lexically near-disjoint

Terms-only keys both sides, N = 2,339 htfl gold terms:

| measure | count | % of htfl terms |
|---|--:|--:|
| **type overlap** — also a training gold entry | **10** | **0.4%** |
| occurrence-weighted type overlap | 31 / 9,243 | 0.3% |
| **text overlap** — sequence occurs in training text, any label | 99 | 4.2% |
| … seen in text but never a labelled term there | 89 | 3.8% |
| **head overlap** — final token matches a training term's | 302 | 23.1% (N=1,310 MW) |

**Scope caveat.** These were computed with corp + equi + wind on the training
side, before the split was fixed. Under the adopted split (corp + wind only) the
figures are lower — equi alone contributed 6 of the 10 type overlaps. Not
recomputed: the conclusion holds *a fortiori*. **Do not quote 0.4% as a
split-matched number.** See `data_stats.md` §8.

All ten type overlaps are generic: `bpm`, `chest`, `compliance`, `contracting`,
`muscle`, `muscles`, `muscular`, `pad`, `rna`, `remote monitoring`. Nine are
single-token. Overlap at 3+ tokens is exactly zero. Several are polysemes rather
than shared terms — *compliance* is regulatory in corp and ventricular or
adherence-related in htfl; *pad* is a wind component and, in cardiology,
peripheral artery disease — so the measured overlap overstates the real one.

**Essentially no reported score can be lexical memorisation.** Every result on
htfl is a genuine cross-domain generalisation result, with no caveat required.

**Head overlap is the transfer channel that does exist.** 23% of htfl's
multi-word terms share a final token with a training term — a model that
learned `... failure` occupies a term-final position generalises to compounds
it has never seen.

**Prediction, checkable against per-length recall:** recall on multi-word htfl
terms should exceed recall on single-word terms. Single words are 44% of the
htfl key and have essentially no transfer channel; failures should concentrate
there. **Logged as contrarian** — the standard ATE finding is the opposite, and
htfl's single-word terms are also its most frequent. See `data_stats.md` §8.2.

On the terms+NE key overlap roughly triples (type 1.3%, text 4.8%) — named
entities recur across domains. Keys are never mixed across the two sides.

### 9.4 htfl is harder than the training domains on every measured axis

- twice the term inventory: 2,339 gold terms against ~1,100 per training domain
- highest hapax rate: 47.7%, with no unannotated text to improve the estimate
- highest term density: 26% positive tokens
- 99.6% of its terms never seen as labelled training examples
- a different term profile: its most frequent entries include `p`, `ci`, `hr`,
  `hf` — statistical notation and research boilerplate from clinical abstracts,
  a term type the training domains contain almost none of