# Data Statistics — T2

Statistics over the annotated portion of ACTER v1.5, English. Computed by
`src/stats/`, re-runnable. Data facts live in `docs/data_layout.md`; task
planning in `Tasks.md`. This file records what was measured and what each
number decides.

---

## 1. The problem this task exists to solve

### 1.1 What I thought the question was

The obvious reading of `data_layout.md` §7 item 1 is: measure how long
sentences are, take a high percentile, set `max_length` to it.

That framing is wrong in two ways.

### 1.2 Small concern: train/test length mismatch

Setting `max_length` from the training domains alone can truncate longer
sentences at test time. Real, but minor: `htfl` is measured here too, so the
distribution is known on both sides.

### 1.3 The actual problem: dataset tokens are not model tokens

ACTER's token stream is whitespace-tokenised, one token per line in the
sequential annotation file. BERT does not consume that stream. Its WordPiece
tokenizer splits each token into subword pieces drawn from a fixed 30k
vocabulary.

Domain vocabulary is not in that vocabulary. `remunerated`, a single token in
`corp`, becomes something like `rem ##une ##rated` — one dataset token, three
model positions. The split exists so the model can build a representation for
a word it never saw during pretraining, out of pieces it did see.

Two consequences follow, and both change what T2 has to measure.

**Consequence 1 — the length budget is in the wrong unit.**

BERT's 512-position limit counts subwords, not words. A sentence of 400
dataset tokens may be 500+ subwords. Measuring whitespace tokens says nothing
about whether a sentence fits.

Worse, the inflation ratio is not constant. Function words inflate at ~1.0;
domain terms inflate at 2–3x. So the tokens that consume the most budget are
exactly the tokens carrying positive labels. The measurement must therefore
report the inflation ratio *split by whether the token is inside a gold span*,
not as a single corpus-wide average.

**Consequence 2 — one label, several positions.**

A gold label attaches to a dataset token. After tokenization that token spans
several model positions. The label array does not change shape; what changes
is which position carries it.

Resolution, applied identically at training and inference time:

- The **first** subword of each dataset token carries that token's label.
- Every other subword is set to `-100`, which the loss function ignores.
  Nothing is predicted there, nothing is scored there, no gradient flows from
  there. `-100` is an exclusion marker, not a penalty.
- At inference, the prediction for a dataset token is read from its **first**
  subword position. The remaining positions are discarded unread.

The mapping is not recovered by string matching. `word_ids()` from the fast
tokenizer returns, for each subword position, the index of the input token it
came from. This requires passing the pre-split token list with
`is_split_into_words=True`, never a re-joined string.

**Consequence 3 — subwords never reach the output.**

The decoded term string is built by joining the **original dataset tokens**
identified by a span, not by detokenizing wordpieces. Reassembling
`self-employed` from `self`, `-`, `employed` yields `self - employed`, which
does not match the gold list entry — a silent false negative plus a false
positive, landing hardest on the fragmented domain vocabulary the task exists
to extract.

The model's job is to say *which* tokens form a term. It is never asked what
those tokens say; that is already on disk in the exact form the gold list was
built from.

## 2. The loader

All seven T2 statistics read the corpus through `src/stats/loading.py`. A
loader bug is therefore a bug in every statistic simultaneously, in the same
direction, with no crash — which is why it is documented and tested before any
number is reported.

### 2.1 Design rules

**Iteration starts from annotation files.** The paired text path is derived
from each annotation file and asserted to exist, never the reverse. An
unannotated text has no annotation file and therefore cannot enter the
pipeline. `data_layout.md` §1 makes this the structural defence against the
wind-energy trap.

**The token stream comes from the annotation file only.** Never from
`texts/` or `texts_tokenised/`. Only the annotation file has tokens and labels
aligned one-to-one; reading tokens from anywhere else would require
re-aligning them to labels, which is a class of bug with no crash attached.

**Sentence boundary is a blank line.** Per the locked decision in
`data_layout.md` §6. A run of blank lines is one boundary; a trailing blank at
EOF produces no empty sentence.

**Everything raises.** There is no warn-and-continue path. A warning scrolls
past; an exception stops the run.

**Checks guard the config, not the corpus.** ACTER v1.5 is pinned at a tag and
was inspected by hand, so the data cannot change. What can change is a path or
a config value pointing somewhere wrong — the failure `Tasks.md` names as the
T2 risk, and the one that produces plausible numbers rather than an error.

### 2.2 Interface

```python
load_domain(domain, cfg) -> list[Document]
# Document.file_id    e.g. "corp_en_01"
# Document.sentences  list of (tokens, labels) — two parallel lists per sentence
```

Files are processed in sorted order, so output is deterministic across
machines.

### 2.3 The two checks that matter

**The B-label assertion.** `iob_annotations/` and `io_annotations/` are
sibling directories with near-identical names. In an IO file every line parses
cleanly, because `I` and `O` are valid labels in both schemes — there is simply
never a `B`. Nothing crashes, and every span-based statistic comes out
silently wrong. Asserted at domain level rather than per file, since a single
file may legitimately contain no terms.

**The inventory ratio.** Paired token count divided by whole-domain corpus
token count, both whitespace-tokenised so the unit cancels. No external table
and no version drift. `data_layout.md` §1 puts wind at ~52k annotated of ~314k
total, so a ratio anywhere near 1.0 means the unannotated corpus was loaded.

A superseded check compared the loaded count against §1's published word
counts with a ±15% tolerance. That comparison was invalid: §1's figures are
`wc -w` over raw non-tokenised `texts/`, while the loader counts the annotated
stream with punctuation split into separate tokens, which inflates every
domain by 10–18%. The tolerance was being asked to absorb a unit mismatch in a
tripwire built to catch a 6× error. Replaced by the ratio above.

---

## 3. Evidence the loader is correct

Unit tests covering the parser, loader wiring, and the inventory ratio live in
`tests/test_loading.py` and all pass. The results below are from the real
corpus.

**Inventory:** corp 12, equi 34, wind 5, htfl 190 annotation files.

**Wind inventory ratio:** 0.1806, below the 0.30 bound (paired 57,766 /
corpus 319,887). Independently reproduces §1's ~52k-of-~314k figure. The
unannotated corpus was not loaded.

**Totals:**

| domain | documents | sentences | tokens |
|---|---|---|---|
| corp | 12 | 2,002 | 50,845 |
| equi | 34 | 3,090 | 58,203 |
| wind | 5 | 6,638 | 57,766 |
| htfl | 190 | 2,432 | 55,467 |
| **pooled** | **241** | **14,162** | **222,281** |

Zero-token sentences: 0 in every domain.

**Token identity.** The loader's token stream was verified string-for-string,
per file and in order, against `texts_tokenised/`: 241 of 241 files identical.
This is the evidence that decoded term strings will match the gold list by
construction, since both are joins over the same token stream. It also answers
`data_layout.md` §7 item 2 — sentence boundaries agree between the blank lines
of the annotation file and the line structure of `texts_tokenised`.

> The identity check was removed from `loading.py` after this run. The result
> above stands as a one-time verification. It is not re-run on every
> invocation.

---

## 4. s01 — sentence length (whitespace tokens)

Scope: English, annotated portion only, sequential IOB annotations,
`without_named_entities` labels. Sentence length = number of `(token, label)`
rows in the sentence block; punctuation and numbers are separate tokens and
are counted; nothing is filtered.

Percentiles: sort ascending, nearest-rank; 0-based index
`ceil(p/100 * n) - 1`, clamped to `[0, n-1]`, integer arithmetic. Percentage
columns are of that row's `n_sentences` (`N`).

| domain | n_docs | n_sentences | n_tokens | min | p50 | p90 | p95 | p99 | max | sents ≤2 tok (n) | sents ≤2 tok (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corp | 12 | 2,002 | 50,845 | 1 | 22 | 49 | 61 | 98 | 281 | 228 | 11.39% (N=2,002) |
| equi | 34 | 3,090 | 58,203 | 1 | 18 | 33 | 39 | 52 | 95 | 255 | 8.25% (N=3,090) |
| wind | 5 | 6,638 | 57,766 | 1 | 1 | 27 | 36 | 62 | 258 | 4,048 | 60.98% (N=6,638) |
| htfl | 190 | 2,432 | 55,467 | 1 | 20 | 43 | 54 | 78 | 171 | 338 | 13.90% (N=2,432) |
| pooled | 241 | 14,162 | 222,281 | 1 | 13 | 36 | 46 | 72 | 281 | 4,869 | 34.38% (N=14,162) |

### 4.1 The four domains are four different text types

Token counts are balanced by design, but nothing else is:

| | corp | equi | wind | htfl |
|---|---:|---:|---:|---:|
| tokens / document | 4,237 | 1,712 | 11,553 | 292 |
| sentences / document | 167 | 91 | 1,328 | 13 |
| p99 / p50 | 4.5× | 2.9× | 62× | 3.9× |

- **equi** is the only single-peaked distribution: p50 18, p99 52, max 95.
  Ordinary prose, no tail.
- **corp** is legal text — p50 22 with a max of 281 (an EU declaration-of-
  interests sentence). 1% of sentences exceed 98 tokens.
- **htfl** has a similar shape to corp but arrives as 190 documents of ~13
  sentences: short clinical notes or abstracts, not articles.
- **wind** is not prose at all. See §4.3.

**"Document" is not a stable unit across this corpus** — 292 tokens in htfl
against 11,553 in wind, a 40× spread. Any design that consumes document-level
context has to hold at both ends.

### 4.2 The pooled row describes no domain

p50 of 13 is the average of wind's 1 and the other three domains' 18–22. The
34.38% ≤2-token figure is wind's 61% diluted by three domains at 8–14%. The
row is retained for completeness and is not cited anywhere. **Every conclusion
in this project is per-domain.**

### 4.3 wind is bimodal — 61% of its sentences are fragments

```
p50 = 1        61% of sentences ≤ 2 tokens (4,048 of 6,638)      max = 258
```

Five documents, ~1,328 sentences each. These are long technical reports whose
tables and lists were segmented one cell per line; the loader read each
resulting blank line as a sentence boundary, correctly, per the locked
decision in `data_layout.md` §6.

The distribution is two distributions stacked: a mass of fragments at 1–2
tokens, and real prose out to 258. The percentiles average across both and
describe neither, which is why the ≤2-token column sits beside them. For wind,
that column is the informative one.

**Effect on the training mix.** Under sentence-level batching, wind supplies
6,638 of 14,162 examples — 47% of the corpus by example count against 26% by
token count. Most of those examples carry one or two tokens.

### 4.4 max_length: superseded by the measurement in §5

The whitespace numbers above suggested a `max_length` question that the
wordpiece measurement settles directly. See §5.

### 4.5 Inventory ratios

Paired-file tokens over whole-domain-corpus tokens, both whitespace-split
(annotated `texts_tokenised/` plus `unannotated_texts/`).

| domain | paired tokens | corpus tokens | ratio | bound |
|---|---:|---:|---:|---|
| corp | 50,845 | 181,923 | 0.2795 | — |
| equi | 58,203 | 109,387 | 0.5321 | — |
| wind | 57,766 | 319,887 | 0.1806 | < 0.30 |
| htfl | 55,467 | 55,467 | 1.0000 | — |

corp's 0.28 matches `data_layout.md` §1: annotations exist only in the
parallel corruption corpus, and the comparable corpus is entirely unannotated.
wind's 0.18 independently reproduces §1's ~52k-of-~314k figure. htfl is 1.0,
as §1 states.

**Relevant to T4.** There are ~445k unannotated English tokens available as
reference material (corp ~131k, equi ~51k, wind ~262k) — roughly triple the
annotated training text. Whether that is worth using for C-Value's frequency
counts depends on the hapax proportion measured in `data_layout.md` §7 item 4.
Khaled's open decision; the numbers above are the input to it.

---

## 5. s02 — wordpieces (four tokenizers)

Wordpieces per sentence including special tokens, measured with
`tok(tokens, is_split_into_words=True)` on the pre-split token list, no
truncation. Inflation = subwords per dataset token, split by whether the token
carries a positive label.

| tokenizer | htfl p99 | corpus max | inside-span inflation |
|---|---:|---:|---:|
| bert-base-cased | 118 | 1,074 | 1.636 |
| roberta-base | 112 | 399 | 1.445 |
| deberta-v3-base | 106 | 1,061 | 1.242 |
| xlm-roberta-base | 120 | 488 | 1.795 |

### 5.1 max_length = 256. Settled.

Sentences exceeding 256 wordpieces, out of 14,162:

| tokenizer | >256 | >512 | htfl >256 |
|---|---:|---:|---:|
| bert-base-cased | 8 | 1 | 1 |
| roberta-base | 5 | 0 | 0 |
| deberta-v3-base | 6 | 1 | 0 |
| xlm-roberta-base | 6 | 0 | 0 |

htfl p99 is 106–120 across all four tokenizers, so 256 leaves more than
double the headroom on the test domain. Under BERT exactly **one** htfl
sentence exceeds it.

**Decision:** `max_length = 256`, used purely as a truncation threshold, with
dynamic per-batch padding. Attention is quadratic in sequence length, so this
costs roughly a quarter of what 512 would.

### 5.2 The inflation estimate used earlier was wrong

Planning assumed ~1.8× subword inflation. Measured overall inflation is
**1.12–1.40**, and p50 barely moves: 22 whitespace tokens becomes 27 BERT
wordpieces.

The earlier concern that a 281-token sentence would consume ~506 of BERT's 512
positions was overstated by 30–60%. Under BERT that sentence is 325
wordpieces.

Recorded because the corrected figure is what future length estimates in this
project should use — and because the failure mode was assuming a ratio instead
of measuring one.

### 5.3 Terms fragment far more than the text around them

The overall ratio hides the effect the split was designed to find. On htfl:

| tokenizer | inside span (B/I) | outside (O) | ratio |
|---|---:|---:|---:|
| bert-base-cased | 2.043 | 1.215 | **1.68×** |
| roberta-base | 1.722 | 1.141 | 1.51× |
| deberta-v3-base | 1.382 | 1.102 | 1.25× |
| xlm-roberta-base | 2.031 | 1.367 | 1.49× |

Under BERT, a htfl gold-term token becomes **2.04 wordpieces on average**
while ordinary text becomes 1.22 — terms fragment at 68% above the rate of
their surrounding text, in the only domain where recall is scored.

This is a direct consequence of vocabulary coverage: WordPiece's 30k pieces
were learned on general English, and clinical terminology is not in them.

**wind is the exception.** Its B/I inflation is 1.443 against O at 1.299, a
gap of only 1.11× versus htfl's 1.68×. wind terms fragment barely more than
wind text — consistent with them being short table-cell nouns rather than long
technical compounds. Further evidence for §6.1.

### 5.4 BERT vs DeBERTa-v3

The largest gap in the table sits exactly where the task is hardest.
DeBERTa-v3 fragments htfl terms at **1.382 against BERT's 2.043** — a third
fewer pieces per term. Its vocabulary is 128k SentencePiece Unigram pieces
against BERT's 30k WordPiece.

Why this could matter: with first-subword labelling, the model classifies a
term from its first fragment plus context. `hyper` as the opening piece of a
fragmented medical compound carries less than the near-whole token DeBERTa
would produce. Whether that converts to F1 is unknown without training.

XLM-R is both the longest and the worst on htfl terms. Relevant to T5: Tran
et al. (2024) worked under a heavier length budget than this project will.

**Decision: train BERT this month.** It is the ATE reference point, it keeps
results comparable, and switching now costs time the schedule does not have.

**Logged as the first lever if htfl recall underperforms:** DeBERTa-v3 is a
general-purpose encoder with a larger vocabulary, not a biomedical one. Using
it introduces no leakage into the cross-domain claim, unlike BioBERT or
PubMedBERT, which must not be used.

---

## 6. s04 — label distribution

Dataset-token space, no tokenizer. B, I, O counted separately — B is the term
*occurrence* count, which §7 item 4 needs.

| domain | n_tokens | B | I | O | pos. rate | mean occ. len | zero-pos sents |
|---|--:|--:|--:|--:|--:|--:|--:|
| corp | 50,845 | 4,180 (8.2%) | 2,239 (4.4%) | 44,426 (87.4%) | 0.1262 | 1.54 | 607 (30.3%) |
| equi | 58,203 | 8,662 (14.9%) | 1,940 (3.3%) | 47,601 (81.8%) | 0.1822 | 1.22 | 467 (15.1%) |
| wind | 57,766 | 5,053 (8.7%) | 3,354 (5.8%) | 49,359 (85.4%) | 0.1455 | 1.66 | 4,905 (73.9%) |
| htfl | 55,467 | 9,636 (17.4%) | 4,806 (8.7%) | 41,025 (74.0%) | 0.2604 | 1.50 | 443 (18.2%) |
| pooled | 222,281 | 27,531 (12.4%) | 12,339 (5.6%) | 182,411 (82.1%) | 0.1794 | 1.45 | 6,422 (45.3%) |

### 6.1 These rates are an alignment invariant

The imbalance itself needs no handling — 13–26% positive requires no loss
weighting. The value of these numbers is as a **fixed reference constant**.

The week-2 dataloader will tokenize, apply first-subword labelling, and set
continuation subwords to `-100`. Exactly one position per dataset token is
scored, so the positive rate computed over the batched tensors (ignoring
`-100`) must reproduce the table above **exactly**. htfl must come out at
0.2604.

If it does not, label alignment is broken — and nothing crashes when that
happens. Assert against these four numbers once the dataloader exists.

### 6.2 wind's ≤2-token sentences are filterable

Short (≤2 dataset tokens) versus prose (≥3), per domain:

| domain | bucket | n_sent | % dom sent | n_tok | B | pos. rate | B % of domain B |
|---|---|--:|--:|--:|--:|--:|--:|
| corp | short | 228 | 11.4% | 429 | 14 | 0.0396 | 0.3% |
| corp | prose | 1,774 | 88.6% | 50,416 | 4,166 | 0.1270 | 99.7% |
| equi | short | 255 | 8.3% | 470 | 77 | 0.2043 | 0.9% |
| equi | prose | 2,835 | 91.7% | 57,733 | 8,585 | 0.1820 | 99.1% |
| wind | short | 4,048 | 61.0% | 4,275 | 44 | 0.0122 | 0.9% |
| wind | prose | 2,590 | 39.0% | 53,491 | 5,009 | 0.1562 | 99.1% |
| htfl | short | 338 | 13.9% | 674 | 2 | 0.0030 | 0.0% |
| htfl | prose | 2,094 | 86.1% | 54,793 | 9,634 | 0.2635 | 100.0% |

**wind's short bucket holds 0.9% of wind's B count**, at a positive rate of
0.0122 — 13× below wind's prose. It is 61% of wind's sentences and 7.4% of its
tokens.

**Decision: filter sentences of ≤2 dataset tokens from wind at training time.**
Cost is 0.9% of wind's term occurrences; benefit is removing 4,048 examples
that carry almost no signal.

The content confirms it. Of the 4,048 short sentences, **3,499 are a single
period** — a segmentation artifact, not text. The rest are stray digits, list
markers, single letters, and a document code:

| count | tokens |
|--:|---|
| 3,499 | `.` |
| 13 | `2` |
| 9 | `λ` |
| 9 | `D-80.1-GP.SD.03-A-A-GB` |
| 8 | `2002-09-10` |
| 6 | `8` |
| 5 | `Cp =` |

This also resolves the question of whether wind's terminology lives in table
cells: it does not.

### 6.3 htfl has a higher term density than any training domain

| | corp | equi | wind | **htfl** |
|---|--:|--:|--:|--:|
| positive rate | 0.126 | 0.182 | 0.146 | **0.260** |
| mean occurrence length | 1.54 | 1.22 | 1.66 | 1.50 |

The test domain runs at 26% positive tokens against a training-domain average
near 15%. A model fits the label prior of its training distribution, so this
is a distribution shift in the prior itself, independent of any lexical
question.

**Predicted effect: recall suppressed on htfl, precision unaffected.** Recorded
here so it can be checked against actual per-domain results rather than
discovered afterwards.

Consequence for T3: a decode threshold tuned on training-domain data inherits
the wrong prior. Either use argmax and report this shift as a known
limitation, or tune on a held-out training domain and state the value used.
Tuning on htfl is fitting the test set.

`equi` is the opposite extreme — 14.9% B against 3.3% I, mean occurrence
length 1.22. Dressage terms are overwhelmingly single words.

---

## 7. s05 — term length and frequency (gold unique lists)

Tokenised gold lists only. A non-tokenised `*_terms.tsv` variant exists in
every domain and is ignored: only the tokenised list matches the corpus token
stream that `decode()` will join from.

**Both format assumptions in `data_layout.md` §4 confirmed:** 0 entries
contain an uppercase character, 0 duplicates after lowercasing, in all eight
files. Dedup-at-lowercase is safe.

### 7.1 Term length — ≥95% of terms are ≤4 tokens

Length in whitespace tokens, cumulative percentage at ≤4:

| domain | key | N | 1 token | cum ≤4 | max |
|---|---|--:|--:|--:|--:|
| corp | terms | 926 | 42.0% | 98.4% | 7 |
| corp | terms+NE | 1,172 | 42.8% | 95.2% | 13 |
| equi | terms | 1,146 | 55.7% | 99.7% | 5 |
| equi | terms+NE | 1,561 | 55.9% | 99.0% | 7 |
| wind | terms | 1,092 | 29.1% | 98.9% | 7 |
| wind | terms+NE | 1,529 | 36.8% | 98.0% | 10 |
| htfl | terms | 2,339 | 44.0% | 97.3% | 8 |
| htfl | terms+NE | 2,556 | 44.3% | 96.1% | 13 |

**For T4:** candidate generation capped at 4 tokens loses 0.3–2.7% of gold
terms on the terms-only keys. Going beyond 5 is not worth the precision cost.

**Bears on §5.4.** Tran et al. (2024) report that the best models under both
BIO and NOBI predicted terms only up to 4 tokens (English). On these keys that
ceiling costs at most 2.7% of gold types, so it is not a material limit for
this project's setup.

Domains differ in compositionality: equi is 55.7% single-word terms, wind only
29.1%. The longest entries are almost entirely organisation names on the
terms+NE keys.

### 7.2 Annotation is not exhaustive — a third to a half of occurrences are untagged

Three counts, deliberately distinguished:

- **(a)** decoded gold-BIO span occurrences
- **(b)** the term's token sequence in the annotated token stream, any label
- **(c)** same over the whole corpus (`texts_tokenised/` + `unannotated_texts/`)

| domain | (a) total | (b) total | **(b)/(a)** | (c) total |
|---|--:|--:|--:|--:|
| corp | 4,116 | 6,632 | **1.61** | 19,072 |
| equi | 8,546 | 12,606 | **1.48** | 21,619 |
| wind | 4,982 | 9,304 | **1.87** | 31,306 |
| htfl | 9,243 | 13,890 | **1.50** | 13,890 |

Individual cases are stark. In wind, `wind` is tagged 57 times and appears
642; `power` 63 against 399; `rotor` 99 against 284. In corp,
`anti-corruption` is tagged 44 times and appears 146. In htfl,
`heart failure` is tagged 350 times and appears 530.

**Two mechanisms, with different implications:**

*Nesting.* `wind` occurs inside `wind turbine`, `wind speed`, `wind energy`.
BIO marks only the longest span, so the inner occurrence carries no
independent label. This is §5.1's recall ceiling measured from the other
direction, and it explains most of wind's 1.87 — wind's terms are the most
compositional in the corpus (§7.1).

*Genuinely unannotated occurrences,* where a term appears standalone and was
not marked.

These numbers do not separate the two. The distinction matters: nesting is a
representational limit to report, non-exhaustive annotation is training noise.

**Consequence for training.** The model will see `wind` labelled `B` in one
sentence and `O` in another, distinguished only by whether the occurrence sits
inside a longer term. That is contradictory supervision unless the model
learns the longest-match rule. **Training loss will not approach zero, and
that is not a bug.**

### 7.3 Hapax proportion — the ceiling on any frequency-based method

| domain | hapax % under (a) | under (b) | under (c) |
|---|--:|--:|--:|
| corp | 44.5% | 44.7% | 30.1% |
| equi | 41.8% | 38.8% | 32.3% |
| wind | 43.9% | 43.0% | 30.4% |
| htfl | 47.7% | 46.0% | 46.0% |

**For T4's open decision** (which corpus supplies frequency counts): using the
unannotated reference material moves hapax from ~44% to ~30% in corp and wind,
and wind's max frequency from 212 to 4,560. That is a substantial improvement
in frequency estimates and a real argument for using it.

But **~30% of gold terms still occur exactly once even across 314k words.**
C-Value cannot rank a hapax by definition, and the metric is F1 over a
deduplicated list, which weights a hapax the same as a 598-occurrence term.
The statistical baseline is therefore capped near 70% recall before any code
is written.

htfl's (c) equals its (b) exactly, since htfl has no unannotated text — this
confirms `data_layout.md` §1 and validates the counting code.

### 7.4 htfl is harder than the training domains on every axis measured

- **Twice the term inventory:** 2,339 gold terms against ~1,100 per training
  domain.
- **Highest hapax rate:** 47.7%, with no unannotated text to improve the
  estimate.
- **Highest term density:** 26% positive tokens (§6.3).
- **Different term profile:** its most frequent entries include `p`, `ci`,
  `hr`, `hf` — statistical notation and abbreviations from clinical abstracts.
  The training domains contain essentially none of this term type.

---

## 8. s06 — term-set overlap, training domains vs htfl

Terms-only keys (`without_named_entities`) on both sides. N = 2,339 htfl gold
terms. Training side = corp + equi + wind. Token-sequence matching,
lowercased.

`Tasks.md` calls this the number that matters most: it bounds how much of any
reported score could be memorisation rather than cross-domain generalisation.

| measure | count | % of htfl terms |
|---|--:|--:|
| **type overlap** — htfl term is also a training gold entry | **10** | **0.4%** (N=2,339) |
| occurrence-weighted type overlap | 31 / 9,243 occ | 0.3% |
| **text overlap** — sequence occurs in training annotated text, any label | 99 | 4.2% (N=2,339) |
| … seen in text but never a labelled term there | 89 | 3.8% (N=2,339) |
| **head overlap** — final token matches a training term's final token | 302 | 23.1% (N=1,310 multi-word) |
| … not already a type match | 301 | 23.0% (N=1,310) |

Per training domain alone: corp 2 type / 39 text, equi 6 / 58, wind 2 / 46.

### 8.1 The domains are lexically near-disjoint

**10 of 2,339 htfl gold terms appear in the training gold lists.** All ten are
generic: `bpm`, `chest`, `compliance`, `contracting`, `muscle`, `muscles`,
`muscular`, `pad`, `rna`, `remote monitoring` — anatomy and ordinary English
that happens to be annotated in dressage or wind energy. None is medical
terminology.

By length: 9 of the 10 are single-token. Type overlap at 3+ tokens is exactly
zero.

Text overlap is an order of magnitude higher in relative terms (4.2% vs 0.4%)
and still negligible in absolute terms — the model saw 99 htfl term strings
somewhere in training, 89 of them never as labelled examples.

**Consequence: essentially no reported score can be lexical memorisation.**
Every result on htfl is a genuine cross-domain generalisation result, with no
caveat required. This is the question a sharp reader asks first, and the answer
is that it is not a concern.

### 8.2 Head overlap is the transfer channel that does exist

**23.1% of htfl's multi-word terms share a final token with some training
term**, and 23.0% are not covered by type overlap at all.

That is the mechanism by which anything transfers: a model that learned
`... failure` or `... system` occupies a term-final position generalises to
compounds it has never seen. `heart failure` carries a head flag despite no
type or text overlap.

**Prediction, checkable against per-length recall later:** recall on
multi-word htfl terms should exceed recall on single-word terms. Multi-word
terms have head-position and syntactic cues that transfer; single words, 44%
of the htfl key, have essentially no transfer channel at all. Failures should
concentrate there.

### 8.3 htfl's frequent terms are clinical register, not terminology

None of htfl's 20 most frequent terms is a type overlap. What they are:
`patients`, `p`, `ci`, `mortality`, `outcomes`, `clinical`, `hr`, `follow-up`,
`baseline`, `significantly`, `significant`, `death`.

Most is research-abstract register — statistical notation and boilerplate —
rather than domain terminology. ACTER's scheme counts Common Terms and
Out-of-Domain Terms as positive (`data_layout.md` §4), so these are gold.

The small text overlap that exists is concentrated here: `patients`, `p`,
`baseline`, `death`, `significant` appear in corp and equi because they are
ordinary English. The abbreviations `hf`, `hfpef`, `chf` have no overlap of
any kind, and the training domains contain almost nothing of that term type.

### 8.4 Named entities inflate every measure

terms+NE key on both sides, N = 2,556: type overlap 32 (1.3%), text overlap
122 (4.8%), head overlap 356 of 1,424 (25.0%).

Roughly triple the type overlap, as expected — organisation and place names
recur across domains. Keys are never mixed across the two sides; doing so
would produce a misleadingly high figure.

---

## 9. Open questions

### 9.1 [CLOSED] Do wind's ≤2-token sentences contain terms?

No. They hold 0.9% of wind's B count and 3,499 of the 4,048 are a single
period. Resolved in §6.2; filtering decision recorded there.

### 9.2 [CLOSED] Is wind the sharpest test of the document-context hypothesis?

Premise was that wind's terms live in context-free table cells. They do not —
see §6.2. wind's terms are in its prose, like every other domain.

### 9.3 [OPEN] Is cross-sentence context actually unexplored for ATE?

**Not established. Do not assume it.** Cross-sentence and document-level
context for sequence labelling is a worked area in NLP, and ACTER-specific
work exists (Tran et al. 2024, same test set — `data_layout.md` §5.4).

What may be unclaimed is the narrower framing: first-occurrence locality and
multi-scale chunking, on ACTER, under the TermEval 2020 protocol. Establishing
that is **T5's job**. No novelty claim goes in the writeup until T5 reports.

### 9.4 [OPEN, low priority] One wind sentence tokenizes to 1,074 pieces

Corpus max under BERT is 1,074 wordpieces, from a wind sentence whose
whitespace length is far below that. The four tokenizers disagree by 2.7× on
it (BERT 1,074, DeBERTa 1,061, XLM-R 488, RoBERTa 399), which is the signature
of a long unbroken character sequence — a URL, a numeric string, or a table
row rather than text.

One sentence out of 14,162, past a `max_length` of 256 that already truncates
almost nothing. Not worth weight. Excluding it, the corpus max under BERT is
325.

Noted only so the figure in §5 is not mistaken for a property of the prose.

---

## 10. Still to measure

| item | statistic | decides |
|---|---|---|
| ~~§7 T2-1b~~ | ~~wordpieces per sentence~~ | **done — §5, max_length = 256** |
| ~~§7 T2-1c~~ | ~~inflation ratio, inside vs outside span~~ | **done — §5.3** |
| ~~§7 T2-3~~ | ~~term length distribution~~ | **done — §7.1, cap at 4** |
| ~~§7 T2-4~~ | ~~term frequency, hapax proportion~~ | **done — §7.2, §7.3** |
| ~~§7 T2-5~~ | ~~term-set overlap, train ↔ htfl~~ | **done — §8, overlap is 0.4%** |
| ~~§7 T2-6~~ | ~~positive-label proportion~~ | **done — §6** |
| §7 T2-7 | nested-term count, split 1-token vs ≥2-token | NOBI's expected effect size — **deferred to week 3** |