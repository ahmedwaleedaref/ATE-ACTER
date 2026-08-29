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

### 1.4 Truncation is a third recall ceiling

`data_layout.md` §5 records two ceilings, from nested terms and from
discontinuous-term fragments. Truncation adds a third.

A gold term whose every occurrence falls past `max_length` can never be
predicted. It is not a model failure and must not be reported as one. Unlike
the other two, this ceiling is a property of the configuration rather than the
annotation scheme, which makes it the only one that can be engineered away —
and the only one that is invisible if not deliberately measured.

Counted at the **type** level, not the occurrence level: a term with eight
occurrences, one of them truncated, loses nothing, because the metric is F1
over a deduplicated list.

---

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

### 4.4 max_length: 512 is probably safe, 256 may be sufficient

Longest sentence in the corpus is 281 whitespace tokens (corp). At a plausible
subword inflation of ~1.8× that is ~506 positions — inside BERT's 512 limit,
but by six positions on a ratio that has not yet been measured. **Provisional,
not settled.**

The more useful figures are the p99s: corp 98, htfl 78. At ~1.8× those are
~176 and ~140 subwords. A `max_length` of **256 may truncate under 1% of
sentences at roughly a quarter of the attention cost of 512**, since attention
is quadratic in sequence length. Padding should be per-batch (dynamic), with
`max_length` acting purely as a truncation threshold.

Deciding this requires the wordpiece measurement in §5, not this table.

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

## 5. Open questions raised by s01

### 5.1 [OPEN] Do wind's 4,048 fragments contain terms?

wind's ≤2-token "sentences" are table cells. Whether they can be dropped from
training depends on how many gold terms live in them — a number not yet
measured.

- **Few terms (~5% of wind's occurrences)** → structural noise. Filter them
  out with a one-line rule.
- **Many terms (20%+)** → wind's terminology lives in tables. Dropping them
  discards a quarter of wind's training signal.

The second case is likely, not hypothetical. A cell reading `Rotor diameter`
is entirely term. Tables are where terminology concentrates in technical
documents, so fragments may have *higher* positive-label density than prose,
not lower.

**Resolves in §7 item 6** (class imbalance) — compute the ≤2-token subset
separately there. **Do not filter before measuring:** it would silently change
every wind number.

### 5.2 [OPEN] Is wind the best test of the document-context hypothesis?

A table cell has no sentence context. So if wind's terms concentrate in
fragments, wind is where sentence-level input is weakest — and therefore where
document-level context has the most room to help.

That would make wind the project's sharpest evidence rather than its noisiest
domain. It would also be a per-domain effect that pooled reporting hides,
reinforcing §4.2.

Depends on 5.1. Not actionable this week.

### 5.3 [OPEN] Is cross-sentence context actually unexplored for ATE?

**Not established. Do not assume it.** Cross-sentence and document-level
context for sequence labelling is a worked area in NLP, and ACTER-specific
work exists (Tran et al. 2024, same test set — `data_layout.md` §5.4).

What may be unclaimed is the narrower framing: first-occurrence locality and
multi-scale chunking, on ACTER, under the TermEval 2020 protocol. Establishing
that is **T5's job**. No novelty claim goes in the writeup until T5 reports.

---

## 6. Still to measure

| item | statistic | decides |
|---|---|---|
| §7 T2-1b | wordpieces per sentence, three tokenizers | `max_length`; encoder choice |
| §7 T2-1c | inflation ratio, inside-span vs outside-span | whether domain terms dominate the budget |
| §7 T2-1d | htfl gold term *types* with all occurrences past 128/256/512 | the truncation ceiling (§1.4) |
| §7 T2-3 | term length distribution | T4 candidate n-gram cap |
| §7 T2-4 | term frequency, hapax proportion | C-Value's ceiling; T4 reference-corpus decision |
| §7 T2-5 | term-set overlap, train ↔ htfl | how much of any score is memorisation |
| §7 T2-6 | positive-label proportion; **wind fragment subset (§5.1)** | training invariant; wind filtering |
| §7 T2-7 | nested-term count, split 1-token vs ≥2-token | NOBI's expected effect size |