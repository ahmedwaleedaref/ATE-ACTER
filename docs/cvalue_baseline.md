# T4 — C-Value Baseline

A no-training statistical reference point for term extraction on `htfl`
(heart failure), giving a lower bound the neural model has to beat. No gold
labels are read anywhere in this pipeline — only raw token streams. Code in
`src/models/cvalue.py` (pure functions) and `src/models/run_cvalue.py`
(orchestration). Config: `configs/cvalue.json`. Tests: `tests/test_cvalue.py`.

---

## 1. Decisions locked (khaled, recorded per Tasks.md T4)

### 1.1 Reference corpus: `target_only`

**Question:** which corpus do the frequency counts for `htfl` come from?

**Finding that shaped this decision:** `htfl` has **no `unannotated_texts/`
directory at all** — confirmed directly against the extracted corpus. Its
45,507 whitespace tokens of annotated text are the *entire* in-domain text
available, not a subset of a larger in-domain pool. This is different from
what `docs/Data_stats.md` §4.5 discusses for the *training* domains (corp,
equi, wind), which do each have a substantial unannotated reserve.

**Decision:** `reference_corpus = "target_only"`. Frequency counts for every
candidate come from `htfl`'s own annotated text only. No text from corp,
equi, or wind — annotated or not — is mixed in.

**Why, given the alternative was legitimate in principle:** the open-decision
note in `data_layout.md` §1 and `Tasks.md` T4 both point out that using
training-domain text as reference material is legitimate practice (TermEval
participants did it). That argument applies to enriching frequency estimates
with more text *of the same kind*. Here the only available enrichment is
**cross-domain** text (corruption/dressage/wind vocabulary), not more
heart-failure text. Mixing it in would let a legal or engineering term's
frequency in an unrelated corpus affect whether a cardiology term clears the
C-Value threshold — a different and noisier proposition than the "more
reference text, same domain" case the open decision was written for.

**Consequence:** `run_cvalue.py` only implements `target_only` and asserts on
any other value. `reference_corpus` is still a config field (not hardcoded)
so a `target_plus_train_annotated` mode can be added later as a second run to
compare against, without touching the pipeline code — but that comparison
should wait for `docs/Data_stats.md` §7 item 4 (hapax proportion), per the
note already in that file: *"Whether that is worth using for C-Value's
frequency counts depends on the hapax proportion measured [...] Khaled's open
decision; the numbers above are the input to it."* That number isn't
computed yet.

**Update — now measured, decision confirmed rather than changed.**
`Data_stats.md` §7.3 gives the hapax proportion under three counts: (a)
decoded gold-BIO spans, (b) the term's token sequence anywhere in the
*annotated* stream, (c) the same over the *whole* corpus (annotated +
unannotated). For htfl specifically:

> "htfl's (c) equals its (b) exactly, since htfl has no unannotated text —
> this confirms `data_layout.md` §1 and validates the counting code."

This is the direct, now-cited confirmation of what §1.1 above found by
inspecting the corpus directly: there is no unannotated htfl text to add as
reference material, for anyone, under any counting method. The
`target_plus_train_annotated` question that motivated waiting for this
number was about whether cross-domain text could still help despite htfl
having no in-domain reserve of its own — §7.3 doesn't directly answer that
narrower question, but it removes the possibility that htfl was quietly
sitting on an untapped in-domain corpus this whole time. `target_only`
stands. `target_plus_train_annotated` remains a real, not-yet-implemented
option per the original reasoning in §1.1, not a resolved question.

### 1.2a Recall ceiling from hapax terms — a limit no threshold or n-gram
tuning can fix

`Data_stats.md` §7.3 also gives htfl's own hapax rate directly, and it is
the domain's worst-case number in the table: **47.7%** under (a) and
**46.0%** under (b)/(c) (identical, per the no-unannotated-text point
above). Compare to corp (44.5%/44.7%/30.1%), equi (41.8%/38.8%/32.3%), and
wind (43.9%/43.0%/30.4%) — every training domain's hapax rate drops
substantially once unannotated reference text is folded in; htfl's cannot,
because there is none to fold in.

`Data_stats.md` states the general consequence plainly: *"C-Value cannot
rank a hapax by definition, and the metric is F1 over a deduplicated list,
which weights a hapax the same as a 598-occurrence term. The statistical
baseline is therefore capped near 70% recall before any code is written."*
That 70% figure is illustrative of the general class of domains it was
computed from (closer to wind's and corp's ~30% hapax-under-(c)); **htfl's
own number is worse, not better** — with roughly 46–48% of its gold terms
occurring exactly once in the only text available to this baseline, the
recall ceiling this pipeline faces on htfl specifically sits closer to
**52–54%**, not 70%, before any threshold or `max_n` choice is made.

This is a structural limit, not a bug and not something §1.4's `max_n`
discussion or a future threshold grid search (§6.3) can move. It belongs
next to the nested-term and discontinuous-term ceilings in
`data_layout.md` §5, and should be reported alongside real F1 once T3
exists, the same way those two are — not discovered by a confused reader
asking why recall tops out well short of 100% no matter what gets tuned.

### 1.2 Candidate generation: stopword-boundary n-grams, no POS tagger

**Question:** classic C-Value candidate generation uses a linguistic filter
(POS-based noun-phrase chunking). `requirements.txt` pins no POS tagger, and
`CLAUDE.md` requires asking before adding a dependency.

**Decision:** no new dependency. Candidates are generated as n-grams
(`min_n`..`max_n` tokens, sliding window, never crossing a sentence
boundary) and kept only if their **first and last token** are neither
punctuation-only, a bare numeric/statistical fragment, nor in a hand-curated
stopword list (`DEFAULT_STOPWORDS` in `cvalue.py`). Interior tokens are
unrestricted — `"abuse of functions"` and `"right to a fair trial"` both
survive despite an interior stopword.

**Consequence, stated plainly:** this is a coarser substitute for POS
chunking. It cannot tell a noun phrase from any other phrase shape that
happens to start and end on a content word — expect lower candidate-list
precision than a POS-filtered system would get. Section 3 below reports the
actual numbers against the real corpus so this isn't just asserted.

**Revisit trigger:** if T4's numbers turn out to gate a decision (e.g. "is
the neural model worth building"), add a POS tagger as a proposed dependency
change and re-ask rather than silently installing one — this file is not
itself the approval.

### 1.3 `min_n = 2`: unigrams excluded from generation, not just scoring

Classic C-Value: `c_value(a) = log2(|a|) * f(a)` when `a` has no longer
candidates containing it. For `|a| = 1`, `log2(1) = 0`, so **every unigram
scores exactly 0**, regardless of frequency. This is a known property of the
method (Frantzi et al., 2000 was designed for multi-word term extraction;
single-word terms need a different technique, e.g. NC-value with context
words — out of scope for T4).

`compute_cvalue` does not special-case this away — a caller who sets
`min_n=1` would just get a pile of zero-scored, never-thresholded unigram
candidates sitting in the output of `generate_candidates`, silently dead
weight. Excluding them at generation time (`min_n=2`, the config default)
makes the limitation visible instead of latent. This is a real recall cost:
the gold list contains single-word terms (`"failure"`, `"myocyte"`, ...) that
this baseline can never produce, on top of the recall ceiling from nested
terms described in `data_layout.md` §5.1 — a separate ceiling, not this one.

### 1.4 `max_n = 4` (final — 5 was tried, measured, and reverted)

**History, kept for the record.** `max_n` was raised from 4 to 5 on a
verbal report that Ahmed's T2-3 work found 5-token gold terms in htfl.
`Data_stats.md` §7.1 ("Term length and frequency, gold unique lists") later
gave the actual measurement:

| domain | key | N | 1-token | cum ≤4 | max |
|---|---|--:|--:|--:|--:|
| htfl | terms | 2,339 | 44.0% | **97.3%** | 8 |
| htfl | terms+NE | 2,556 | 44.3% | 96.1% | 13 |

`Data_stats.md` states the conclusion directly: *"candidate generation
capped at 4 tokens loses 0.3–2.7% of gold terms on the terms-only keys.
Going beyond 5 is not worth the precision cost."* That is weaker support
for moving to 5 than the original verbal report implied — it quantifies
capping at 4 as a *small* recall cost, not a case for extending past it.

**Measured, before reverting** (informal exact-match check against
`htfl_en_tokenised_terms.tsv`, same caveat as §3 — not T3's real scorer):

| | max_n=4 | max_n=5 |
|---|---:|---:|
| raw candidates | 33,284 | 45,821 |
| after `min_frequency>=2` | 3,319 | 3,669 |
| terms in output | 2,454 | 2,494 |
| overall precision | 0.1972 | 0.1941 |
| overall recall | 0.2069 | 0.2069 |
| precision *within the 5-token band alone* | — | **0.0171** (6/350 correct) |

Within the 5-token band itself, 344 of 350 predictions (98.3%) were wrong —
overwhelmingly grammatically incomplete fragments (`"time to death or
first"`, `"used to assess the relationship"`, `"total expenditure on health
per"`), not the numeric-fragment noise §3.1 already handles. The boundary
filter (§1.2) checks only that a candidate's first and last token aren't
stopwords/punctuation/numeric — it has no way to know whether a phrase is
semantically whole, and that gap widens with every extra token allowed.

**Decision: reverted to `max_n = 4`.** The recall upside of 5 was capped at
~2.7% of gold types by Ahmed's own citation, and only 6 of the 350
candidates that capability actually produced were correct. Weighed against
a documented, concentrated precision cost, khaled's call was that the trade
isn't worth it — `max_n = 4` is the committed value going forward, not a
placeholder. Revisiting this again would need either a better candidate
filter (§6.2, POS-based chunking) that could make 5-token candidates
trustworthy, or a length-dependent threshold (§6.3) that scores the 5-token
band separately from shorter candidates rather than sharing one global
cutoff.

---

## 2. Algorithm

```
sentences (tokens, labels)        -- labels unused; loaded via src.statistics.loading
   |
   | [generate_candidates(min_n, max_n, stopwords)]
   v  
n-gram windows per sentence, boundary-filtered, aggregated to
{ candidate_string: Candidate(tokens, freq) }
   |
   | [apply_min_frequency(min_frequency)]
   v  
drop rare types before nesting is computed, so noise can't inflate
another candidate's nested-term set
   |
   | [compute_nesting]
   v  
for every candidate a, the set of strictly-longer candidates b whose
token sequence contains a's as a contiguous sub-sequence (type-level,
not occurrence-level)
   |
   | [compute_cvalue]
   v  
c_value(a) = log2(|a|) * f(a)                                         if T_a empty
c_value(a) = log2(|a|) * (f(a) - (1/|T_a|) * sum_{b in T_a} f(b))     otherwise
   |
   | [threshold_terms(threshold)]
   v  
keep candidates with score >= threshold, dedup, sort
   |
   | [write_term_list]
   v  
one term per line, lowercased, deduplicated, UTF-8,
no header, no index column
```

**Token-string representation.** Candidate strings are built by joining the
annotation-file token stream with single spaces — the same stream `Data_stats.md`
§3 verified byte-identical to `texts_tokenised/`. This means T4's output
matches the **`_tokenised_terms(.tsv|_nes.tsv)`** gold key's spacing
convention (`"expenditure budget line ( s )"`), not the raw `_terms.tsv`
key's (`"expenditure budget line(s)"`). **When T3 lands, score this output
against the tokenised key, or every multi-token term touching punctuation
will miss on formatting alone, not content.**

---

## 3. Result on the real corpus (informal check, not the real scorer)

Run: `python -m src.models.run_cvalue`, config as committed in
`configs/cvalue.json` (`min_n=2, max_n=4, min_frequency=2, threshold=1.0`).
`max_n` was tried at 5 and reverted back to 4 — see §1.4 for the full
before/after comparison and why.

| | value |
|---|---:|
| htfl annotated tokens | 55,467 |
| raw candidates (types) | 33,284 |
| candidates after `min_frequency>=2` | 3,319 |
| terms in output (`threshold>=1.0`) | 2,454 |

Checked informally against `htfl_en_tokenised_terms.tsv` (2,339 entries) by
exact string match — **not** the real T3 scorer, no P/R/F1 claim is made
here, just a sanity check that the pipeline produces signal and not noise:

- overlap: 484 terms
- naive precision ≈ 0.197, naive recall ≈ 0.207

This is in the range expected for a no-training baseline with a
non-linguistic candidate filter — a real lower bound, not competitive with a
POS-filtered C-Value system, let alone the trained model. **Real P/R/F1
against both gold keys, with the ceiling numbers from `data_layout.md` §5.3
alongside them, is T3's job once the harness lands.**

### 3.1 Residual noise, observed and accepted

An early run surfaced numeric/statistical fragments as candidates (e.g.
`"0.11 mg / l"`, `"1.73 m ( 2"`) — clinical papers are full of confidence
intervals and p-values, and bare digits are neither stopwords nor
punctuation-only. Added a boundary check rejecting purely
numeric/decimal/percent/± tokens (`_NUMERIC_ONLY_RE` in `cvalue.py`); this
measurably improved naive precision (0.184 → 0.197) without moving recall.
Not exhaustive — e.g. `"2·9 , p"` (a middle-dot decimal from one paper's
formatting) still slips through. Chasing every such case by hand is a
diminishing-returns substitute for the POS tagger declined in §1.2; this is
recorded as a known limitation rather than patched indefinitely.

---

## 4. Config reference (`configs/cvalue.json`)

| key | meaning |
|---|---|
| `target_domain` | domain to generate candidates and predictions for (`"htfl"`) |
| `reference_corpus` | `"target_only"` — see §1.1; only value currently implemented |
| `min_n` / `max_n` | candidate length bounds in tokens (2–4, final — 5 was tried and reverted, see §1.4) |
| `min_frequency` | drop candidate types below this raw frequency before nesting/scoring |
| `threshold` | minimum C-Value score to appear in the output term list |
| `stopwords_path` | `null` for the built-in list, or a path to a custom one-word-per-line file |
| `output_path` | where the term list is written |
| `run_info_path` | where per-run provenance (counts, config used) is written |

Nothing in `src/models/cvalue.py` or `run_cvalue.py` hardcodes a domain,
threshold, or n-gram bound — every run is reproducible from its config file
alone, per the T4 definition of done.

---

## 5. Still open

- `reference_corpus = "target_plus_train_annotated"` remains a real,
  unimplemented option. `Data_stats.md` §7.3 (hapax proportion) is now
  cited (§1.1) and confirms `target_only` is correct for htfl specifically
  — there is no in-domain material to add, for anyone. It does not resolve
  whether cross-domain (corp/equi/wind) text would still help; that
  narrower question is still open if anyone wants to pursue it.
- `threshold = 1.0` is unvalidated against any scorer — Tasks.md's own plan
  ("swap in the real scorer when T3 lands") applies here directly. Treat the
  current value as a placeholder, not a tuned choice.
- The hapax recall ceiling (§1.2a, ~52–54% for htfl) is not a to-do item —
  it's structural — but it isn't reported anywhere yet alongside real F1.
  Add it next to the `max_recall`/`max_precision` numbers from
  `data_layout.md` §5.3 once T3 produces them (§6.3).

## 5.1 Evaluating now, against a stub — `src/models/evaluate_cvalue.py`

Built ahead of T3 so scoring, threshold-tuning, and the tokenised-vs-raw gold
key question (§2) are exercised end-to-end before the real harness exists,
per Tasks.md's own plan for T4: *"Test against a hand-made 10-line gold
file; swap in the real scorer when T3 lands."*

`src/models/_stub_scorer.py` implements exactly the signature Tasks.md
specifies for T3 — `score(pred_list_path, gold_list_path) -> (precision,
recall, F1)` — as plain set-based overlap over two lowercased,
deduplicated term lists. **It is not T3.** It does not know whatever
edge-case handling Ahmed's real harness has (malformed lines, tie-breaking,
etc.), and `evaluate_cvalue.py` prints a loud warning every run so a stub
number is never mistaken for a reportable one.

`evaluate_cvalue.py` has one marked **SWAP POINT** at the top of the file —
a single import line. Once T3 lands, replace

```python
from src.models._stub_scorer import score
_USING_STUB = True
```

with

```python
from src.eval.<his_module> import score
```

confirming the signature matches. Nothing else in this file, in
`cvalue.py`, or in `run_cvalue.py` needs to change.

**Run against the real corpus today** (stub scorer, both gold keys, correctly
pointed at the *tokenised* key per §2):

```
$ python -m src.models.evaluate_cvalue
```

| gold key | precision | recall | F1 |
|---|---:|---:|---:|
| terms_only | 0.1972 | 0.2069 | 0.2020 |
| terms_plus_nes | 0.2107 | 0.2023 | 0.2064 |

Matches the informal check in §3 to four decimal places — cross-check that
both are computing the same thing. **These are stub numbers.** Once T3
lands: rerun, confirm the numbers move (if they don't move at all,
something is wired wrong), then do the threshold/n-gram grid search from
§6.3 against the real ones.

## 6. Review notes (khaled) — action items for later weeks

Recorded here so they don't live only in chat history. None of these are
implemented yet; §1.3's "excluded, not special-cased away" stance still
holds until one of these is deliberately built.

**6.1 — Unigram ceiling is concrete, not hypothetical, for this domain.**
`htfl` core vocabulary is dense with single-word terms that this baseline
can never produce by construction — `hypertension`, `dyspnea`,
`cardiomyopathy`, `arrhythmia` and similar are exactly the kind of term
`log2(1) = 0` locks out. This sits on top of, and is separate from, the
nested-term recall ceiling in `data_layout.md` §5.1 — two independent causes
of the same symptom (missed single-token gold terms), and any recall
analysis later should be able to tell them apart, not lump them into one
number.

*Candidate fix, not yet built:* a modified C-Value that assigns unigrams a
non-zero score by a different rule than `log2(|a|) * f(a)` — e.g. a flat
frequency-based or corpus-frequency-contrast score, gated by its own
threshold rather than sharing the multi-word one. Worth trying once T3
exists to measure whether it actually buys recall without wrecking
precision, rather than assuming it will.

**6.2 — POS-based candidate generation as the precision fix.**
The stopword-boundary filter is confirmed noisy (§3.1) by design, not by
accident, and §1.4's `max_n=5` experiment showed the noise concentrates
sharply at longer candidate lengths — 98.3% wrong within the 5-token band
before that setting was reverted. This raises the priority of the standard
next step: `spaCy` (or similar) with an `Adj*Noun+`-style chunk pattern in
place of the boundary heuristic, restricting candidates to actual noun
phrases — the kind of filter that could make longer candidate lengths
trustworthy enough to revisit `max_n=5`. This is the dependency change
flagged as a "revisit trigger" in §1.2 — still needs an explicit
ask-and-approve before adding it to `requirements.txt`, per `CLAUDE.md`,
not a silent swap.

**6.3 — Threshold and n-gram bounds: tune only once T3 exists.**
`threshold = 1.0` stays fixed until `T3`'s `score()` is available. Then: a
small grid search over `threshold` (and possibly `min_n`/`max_n`) against
real P/R/F1 on `htfl`, reported against both gold keys per
`data_layout.md` §4, alongside the `max_recall`/`max_precision` ceilings
from §5.3 — a tuned threshold means nothing without the ceiling it's being
tuned under. No tuning against a proxy metric before then.