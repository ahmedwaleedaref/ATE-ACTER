# C-Value baseline: unique-list F1 on htfl

This is a first look at a number, not a result.

- metric: unique-list F1 (TermEval 2020 protocol; see configs/eval.yaml)
- domain: htfl
- prediction file: results/term_lists/htfl_cvalue_terms.txt
- keys scored (one predicted list, both keys, tokenised gold): ANN, NES
- entries in the prediction that look non-tokenised (apostrophe/hyphen, no surrounding space; informational, not a failure): 247

**Frequency-corpus decision: UNRECORDED.** Which corpus C-Value counted term frequencies over -- annotated text only, or including the unannotated portion as reference material (Tasks.md, T4) -- is not recorded for this prediction file. Both are legitimate and are not comparable to each other. This is stated as open, not guessed at.

**BIO-tagger ceilings, for orientation only:** on htfl, max_recall is 0.9316 (ANN) / 0.8549 (NES); max_precision is 0.8887 (ANN) / 0.8911 (NES) (results/ceilings.md). These are the caps for a BIO sequence tagger under this annotation scheme. C-Value is not a tagger, produces no spans, and is not bound by them -- listed here for context, not as a denominator for the score below.

| key | precision | recall | f1 | n_predicted | n_gold |
|---|---|---|---|---|---|
| ANN | 0.1941 | 0.2069 | 0.2003 | 2494 | 2339 |
| NES | 0.2073 | 0.2023 | 0.2048 | 2494 | 2556 |
