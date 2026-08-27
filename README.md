# ate-acter

Automatic Term Extraction (ATE): given a specialised document, output the list
of technical terms it contains. Approach: BIO token-level sequence labelling
with a pretrained transformer encoder.

## Dataset

[ACTER](https://github.com/AylaRT/ACTER) — Annotated Corpora for Term
Extraction Research, Ghent University. Pinned to tag **v1.5**.

Benchmark: the TermEval 2020 shared task. Four domains (corruption, dressage,
heart failure, wind energy), three languages.

## Train / test split

- **Train:** corruption + dressage + wind energy
- **Test:** heart failure (held-out domain)
- **Track:** English
- **Metric:** F1 over a deduplicated list of unique terms

## Licensing

ACTER is released under **CC BY-NC-SA 4.0** (non-commercial, share-alike).
Nothing under `data/` is committed to this repository. Any downstream
commercial use of models trained on ACTER is an open question to resolve
separately.
