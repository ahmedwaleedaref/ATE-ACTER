# T6 — alignment report

- date: 2026-09-07
- tokenizer: bert-base-cased
- max_length: 256
- labels: without_named_entities, scheme: iob_annotations
- split: train ['corp', 'wind'], dev 'equi', test 'htfl'
- short-sentence filter: <= 2 dataset tokens, domains ['wind'], splits ['train']

The gate is `tests/test_alignment_gate.py` — per-example exact sequence
equality between the labels recovered through `word_ids()` and the labels
`load_domain` returned. Nothing on this page is asserted.

## Positive rate, truncation DISABLED

Comparable to `Data_stats.md` §6.1, which measured the same quantity in
dataset-token space with no tokenizer involved.

| domain | filter | n_sent | dataset tokens | scored positions | zero-piece | positives | rate | §6.1 | delta |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | none | 2,002 | 50,845 | 50,845 | 0 | 6,419 | 0.1262 | 0.1262 | +0.000046 |
| equi | none | 3,090 | 58,203 | 58,203 | 0 | 10,602 | 0.1822 | 0.1822 | -0.000044 |
| wind | ≤2 dropped | 2,590 | 53,491 | 53,450 | 41 | 8,355 | 0.1563 | 0.1562 | +0.000114 |
| wind | none | 6,638 | 57,766 | 57,725 | 41 | 8,407 | 0.1456 | 0.1455 | +0.000139 |
| htfl | none | 2,432 | 55,467 | 55,467 | 0 | 14,442 | 0.2604 | 0.2604 | -0.000029 |

`scored positions` is dataset tokens minus zero-piece tokens: a dataset
token that tokenizes to nothing gets no model position, so its label has
nowhere to land. Where that count is non-zero the rate cannot equal §6.1
exactly — §6.1's denominator is dataset tokens, this one's is positions.

## Truncation loss, max_length 256

A dataset token with no model position — cut by the truncation frontier,
or tokenizing to zero wordpieces — is predicted `O` by
`recover_token_labels`. It can only cost recall. Reported, never asserted.

`sentences truncated` counts untruncated length > max_length, the same
definition `data_layout.md` §8.2 measured. `tokens lost to truncation`
differences the two passes per sentence, so a sentence that merely ENDS
in a zero-piece character is not miscounted as truncated.

| domain | filter | rate | sentences truncated | tokens lost to truncation | zero-piece | positive labels lost |
|---|---|--:|--:|--:|--:|--:|
| corp | none | 0.1262 | 2 | 113 | 0 | 16 |
| equi | none | 0.1822 | 0 | 0 | 0 | 0 |
| wind | ≤2 dropped | 0.1564 | 5 | 231 | 41 | 32 |
| wind | none | 0.1457 | 5 | 231 | 41 | 32 |
| htfl | none | 0.2604 | 1 | 2 | 0 | 0 |
