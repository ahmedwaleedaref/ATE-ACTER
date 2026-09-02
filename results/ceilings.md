# List-metric ceilings (max_recall / max_precision)

- headline metric: unique_list_f1
- secondary metric: exact_span_f1
- scheme: bio
- labels: without_named_entities
- keys scored (one predicted list, both keys): ann, nes
- dangling-I policy: drop (seqeval mode='strict', scheme='IOB2')

| domain | key | precision | recall | f1 | n_spans | n_unique |
|---|---|---|---|---|---|---|
| corp | ANN | 0.9668 | 0.9438 | 0.9552 | 4180 | 904 |
| corp | NES | 0.9668 | 0.7457 | 0.8420 | 4180 | 904 |
| equi | ANN | 0.9294 | 0.9764 | 0.9523 | 8662 | 1204 |
| equi | NES | 0.9294 | 0.7168 | 0.8094 | 8662 | 1204 |
| wind | ANN | 0.9468 | 0.9295 | 0.9381 | 5053 | 1072 |
| wind | NES | 0.9468 | 0.6638 | 0.7805 | 5053 | 1072 |
| htfl | ANN | 0.8887 | 0.9316 | 0.9096 | 9636 | 2452 |
| htfl | NES | 0.8911 | 0.8549 | 0.8726 | 9636 | 2452 |
