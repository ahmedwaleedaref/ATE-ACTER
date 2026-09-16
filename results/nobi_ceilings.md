# NOBI list-metric ceilings

Gold NOBI labels are derived from ACTER BIO spans plus the tokenized terms-only key. Same-start nested terms are ignored; overlapping nested terms use left-to-right, longest-match-first resolution.

| domain | key | precision | recall | f1 | spans | unique types | nested spans | nested tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corp | ANN | 0.9680 | 0.9795 | 0.9737 | 5073 | 937 | 893 | 954 |
| corp | NES | 0.9680 | 0.7739 | 0.8601 | 5073 | 937 | 893 | 954 |
| equi | ANN | 0.9180 | 0.9773 | 0.9467 | 9909 | 1220 | 1247 | 1353 |
| equi | NES | 0.9180 | 0.7175 | 0.8055 | 9909 | 1220 | 1247 | 1353 |
| wind | ANN | 0.9486 | 0.9643 | 0.9564 | 6414 | 1110 | 1361 | 1819 |
| wind | NES | 0.9486 | 0.6887 | 0.7980 | 6414 | 1110 | 1361 | 1819 |
| htfl | ANN | 0.8790 | 0.9628 | 0.9190 | 11289 | 2562 | 1653 | 2363 |
| htfl | NES | 0.8813 | 0.8834 | 0.8824 | 11289 | 2562 | 1653 | 2363 |
