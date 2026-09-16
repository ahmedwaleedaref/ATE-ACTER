# Document-context fill rate

Fill rate = context tokens available within the sentence's own document,
divided by the window requested (both sides). 1.0 means the window is full;
0.5 means half the requested context does not exist.

Distribution, not a mean -- a mean hides whether the shortfall is a few
starved sentences or all of them. Percentiles are nearest-rank.

## Window 32 tokens per side (64 total)

| domain | role | tokens/doc | n_sent | p0 | p10 | p25 | p50 | p75 | p90 | p100 | full | < 0.5 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | train | 4,237 | 2,002 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.00 |
| equi | dev | 1,712 | 3,090 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.00 |
| wind | train | 11,553 | 6,638 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.00 |
| htfl | test | 292 | 2,432 | 0.50 | 0.50 | 0.78 | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 | 0.00 |

## Window 64 tokens per side (128 total)

| domain | role | tokens/doc | n_sent | p0 | p10 | p25 | p50 | p75 | p90 | p100 | full | < 0.5 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | train | 4,237 | 2,002 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.00 |
| equi | dev | 1,712 | 3,090 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 0.00 |
| wind | train | 11,553 | 6,638 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.00 |
| htfl | test | 292 | 2,432 | 0.27 | 0.50 | 0.64 | 0.91 | 1.00 | 1.00 | 1.00 | 0.43 | 0.01 |

## Window 128 tokens per side (256 total)

| domain | role | tokens/doc | n_sent | p0 | p10 | p25 | p50 | p75 | p90 | p100 | full | < 0.5 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | train | 4,237 | 2,002 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 |
| equi | dev | 1,712 | 3,090 | 0.36 | 0.78 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 0.00 |
| wind | train | 11,553 | 6,638 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.00 |
| htfl | test | 292 | 2,432 | 0.13 | 0.50 | 0.56 | 0.70 | 0.88 | 1.00 | 1.00 | 0.14 | 0.05 |

## Window 210 tokens per side (420 total)

| domain | role | tokens/doc | n_sent | p0 | p10 | p25 | p50 | p75 | p90 | p100 | full | < 0.5 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | train | 4,237 | 2,002 | 0.50 | 0.81 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | 0.00 |
| equi | dev | 1,712 | 3,090 | 0.22 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.76 | 0.01 |
| wind | train | 11,553 | 6,638 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.00 |
| htfl | test | 292 | 2,432 | 0.08 | 0.41 | 0.50 | 0.59 | 0.69 | 0.78 | 1.00 | 0.02 | 0.15 |

