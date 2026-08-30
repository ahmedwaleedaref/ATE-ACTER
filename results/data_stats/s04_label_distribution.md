# s04 -- label distribution and class imbalance

Dataset-token space, no tokenizer. B / I / O counted separately -- B is the term-*occurrence* count (item 4 needs it). **Part A is a fixed invariant of ACTER v1.5, not a modelling signal**: a different positive rate downstream means broken label alignment.

## Part A -- per domain and pooled

B/I/O `%` is of the row's `n_tokens`; zero-positive `%` is of its `n_sentences`.

| domain | n_docs | n_sent | n_tokens | B | I | O | pos. rate | mean occ. len | zero-pos sents |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | 12 | 2002 | 50845 | 4180 (8.2%) | 2239 (4.4%) | 44426 (87.4%) | 0.1262 | 1.54 | 607 (30.3%) |
| equi | 34 | 3090 | 58203 | 8662 (14.9%) | 1940 (3.3%) | 47601 (81.8%) | 0.1822 | 1.22 | 467 (15.1%) |
| wind | 5 | 6638 | 57766 | 5053 (8.7%) | 3354 (5.8%) | 49359 (85.4%) | 0.1455 | 1.66 | 4905 (73.9%) |
| htfl | 190 | 2432 | 55467 | 9636 (17.4%) | 4806 (8.7%) | 41025 (74.0%) | 0.2604 | 1.50 | 443 (18.2%) |
| pooled | 241 | 14162 | 222281 | 27531 (12.4%) | 12339 (5.6%) | 182411 (82.1%) | 0.1794 | 1.45 | 6422 (45.3%) |

## Part B -- short (<=2 tokens) vs prose (>=3 tokens)

| domain | bucket | n_sent | % dom sent | n_tok | % dom tok | B | I | O | pos. rate | B % of domain B |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | short | 228 | 11.4% | 429 | 0.8% | 14 | 3 | 412 | 0.0396 | 0.3% |
| corp | prose | 1774 | 88.6% | 50416 | 99.2% | 4166 | 2236 | 44014 | 0.1270 | 99.7% |
| equi | short | 255 | 8.3% | 470 | 0.8% | 77 | 19 | 374 | 0.2043 | 0.9% |
| equi | prose | 2835 | 91.7% | 57733 | 99.2% | 8585 | 1921 | 47227 | 0.1820 | 99.1% |
| wind | short | 4048 | 61.0% | 4275 | 7.4% | 44 | 8 | 4223 | 0.0122 | 0.9% |
| wind | prose | 2590 | 39.0% | 53491 | 92.6% | 5009 | 3346 | 45136 | 0.1562 | 99.1% |
| htfl | short | 338 | 13.9% | 674 | 1.2% | 2 | 0 | 672 | 0.0030 | 0.0% |
| htfl | prose | 2094 | 86.1% | 54793 | 98.8% | 9634 | 4806 | 40353 | 0.2635 | 100.0% |

**Decision (wind).** The short bucket holds **0.9%** of wind's total B count: under ~5% -- structural noise, filterable at training time. This statistic measures; it does not filter.

## 20 most frequent wind short-bucket sentences

| count | tokens | labels |
|--:|---|---|
| 3499 | . | O |
| 13 | 2 | O |
| 9 | λ | O |
| 9 | D-80.1-GP.SD.03-A-A-GB | O |
| 8 | 2002-09-10 | O |
| 6 | 8 | O |
| 6 | 1 | O |
| 5 | 3 | O |
| 5 | 4 | O |
| 5 | Cp = | B O |
| 5 | h | O |
| 4 | 5 . | O O |
| 4 | i | O |
| 4 | 5 | O |
| 4 | 2 . | O O |
| 4 | • | O |
| 4 | 3 . | O O |
| 4 | 4 . | O O |
| 4 | 6 . | O O |
| 3 | v | O |

