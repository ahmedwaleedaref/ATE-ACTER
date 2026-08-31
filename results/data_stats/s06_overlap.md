# s06 -- term-set overlap, training domains vs htfl

Terms-only keys (`without_named_entities`), N = 2339 htfl gold terms; a terms+NE table follows. Training side = corp + equi + wind gold lists / annotated streams combined. Token-sequence matching, lowercased. Every `%` carries its `N`.

## 1. Type overlap -- htfl terms that are also training gold entries

| measure | count | pct |
|---|--:|--:|
| htfl ∩ training gold | 10 | 0.4% (N=2339) |

### by htfl term length

| length | overlap | htfl terms | pct |
|---|--:|--:|--:|
| 1 | 9 | 1029 | 0.9% (N=1029) |
| 2 | 1 | 754 | 0.1% (N=754) |
| 3 | 0 | 366 | 0.0% (N=366) |
| 4+ | 0 | 190 | 0.0% (N=190) |

## 2. Text overlap -- htfl term sequence occurs in the training annotated token stream (any label)

| measure | count | pct |
|---|--:|--:|
| htfl term sequence seen in training text | 99 | 4.2% (N=2339) |
| ... of those, not in any training gold list | 89 | 3.8% (N=2339) |

## 3. Head overlap -- htfl term final token = a training gold term final token

Restricted to htfl terms of ≥2 tokens (N = 1310).

| measure | count | pct |
|---|--:|--:|
| shared final token | 302 | 23.1% (N=1310) |
| ... not already in type overlap | 301 | 23.0% (N=1310) |

## 4. Per training domain alone, vs htfl

| domain | type overlap | pct | text overlap | pct |
|---|--:|--:|--:|--:|
| corp | 2 | 0.1% (N=2339) | 39 | 1.7% (N=2339) |
| equi | 6 | 0.3% (N=2339) | 58 | 2.5% (N=2339) |
| wind | 2 | 0.1% (N=2339) | 46 | 2.0% (N=2339) |

## 5. 20 most frequent htfl terms by (a), with overlap flags

| term | (a) | type | text | head |
|---|--:|:-:|:-:|:-:|
| `patients` | 598 | — | yes | — |
| `heart failure` | 350 | — | — | yes |
| `hf` | 293 | — | — | — |
| `p` | 220 | — | yes | — |
| `ci` | 114 | — | yes | — |
| `mortality` | 109 | — | — | — |
| `outcomes` | 100 | — | — | — |
| `clinical` | 98 | — | — | — |
| `patient` | 73 | — | yes | — |
| `hr` | 71 | — | — | — |
| `hfpef` | 64 | — | — | — |
| `chf` | 60 | — | — | — |
| `follow-up` | 60 | — | yes | — |
| `baseline` | 57 | — | yes | — |
| `significantly` | 57 | — | yes | — |
| `death` | 55 | — | yes | — |
| `cardiac` | 54 | — | — | — |
| `hospitalization` | 53 | — | yes | — |
| `significant` | 52 | — | yes | — |
| `therapy` | 50 | — | — | — |

## 6. Occurrence-weighted type overlap

htfl annotated term occurrences (s05 count (a)) belonging to terms present in the training gold lists.

| set | occurrences | pct |
|---|--:|--:|
| all htfl gold terms | 9243 | 100.0% (N=9243) |
| in training gold types | 31 | 0.3% (N=9243) |

## terms+NE key (both sides) -- comparison only

htfl gold terms+NE: N = 2556. Named entities recur across domains and inflate overlap; keys are not mixed across the two sides.

| measure | count | pct |
|---|--:|--:|
| type overlap | 32 | 1.3% (N=2556) |
| text overlap | 122 | 4.8% (N=2556) |
| ... in text but not in training gold | 90 | 3.5% (N=2556) |
| head overlap (N_mw = 1424) | 356 | 25.0% (N=1424) |
| ... not already in type overlap | 350 | 24.6% (N=1424) |

