# s04 -- term length and term frequency (gold unique lists)

Tokenised gold lists only; the non-tokenised `*_terms.tsv` / `*_terms_nes.tsv` variant exists in every domain and is ignored. Length = whitespace tokens; percentiles s01 nearest-rank; every `%` is of the row's `N`.

## Part A -- term length

`terms` / `terms+NE` = gold key excluding / including named entities. Bucket cells `count (pct)`. `upper` = entries with an uppercase char (§4 says the lists are lowercased); `dup` = entries minus distinct lowercased forms.

| domain | key | N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ | cum ≤4 | p50 | p95 | p99 | max | upper | dup |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corp | terms | 926 | 389 (42.0%) | 376 (40.6%) | 115 (12.4%) | 31 (3.3%) | 9 (1.0%) | 4 (0.4%) | 2 (0.2%) | 0 (0.0%) | 98.4% | 2 | 3 | 5 | 7 | 0 | 0 |
| corp | terms+NE | 1172 | 502 (42.8%) | 418 (35.7%) | 143 (12.2%) | 53 (4.5%) | 31 (2.6%) | 11 (0.9%) | 6 (0.5%) | 8 (0.7%) | 95.2% | 2 | 4 | 7 | 13 | 0 | 0 |
| equi | terms | 1146 | 638 (55.7%) | 417 (36.4%) | 70 (6.1%) | 17 (1.5%) | 4 (0.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 99.7% | 1 | 3 | 4 | 5 | 0 | 0 |
| equi | terms+NE | 1561 | 873 (55.9%) | 534 (34.2%) | 102 (6.5%) | 36 (2.3%) | 12 (0.8%) | 3 (0.2%) | 1 (0.1%) | 0 (0.0%) | 99.0% | 1 | 3 | 5 | 7 | 0 | 0 |
| wind | terms | 1092 | 318 (29.1%) | 525 (48.1%) | 198 (18.1%) | 39 (3.6%) | 11 (1.0%) | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) | 98.9% | 2 | 3 | 5 | 7 | 0 | 0 |
| wind | terms+NE | 1529 | 562 (36.8%) | 631 (41.3%) | 243 (15.9%) | 63 (4.1%) | 16 (1.0%) | 3 (0.2%) | 6 (0.4%) | 5 (0.3%) | 98.0% | 2 | 4 | 5 | 10 | 0 | 0 |
| htfl | terms | 2339 | 1029 (44.0%) | 754 (32.2%) | 366 (15.6%) | 128 (5.5%) | 32 (1.4%) | 22 (0.9%) | 6 (0.3%) | 2 (0.1%) | 97.3% | 2 | 4 | 6 | 8 | 0 | 0 |
| htfl | terms+NE | 2556 | 1132 (44.3%) | 787 (30.8%) | 386 (15.1%) | 152 (5.9%) | 42 (1.6%) | 34 (1.3%) | 9 (0.4%) | 14 (0.5%) | 96.1% | 2 | 4 | 6 | 13 | 0 | 0 |

### 10 longest terms per domain and key

- **corp / terms**: `advisory committee on the conduct of members` (7); `closed , with no further action taken` (7); `confiscation of the proceeds of crime` (6); `expenditure budget line ( s )` (6); `laundering of the proceeds of crime` (6); `primacy of the rule of law` (6); `aut dedere aut judicare rule` (5); `breach of a statutory duty` (5); `code of good administrative behaviour` (5); `criminal expertise and liaison unit` (5)
- **corp / terms+NE**: `ad hoc committee for the negotiations of a united nations convention against corruption` (13); `oecd 's convention on combating bribery of foreign officials in international business transactions` (13); `oecd convention on combating bribery of foreign public officials in international business transactions` (13); `convention on the protection of the european communities ' financial interests` (11); `network of expertise on economic , financial and fiscal crime` (10); `evaluation of the activities of the european anti-fraud office` (9); `federal government department of the budget and management control` (9); `treaty on the functioning of the european union` (8); `advisory committee on the conduct of members` (7); `bureau of official ethics and good practice` (7)
- **equi / terms**: `change rein across the diagonal` (5); `grand prix freestyle to music` (5); `half turn on the haunches` (5); `quarter turn on the haunches` (5); `acceptance of the bit` (4); `airs above the ground` (4); `change through the diagonal` (4); `coiling of the loins` (4); `fall to the outside` (4); `flying changes in sequence` (4)
- **equi / terms+NE**: `people for the ethical treatment of animals` (7); `humane society of the united states` (6); `mexican haute école of riders domecq` (6); `royal andalusian school of equestrian art` (6); `annual pony club national championships` (5); `change rein across the diagonal` (5); `escola portuguesa de arte equestre` (5); `françois robichon de la guérinière` (5); `grand prix freestyle to music` (5); `half turn on the haunches` (5)
- **wind / terms**: `enfield-andreau type of horizontal axis wind turbine` (7); `cut - out wind speed` (5); `direct drive synchronous annular generator` (5); `distribution of angle of attack` (5); `free stream velocity of wind` (5); `horizontal-axis wind turbine blade design` (5); `planetary / spur gear system` (5); `prandtl 's tip correction factor` (5); `rate of decrease of momentum` (5); `tip-controlled horizontal axis wind turbines` (5)
- **wind / terms+NE**: `baltic coast environmental research and planning institute of klaipeda university` (10); `department of defense , executive services and communications directorate` (9); `joint implementation project design document form for small-scale projects` (9); `national program of increasing efficiency of energy consumption` (8); `state commission for control on prices and energy` (8); `enfield-andreau type of horizontal axis wind turbine` (7); `graduate school of natural and applied sciences` (7); `journal of wind engineering and industrial aerodynamics` (7); `office of energy efficiency & renewable energy` (7); `vattenfall europe berlin ag & co. kg` (7)
- **htfl / terms**: `heart failure with preserved left ventricular ejection fraction` (8); `l-type voltage-gated ca ( 2 + ) channels` (8); `area under the receiver operating characteristic curve` (7); `fully magnetically levitated left ventricular assist system` (7); `heart failure with a reduced ejection fraction` (7); `hypertensive heart failure with preserved ejection fraction` (7); `raf-mek1 / 2-erk1 / 2 scaffold proteins` (7); `raf-mek1 / 2-erk1 / 2 signalling pathway` (7); `ca ( 2 + ) handling` (6); `ca ( 2 + ) sensitivity` (6)
- **htfl / terms+NE**: `tools for economic analysis of patient management interventions in heart failure cost-effectiveness model` (13); `candesartan in heart failure assessment of reduction in mortality and morbidity` (11); `heart failure - a controlled trial investigating outcomes in exercise training` (11); `heart failure : a controlled trial investigating outcomes of exercise training` (11); `eplerenone post myocardial infarction heart failure efficacy and survival study` (10); `national cardiovascular disease registry practice innovation and clinical excellence registry` (10); `systolic heart failure treatment with the if inhibitor ivabradine trial` (10); `early treatment of atrial fibrillation for stroke prevention trial` (9); `cochrane effective practice and organisation of care taxonomy` (8); `grading of recommendations assessment , development and evaluation` (8)

## Part B -- term frequency (terms-only key)

(a) decoded gold-BIO span occurrences (tokens space-joined, lowercased). (b) the term's token sequence in the annotated token stream, any label. (c) same over the whole corpus (`texts_tokenised/` + `unannotated_texts/`, htfl has none). Every gold term counted, zeros included.

### corp (N=926 terms)

`I` with no preceding `B` during decode: 0. total (b) / total (a) = 6632 / 4116 = 1.61.

| stat | (a) annotated occ | (b) surface annotated | (c) surface corpus |
|---|--:|--:|--:|
| total occurrences | 4116 | 6632 | 19072 |
| hapax (freq = 1) | 412 | 414 | 279 |
| hapax % (N=926) | 44.5% | 44.7% | 30.1% |
| p50 freq | 1 | 2 | 3 |
| p90 freq | 10 | 15 | 45 |
| p99 freq | 42 | 98 | 292 |
| max freq | 300 | 485 | 873 |

10 most frequent terms by (a):

| term | (a) | (b) | (c) |
|---|--:|--:|--:|
| `corruption` | 300 | 485 | 873 |
| `fraud` | 81 | 98 | 315 |
| `financial interests` | 63 | 66 | 204 |
| `rules` | 57 | 77 | 204 |
| `decision` | 55 | 98 | 185 |
| `companies` | 46 | 46 | 77 |
| `legal person` | 46 | 46 | 61 |
| `anti-corruption` | 44 | 146 | 241 |
| `political` | 42 | 64 | 278 |
| `regulation` | 42 | 43 | 152 |

### equi (N=1146 terms)

`I` with no preceding `B` during decode: 0. total (b) / total (a) = 12606 / 8546 = 1.48.

| stat | (a) annotated occ | (b) surface annotated | (c) surface corpus |
|---|--:|--:|--:|
| total occurrences | 8546 | 12606 | 21619 |
| hapax (freq = 1) | 479 | 445 | 370 |
| hapax % (N=1146) | 41.8% | 38.8% | 32.3% |
| p50 freq | 2 | 2 | 3 |
| p90 freq | 15 | 19 | 32 |
| p99 freq | 79 | 147 | 261 |
| max freq | 1040 | 1250 | 2429 |

10 most frequent terms by (a):

| term | (a) | (b) | (c) |
|---|--:|--:|--:|
| `horse` | 1040 | 1080 | 1870 |
| `rider` | 284 | 293 | 527 |
| `horses` | 173 | 184 | 252 |
| `walk` | 130 | 182 | 271 |
| `trot` | 123 | 204 | 327 |
| `ride` | 116 | 120 | 194 |
| `riding` | 116 | 173 | 261 |
| `training` | 108 | 162 | 295 |
| `balance` | 107 | 112 | 177 |
| `riders` | 89 | 97 | 152 |

### wind (N=1092 terms)

`I` with no preceding `B` during decode: 2. total (b) / total (a) = 9304 / 4982 = 1.87.

| stat | (a) annotated occ | (b) surface annotated | (c) surface corpus |
|---|--:|--:|--:|
| total occurrences | 4982 | 9304 | 31306 |
| hapax (freq = 1) | 479 | 470 | 332 |
| hapax % (N=1092) | 43.9% | 43.0% | 30.4% |
| p50 freq | 1 | 2 | 3 |
| p90 freq | 10 | 18 | 46 |
| p99 freq | 46 | 112 | 396 |
| max freq | 212 | 642 | 4560 |

10 most frequent terms by (a):

| term | (a) | (b) | (c) |
|---|--:|--:|--:|
| `blade` | 212 | 521 | 969 |
| `blades` | 114 | 143 | 350 |
| `rotor` | 99 | 284 | 735 |
| `airfoil` | 96 | 100 | 174 |
| `tip-speed ratio` | 65 | 65 | 67 |
| `power` | 63 | 399 | 1383 |
| `wind turbine` | 60 | 129 | 1073 |
| `wind turbines` | 60 | 112 | 541 |
| `power coefficient` | 57 | 70 | 83 |
| `wind` | 57 | 642 | 4560 |

### htfl (N=2339 terms)

`I` with no preceding `B` during decode: 1. total (b) / total (a) = 13890 / 9243 = 1.50.

| stat | (a) annotated occ | (b) surface annotated | (c) surface corpus |
|---|--:|--:|--:|
| total occurrences | 9243 | 13890 | 13890 |
| hapax (freq = 1) | 1115 | 1076 | 1076 |
| hapax % (N=2339) | 47.7% | 46.0% | 46.0% |
| p50 freq | 1 | 2 | 2 |
| p90 freq | 7 | 10 | 10 |
| p99 freq | 40 | 70 | 70 |
| max freq | 598 | 647 | 647 |

10 most frequent terms by (a):

| term | (a) | (b) | (c) |
|---|--:|--:|--:|
| `patients` | 598 | 598 | 598 |
| `heart failure` | 350 | 530 | 530 |
| `hf` | 293 | 346 | 346 |
| `p` | 220 | 229 | 229 |
| `ci` | 114 | 114 | 114 |
| `mortality` | 109 | 127 | 127 |
| `outcomes` | 100 | 108 | 108 |
| `clinical` | 98 | 101 | 101 |
| `patient` | 73 | 73 | 73 |
| `hr` | 71 | 89 | 89 |

