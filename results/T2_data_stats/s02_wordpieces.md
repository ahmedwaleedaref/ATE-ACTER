# s02 -- wordpiece length distribution

Four tokenizers over the same annotated portion as s01 (English, sequential IOB, `without_named_entities`). Each sentence tokenized with `is_split_into_words=True` and no truncation; the wordpiece count includes special tokens. `infl` = mean subwords per input token, split by whether the label is inside a span (`B`/`I`) or outside (`O`).

## Summary

| tokenizer | htfl p99 | corpus max | inside-span inflation |
|---|---:|---:|---:|
| bert-base-cased | 118 | 1074 | 1.636 |
| roberta-base | 112 | 399 | 1.445 |
| deberta-v3-base | 106 | 1061 | 1.242 |
| xlm-roberta-base | 120 | 488 | 1.795 |

## bert-base-cased

`google-bert/bert-base-cased` @ `cd5ef92a9fb2f889e972770a36d4ed042daf221e`

| domain | n_sent | min | p50 | p90 | p95 | p99 | max | infl B/I | infl O | infl all | >512 | >256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corp | 2002 | 3 | 27 | 58 | 71 | 115 | 325 | 1.367 | 1.104 | 1.137 | 0 | 2 |
| equi | 3090 | 3 | 22 | 40 | 47 | 62 | 136 | 1.399 | 1.075 | 1.134 | 0 | 0 |
| wind | 6638 | 3 | 3 | 35 | 48 | 85 | 1074 | 1.443 | 1.299 | 1.320 | 1 | 5 |
| htfl | 2432 | 4 | 30 | 64 | 80 | 118 | 260 | 2.043 | 1.215 | 1.431 | 0 | 1 |
| all | 14162 | 3 | 17 | 47 | 59 | 98 | 1074 | 1.636 | 1.174 | 1.257 | 1 | 8 |

Sentences over 512 wordpieces (1):
- `wind_en_32` sentence 6: 1074

20 dataset tokens with the most subwords:

| token | subwords | occurrences | label |
|---|---:|---:|---|
| `...............................................................................................................` | 111 | 1 | O |
| `..............................................................................................................` | 110 | 1 | O |
| `............................................................................................................` | 108 | 1 | O |
| `..........................................................................................................` | 106 | 2 | O |
| `........................................................................................................` | 104 | 1 | O |
| `.....................................................................................................` | 101 | 1 | O |
| `...................................................................................................` | 99 | 1 | O |
| `...............................................................................................` | 95 | 1 | O |
| `........................................................................................` | 88 | 2 | O |
| `.................................................................................` | 81 | 1 | O |
| `.............................................................................` | 77 | 1 | O |
| `......................................................................` | 70 | 1 | O |
| `..............................................................` | 62 | 1 | O |
| `http://www.dnv.com/publications/oilgas_news/oilgasnews-32006/NewDNVstandardsforwindturbinedesign.asp` | 45 | 1 | O |
| `http://www.fluid.mech.ntua.gr/wind/avis/avigotem.html` | 29 | 1 | O |
| `http://www.gl-group.com/brochurepdf/0E504.pdf` | 28 | 1 | O |
| `http://www.ieawind.org/Annex%20XXIII/Subtask2.html` | 27 | 1 | O |
| `A-K-V-E-S-H-C-M-R-B-P-F` | 23 | 1 | B |
| `http://www.beatricewind.co.uk/home/default.asp` | 20 | 1 | O |
| `1-alkyl-2-acetyl-sn-glycerol` | 19 | 1 | B |

## roberta-base

`FacebookAI/roberta-base` @ `e2da8e2f811d1448a5b465c236feacd80ffbac7b`

| domain | n_sent | min | p50 | p90 | p95 | p99 | max | infl B/I | infl O | infl all | >512 | >256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corp | 2002 | 3 | 25 | 55 | 66 | 105 | 309 | 1.168 | 1.062 | 1.076 | 0 | 2 |
| equi | 3090 | 3 | 21 | 39 | 45 | 60 | 125 | 1.315 | 1.058 | 1.105 | 0 | 0 |
| wind | 6638 | 3 | 3 | 34 | 45 | 80 | 399 | 1.344 | 1.219 | 1.237 | 0 | 3 |
| htfl | 2432 | 3 | 27 | 59 | 73 | 112 | 232 | 1.722 | 1.141 | 1.292 | 0 | 0 |
| all | 14162 | 3 | 17 | 44 | 56 | 92 | 399 | 1.445 | 1.121 | 1.179 | 0 | 5 |

20 dataset tokens with the most subwords:

| token | subwords | occurrences | label |
|---|---:|---:|---|
| `http://www.dnv.com/publications/oilgas_news/oilgasnews-32006/NewDNVstandardsforwindturbinedesign.asp` | 38 | 1 | O |
| `http://www.fluid.mech.ntua.gr/wind/avis/avigotem.html` | 25 | 1 | O |
| `A-K-V-E-S-H-C-M-R-B-P-F` | 23 | 1 | B |
| `http://www.ieawind.org/Annex%20XXIII/Subtask2.html` | 22 | 1 | O |
| `http://www.gl-group.com/brochurepdf/0E504.pdf` | 20 | 1 | O |
| `http://www.beatricewind.co.uk/home/default.asp` | 18 | 1 | O |
| `D-80.1-GP.SD.03-A-A-GB` | 17 | 9 | O |
| `www.bwea.com/energy/briefing-sheets.html` | 17 | 1 | O |
| `A-K-E-H-C-M-B-F` | 15 | 1 | B |
| `1-alkyl-2-acetyl-sn-glycerol` | 15 | 1 | B |
| `4.29±0.5-4.55±0.49` | 15 | 1 | O |
| `4.32±0.54-4.31±0.49` | 15 | 1 | O |
| `13.1-34015(17.1-9.3` | 14 | 1 | O |
| `www.bwea.com/edu/calcs.html` | 14 | 1 | O |
| `(123)I-meta-iodobenzylguanidine` | 14 | 1 | B |
| `9.14.5.)-V4-3168` | 12 | 2 | O |
| `faydalanılmıştır` | 12 | 1 | O |
| `http://raphael.mit.edu/xfoil` | 12 | 1 | O |
| `http://www.ntis.gov/ordering.htm` | 12 | 1 | O |
| `CHA₂DS₂VASc` | 12 | 1 | B |

## deberta-v3-base

`microsoft/deberta-v3-base` @ `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`

| domain | n_sent | min | p50 | p90 | p95 | p99 | max | infl B/I | infl O | infl all | >512 | >256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corp | 2002 | 3 | 25 | 54 | 64 | 104 | 298 | 1.100 | 1.040 | 1.048 | 0 | 2 |
| equi | 3090 | 3 | 21 | 38 | 44 | 59 | 116 | 1.177 | 1.040 | 1.065 | 0 | 0 |
| wind | 6638 | 3 | 3 | 32 | 42 | 77 | 1061 | 1.193 | 1.192 | 1.192 | 1 | 4 |
| htfl | 2432 | 3 | 25 | 55 | 69 | 106 | 224 | 1.382 | 1.102 | 1.175 | 0 | 0 |
| all | 14162 | 3 | 16 | 42 | 53 | 88 | 1061 | 1.242 | 1.095 | 1.121 | 1 | 6 |

Sentences over 512 wordpieces (1):
- `wind_en_32` sentence 6: 1061

20 dataset tokens with the most subwords:

| token | subwords | occurrences | label |
|---|---:|---:|---|
| `...............................................................................................................` | 111 | 1 | O |
| `..............................................................................................................` | 110 | 1 | O |
| `............................................................................................................` | 108 | 1 | O |
| `..........................................................................................................` | 106 | 2 | O |
| `........................................................................................................` | 104 | 1 | O |
| `.....................................................................................................` | 101 | 1 | O |
| `...................................................................................................` | 99 | 1 | O |
| `...............................................................................................` | 95 | 1 | O |
| `........................................................................................` | 88 | 2 | O |
| `.................................................................................` | 81 | 1 | O |
| `.............................................................................` | 77 | 1 | O |
| `......................................................................` | 70 | 1 | O |
| `..............................................................` | 62 | 1 | O |
| `http://www.dnv.com/publications/oilgas_news/oilgasnews-32006/NewDNVstandardsforwindturbinedesign.asp` | 36 | 1 | O |
| `http://www.fluid.mech.ntua.gr/wind/avis/avigotem.html` | 24 | 1 | O |
| `http://www.ieawind.org/Annex%20XXIII/Subtask2.html` | 24 | 1 | O |
| `A-K-V-E-S-H-C-M-R-B-P-F` | 23 | 1 | B |
| `http://www.gl-group.com/brochurepdf/0E504.pdf` | 22 | 1 | O |
| `http://www.beatricewind.co.uk/home/default.asp` | 19 | 1 | O |
| `D-80.1-GP.SD.03-A-A-GB` | 17 | 9 | O |

## xlm-roberta-base

`FacebookAI/xlm-roberta-base` @ `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`

| domain | n_sent | min | p50 | p90 | p95 | p99 | max | infl B/I | infl O | infl all | >512 | >256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corp | 2002 | 3 | 31 | 67 | 81 | 131 | 379 | 1.711 | 1.258 | 1.315 | 0 | 3 |
| equi | 3090 | 3 | 25 | 46 | 54 | 72 | 128 | 1.594 | 1.254 | 1.316 | 0 | 0 |
| wind | 6638 | 3 | 4 | 38 | 51 | 86 | 488 | 1.705 | 1.396 | 1.441 | 0 | 3 |
| htfl | 2432 | 3 | 33 | 69 | 86 | 120 | 251 | 2.031 | 1.367 | 1.540 | 0 | 0 |
| all | 14162 | 3 | 19 | 52 | 66 | 102 | 488 | 1.795 | 1.319 | 1.404 | 0 | 6 |

20 dataset tokens with the most subwords:

| token | subwords | occurrences | label |
|---|---:|---:|---|
| `http://www.dnv.com/publications/oilgas_news/oilgasnews-32006/NewDNVstandardsforwindturbinedesign.asp` | 37 | 1 | O |
| `http://www.fluid.mech.ntua.gr/wind/avis/avigotem.html` | 24 | 1 | O |
| `A-K-V-E-S-H-C-M-R-B-P-F` | 23 | 1 | B |
| `http://www.ieawind.org/Annex%20XXIII/Subtask2.html` | 22 | 1 | O |
| `http://www.gl-group.com/brochurepdf/0E504.pdf` | 19 | 1 | O |
| `http://www.beatricewind.co.uk/home/default.asp` | 18 | 1 | O |
| `www.bwea.com/energy/briefing-sheets.html` | 18 | 1 | O |
| `D-80.1-GP.SD.03-A-A-GB` | 15 | 9 | O |
| `A-K-E-H-C-M-B-F` | 15 | 1 | B |
| `www.bwea.com/edu/calcs.html` | 14 | 1 | O |
| `1-alkyl-2-acetyl-sn-glycerol` | 14 | 1 | B |
| `http://raphael.mit.edu/xfoil` | 13 | 1 | O |
| `(123)I-meta-iodobenzylguanidine` | 13 | 1 | B |
| `http://www.ntis.gov/ordering.htm` | 12 | 1 | O |
| `Sandy_Butterfield@NREL.Gov` | 12 | 1 | O |
| `renin-angiotensin-aldosterone` | 11 | 2 | B |
| `ash.sharma@nefco.fi` | 11 | 1 | O |
| `http://www.offshorewindenergy.org` | 11 | 1 | O |
| `Renin-angiotensin-aldosterone` | 11 | 1 | B |
| `renin-angiotensin-aldosteron` | 11 | 1 | B |

