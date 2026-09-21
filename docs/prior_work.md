# Prior work and comparison table

What this project's numbers sit next to, and in what unit. Originally T5.

## TermEval 2020, English, heart-failure test set

Rigouts Terryn et al. (2020), Table 4. Percentages. **Verify against the paper
before use — do not take these on trust from this file.**

| Rank | Team | Method | P | R | F1 incl NE | F1 excl NE |
|---|---|---|--:|--:|--:|--:|
| 1 | TALN-LS2N | BERT binary classification | 34.8 | 70.9 | 46.7 | 45.0 |
| 2 | RACAI | TextRank + TFIDF + embeddings | 42.4 | 40.3 | 41.3 | 39.3 |
| 3 | NYU | Termolator, chunking + TFIDF | 43.5 | 23.6 | 30.6 | 31.5 |
| 4 | e-Terminology | TSR filtering, statistical | 34.4 | 14.2 | 20.1 | 21.4 |
| 5 | NLPLab UQAM | BiLSTM + GloVe | 21.4 | 15.6 | 18.1 | 17.8 |

**The unit is settled for this paper: these are unique-list scores.** ACTER 1.2
shipped annotations only as flat lists of unique terms, so no participant could
have reported a span-level number. This project's headline metric is in the same
unit.

**The supervision asymmetry, confirmed from the source.** Sequential span
annotations arrived in v1.5. Same test set, same metric, **more supervision
available to us** — state it in any comparison, do not gloss it.

English hapax terms were 43% of the gold with NEs included, and recall was lowest
on hapax terms across every system. Consistent with this project's measured 47.7%
(`Data_stats.md` §7.3).

## This project, for the same table

| system | equi (dev) | htfl (test) | unit |
|---|--:|--:|---|
| C-Value baseline | — | 0.2003 | unique-list, ANN |
| deberta-v3-base, sentence-level | 0.5592 ± 0.0090 | 0.5782 ± 0.0230 | unique-list, ANN |
| deberta-v3-base + document context | 0.5866 ± 0.0031 | 0.6013 ± 0.0058 | unique-list, ANN |
| **BIO decode ceiling** | **0.9523** | **0.9096** | unique-list, ANN |

The ceiling row belongs in any published comparison: a reader weighing an F1
against prior work needs to know the cap this annotation scheme imposes.

## Still open — T14

**Tran et al. (2024)**, *Can cross-domain term extraction benefit from
cross-lingual transfer and nested term labeling?* (*Machine Learning*). Needed:
English heart-failure F1 on both keys (they call them ANN and NES), the
BIO-vs-NOBI recall delta, and **which unit they report in** — they do sequence
labelling and could report either. Also their per-length recall, which is the
direct comparison for E08's breakdown.

**Lang et al. (2021)** — the origin of the train/validation/test split used here.

**Secondary:** does either paper discuss `equi` being an unrepresentative
validation domain for this test set (`Data_stats.md` §8.5)? If nobody has, it is
worth a paragraph.

Every number added to the table above must be labelled with its metric unit and
its key. Mixing units silently would make the table wrong.
