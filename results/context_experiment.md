# Document-level context (FLERT) — result

`microsoft/deberta-v3-base`, LR 1e-5, 32 context tokens per side, 6 epochs,
seeds 42–46. Context is attended (mask 1) and never scored (label `-100`).
Full record in `docs/EXPERIMENTS.md` E09.

## Verdict

**Context helps on dev decisively and is not established on test at n = 5.**

| effect (paired on seed, df = 4) | equi | htfl |
|---|---|---|
| epochs alone, 6 vs 5 | −0.0000, t 0.01 | +0.0096, t 0.84 |
| **context alone, 32 vs 0** | **+0.0274, t 7.71, 5+/0−** | +0.0136, t 1.88, 5+/0− |

| cell | equi | htfl |
|---|--:|--:|
| ctx 0, 5 ep (E08) | 0.5592 ± 0.0090 | 0.5782 ± 0.0230 |
| ctx 0, 6 ep (control) | 0.5592 ± 0.0056 | 0.5877 ± 0.0178 |
| **ctx 32, 6 ep** | **0.5866 ± 0.0031** | **0.6013 ± 0.0058** |

Ceilings: equi 0.9523, htfl 0.9096. Context reaches 0.616 and 0.661 of them.

A control at context 0 / 6 epochs was required because the first context run
changed epochs too, and 6 epochs is also a different LR schedule (1,722 steps
against 1,435). The epoch effect is zero on equi, so the dev gain is context's
alone; the two isolated effects sum to the combined one, so they do not interact.

## Why 32 and not wider

`results/context_fill_rate.md`. Fraction of sentences receiving a full window:

| window/side | corp | wind | equi | **htfl** |
|---|--:|--:|--:|--:|
| 32 | 0.96 | 0.99 | 0.94 | **0.64** |
| 64 | 0.94 | 0.99 | 0.91 | **0.43** |
| 128 | 0.90 | 0.98 | 0.85 | **0.14** |
| 210 | 0.86 | 0.97 | 0.76 | **0.02** |

htfl documents average 292 tokens against wind's 11,553. **The window is not a
free hyperparameter**: widening it widens the train/test shift in the feature
being added. Measured before implementation, and it predicted the result —
the domain with the least context is the domain where the gain does not hold up.

## Variance

equi std **0.0090 → 0.0031**, htfl **0.0230 → 0.0058**. Best epoch goes from
`4,3,1,1,5` (E08) and `5,6,1,6,2` (control) to `5,5,5,5,6` with context. Six
epochs without context leaves it scattered, so the stabilisation is context's.

## What context actually improved

Largest move by a factor of three is **long terms**: 4+ recall 0.2621 → 0.3368.
That is where training is thinnest — 4+ is 1.6% of training occurrences against
8.1% of the htfl key.

The pre-run prediction was that context would lift *rare* terms most. It did not:
singletons +0.011 against frequent terms +0.021. A singleton occurs once, so
context supplies topic, never a second sighting of the term.

## Correctness

The feature is gated, not assumed. With context on, the positive rate over scored
positions is byte-identical to sentence-level in all four domains — corp 0.126211,
equi 0.182156, wind 0.156257, htfl 0.260371 — with 622,022 context tokens added
and zero context labels reaching the loss. Sentence-label recovery is exact over
all 10,114 examples with truncation off.

The sentence is budgeted against `max_length` before context fills the remainder,
so turning context on truncates nothing the baseline kept: the sentences that lose
labels are the same set either way (`corp_en_02:20`, `corp_en_08:63`,
`wind_en_01:5246/5257`, `wind_en_32:6/7`), and equi and htfl lose nothing.

## Open

- **Within-htfl fill rate.** Whether the gain tracks fill rate across htfl's 190
  documents is what separates "context does not help there" from "context was not
  there". Needs per-document predictions; the term dump is deduplicated
  corpus-wide and cannot answer it.
- **Window 64.** To be run only if 32 helped. It does, on dev.
- **Control artifacts lost.** ctx 0 / 6 ep is console-transcribed only — no run
  JSONs or term lists, so no breakdown and `src/aggregate.py` cannot read it.
