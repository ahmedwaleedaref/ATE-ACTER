# Model selection — week 2 final

**Selected: `microsoft/deberta-v3-base`, LR 1e-5, 5 epochs, seed 42.**

## The three encoders

Each at its own selected config — LR tuned per encoder, as T10 requires.
Mean ± std over seeds 42–46, per-seed statistic = best epoch on equi,
ANN unique-list F1.

| encoder | config | equi | htfl |
|---|---|--:|--:|
| bert-base-cased | 3e-5 / 5ep | 0.4811 ± 0.0179 | 0.5278 ± 0.0156 |
| roberta-base | 3e-5 / 5ep | 0.5063 ± 0.0266 | 0.5632 ± 0.0124 |
| **deberta-v3-base** | **1e-5 / 5ep** | **0.5590 ± 0.0086** | **0.5784 ± 0.0228** |

Ceilings: equi 0.9523, htfl 0.9096 (`results/T3_eval_harness/ceilings.md`). DeBERTa reaches
0.587 of the equi ceiling and 0.636 of htfl's.

DeBERTa wins both domains, which resolves T10's logged prediction in its
favour: `data_layout.md` §8.1 predicted it on fragmentation grounds — deberta
splits htfl terms at 1.382 against bert's 2.043, and first-subword labelling
makes the first piece carry the classification.

## The run chosen for each encoder

Each is the single concrete model behind its row. All have artifacts under
`results/runs/`.

| encoder | seed | best epoch | equi | htfl | rank in its cell | chosen by |
|---|--:|--:|--:|--:|---|---|
| bert-base-cased | 42 | 4 | 0.5081 | 0.5441 | #1 equi, #1 htfl | highest equi |
| roberta-base | 46 | 2 | 0.5276 | 0.5710 | #2 equi, #2 htfl | balanced |
| **deberta-v3-base** | **42** | **4** | **0.5657** | **0.6039** | **#1 equi, #1 htfl** | highest equi |

bert and deberta were picked by the highest-equi rule; both happened to top
htfl as well. roberta's seed 46 was picked as the balanced run instead — seed
44 tops its equi (0.5328) but is last on htfl, and seed 42 tops its htfl
(0.5755) but is last on equi.

**None of these per-seed numbers is its config's score.** Each is a maximum
over five on dev and carries that bias. The config figures are the means in
the table above.

## Records

| | |
|---|---|
| bert | `results/t9_selected.md`, docs/EXPERIMENTS.md E03 |
| deberta | `results/deberta_selected.md`, docs/EXPERIMENTS.md E04–E05 |
| roberta | `results/roberta_selected.md`, docs/EXPERIMENTS.md E06–E07 |

## Three caveats this table cannot show

**The comparison spans two environments.** bert and roberta ran locally (RTX
3050, torch 2.14.0+cu130, Python 3.14.4); deberta ran on Colab and Kaggle T4s
at torch 2.10–2.11, Python 3.13. `transformers==5.16.1` throughout. Only the
bert-vs-roberta pair is confound-free — and that is the pair whose winner was
not selected.

**roberta's LR is not really tuned.** Its grid (E07, 30 runs) returned all six
cells inside the tie set on equi, every paired t between 0.38 and 1.15, with
the spread of cell means a third of the within-cell noise. equi at n = 5 has no
power over roberta's hyperparameters. Its config is bert's, chosen from a tie
set for continuity.

**The selected model cannot train on this machine.** deberta-v3-base in fp32
needs 2.94 GB for weights, gradients and two AdamW moments against 3.65 GB
usable, and OOMs at every batch size tried, with gradient checkpointing and
`expandable_segments` on. Its checkpoint also ships fp16, which silently NaNs
under AdamW without the `dtype=torch.float32` load added in `build_model`.
Every deberta run so far has been on rented T4s.

## What T11 needs next

T11 wants recall by term length, recall by gold frequency, and exact-span F1
alongside unique-list F1, over all five seeds of the selected model. None of
that is in any run JSON — they hold eight aggregate floats per domain. It needs
per-span predictions dumped, which means re-running deberta `{1e-5, 5}` on
Colab or Kaggle with `--save-weights` or a prediction dump, and wiring the
`score_exact_spans` call that `evaluate()` currently accepts and ignores
(`src/models/train_loop.py:111`, deferred there "until T11 needs it").
