# T8 — Seed variance

`bert-base-cased`, lr 3e-05, 5 epochs, effective batch 16, warmup 0.1. Seeds 42, 43, 44, 45, 46.

Per-seed statistic: **best epoch on equi**, ANN unique-list F1. htfl is evaluated once per run, on the best-equi weights.

Encoder sha256 (identical across all five): `963a5823bad227c274d31d4deb86f0ee...`

| seed | best epoch | equi F1 | htfl F1 | collapsed | first 10 sampled indices |
|---|---|---|---|---|---|
| 42 | 4 | 0.5081 | 0.5441 | — | 4538, 2270, 2383, 1987, 4046, 891, 191, 373, 3066, 2524 |
| 43 | 3 | 0.4882 | 0.5356 | — | 1007, 528, 2580, 4216, 203, 4439, 3652, 2361, 1131, 1700 |
| 44 | 3 | 0.4755 | 0.5029 | — | 2557, 356, 3148, 62, 17, 2817, 87, 2970, 1183, 4147 |
| 45 | 5 | 0.4729 | 0.5241 | — | 2485, 3326, 3506, 2311, 3903, 748, 1214, 1826, 3241, 2976 |
| 46 | 3 | 0.4609 | 0.5322 | — | 34, 2901, 4164, 1566, 2132, 4534, 1860, 3599, 2797, 776 |

| domain | mean | std (ddof=1) | ceiling | mean / ceiling |
|---|---|---|---|---|
| equi | 0.4811 | 0.0179 | 0.9523 | 0.505 |
| htfl | 0.5278 | 0.0156 | 0.9096 | 0.580 |

## What this std makes detectable

Against a second config with the same std (s = 0.0179 on equi), `SE(gap) = s·sqrt(2/5) = 0.0113`:

- gap below **0.0179** F1 (t < 1.58) — no difference detected at n=5; not the same as no difference
- gap above **0.0358** F1 (t > 3.16) — too large to be seed luck, scoped to equi, this metric, this n

Both configs contribute their own std in T9; the equal-std figures above are an indication, not the test.
