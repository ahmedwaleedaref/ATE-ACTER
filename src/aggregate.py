"""T8 aggregation: the five seed runs -> results/t8_seed_variance.md.

Generated scaffolding. It reads what ``run_train.py`` wrote and does arithmetic
on it; it never re-scores anything, and it never touches a model.

Run the five seeds and aggregate:
    for s in 42 43 44 45 46; do python -m src.models.run_train --seed $s --reason "T8 seed variance" || break; done && python -m src.aggregate

Reads:  results/runs/seed_*.json
Writes: results/t8_seed_variance.md   (also printed to stdout)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNS_JSON = _REPO_ROOT / "results" / "runs"
_OUT = _REPO_ROOT / "results" / "t8_seed_variance.md"

SEEDS = (42, 43, 44, 45, 46)

# results/ceilings.md, ANN rows -- the max_recall/max_precision list ceilings.
# Measured values, copied verbatim. A model's mean is only interpretable as a
# fraction of what the list metric can award on that domain at all.
CEILING = {"equi": 0.9523, "htfl": 0.9096}

# T8, comparing ONE cell's mean against nothing else: the unpaired bands, which
# with equal stds are the 1-std and 2-std gaps exactly.
T_TIE, T_SERIOUS = 1.58, 3.16

# E03 step 2 compares two cells that ran the SAME five seeds. set_seed(n) runs
# before from_pretrained, so seed n has an identical classifier-head init and an
# identical shuffle order in every cell -- same subject, two treatments. The test
# is therefore paired: five per-seed differences, one-sample t against zero,
# df = 4. These are real t-distribution points, not the rescaled std bands above.
T_PAIRED_SUGGESTIVE, T_PAIRED_SERIOUS = 2.132, 2.776   # two-tailed p = 0.10, 0.05
_PAIRED_DF = 4


# A grid is (learning_rate, num_epochs) -> the directory holding that cell's five
# seeds. One cell of each grid is inherited rather than recomputed, so it points
# somewhere else: bert's {3e-5, 5} is E02, and deberta's is the T10 encoder cell.
GRIDS = {
    "bert": {
        "model": "bert-base-cased",
        "lrs": (2e-5, 3e-5, 5e-5),
        "epochs": (3, 5),
        "inherited": {(3e-5, 5): ("results/runs", "E02")},
        "root": "t9",
        "cell": "lr{lr:g}_e{ep}",
        # historical name: EXPERIMENTS.md E03 and results/t9_selected.md cite it
        "out": "t9_grid",
    },
    "deberta": {
        "model": "microsoft/deberta-v3-base",
        # 3e-5 is NOT a cell of this grid. {3e-5,5} is E04 -- a different question
        # (deberta at BERT's config, for the encoder comparison) whose run JSONs
        # were lost with the Colab session, and {3e-5,3} was never run. E04's
        # numbers are recorded in EXPERIMENTS.md and cited there, not inherited
        # here: a grid cell has to be backed by artifacts this tool can read.
        "lrs": (1e-5, 2e-5),
        "epochs": (3, 5),
        "inherited": {},
        "root": "deberta_grid",
        "cell": "deberta_lr{lr:g}_e{ep}",
        "out": "deberta_grid",
    },
    "roberta": {
        "model": "FacebookAI/roberta-base",
        # Same axes bert got in T9/E03, so the two encoders are tuned alike.
        # {3e-5, 5} is E06 and is reused, not recomputed: it ran on this machine
        # at this stack, so unlike deberta's E04 it has artifacts aggregate reads.
        "lrs": (1e-5, 2e-5, 3e-5),
        "epochs": (3, 5),
        "inherited": {(3e-5, 5): ("results/runs/t10/roberta-base", "E06")},
        "root": "roberta_grid",
        "cell": "roberta_lr{lr:g}_e{ep}",
        "out": "roberta_grid",
    },
}


def cell_dir(encoder: str, lr: float, epochs: int) -> Path:
    g = GRIDS[encoder]
    if (lr, epochs) in g["inherited"]:
        return _REPO_ROOT / g["inherited"][(lr, epochs)][0]
    return _RUNS_JSON / g["root"] / g["cell"].format(lr=lr, ep=epochs)


# kept so nothing that already calls it breaks
T9_LRS, T9_EPOCHS = GRIDS["bert"]["lrs"], GRIDS["bert"]["epochs"]


def t9_cell_dir(lr: float, epochs: int) -> Path:
    return cell_dir("bert", lr, epochs)


def load_runs(runs_dir: Path | None = None) -> list[dict]:
    runs_dir = runs_dir or _RUNS_JSON
    paths = sorted(runs_dir.glob("seed_*.json"))
    assert paths, f"no seed_*.json in {runs_dir} -- run the five seeds first"
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    del paths

    found = sorted(r["seed"] for r in runs)
    assert found == list(SEEDS), (
        f"T8 is five fixed seeds {list(SEEDS)}; found {found}. "
        "A partial set is not a seed-variance result -- do not aggregate it.")

    # five runs of ONE config, or the std is measuring the config axis too
    baseline = {k: v for k, v in runs[0]["config"].items() if k != "seed"}
    for r in runs[1:]:
        other = {k: v for k, v in r["config"].items() if k != "seed"}
        differing = sorted(k for k in baseline.keys() | other.keys()
                           if baseline.get(k) != other.get(k))
        assert not differing, (
            f"seed {r['seed']} was run at a different config than seed {runs[0]['seed']}: "
            f"{differing}. These are not five samples of one config.")

    # the tripwire from item 5: same pretrained encoder in every run
    hashes = {r["seed"]: r["encoder_weight_hash"] for r in runs}
    if len(set(hashes.values())) != 1:
        lines = "\n".join(f"    seed {s}: {h}" for s, h in sorted(hashes.items()))
        raise AssertionError(
            "encoder_weight_hash differs across seeds. The pretrained encoder was "
            "not identical between runs, so the spread below is not seed variance:\n"
            + lines)
    return sorted(runs, key=lambda r: r["seed"])


def stats(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)  # ddof=1


def cell_stats(runs: list[dict]) -> dict:
    """mean/std of one cell on both domains, plus what step 2 needs from it."""
    equi = [r["best_equi_f1"] for r in runs]
    htfl = [r["htfl_f1"] for r in runs if r["htfl_f1"] is not None]
    out = {"n": len(runs),
           "equi_mean": statistics.fmean(equi), "equi_std": statistics.stdev(equi),
           "collapsed": [r["seed"] for r in runs if r["collapsed"]],
           "best_epochs": [r["best_epoch"] for r in runs]}
    if len(htfl) == len(runs):
        out["htfl_mean"] = statistics.fmean(htfl)
        out["htfl_std"] = statistics.stdev(htfl)
    return out


def t9_grid(encoder: str = "bert") -> None:
    """Step 1: the LR x epochs table. Measurement only -- no cell is compared to
    another here, and nothing is selected."""
    g = GRIDS[encoder]
    T9_LRS, T9_EPOCHS = g["lrs"], g["epochs"]
    cells, missing = {}, []
    for lr in T9_LRS:
        for ep in T9_EPOCHS:
            d = cell_dir(encoder, lr, ep)
            try:
                cells[(lr, ep)] = cell_stats(load_runs(d))
            except AssertionError as e:
                missing.append(f"  lr {lr:g} / {ep} epochs -> {e}")
    if missing:
        print("cells not yet measured:\n" + "\n".join(missing) + "\n")

    out = [f"# Hyperparameter grid, step 1 ({g['model']})", "",
           "Each cell is `mean ± std` of five best-epoch equi ANN F1 scores, "
           "ddof=1. Measurement only: nothing here is a comparison.", "",
           "| | " + " | ".join(f"epochs {e}" for e in T9_EPOCHS) + " |",
           "|---|" + "---|" * len(T9_EPOCHS)]
    for lr in T9_LRS:
        row = [f"**LR {lr:g}**"]
        for ep in T9_EPOCHS:
            c = cells.get((lr, ep))
            if c is None:
                row.append("—")
            else:
                mark = f" **{len(c['collapsed'])}/{c['n']} COLLAPSED**" if c["collapsed"] else ""
                note = (f" ({g['inherited'][(lr, ep)][1]})"
                        if (lr, ep) in g["inherited"] else "")
                row.append(f"{c['equi_mean']:.4f} ± {c['equi_std']:.4f}{note}{mark}")
        out.append("| " + " | ".join(row) + " |")

    out += ["", "## Per-cell detail", "",
            "| LR | epochs | equi mean ± std | htfl mean ± std | best epochs | collapsed |",
            "|---|---|---|---|---|---|"]
    for (lr, ep), c in sorted(cells.items()):
        htfl = (f"{c['htfl_mean']:.4f} ± {c['htfl_std']:.4f}"
                if "htfl_mean" in c else "— (not scored)")
        out.append(f"| {lr:g} | {ep} | {c['equi_mean']:.4f} ± {c['equi_std']:.4f} | {htfl} | "
                   f"{', '.join(str(b) for b in c['best_epochs'])} | "
                   f"{len(c['collapsed']) or '—'} |")
    out += ["", "Selection happens on equi. htfl is recorded per run and must not "
            "enter the comparison in step 2.", ""]

    text = "\n".join(out) + "\n"
    dest = _REPO_ROOT / "results" / f"{g['out']}_step1.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {dest.relative_to(_REPO_ROOT)}")


def paired_compare(ref: list[dict], other: list[dict]) -> dict:
    """Paired t on five per-seed differences, plus the unpaired t for contrast.

    The per-seed difference is the whole dataset once it is taken: the seed's own
    strength appears on both sides and cancels, which is exactly the spread the
    unpaired test would otherwise charge to its own uncertainty, twice.
    """
    a = {r["seed"]: r["best_equi_f1"] for r in other}
    b = {r["seed"]: r["best_equi_f1"] for r in ref}
    assert set(a) == set(b), (
        f"cells do not share seeds -- {sorted(set(a) ^ set(b))} in one but not the "
        "other. Pairing is only valid on the same seeds.")
    seeds = sorted(a)
    d = [b[s] - a[s] for s in seeds]

    mean_d = statistics.fmean(d)
    s_d = statistics.stdev(d)
    se_paired = s_d / math.sqrt(len(d))
    # identical differences would divide by zero; report it rather than crash
    t_paired = abs(mean_d) / se_paired if se_paired > 0 else float("inf")

    s_a, s_b = statistics.stdev(list(a.values())), statistics.stdev(list(b.values()))
    se_unpaired = math.sqrt(s_a**2 / len(a) + s_b**2 / len(b))
    t_unpaired = abs(mean_d) / se_unpaired if se_unpaired > 0 else float("inf")

    n_pos = sum(1 for x in d if x > 0)
    n_neg = sum(1 for x in d if x < 0)
    # all five sharing a sign has probability 2 * (1/2)^5 under the null. At n=5
    # this is the strongest statement available that assumes no distribution --
    # worth having, because df=4 cannot verify the t-test's normality.
    sign_p = 0.0625 if (n_pos == len(d) or n_neg == len(d)) else None

    return {"seeds": seeds, "d": d, "mean_d": mean_d, "s_d": s_d,
            "t_paired": t_paired, "t_unpaired": t_unpaired,
            "se_paired": se_paired, "se_unpaired": se_unpaired,
            "n_pos": n_pos, "n_neg": n_neg, "sign_p": sign_p}


def reading(t: float) -> str:
    if t < T_PAIRED_SUGGESTIVE:
        return "no difference detected"
    if t < T_PAIRED_SERIOUS:
        return "suggestive"
    return "take seriously"


def t9_compare(encoder: str = "bert") -> None:
    """Step 2. Highest-mean cell paired against each of the others."""
    g = GRIDS[encoder]
    T9_LRS, T9_EPOCHS = g["lrs"], g["epochs"]
    cells = {}
    for lr in T9_LRS:
        for ep in T9_EPOCHS:
            cells[(lr, ep)] = load_runs(cell_dir(encoder, lr, ep))
    assert len(cells) == len(T9_LRS) * len(T9_EPOCHS), "step 2 needs every cell"

    collapsed = {k: [r["seed"] for r in v if r["collapsed"]] for k, v in cells.items()}
    means = {k: statistics.fmean(r["best_equi_f1"] for r in v) for k, v in cells.items()}
    ref = max(means, key=means.get)

    out = [f"# Hyperparameter grid, step 2 — paired comparison ({g['model']})", "",
           f"Reference cell: **LR {ref[0]:g}, {ref[1]} epochs** — highest equi mean "
           f"({means[ref]:.4f}). Compared against the other {len(cells) - 1}.", "",
           "Paired t on five per-seed differences, df = 4. The same seeds ran in "
           "every cell with identical head init and shuffle order, so the seed's own "
           "strength cancels in the difference. Unpaired t shown alongside for "
           "contrast only — it discards the pairing and is not the test.", "",
           f"Bands: t < {T_PAIRED_SUGGESTIVE} not detected, "
           f"{T_PAIRED_SUGGESTIVE}–{T_PAIRED_SERIOUS} suggestive, "
           f"> {T_PAIRED_SERIOUS} take seriously (two-tailed p = 0.10, 0.05).", "",
           "| cell | mean | gap to ref | s_d | **t paired** | t unpaired | signs | reading |",
           "|---|--:|--:|--:|--:|--:|--:|---|"]

    ties = []
    for k in sorted(cells, key=lambda k: -means[k]):
        if k == ref:
            out.append(f"| **LR {k[0]:g} / {k[1]}ep (ref)** | **{means[k]:.4f}** | — | — | — | — | — | — |")
            continue
        if collapsed[ref] or collapsed[k]:
            out.append(f"| LR {k[0]:g} / {k[1]}ep | {means[k]:.4f} | {means[ref]-means[k]:+.4f} "
                       f"| — | — | — | — | **collapse in cell — no t** |")
            continue
        c = paired_compare(cells[ref], cells[k])
        sign = f"{c['n_pos']}+/{c['n_neg']}−"
        if c["sign_p"]:
            sign += f" (p={c['sign_p']})"
        verdict = reading(c["t_paired"])
        if verdict == "no difference detected":
            ties.append(k)
        out.append(f"| LR {k[0]:g} / {k[1]}ep | {means[k]:.4f} | {c['mean_d']:+.4f} "
                   f"| {c['s_d']:.4f} | **{c['t_paired']:.2f}** | {c['t_unpaired']:.2f} "
                   f"| {sign} | {verdict} |")

    out += ["", "## Per-seed differences (reference minus cell)", "",
            "| cell | " + " | ".join(f"seed {s}" for s in sorted(
                r["seed"] for r in cells[ref])) + " |",
            "|---|" + "---|" * len(cells[ref])]
    for k in sorted(cells, key=lambda k: -means[k]):
        if k == ref or collapsed[ref] or collapsed[k]:
            continue
        c = paired_compare(cells[ref], cells[k])
        out.append(f"| LR {k[0]:g} / {k[1]}ep | "
                   + " | ".join(f"{x:+.4f}" for x in c["d"]) + " |")

    n_tests = len(cells) - 1
    # Bonferroni: the same family-wise 0.05 split across the tests actually run.
    # Computed, not quoted -- it moves with the number of cells.
    from scipy import stats
    t_bonf = stats.t.ppf(1 - 0.05 / (2 * n_tests), _PAIRED_DF)
    out += ["", "## Multiple comparisons", "",
            f"{n_tests} tests were run against one reference. Bonferroni at "
            f"df = {_PAIRED_DF} would demand t > {t_bonf:.3f} rather than "
            f"{T_PAIRED_SERIOUS}. The raw t is reported above and the count is stated "
            "here rather than a correction being applied silently; judge the family "
            "accordingly.", ""]

    if ties:
        # Tiebreak: highest mean on the selection domain. The reference cell is the
        # highest mean by construction, so it takes the tie set. (The earlier rule
        # was "cheapest in the tie set"; it did not change T9's outcome, whose tie
        # set member cost the same 5 epochs as its reference.)
        pool = ties + [ref]
        cheapest = max(pool, key=lambda k: means[k])
        out += ["## Tie set", "",
                "Not distinguishable from the reference at n = 5: "
                + ", ".join(f"LR {k[0]:g}/{k[1]}ep" for k in ties) + ".", "",
                "Not the same as identical — this experiment cannot separate them. "
                f"Tiebreak is highest mean on the selection domain, which takes "
                f"**LR {cheapest[0]:g}, {cheapest[1]} epochs**.", ""]
    else:
        out += ["## Tie set", "", "Every cell separated from the reference.", ""]

    text = "\n".join(out) + "\n"
    dest = _REPO_ROOT / "results" / f"{g['out']}_step2.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {dest.relative_to(_REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="T8 seed variance / T9 grid step 1.")
    ap.add_argument("--cell", default=None,
                    help="directory of one cell's five seeds (default: results/runs/)")
    ap.add_argument("--grid", nargs="?", const="bert", choices=sorted(GRIDS),
                    help="step 1: walk every cell of an encoder's grid into its table")
    ap.add_argument("--compare", nargs="?", const="bert", choices=sorted(GRIDS),
                    help="step 2: paired comparison, best cell against the rest")
    args = ap.parse_args()
    if args.grid:
        t9_grid(args.grid)
        return
    if args.compare:
        t9_compare(args.compare)
        return
    runs = load_runs(Path(args.cell) if args.cell else None)
    # --cell used to write results/t8_seed_variance.md whatever it was pointed at,
    # so inspecting any other cell silently overwrote T8's committed report with
    # that cell's numbers, under T8's heading. Named per cell now.
    global _OUT
    if args.cell:
        _OUT = _REPO_ROOT / "results" / f"cell_{Path(args.cell).name}.md"
    collapsed = [r["seed"] for r in runs if r["collapsed"]]

    for r in runs:
        assert r["htfl_f1"] is not None, (
            f"seed {r['seed']} has no htfl score -- it was run with --skip-test")

    series = {
        "equi": [r["best_equi_f1"] for r in runs],
        "htfl": [r["htfl_f1"] for r in runs],
    }

    cfg = runs[0]["config"]
    out = [
        "# T8 — Seed variance",
        "",
        f"`{cfg['model_name']}`, lr {cfg['learning_rate']:g}, {cfg['num_epochs']} epochs, "
        f"effective batch {cfg['effective_train_batch_size']}, warmup {cfg['warmup_ratio']:g}. "
        f"Seeds {', '.join(str(s) for s in SEEDS)}.",
        "",
        "Per-seed statistic: **best epoch on equi**, ANN unique-list F1. htfl is "
        "evaluated once per run, on the best-equi weights.",
        "",
        f"Encoder sha256 (identical across all five): `{runs[0]['encoder_weight_hash'][:32]}...`",
        "",
        "| seed | best epoch | equi F1 | htfl F1 | collapsed | first 10 sampled indices |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs:
        first10 = ", ".join(str(i) for i in r["first_batch_indices"])
        out.append(f"| {r['seed']} | {r['best_epoch']} | {r['best_equi_f1']:.4f} | "
                   f"{r['htfl_f1']:.4f} | {'YES' if r['collapsed'] else '—'} | {first10} |")

    out += ["", "| domain | mean | std (ddof=1) | ceiling | mean / ceiling |", "|---|---|---|---|---|"]
    summary = {}
    for domain, values in series.items():
        mean, std = stats(values)
        summary[domain] = (mean, std)
        ceiling = CEILING[domain]
        out.append(f"| {domain} | {mean:.4f} | {std:.4f} | {ceiling:.4f} | "
                   f"{mean / ceiling:.3f} |")

    out.append("")
    if collapsed:
        out += [
            f"## {len(collapsed)} of {len(runs)} runs COLLAPSED — seeds "
            f"{', '.join(str(s) for s in collapsed)}",
            "",
            "The mean and std above include the zeros, which is the honest estimate "
            "of what this config does under the seed lottery. **No t-statistic is "
            "reported:** the std is dominated by one point and the distribution is "
            "not remotely normal. The collapse rate is a property of the config and "
            "matters more than the mean.",
            "",
            f"Collapse rate: {len(collapsed)}/{len(runs)}.",
        ]
    else:
        equi_std = summary["equi"][1]
        se_factor = math.sqrt(2 / len(runs))  # equal-std case
        out += [
            "## What this std makes detectable",
            "",
            f"Against a second config with the same std (s = {equi_std:.4f} on equi), "
            f"`SE(gap) = s·sqrt(2/{len(runs)}) = {equi_std * se_factor:.4f}`:",
            "",
            f"- gap below **{T_TIE * equi_std * se_factor:.4f}** F1 (t < {T_TIE}) — "
            "no difference detected at n=5; not the same as no difference",
            f"- gap above **{T_SERIOUS * equi_std * se_factor:.4f}** F1 (t > {T_SERIOUS}) — "
            "too large to be seed luck, scoped to equi, this metric, this n",
            "",
            "Both configs contribute their own std in T9; the equal-std figures above "
            "are an indication, not the test.",
        ]

    text = "\n".join(out) + "\n"
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {_OUT.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
