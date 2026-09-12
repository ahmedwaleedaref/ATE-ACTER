"""T8 aggregation: the five seed runs -> results/t8_seed_variance.md.

Generated scaffolding. It reads what ``run_train.py`` wrote and does arithmetic
on it; it never re-scores anything, and it never touches a model.

Run the five seeds and aggregate:
    for s in 42 43 44 45 46; do python -m src.models.run_train --seed $s --reason "T8 seed variance" || break; done && python -m src.aggregate

Reads:  results/runs/seed_*.json
Writes: results/t8_seed_variance.md   (also printed to stdout)
"""

from __future__ import annotations

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

# Tasks_week2.md T8: t = |mean_A - mean_B| / sqrt(s_A^2/n + s_B^2/n).
# With equal stds these two boundaries are the 1-std and 2-std bands exactly.
T_TIE, T_SERIOUS = 1.58, 3.16


def load_runs() -> list[dict]:
    paths = sorted(_RUNS_JSON.glob("seed_*.json"))
    assert paths, f"no seed_*.json in {_RUNS_JSON} -- run the five seeds first"
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

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


def main() -> None:
    runs = load_runs()
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
