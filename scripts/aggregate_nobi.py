"""Aggregate completed NOBI seed runs without mixing them with BIO runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNS = _REPO_ROOT / "results" / "runs" / "nobi"
_OUT = _REPO_ROOT / "results" / "nobi_seed_summary.md"
_SEEDS = (42, 43, 44, 45, 46)


def load_runs() -> list[dict]:
    paths = sorted(_RUNS.glob("seed_*.json"))
    assert paths, f"no NOBI runs found in {_RUNS}"
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert all(run.get("scheme") == "nobi" for run in runs), "NOBI directory contains a non-NOBI run"
    assert sorted(run["seed"] for run in runs) == list(_SEEDS), (
        f"NOBI aggregation requires seeds {_SEEDS}; found "
        f"{sorted(run['seed'] for run in runs)}"
    )
    return sorted(runs, key=lambda run: run["seed"])


def main() -> None:
    runs = load_runs()
    equi = [run["best_equi_f1"] for run in runs]
    htfl = [run["htfl_f1"] for run in runs if run["htfl_f1"] is not None]
    assert len(htfl) == len(runs), "every NOBI run must include an htfl score"
    lines = [
        "# NOBI seed summary",
        "",
        "Per-seed statistic: best epoch on equi, ANN unique-list F1.",
        "",
        "| seed | best epoch | equi ANN F1 | htfl ANN F1 | collapsed |",
        "|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['best_equi_f1']:.4f} | "
            f"{run['htfl_f1']:.4f} | {'yes' if run['collapsed'] else 'no'} |"
        )
    lines += [
        "",
        "| domain | mean | std (ddof=1) |", "|---|---:|---:|",
        f"| equi | {statistics.fmean(equi):.4f} | {statistics.stdev(equi):.4f} |",
        f"| htfl | {statistics.fmean(htfl):.4f} | {statistics.stdev(htfl):.4f} |",
        "",
    ]
    _OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {_OUT.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()