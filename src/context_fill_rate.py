"""How much document context actually exists around each sentence.

Measured BEFORE any FLERT implementation, because the answer decides whether the
feature can exist on the test domain at all. Documents are wildly uneven --
wind 11,553 tokens each, htfl 292 (Data_stats.md 4.1) -- so a context window is
a different feature in each domain.

Context is taken from the sentence's OWN document and never crosses a document
boundary. It counts every dataset token in the document, including the wind
sentences the training filter drops: those tokens are still on the page, and a
FLERT window would read them.

Reported as a distribution per domain, never as a mean: a mean fill rate hides
whether the shortfall is a few starved sentences or all of them.

    python -m src.context_fill_rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.stats.loading import load_config, load_domain

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "results" / "context_fill_rate.md"

# Dataset tokens per side. 512 subwords is the stated budget; at deberta's 1.12
# overall inflation that is ~450 dataset tokens, of which the sentence itself
# takes ~25 at the median. So ~210 per side is the largest window that fits.
WINDOWS = (32, 64, 128, 210)
DECILES = (0, 10, 25, 50, 75, 90, 100)


def percentile(sorted_values: list[float], p: int) -> float:
    """Nearest-rank, matching src/stats/s01_lengths.py."""
    import math
    if not sorted_values:
        return float("nan")
    i = max(0, min(len(sorted_values) - 1, math.ceil(p / 100 * len(sorted_values)) - 1))
    return sorted_values[i]


def fill_rates(domain: str, window: int) -> list[float]:
    """One fill rate per sentence: available context tokens / requested."""
    rates = []
    for doc in load_domain(domain):
        lengths = [len(tokens) for tokens, _ in doc.sentences]
        total = sum(lengths)
        before = 0
        for n in lengths:
            after = total - before - n
            available = min(window, before) + min(window, after)
            rates.append(available / (2 * window))
            before += n
    return rates


def main() -> None:
    ap = argparse.ArgumentParser(description="Document-context fill rate per domain.")
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    args = ap.parse_args()

    cfg = load_config()
    domains = cfg.domains
    role = {**{d: "train" for d in cfg.train_domains},
            cfg.dev_domain: "dev", cfg.test_domain: "test"}

    L = ["# Document-context fill rate", "",
         "Fill rate = context tokens available within the sentence's own document,",
         "divided by the window requested (both sides). 1.0 means the window is full;",
         "0.5 means half the requested context does not exist.",
         "",
         "Distribution, not a mean -- a mean hides whether the shortfall is a few",
         "starved sentences or all of them. Percentiles are nearest-rank.",
         ""]

    for window in WINDOWS:
        L.append(f"## Window {window} tokens per side ({2 * window} total)")
        L.append("")
        L.append("| domain | role | tokens/doc | n_sent | " +
                 " | ".join(f"p{p}" for p in DECILES) + " | full | < 0.5 |")
        L.append("|---|---|--:|--:|" + "--:|" * (len(DECILES) + 2))
        for domain in domains:
            docs = load_domain(domain)
            tok_per_doc = sum(len(t) for d in docs for t, _ in d.sentences) / len(docs)
            rates = sorted(fill_rates(domain, window))
            n = len(rates)
            full = sum(1 for r in rates if r >= 0.999) / n
            starved = sum(1 for r in rates if r < 0.5) / n
            cells = " | ".join(f"{percentile(rates, p):.2f}" for p in DECILES)
            L.append(f"| {domain} | {role[domain]} | {tok_per_doc:,.0f} | {n:,} | "
                     f"{cells} | {full:.2f} | {starved:.2f} |")
        L.append("")

    report = "\n".join(L)
    out = Path(args.out)
    out = out if out.is_absolute() else _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nwrote {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
