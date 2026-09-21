"""T6 report: positive rates, truncation loss, filter state.

Nothing here asserts. The gate is the exact-sequence-equality check in
``tests/test_alignment_gate.py``; this writes the numbers that go in the
results and that week-1's ``Data_stats.md`` section 6.1 can be read against.

Run:  python -m src.data.run_gate
Out:  results/t6_alignment.md
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.data.align import positive_rate
from src.data.dataset import (
    ATEDataset,
    build_dataloader,
    build_examples,
    get_tokenizer,
    load_train_config,
)
from src.statistics.loading import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO_ROOT / "results" / "t6_alignment.md"

# Data_stats.md section 6.1, the numbers the dataloader will actually produce:
# wind post-filter, everything else unfiltered. Printed beside the measured
# value as a reference, never asserted against here.
REFERENCE_POSITIVE_RATE = {
    ("corp", False): 0.1262,
    ("equi", False): 0.1822,
    ("wind", True): 0.1562,
    ("wind", False): 0.1455,
    ("htfl", False): 0.2604,
}

# (domain, filter applied). wind appears twice: once as it enters training and
# once unfiltered, since only the first is comparable to the reference constant
# and only the second is comparable to the week-1 section 6 table.
ROWS = [("corp", False), ("equi", False), ("wind", True), ("wind", False), ("htfl", False)]


def _build(domain, *, filtered, truncation, tokenizer, train_cfg, data_cfg):
    return build_examples(
        domain,
        tokenizer=tokenizer,
        truncation=truncation,
        max_length=train_cfg.max_length if truncation else None,
        filter_max_tokens=train_cfg.filter_max_tokens if filtered else None,
        data_cfg=data_cfg,
    )


def _rate(examples, *, tokenizer, train_cfg):
    loader = build_dataloader(
        ATEDataset(examples),
        tokenizer=tokenizer,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        length_grouped=False,
    )
    return positive_rate(loader)


def measure(domain: str, *, filtered: bool, tokenizer, train_cfg, data_cfg) -> dict:
    """One domain, both truncation settings, causes separated by differencing.

    A dataset token can end up with no model position two ways -- the
    truncation frontier cut it, or it tokenized to zero wordpieces -- and they
    need different treatment: the frontier is a length policy this project
    chose, the other is a fact about the tokenizer. Position in the sentence
    does not tell them apart, because 4 wind sentences end in a zero-piece
    character.

    With truncation OFF there is no frontier, so every token without a position
    is a zero-piece token. Differencing the two passes, per sentence, gives the
    truncation cost exactly. Sentences truncated is counted independently as
    ``n_subwords > max_length`` on the untruncated pass -- the same definition
    ``data_layout.md`` section 8.2 measured (8 corpus-wide under BERT).
    """
    off = _build(domain, filtered=filtered, truncation=False,
                 tokenizer=tokenizer, train_cfg=train_cfg, data_cfg=data_cfg)
    on = _build(domain, filtered=filtered, truncation=True,
                tokenizer=tokenizer, train_cfg=train_cfg, data_cfg=data_cfg)
    assert len(off) == len(on), "the two passes must be the same sentences in the same order"

    rate_off, n_positive, n_scored = _rate(off, tokenizer=tokenizer, train_cfg=train_cfg)
    rate_on, _, _ = _rate(on, tokenizer=tokenizer, train_cfg=train_cfg)

    zero_piece = sum(example.n_tokens_without_position for example in off)
    truncated_tokens = sum(after.n_tokens_without_position - before.n_tokens_without_position
                           for before, after in zip(off, on))
    truncated_sentences = sum(1 for example in off
                              if example.n_subwords > train_cfg.max_length)

    return {
        "domain": domain,
        "filtered": filtered,
        "n_sentences": len(off),
        "n_dataset_tokens": sum(len(example.tokens) for example in off),
        "n_scored": n_scored,
        "n_positive": n_positive,
        "rate": rate_off,
        "rate_truncated": rate_on,
        "zero_piece_tokens": zero_piece,
        "n_sentences_truncated": truncated_sentences,
        "dropped_tokens": truncated_tokens,
        "positive_labels_lost": sum(example.n_positive_labels_without_position
                                    for example in on),
    }


def _render(rows: list[dict], train_cfg, data_cfg) -> str:
    lines = [
        "# T6 — alignment report",
        "",
        f"- date: {date.today().isoformat()}",
        f"- tokenizer: {train_cfg.model_name}",
        f"- max_length: {train_cfg.max_length}",
        f"- labels: {data_cfg.ne_variant}, scheme: {data_cfg.sequential_scheme}",
        f"- split: train {list(data_cfg.train_domains)}, dev {data_cfg.dev_domain!r}, "
        f"test {data_cfg.test_domain!r}",
        f"- short-sentence filter: <= {train_cfg.filter_max_tokens} dataset tokens, "
        f"domains {list(train_cfg.filter_domains)}, splits {list(train_cfg.filter_splits)}",
        "",
        "The gate is `tests/test_alignment_gate.py` — per-example exact sequence",
        "equality between the labels recovered through `word_ids()` and the labels",
        "`load_domain` returned. Nothing on this page is asserted.",
        "",
        "## Positive rate, truncation DISABLED",
        "",
        "Comparable to `Data_stats.md` §6.1, which measured the same quantity in",
        "dataset-token space with no tokenizer involved.",
        "",
        "| domain | filter | n_sent | dataset tokens | scored positions | zero-piece | positives | rate | §6.1 | delta |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        reference = REFERENCE_POSITIVE_RATE.get((row["domain"], row["filtered"]))
        ref_text = f"{reference:.4f}" if reference is not None else "—"
        delta_text = f"{row['rate'] - reference:+.6f}" if reference is not None else "—"
        lines.append(
            f"| {row['domain']} | {'≤2 dropped' if row['filtered'] else 'none'} | "
            f"{row['n_sentences']:,} | {row['n_dataset_tokens']:,} | {row['n_scored']:,} | "
            f"{row['zero_piece_tokens']} | {row['n_positive']:,} | "
            f"{row['rate']:.4f} | {ref_text} | {delta_text} |"
        )
    lines += [
        "",
        "`scored positions` is dataset tokens minus zero-piece tokens: a dataset",
        "token that tokenizes to nothing gets no model position, so its label has",
        "nowhere to land. Where that count is non-zero the rate cannot equal §6.1",
        "exactly — §6.1's denominator is dataset tokens, this one's is positions.",
    ]

    lines += [
        "",
        f"## Truncation loss, max_length {train_cfg.max_length}",
        "",
        "A dataset token with no model position — cut by the truncation frontier,",
        "or tokenizing to zero wordpieces — is predicted `O` by",
        "`recover_token_labels`. It can only cost recall. Reported, never asserted.",
        "",
        "`sentences truncated` counts untruncated length > max_length, the same",
        "definition `data_layout.md` §8.2 measured. `tokens lost to truncation`",
        "differences the two passes per sentence, so a sentence that merely ENDS",
        "in a zero-piece character is not miscounted as truncated.",
        "",
        "| domain | filter | rate | sentences truncated | tokens lost to truncation | zero-piece | positive labels lost |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {'≤2 dropped' if row['filtered'] else 'none'} | "
            f"{row['rate_truncated']:.4f} | {row['n_sentences_truncated']} | "
            f"{row['dropped_tokens']} | {row['zero_piece_tokens']} | "
            f"{row['positive_labels_lost']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="T6 alignment report.")
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    train_cfg = load_train_config()
    data_cfg = load_config()
    tokenizer = get_tokenizer(train_cfg)

    rows = [measure(domain, filtered=filtered, tokenizer=tokenizer,
                    train_cfg=train_cfg, data_cfg=data_cfg)
            for domain, filtered in ROWS]

    report = _render(rows, train_cfg, data_cfg)
    print(report)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
