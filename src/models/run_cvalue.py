"""T4 entry point. Loads the target domain through the shared T2 loader,
runs candidate generation -> C-Value -> threshold, and writes:

  <output_path>    the term list, one lowercased term per line
  <run_info_path>  provenance: which config, which corpus, how many tokens,
                    how many candidates survived each stage -- so the
                    frequency-corpus decision (docs/Tasks.md T4) is recorded
                    against a specific run, not just asserted in docs.

Run:  python -m src.models.run_cvalue                      (uses configs/cvalue.json)
      python -m src.models.run_cvalue --config path/to.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.cvalue import (  # noqa: E402
    apply_min_frequency,
    compute_cvalue,
    compute_nesting,
    generate_candidates,
    load_stopwords,
    threshold_terms,
    write_term_list,
)
from src.statistics.loading import load_config, load_domain  # noqa: E402

_DEFAULT_CVALUE_CONFIG = _REPO_ROOT / "configs" / "cvalue.json"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_cvalue_config(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else _DEFAULT_CVALUE_CONFIG
    _check(path.is_file(), f"cvalue config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "target_domain", "reference_corpus", "min_n", "max_n",
        "min_frequency", "threshold", "output_path", "run_info_path",
    }
    missing = required - cfg.keys()
    _check(not missing, f"cvalue config missing keys: {sorted(missing)}")

    # T4's open decision, made explicit here rather than left implicit in
    # code. "target_only" is the only mode currently implemented -- htfl has
    # no unannotated_texts/ directory at all, so there is no in-domain
    # reference material to add, and mixing in cross-domain text was
    # declined. See docs/cvalue_baseline.md.
    _check(
        cfg["reference_corpus"] == "target_only",
        f"reference_corpus={cfg['reference_corpus']!r} is not implemented; "
        f"only 'target_only' is wired up (see docs/cvalue_baseline.md for why)",
    )
    return cfg


def run(cvalue_config_path: Path | None = None, data_config_path: Path | None = None) -> dict:
    cvalue_cfg = load_cvalue_config(cvalue_config_path)
    data_cfg = load_config(data_config_path)

    domain = cvalue_cfg["target_domain"]
    documents = load_domain(domain, data_cfg)  # asserts pairing, B-label, etc.
    sentences = [s for doc in documents for s in doc.sentences]
    n_tokens = sum(len(tokens) for tokens, _labels in sentences)

    stopwords = load_stopwords(cvalue_cfg["stopwords_path"])

    raw_candidates = generate_candidates(
        sentences,
        min_n=cvalue_cfg["min_n"],
        max_n=cvalue_cfg["max_n"],
        stopwords=stopwords,
    )
    frequent_candidates = apply_min_frequency(raw_candidates, cvalue_cfg["min_frequency"])
    nesting = compute_nesting(frequent_candidates)
    scores = compute_cvalue(frequent_candidates, nesting)
    terms = threshold_terms(scores, cvalue_cfg["threshold"])

    output_path = _REPO_ROOT / cvalue_cfg["output_path"]
    write_term_list(terms, output_path)

    run_info = {
        "config_used": cvalue_cfg,
        "target_domain": domain,
        "reference_corpus": cvalue_cfg["reference_corpus"],
        "n_documents": len(documents),
        "n_sentences": len(sentences),
        "n_tokens": n_tokens,
        "n_candidates_raw": len(raw_candidates),
        "n_candidates_after_min_frequency": len(frequent_candidates),
        "n_terms_output": len(terms),
        "output_path": str(output_path),
    }
    run_info_path = _REPO_ROOT / cvalue_cfg["run_info_path"]
    run_info_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(description="T4 -- C-Value baseline")
    parser.add_argument("--config", type=Path, default=None,
                         help="path to configs/cvalue.json (default: repo config)")
    parser.add_argument("--data-config", type=Path, default=None,
                         help="path to configs/data.json (default: repo config)")
    args = parser.parse_args()

    info = run(args.config, args.data_config)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
