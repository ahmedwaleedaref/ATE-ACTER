# Project rules
- NEVER commit anything under data/ — ACTER is CC BY-NC-SA 4.0 (non-commercial)
- Pin every dependency version; no unpinned installs
- Hand-written by the author — do not generate or refactor:
  - the evaluation harness (src/eval/)
  - the loader (src/stats/loading.py)
  - src/data/align.py — align_labels, recover_token_labels, positive_rate
  Exception, week 2 onward: the rest of src/data/ (dataset.py, run_gate.py)
  and the training loop are generated scaffolding around those. The rule is
  that nothing which measures or aligns is generated, so a gate never checks
  generated code against generated code.
- Every experiment writes config + seed + result to results/
- Do not add dependencies without asking
- Assertions encode measured facts, not guesses. When one fails, diagnose the
  cause. Never weaken an assertion, add a tolerance, or adjust an expected
  value to make a test pass. Expected values change only when a new
  measurement is recorded in docs/.