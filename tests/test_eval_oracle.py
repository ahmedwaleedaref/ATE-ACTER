"""The eval path, checked against numbers week-1 code produced by another route.

``evaluate`` is fed one-hot logits built from the GOLD labels, so whatever it
returns is a property of the harness, not of a model. Those values are already
known: decoding gold BIO to a term list and scoring it against the gold key is
exactly what ``compute_ceilings`` did in T3, walking the corpus directly -- no
tokenizer, no batching, no padding, no first-subword selection.

So this test compares two independent routes to the same eight numbers. It
covers argmax, recover_token_labels, decode, spans_to_unique_list, score_list,
the example_index round trip, and the fact that padded positions never produce
a span. If it passes, a bad score later is the model, not the plumbing.

The ceilings are measured facts from ``results/ceilings.md`` (docs/data_layout.md
section 5.5). They are not adjustable: if this test fails, the eval path is
wrong, or a decision recorded in docs/ was changed without re-measuring.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.data.dataset import (ATEDataset, ID2LABEL, build_dataloader, build_examples,
                              get_tokenizer, load_train_config)
from src.eval.run_eval import _gold_key_path, load_eval_config
from src.eval.scorers import load_gold_list_into_set
from src.models.train_loop import evaluate
from src.stats.loading import load_config

# domain -> (n_spans, n_types, {key: (max_precision, max_recall)})
CEILINGS = {
    "equi": (8662, 1204, {"ann": (0.9294, 0.9764), "nes": (0.9294, 0.7168)}),
    "htfl": (9636, 2452, {"ann": (0.8887, 0.9316), "nes": (0.8911, 0.8549)}),
}


class OracleModel:
    """Predicts the gold labels exactly, via one-hot logits over the batch's
    own ``labels``. Positions at -100 are clamped to O and never read: the
    first-subword selection in ``recover_token_labels`` skips them."""

    def eval(self):
        return self

    def __call__(self, **batch):
        onehot = F.one_hot(batch["labels"].clamp(min=0), num_classes=len(ID2LABEL)).float()
        return type("Output", (), {"logits": onehot})()


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(load_train_config())


@pytest.mark.parametrize("domain", sorted(CEILINGS))
def test_eval_path_reproduces_the_measured_ceilings(domain, tokenizer):
    cfg = load_train_config()
    data_cfg = load_config()
    eval_cfg = load_eval_config()

    examples = build_examples(domain, tokenizer=tokenizer, truncation=cfg.truncation,
                              max_length=cfg.max_length, filter_max_tokens=None)
    dataset = ATEDataset(examples)
    loader = build_dataloader(dataset, tokenizer=tokenizer, batch_size=cfg.eval_batch_size,
                              shuffle=False, length_grouped=False)
    gold_lists = {key: load_gold_list_into_set(str(_gold_key_path(data_cfg, domain, key)))
                  for key in eval_cfg["keys"]}

    result = evaluate(OracleModel(), loader, dataset, device=torch.device("cpu"),
                      id2label=ID2LABEL, gold_lists=gold_lists)

    n_spans, n_types, per_key = CEILINGS[domain]
    assert result["n_pred_spans"] == n_spans
    assert result["n_pred_types"] == n_types
    for key, (max_p, max_r) in per_key.items():
        assert result[f"list_{key}_p"] == pytest.approx(max_p, abs=5e-5)
        assert result[f"list_{key}_r"] == pytest.approx(max_r, abs=5e-5)
