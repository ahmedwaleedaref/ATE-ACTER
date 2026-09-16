"""NOBI alignment gate over the real ACTER corpus.

This mirrors the BIO gate but uses the derived NOBI labels and five-class
mapping. Truncation is disabled so every represented dataset token can be
checked exactly.
"""

from __future__ import annotations

import pytest

from src.data.align import recover_token_labels
from src.data.dataset import build_examples, get_tokenizer, load_train_config
from src.eval.nobi import NOBI_ID2LABEL
from src.stats.loading import load_config


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(load_train_config())


@pytest.fixture(scope="module")
def data_cfg():
    return load_config()


@pytest.mark.parametrize("domain", ["corp", "equi", "wind", "htfl"])
def test_nobi_labels_survive_subword_round_trip(domain, tokenizer, data_cfg):
    examples = build_examples(
        domain,
        tokenizer=tokenizer,
        truncation=False,
        max_length=None,
        filter_max_tokens=None,
        data_cfg=data_cfg,
        scheme="nobi",
    )

    for example in examples:
        recovered = recover_token_labels(
            example.word_ids,
            example.labels,
            len(example.tokens),
            NOBI_ID2LABEL,
        )
        assert recovered == example.gold_labels, (
            f"{example.file_id}:{example.sent_idx}: NOBI labels changed during "
            f"subword alignment: {recovered} != {example.gold_labels}"
        )