"""T6 gate.

One check: for every sentence in every domain, the labels recovered through
``word_ids()`` equal the labels ``load_domain`` returned.

    recover_token_labels(word_ids, align_labels(word_ids, gold), ...) == gold

That single equality covers misalignment by one position, a dataset token that
tokenized to nothing and lost its label, a length mismatch, and a
first-subword rule that differs between the training path and the inference
path -- and it names the file, sentence and token that failed instead of
reporting a domain-level rate that is slightly off.

It is the one bug the T3 harness cannot catch: misalignment happens upstream of
``decode``, nothing crashes, and the model trains to a plausible score against
shifted labels.

Truncation is DISABLED here. At max_length 256 eight sentences corpus-wide are
cut and their tail tokens correctly become "O", so exact equality would fail for
a reason that is not a bug (``data_layout.md`` §8.2). Truncation loss is
measured separately by ``src/data/run_gate.py``.

The four positive rates are reported by that script, not asserted here. They
cannot all be exact: 41 wind tokens tokenize to zero wordpieces under
bert-base-cased (Private Use Area debris in ``wind_en_01``), so wind's
denominator here is 41 positions short of the dataset-token count §6.1 divided
by. All 41 are gold ``O``, so this equality still holds -- and if one of them
were a ``B``, this test would fail rather than round it away.
"""

from __future__ import annotations

import pytest

from src.data.align import align_labels, recover_token_labels
from src.data.dataset import (
    ID2LABEL,
    LABEL2ID,
    ATEDataset,
    Collator,
    build_examples,
    get_tokenizer,
    load_train_config,
)
from src.statistics.loading import load_config, load_domain

DOMAINS = ["corp", "equi", "wind", "htfl"]


@pytest.fixture(scope="session")
def train_cfg():
    return load_train_config()


@pytest.fixture(scope="session")
def tokenizer(train_cfg):
    return get_tokenizer(train_cfg)


@pytest.fixture(scope="session")
def data_cfg():
    return load_config()


def _first_mismatch(recovered: list[str], gold: list[str]) -> int:
    for index, (left, right) in enumerate(zip(recovered, gold)):
        if left != right:
            return index
    return min(len(recovered), len(gold))


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("domain", DOMAINS)
def test_alignment_gate(domain, tokenizer, data_cfg):
    """Every dataset token's label survives the round trip through wordpieces."""
    examples = build_examples(
        domain,
        tokenizer=tokenizer,
        truncation=False,
        max_length=None,
        filter_max_tokens=None,   # the gate covers every sentence, filtered or not
        data_cfg=data_cfg,
    )

    for example in examples:
        recovered = recover_token_labels(
            example.word_ids, example.labels, len(example.tokens), ID2LABEL)

        if recovered != example.gold_labels:
            index = _first_mismatch(recovered, example.gold_labels)
            context = example.tokens[max(0, index - 3):index + 4]
            pytest.fail(
                f"{example.file_id}:{example.sent_idx} token {index} "
                f"({example.tokens[index] if index < len(example.tokens) else '<past end>'!r}): "
                f"recovered {recovered[index] if index < len(recovered) else '<missing>'!r} "
                f"!= gold {example.gold_labels[index] if index < len(example.gold_labels) else '<missing>'!r}\n"
                f"  lengths: recovered {len(recovered)}, gold {len(example.gold_labels)}, "
                f"tokens {len(example.tokens)}, subwords {example.n_subwords}\n"
                f"  context: {context}"
            )


# --------------------------------------------------------------------------- #
# Scaffolding checks — generated code, not the gate
# --------------------------------------------------------------------------- #
def test_align_labels_on_a_hand_checked_sentence(tokenizer):
    """A fragmenting sentence, worked out by hand rather than by the code.

    ``self-employed`` is one ACTER token and several wordpieces; ``non``, ``-``,
    ``governmental`` are three ACTER tokens. This is the case that breaks if the
    tokens are re-joined into a string instead of passed pre-split.
    """
    tokens = ["The", "self-employed", "non", "-", "governmental", "body", "."]
    labels = ["O", "B", "B", "I", "I", "I", "O"]

    encoding = tokenizer(tokens, is_split_into_words=True)
    word_ids = encoding.word_ids()
    aligned = align_labels(word_ids, labels, LABEL2ID)

    assert len(aligned) == len(word_ids)
    first_positions = [i for i, w in enumerate(word_ids)
                       if w is not None and (i == 0 or word_ids[i - 1] != w)]
    assert len(first_positions) == len(tokens), "one scored position per dataset token"
    assert [aligned[i] for i in first_positions] == [LABEL2ID[label] for label in labels]
    assert all(aligned[i] == -100 for i in range(len(word_ids)) if i not in first_positions)

    recovered = recover_token_labels(word_ids, aligned, len(tokens), ID2LABEL)
    assert recovered == labels


def test_collator_pads_labels_with_ignore_index(tokenizer, data_cfg):
    """Padding must be -100 in ``labels``, not 0.

    Padding with 0 adds fake ``O`` positions: no crash, a diluted positive rate,
    and gradient on nothing.
    """
    examples = build_examples("htfl", tokenizer=tokenizer, truncation=False,
                              max_length=None, filter_max_tokens=None, data_cfg=data_cfg)
    dataset = ATEDataset(sorted(examples, key=lambda e: e.n_subwords)[:8])
    batch = Collator(tokenizer)([dataset[i] for i in range(len(dataset))])

    # example_index is the only link from a batch row back to its Example --
    # tokens, file_id and sent_idx never enter the batch, so if it goes missing
    # the whole inference path loses the identity of what it just predicted.
    # It went missing once, silently, and every other test still passed.
    assert "example_index" in batch, "collator dropped example_index"
    assert batch["example_index"].tolist() == list(range(len(dataset)))
    for row, index in enumerate(batch["example_index"].tolist()):
        example = dataset.examples[index]
        assert batch["input_ids"][row, :example.n_subwords].tolist() == example.input_ids

    lengths = [example.n_subwords for example in dataset.examples]
    assert batch["labels"].shape == (len(dataset), max(lengths))
    for row, length in enumerate(lengths):
        assert (batch["labels"][row, length:] == -100).all()
        assert (batch["attention_mask"][row, length:] == 0).all()


def test_short_sentence_filter_is_explicit_and_wind_only(tokenizer, train_cfg, data_cfg):
    """The filter is a parameter, and the config points it at wind/train only.

    ``Data_stats.md`` §6.2: wind has 6,638 sentences, 4,048 of them ≤2 dataset
    tokens, leaving 2,590. equi and htfl must never be filtered — their sentence
    counts have to match the loader's, which is what makes their scores
    comparable to the measured ceilings.
    """
    assert train_cfg.filter_for("wind", "train") == 2
    assert train_cfg.filter_for("wind", "dev") is None
    assert train_cfg.filter_for("equi", "dev") is None
    assert train_cfg.filter_for("htfl", "test") is None
    assert train_cfg.filter_for("corp", "train") is None

    unfiltered = build_examples("wind", tokenizer=tokenizer, truncation=False,
                                max_length=None, filter_max_tokens=None, data_cfg=data_cfg)
    filtered = build_examples("wind", tokenizer=tokenizer, truncation=False,
                              max_length=None, filter_max_tokens=2, data_cfg=data_cfg)
    assert len(unfiltered) == 6638
    assert len(filtered) == 2590

    for domain, expected in (("equi", 3090), ("htfl", 2432)):
        examples = build_examples(domain, tokenizer=tokenizer, truncation=False,
                                  max_length=None,
                                  filter_max_tokens=train_cfg.filter_for(domain, "dev"),
                                  data_cfg=data_cfg)
        assert len(examples) == expected


def test_sent_idx_survives_the_filter(tokenizer, data_cfg):
    """Filtering must not renumber sentences.

    ``sent_idx`` is the second field of the T3 span key. If it were assigned
    after filtering, every wind span key would shift and exact-span scoring
    would go quietly wrong.
    """
    filtered = build_examples("wind", tokenizer=tokenizer, truncation=False,
                              max_length=None, filter_max_tokens=2, data_cfg=data_cfg)
    by_key = {(example.file_id, example.sent_idx): example for example in filtered}
    assert len(by_key) == len(filtered), "duplicate (file_id, sent_idx)"

    for document in load_domain("wind", data_cfg):
        for sent_idx, (tokens, labels) in enumerate(document.sentences):
            example = by_key.get((document.file_id, sent_idx))
            if example is not None:
                assert example.tokens == tokens
                assert example.gold_labels == labels
