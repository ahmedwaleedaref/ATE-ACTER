"""Label alignment: the three functions written by hand.

Everything else under ``src/data/`` is scaffolding around these. They are
separated into their own module for one reason: the gate compares the output of
``recover_token_labels`` against the labels ``load_domain`` returned, so if the
gate's own measuring functions were generated the gate would be checking
generated code against generated code.

The contract, from ``data_layout.md`` section 8.4 and ``Data_stats.md``
section 1.3:

  * one gold label per DATASET token; after tokenization that token occupies
    several model positions
  * the FIRST subword of each dataset token carries the label
  * every continuation subword and every special token is ``IGNORE_INDEX``
    (-100), which the loss ignores -- no prediction, no gradient, no penalty
  * at inference the prediction for a dataset token is read from its first
    subword; the remaining positions are discarded unread
  * the mapping is ``word_ids()`` from the fast tokenizer, never string matching

``align_labels`` and ``recover_token_labels`` are mirrors of each other and
share their failure modes, which is why the gate runs gold labels through both:
``recover_token_labels(word_ids, align_labels(word_ids, gold), ...) == gold``
holds only if both agree on which position is a token's first subword.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # torch is imported lazily -- this module is importable without it
    from torch.utils.data import DataLoader

IGNORE_INDEX = -100

# The label scheme lives here rather than in dataset.py: it is part of the
# alignment contract, and positive_rate cannot import it from dataset.py --
# dataset.py already imports this module, so that would be a cycle.
LABEL2ID: dict[str, int] = {"O": 0, "B": 1, "I": 2}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


def align_labels(word_ids : Sequence[int | None] 
                 ,labels  : Sequence[str]
                 ,label2id: dict[str , int] ,
) -> list[int]:
    prev : int | None = None 
    sub_words_labels : list[int] = []
    for word_id in word_ids :
        if word_id is None :
            sub_words_labels.append(IGNORE_INDEX)
        elif word_id != prev : 
            sub_words_labels.append(label2id[labels[word_id]])
        else :
            sub_words_labels.append(IGNORE_INDEX)
        prev = word_id
    return sub_words_labels 


def recover_token_labels(
    word_ids: Sequence[int | None],
    position_label_ids: Sequence[int],
    n_tokens: int,
    id2label: dict[int, str],
) -> list[str]:
    prev : int | None = None 
    #in one line we handle that if the sentences was trunicated so there is some word-tokens are not even predicted by model will have a defualt label O
    recover_tokens_labels : list[str] = ["O"] * n_tokens 
    for index,word_id in enumerate(word_ids) :
        if word_id is not None and word_id != prev :
            #word_id reperesnt index of word in orginal tokens list 
            recover_tokens_labels[word_id] = id2label[position_label_ids[index]]
        prev = word_id
    return recover_tokens_labels 


def positive_rate(loader: "DataLoader") -> tuple[float, int, int]:
    B_count : int = 0
    I_count : int = 0
    non_IGNORE_INDEX_label_count : int = 0
    for batch in loader :
        # One reduction over the whole (B, L) tensor -- no row loop, and no
        # unpadding: the collator pads labels with IGNORE_INDEX, so the same
        # filter drops padding, continuation subwords and special tokens
        # together, leaving exactly one scored position per dataset token.
        labels_tensor = batch["labels"] # (B,L)
        B_count += (labels_tensor == LABEL2ID["B"]).sum().item()
        I_count += (labels_tensor == LABEL2ID["I"]).sum().item()
        non_IGNORE_INDEX_label_count += (labels_tensor != IGNORE_INDEX).sum().item()

    # .item() gives Python ints, so an empty loader raises ZeroDivisionError
    # here rather than returning a silent nan.
    return ( (B_count+I_count) / non_IGNORE_INDEX_label_count , (B_count+I_count) , non_IGNORE_INDEX_label_count)
