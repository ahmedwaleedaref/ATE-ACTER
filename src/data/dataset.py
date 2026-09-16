"""Week-2 dataloader: tokenize, align, filter, collate, batch.

Wraps ``src/stats/loading.py``. Nothing here reads the corpus from disk --
``load_domain`` remains the only code that does, and it is untouched. Nothing
here decodes or scores either: predictions go back out through
``src/data/align.py::recover_token_labels`` and then through the T3 harness in
``src/eval/``, so a wrong number has exactly one place it can come from.

Generated scaffolding. The three functions the gate rests on are hand-written in
``src/data/align.py``.

Pipeline, one sentence at a time:

  1. ``load_domain(domain)``          -> (tokens, labels), sentence boundaries
  2. key                              -> (file_id, sent_idx), assigned BEFORE
                                         the filter so keys never shift and keep
                                         matching ``generate_flatten_spans``
  3. filter (explicit parameter)      -> drop sentences of <= N dataset tokens
  4. ``tok(tokens, is_split_into_words=True)``
  5. ``align_labels(word_ids, ...)``  -> first subword labelled, rest -100
  6. collate                          --> dynamic padding, labels padded -100
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from src.data.align import ID2LABEL, IGNORE_INDEX, LABEL2ID, align_labels
from src.data.nobi_labels import build_nobi_labels, load_domain_terms
from src.eval.nobi import NOBI_ID2LABEL, NOBI_LABEL2ID
from src.stats.loading import DataConfig, load_config, load_domain

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRAIN_CONFIG = _REPO_ROOT / "configs" / "train.json"

# LABEL2ID / ID2LABEL are defined in src/data/align.py and re-exported here:
# the label scheme belongs with the alignment contract, not with the
# scaffolding. Imported above, so they stay importable from this module and
# existing call sites (the tests, and the model config's id2label at T7) are
# unchanged.


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    hf_cache_dir: str
    max_length: int
    truncation: bool
    effective_train_batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    eval_batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    max_grad_norm: float
    lr_schedule: str
    warmup_ratio: float
    seed: int
    device: str
    log_every_steps: int
    save_checkpoint: bool
    length_grouped_batching: bool
    filter_max_tokens: int
    filter_domains: tuple[str, ...]
    filter_splits: tuple[str, ...]
    raw: dict           # the file as-loaded, for the per-run results/ header

    def filter_for(self, domain: str, split: str) -> int | None:
        """The ``<= N tokens`` filter threshold for this domain and split, or
        ``None`` for no filtering.

        The rule lives in the config, not in a branch on the domain name, so a
        run with the filter off is a config edit rather than a code edit -- and
        so that "wind only, training only" is greppable in one place.
        """
        if domain in self.filter_domains and split in self.filter_splits:
            return self.filter_max_tokens
        return None


def load_train_config(path=None) -> TrainConfig:
    path = Path(path) if path is not None else _DEFAULT_TRAIN_CONFIG
    assert path.is_file(), f"train config not found: {path}"
    raw = json.loads(path.read_text(encoding="utf-8"))

    per_device = raw["per_device_train_batch_size"]
    accum = raw["gradient_accumulation_steps"]
    effective = raw["effective_train_batch_size"]
    # 16 is the hyperparameter; 8 x 2 is how 4 GB of VRAM reaches it. If these
    # ever disagree the batch size reported in results/ is not the batch size
    # that ran, and T9's grid means something different from what it says.
    assert per_device * accum == effective, (
        f"batch size mismatch: per_device {per_device} x accum {accum} "
        f"!= effective {effective}"
    )

    filt = raw["short_sentence_filter"]
    return TrainConfig(
        model_name=raw["model_name"],
        hf_cache_dir=raw["hf_cache_dir"],
        max_length=raw["max_length"],
        truncation=raw["truncation"],
        effective_train_batch_size=effective,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        eval_batch_size=raw["eval_batch_size"],
        learning_rate=float(raw["learning_rate"]),
        num_epochs=raw["num_epochs"],
        weight_decay=float(raw["weight_decay"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        lr_schedule=raw["lr_schedule"],
        warmup_ratio=float(raw["warmup_ratio"]),
        seed=raw["seed"],
        device=raw["device"],
        log_every_steps=raw["log_every_steps"],
        save_checkpoint=raw["save_checkpoint"],
        length_grouped_batching=raw["length_grouped_batching"],
        filter_max_tokens=filt["max_tokens"],
        filter_domains=tuple(filt["domains"]),
        filter_splits=tuple(filt["splits"]),
        raw=raw,
    )


def get_tokenizer(cfg: TrainConfig):
    """Fast tokenizer only -- ``word_ids()`` does not exist on the slow one.

    ``add_prefix_space=True`` is required for RoBERTa at T10 and harmless
    elsewhere; without it, byte-level BPE with ``is_split_into_words=True``
    either raises or tokenizes unlike its pretraining, which presents as
    "RoBERTa is bad at this task".
    """
    kwargs = {"cache_dir": cfg.hf_cache_dir, "use_fast": True}
    if "roberta" in cfg.model_name:
        kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **kwargs)
    assert tokenizer.is_fast, f"{cfg.model_name} has no fast tokenizer; word_ids() is unavailable"
    return tokenizer




@dataclass(frozen=True)
class Example:
    """One sentence, carrying everything the way back out needs.

    ``tokens`` and ``gold_labels`` are the dataset's own, never reconstructed
    from wordpieces: ``decode`` joins these to build a term string, and
    reassembling ``self-employed`` from its pieces would produce
    ``self - employed``, which matches no gold entry.
    """
    domain: str
    file_id: str
    sent_idx: int               # index into Document.sentences, pre-filter
    tokens: list[str]
    gold_labels: list[str]
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    word_ids: list[int | None]

    @property
    def n_subwords(self) -> int:
        return len(self.input_ids)

    @property
    def represented_indices(self) -> set[int]:
        """Dataset-token indices that got at least one model position, read from
        ``word_ids`` alone so the number does not depend on ``align_labels``
        being right."""
        return {w for w in self.word_ids if w is not None}

    @property
    def indices_without_position(self) -> list[int]:
        """Dataset tokens that got no model position at all.

        Two causes, and this property deliberately does NOT try to tell them
        apart: the truncation frontier cut the tail, or the token tokenized to
        zero wordpieces. Position in the sentence does not separate them -- 4
        wind sentences simply END in a zero-piece character, so "trailing means
        truncated" mis-attributes them and inflates the truncated-sentence
        count. ``run_gate.py`` separates the causes by differencing a
        truncation-off pass against a truncation-on one, which is exact.

        Measured with bert-base-cased, truncation off: 41 tokens, all in
        ``wind_en_01``, all the Private Use Area characters U+F8EF and U+F8FA
        left by PDF extraction, all gold ``O``. corp, equi and htfl have none.
        That count is a property of the tokenizer, not of ACTER -- a byte-level
        BPE keeps those characters -- so it is reported per run, never pinned.

        Every token here loses its label silently: nothing crashes, and the
        model can never predict it. The gate catches the dangerous case anyway,
        because a gold ``B``/``I`` recovered as ``O`` is a sequence mismatch.
        """
        represented = self.represented_indices
        return [i for i in range(len(self.tokens)) if i not in represented]

    @property
    def n_tokens_without_position(self) -> int:
        return len(self.tokens) - len(self.represented_indices)

    @property
    def n_positive_labels_without_position(self) -> int:
        """Positive gold labels the model is given no chance to predict."""
        return sum(1 for i in self.indices_without_position
                   if self.gold_labels[i] != "O")

    @property
    def span_key_prefix(self) -> tuple[str, int]:
        """``(file_id, sent_idx)`` -- the first two fields of the T3 Span key."""
        return (self.file_id, self.sent_idx)


def build_examples(
    domain: str,
    *,
    tokenizer,
    truncation: bool,
    max_length: int | None,
    filter_max_tokens: int | None,
    data_cfg: DataConfig | None = None,
    scheme: str = "bio",
) -> list[Example]:
    """Tokenize and align one domain.

    ``filter_max_tokens`` is explicit rather than inferred from the domain:
    ``None`` filters nothing, an int drops sentences of at most that many
    dataset tokens. Callers pass ``TrainConfig.filter_for(domain, split)``.

    ``truncation=False`` with ``max_length=None`` is the gate's configuration:
    every dataset token reaches a first subword, so recovery can be compared to
    the loader's labels for exact equality. Under truncation the comparison
    fails on the 8 corpus-wide sentences that exceed 256 pieces for a reason
    that is not a bug -- ``data_layout.md`` section 8.2.
    """
    assert scheme in {"bio", "nobi"}, f"unsupported dataset scheme: {scheme}"
    assert truncation or max_length is None, "max_length is meaningless with truncation off"
    data_cfg = data_cfg or load_config()
    documents = load_domain(domain, data_cfg)
    domain_terms = load_domain_terms(data_cfg, domain) if scheme == "nobi" else None

    examples: list[Example] = []
    for doc in documents:
        for sent_idx, (tokens, labels) in enumerate(doc.sentences):
            # sent_idx is bound before the filter: turning the filter on or off
            # must not renumber sentences, or span keys stop matching
            # generate_flatten_spans and every exact-span number goes quietly
            # wrong.
            if filter_max_tokens is not None and len(tokens) <= filter_max_tokens:
                continue

            model_labels = labels
            label2id = LABEL2ID
            if scheme == "nobi":
                model_labels = build_nobi_labels(tokens, labels, domain_terms)
                label2id = NOBI_LABEL2ID

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=truncation,
                max_length=max_length,
            )
            word_ids = encoding.word_ids()
            examples.append(Example(
                domain=domain,
                file_id=doc.file_id,
                sent_idx=sent_idx,
                tokens=list(tokens),
                gold_labels=list(model_labels),
                #input_ids are indices that model will use them to look-up for emmeding vector that correspond to each model-token .
                input_ids=encoding["input_ids"],
                #A binary mask (1 for real tokens, 0 for padding) that instructs the encoder transformer to ignore padded positions during self-attention calculations.
                attention_mask=encoding["attention_mask"],
                labels=align_labels(word_ids, model_labels, label2id),
                word_ids=word_ids,
            ))
    return examples

#The Dataset / DataLoader contract why called DataLoader contract bcs it tell DataLoader what an item is that are going to convert it into a tensor latter .
class ATEDataset(Dataset):
    """Indexable view over ``list[Example]``.
    DataLoader calls __getitem__ repeatedly, gathers the results into a list, and hands that list to collate_fn, which turns it into a batch of tensors.
    so __getitem need to return a dict correspond to each item that habe field needed to build tensor , tensor needed to build tensor are : 
    inputs_ids , attention_mask , labels in sub-word token domain of course .
    """

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
            "example_index": index,#this used if we want to look up for something else use this index . 
        }
    #this is a must we need to know the max-token-size of example in batch bcs we will do dynamic padding 
    @property
    def subword_lengths(self) -> list[int]:
        return [example.n_subwords for example in self.examples]


class Collator:
    """``DataCollatorForTokenClassification`` plus the example index not DefualtDataCollector bcs we will do complex padding .
    The HF collator pads ``input_ids`` with [PAD] Tokens / ``attention_mask`` with 0 /``labels`` with -100 to the
    batch maximum not the global maximum lenght . for memory effieceny .
    """

    def __init__(self, tokenizer):
        self.collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=IGNORE_INDEX,
            return_tensors="pt",
        )
    #rem the pipeline __getitem__ → list of dicts (for 1 batch) → collate_fn → dict of tensors .
    #input is list of dict so each example is dict with required fields to build tensor of this batch .
    def __call__(self, features: list[dict]) -> dict:
        #remove the example_index you cannot input it to collator 
        indices = [feature.pop("example_index") for feature in features] 
        #now input is just a [{"input_ids": [...], "attention_mask": [...], "labels": [...]}, "input_ids": [...], "attention_mask": [...], "labels": [...], "example_index": 12 ]
        #the collator will find longest sentence and do padding depending on it and stack each field into a tensor {"input_ids": tensor(B, L), "attention_mask": tensor(B, L), "labels": tensor(B, L)}
        #B is number of example in batch if you access the first example you will find 1 vector with size L number of subword positions in the longest sequence in this batch the 768 does not get here yet .
        batch = self.collator(features)
        #lets put indices back one index per example those indices are necssary bcs we need to go back to (file_id, sent_idx)
        batch["example_index"] = torch.tensor(indices, dtype=torch.long)
        return batch


class LengthGroupedBatchSampler(Sampler):
    """Batches of similar subword length, in an order that still varies by seed.

    Shuffle all indices, cut into megabatches of ``multiplier * batch_size``,
    sort each megabatch by length descending, cut those into batches, then
    shuffle the batch order. Padding inside a batch stays small, so peak
    activation memory follows the batch's longest sentence rather than the
    corpus maximum -- 4 GB of VRAM makes that binding, and the corpus contains
    one 1,074-piece wind artifact.

    The longest batch lands first inside its megabatch, so an OOM shows up in
    the first epoch rather than after an hour of training.
    """

    def __init__(self, lengths: list[int], batch_size: int, generator: torch.Generator,
                 multiplier: int = 50, drop_last: bool = False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.generator = generator
        #each chunk will have 50 batch aka 800 example if batch_size is 16
        self.multiplier = multiplier
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.lengths)
        order = torch.randperm(n, generator=self.generator).tolist()
        mega = self.batch_size * self.multiplier

        batches: list[list[int]] = []
        for start in range(0, n, mega):
            chunk = order[start:start + mega]
            chunk.sort(key=lambda i: self.lengths[i], reverse=True)
            for offset in range(0, len(chunk), self.batch_size):
                batch = chunk[offset:offset + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        for position in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[position]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def build_dataloader(
    dataset: ATEDataset,
    *,
    tokenizer,
    batch_size: int,
    shuffle: bool, 
    length_grouped: bool,#if it is true it will cal the custom sampler that we did .
    seed: int | None = None,
) -> DataLoader:
    """One DataLoader. ``shuffle`` is training-only; dev and test run in loader
    order, which is sorted-file order, so a rerun batches identically."""
    collate_fn = Collator(tokenizer)
    if shuffle:
        assert seed is not None, "a shuffled loader needs a seed to be reproducible"
        generator = torch.Generator()
        generator.manual_seed(seed)
        if length_grouped:
            return DataLoader(
                dataset,
                batch_sampler=LengthGroupedBatchSampler(
                    dataset.subword_lengths, batch_size, generator),
                collate_fn=collate_fn,
            )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          generator=generator, collate_fn=collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


@dataclass
class Splits:
    """The three splits, plus their datasets for the way back out.

    ``datasets[split].examples[example_index]`` recovers tokens, file_id and
    sent_idx for a row of a batch.
    """
    loaders: dict[str, DataLoader]
    datasets: dict[str, ATEDataset]
    domains: dict[str, tuple[str, ...]]


def build_splits(train_cfg: TrainConfig, data_cfg: DataConfig | None = None,
                 tokenizer=None, seed: int | None = None,
                 scheme: str = "bio") -> Splits:
    """Train (corp + wind, pooled and shuffled), dev (equi), test (htfl).

    The domain assignment comes from ``configs/data.json`` -- one file states
    the split, and week-1 code reads the same one.
    """
    data_cfg = data_cfg or load_config()
    tokenizer = tokenizer or get_tokenizer(train_cfg)
    seed = train_cfg.seed if seed is None else seed

    assert data_cfg.dev_domain, "configs/data.json names no dev_domain"
    domains = {
        "train": tuple(data_cfg.train_domains),
        "dev": (data_cfg.dev_domain,),
        "test": (data_cfg.test_domain,),
    }

    datasets: dict[str, ATEDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split, split_domains in domains.items():
        examples: list[Example] = []
        for domain in split_domains:
            examples.extend(build_examples(
                domain,
                tokenizer=tokenizer,
                truncation=train_cfg.truncation,
                max_length=train_cfg.max_length if train_cfg.truncation else None,
                filter_max_tokens=train_cfg.filter_for(domain, split),
                data_cfg=data_cfg,
                scheme=scheme,
            ))
        dataset = ATEDataset(examples)
        datasets[split] = dataset
        loaders[split] = build_dataloader(
            dataset,
            tokenizer=tokenizer,
            batch_size=(train_cfg.per_device_train_batch_size if split == "train"
                        else train_cfg.eval_batch_size),
            shuffle=(split == "train"),
            length_grouped=train_cfg.length_grouped_batching,
            seed=seed if split == "train" else None,
        )

    return Splits(loaders=loaders, datasets=datasets, domains=domains)
