"""The LR schedule, tested without a model or a GPU.

Two silent failures this catches, both of which corrupt T12 rather than crash:

  * stepping the scheduler per epoch instead of per optimizer step -- "10%
    warmup" then means the first 10% of EPOCHS;
  * computing the horizon from micro-batches instead of optimizer steps -- the
    schedule reaches LR 0 partway through and the rest of the run trains at
    nothing.

Neither raises. Both change what every later number means.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from src.data.dataset import load_train_config
from src.models.train_loop import build_optimizer_and_scheduler


class TinyModel(torch.nn.Module):
    """Real parameter names -- the decay split keys off "bias" and
    "LayerNorm.weight", so a stub without those tests nothing."""

    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 4)
        self.LayerNorm = torch.nn.LayerNorm(4)


@pytest.fixture(scope="module")
def cfg():
    return load_train_config()


@pytest.mark.parametrize("micro_batches,accumulation,epochs,expected", [
    (574, 2, 5, 1435),   # the real T7 run
    (574, 1, 5, 2870),   # no accumulation
    (575, 2, 5, 1440),   # odd: ceil per epoch (288*5), NOT floor over the run (1437)
    (7, 4, 3, 6),        # ceil(7/4)=2 per epoch
])
def test_total_steps_counts_optimizer_steps(cfg, micro_batches, accumulation, epochs, expected):
    small = dataclasses.replace(cfg, gradient_accumulation_steps=accumulation,
                                num_epochs=epochs)
    _, _, total = build_optimizer_and_scheduler(
        TinyModel(), small, micro_batches_per_epoch=micro_batches)
    assert total == expected
    assert total == math.ceil(micro_batches / accumulation) * epochs


def test_warmup_covers_the_stated_fraction_of_optimizer_steps(cfg):
    """Warmup peaks at warmup_ratio x total_steps and the LR ends at 0."""
    model = TinyModel()
    optimizer, scheduler, total = build_optimizer_and_scheduler(
        model, cfg, micro_batches_per_epoch=574)

    lrs = []
    for _ in range(total):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    peak_step = max(range(len(lrs)), key=lambda i: lrs[i]) + 1
    assert peak_step == int(cfg.warmup_ratio * total), (
        f"warmup peaks at step {peak_step}, expected {int(cfg.warmup_ratio * total)} "
        f"({cfg.warmup_ratio:.0%} of {total})")
    assert lrs[peak_step - 1] == pytest.approx(cfg.learning_rate)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)
    # strictly decreasing after the peak: a linear schedule, not a plateau
    assert all(a > b for a, b in zip(lrs[peak_step - 1:], lrs[peak_step:]))


def test_weight_decay_excludes_bias_and_layernorm(cfg):
    """The standard BERT recipe. Recorded because it changes the numbers."""
    model = TinyModel()
    optimizer, _, _ = build_optimizer_and_scheduler(model, cfg, micro_batches_per_epoch=10)

    decayed = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    for name, parameter in model.named_parameters():
        should_decay = not ("bias" in name or "LayerNorm.weight" in name)
        assert (id(parameter) in decayed) == should_decay, name
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))


def test_train_config_reads_every_key_in_the_file(cfg):
    """Every non-underscore key in configs/train.json reaches TrainConfig.

    A key the config states and the code ignores is the quietest failure in this
    project: the file says warmup 0.1 or max_grad_norm 1.0, the run does
    something else, and the JSON log records the file. Underscore keys are
    prose notes and are excluded.
    """
    stated = {k for k in cfg.raw if not k.startswith("_")}
    nested = {"short_sentence_filter"}          # flattened into filter_* fields
    fields = {f.name for f in dataclasses.fields(cfg)}
    missing = (stated - nested) - fields
    assert not missing, f"configs/train.json states {sorted(missing)}, TrainConfig ignores them"
