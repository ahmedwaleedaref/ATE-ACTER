"""T7 training and evaluation loop -- the three functions written by hand.

Same rule as ``src/data/align.py``: everything in this file is the author's,
everything around it (``run_train.py``, the tests) is scaffolding. The reason
is the same one -- these three are where the LR schedule, the gradient
accumulation arithmetic and the inference path live, and T12 measures the
first two.

What this file does NOT do, on purpose:

  * decode or score anything itself. ``evaluate`` calls ``src/eval/`` --
    ``decode``, ``spans_to_unique_list``, ``score_list``, ``score_exact_spans``.
    If a number is wrong there is one place it can be wrong.
  * read config, paths, or the corpus. Gold sets arrive as arguments.
  * know about epochs. The epoch wrapper is in ``run_train.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import transformers

from src.data.align import recover_token_labels
from src.eval.scorers import score_list
from src.eval.spans import decode
from src.eval.surface import spans_to_unique_list

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from src.data.dataset import ATEDataset, TrainConfig

Span = tuple[str, int, int, int]  # (file_id, sent_idx, start, end) -- T3 key


def build_optimizer_and_scheduler(
    model,
    cfg: "TrainConfig",
    *,
    micro_batches_per_epoch: int,
) -> tuple["torch.optim.Optimizer", object, int]:
    
    assert cfg.lr_schedule == "linear"
    
    weight_decay = cfg.weight_decay 
    lr = cfg.learning_rate 
    
    #we want to execlude layer norm wieghts and bais from wieght decaying
    no_decay = ["bias", "LayerNorm.weight"] 
    params_without_decay = [parameter for parameter_name , parameter in model.named_parameters() if  any(k in parameter_name for k in no_decay) ]
    params_with_decay = [parameter for parameter_name , parameter in model.named_parameters() if not any(k in parameter_name for k in no_decay) ]
    groups = [
    {"params": params_with_decay, "weight_decay": weight_decay},
    {"params": params_without_decay, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(groups , lr=lr)
    
    #need to know total_opt_steps 
    n_of_epochs = cfg.num_epochs 
    grad_acc = cfg.gradient_accumulation_steps 
    warmup_ratio = cfg.warmup_ratio
    #care that micro_batches_per_epoch is number of mini-batches not batches this why we must divide by grad_acc
    #ceil per EPOCH, not floor over the whole run: run_train.py steps on each
    #epoch's final partial accumulation window, so an odd micro-batch count
    #makes floor undercount and the run ends with updates at LR 0.
    #574/epoch: both give 1435. 575/epoch: floor 1437, ceil 1440.
    updates_per_epoch = math.ceil(micro_batches_per_epoch / grad_acc)
    total_opt_steps = updates_per_epoch * n_of_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=int(warmup_ratio * total_opt_steps),
                                                             num_training_steps=total_opt_steps)
    return (optimizer , scheduler , total_opt_steps)


def train_step(
    model,
    batch: dict,
    optimizer: "torch.optim.Optimizer",
    scheduler,
    *,
    device: "torch.device",
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    is_update_step: bool,
) -> float:
    batch.pop("example_index") 
    #we need to make data and model on same device 
    batch = {k : t.to(device) for k , t in batch.items() }#now each tensor in dict on same device as model 

    weights = batch.pop("weights")
    output = model(**batch)
    per_position_loss = torch.nn.functional.cross_entropy(
        output.logits.view(-1, output.logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100,
        reduction="none",
    )
    n_scored = (batch["labels"] != -100).sum()
    loss = (per_position_loss * weights.view(-1)).sum() / n_scored
    loss = loss / gradient_accumulation_steps

    loss.backward()
    
    if is_update_step : 
         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # do normal cliping so gradient stay in same direction 
         optimizer.step(); scheduler.step(); optimizer.zero_grad()
    return output.loss.item() 
    


def evaluate(
    model,
    loader: "DataLoader",
    dataset: "ATEDataset",
    *,
    device: "torch.device",
    id2label: dict[int, str],
    gold_lists: dict[str, set[str]],
    #exact-span metric deferred: accepted and IGNORED for now. T7's headline is
    #unique-list F1; span F1 is diagnostic and T11 is what needs it. Adding it is
    #one score_exact_spans call over (file_id, sent_idx, start, end) keys built
    #from the same decoded spans, so nothing else changes shape.
    gold_spans: set[Span] | None = None,
) -> dict:
    model.eval()
    spans_used_for_uniqe_list = [] 
    #no_grad: nothing here is backpropagated, and keeping the graph for 3,090
    #equi sentences is what makes an eval pass run out of 4 GB.
    with torch.no_grad() :
        for batch in loader : 
            idx = batch.pop("example_index").tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**batch).logits.argmax(-1).tolist() #(B , L)so for each sentecnde you have list of indices
            for index , i in enumerate(idx) : 
                example = dataset.examples[i]
                #pred[index] is padded to the batch max; recover_token_labels walks
                #word_ids, which is this example's own length, so padding is never read
                model_labels = recover_token_labels(example.word_ids , pred[index] , len(example.tokens) , id2label)
                spans_used_for_uniqe_list.append(( example.tokens , decode(example.tokens , model_labels , "bio")))

    #spans_to_unique_list returns a 3-tuple, not a set; the counts are the span/type ratio
    uniqe_list_predicted , n_spans , n_unique = spans_to_unique_list(spans_used_for_uniqe_list)

    #score_list(PREDICTED, GOLD) -- that order. One predicted list, scored against both keys.
    results : dict = {"n_pred_spans": n_spans, "n_pred_types": n_unique}
    for key , gold in gold_lists.items() :
        precision , recall , f1 = score_list(uniqe_list_predicted , gold)
        results[f"list_{key}_p"] = precision
        results[f"list_{key}_r"] = recall
        results[f"list_{key}_f1"] = f1
    return results
