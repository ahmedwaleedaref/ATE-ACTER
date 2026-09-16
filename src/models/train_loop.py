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
from src.eval.scorers import score_list , score_exact_spans
from src.eval.spans import decode , count_invalid_tags
from src.eval.surface import spans_to_unique_list , generate_flatten_spans_for_model_prediction

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
    output = model(**batch) 
    loss = output.loss  / gradient_accumulation_steps 
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
    #the decoded GOLD spans for this split, (file_id, sent_idx, start, end).
    #Loaded once by the caller, like gold_lists -- evaluate() never reads the
    #corpus. None gives list metrics only, which is what a smoke run wants.
    gold_spans: set[Span] | None = None,
) -> dict:
    model.eval()
    spans_used_for_uniqe_list = [] 
    spans_used_for_exact_span = []
    invalid_tags_total : dict[str,int] = {"i_sentence_initial": 0, "i_after_o": 0}
    #no_grad: nothing here is backpropagated, and keeping the graph for 3,090
    #equi sentences is what makes an eval pass run out of 4 GB.
    with torch.no_grad() :
        for batch in loader : 
            idx = batch.pop("example_index").tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**batch).logits.argmax(-1).tolist() #(B , L)so for each sentecnde you have list of indices
            for index , i in enumerate(idx) : 
                example = dataset.examples[i]
                file_id : str = example.file_id
                index_sentence_in_file : int = example.sent_idx
                #pred[index] is padded to the batch max; recover_token_labels walks
                #word_ids, which is this example's own length, so padding is never read
                model_labels = recover_token_labels(example.word_ids , pred[index] , len(example.tokens) , id2label)
                #counted on model_labels, the same sequence decode is about to
                #consume -- never on pred, whose continuation subwords and
                #special tokens are discarded unread
                invalid = count_invalid_tags(model_labels , "bio")
                for name in invalid_tags_total :
                    invalid_tags_total[name] += invalid[name]
                spans_list : list[tuple[int,int]] = decode(example.tokens , model_labels , "bio")
                spans_used_for_uniqe_list.append(( example.tokens , spans_list))
                spans_used_for_exact_span.append((file_id,index_sentence_in_file , spans_list ))
    #spans_to_unique_list returns a 3-tuple, not a set; the counts are the span/type ratio
    uniqe_list_predicted , n_spans , n_unique = spans_to_unique_list(spans_used_for_uniqe_list)
    model_flatten_spans = generate_flatten_spans_for_model_prediction(spans_used_for_exact_span)
    
    #score_list(PREDICTED, GOLD) -- that order. One predicted list, scored against both keys.
    results : dict = {
        "n_pred_spans": n_spans,
        "n_pred_types": n_unique,
        #IOB2 violations the model emitted. Gold is strict IOB2 -- 0 I-after-O in
        #222,281 tokens, 3 sentence-initial I corpus-wide -- so these are the
        #model's alone. Free to collect: the dangling-I policy already drops them
        #in decode, so this only reports what was being discarded silently.
        "n_i_sentence_initial": invalid_tags_total["i_sentence_initial"],
        "n_i_after_o": invalid_tags_total["i_after_o"],
    }
    for key , gold in gold_lists.items() :
        precision , recall , f1 = score_list(uniqe_list_predicted , gold)
        results[f"list_{key}_p"] = precision
        results[f"list_{key}_r"] = recall
        results[f"list_{key}_f1"] = f1

    #OUTSIDE the key loop, and with no key suffix. Gold spans are decoded from
    #the without_named_entities labels -- one set per domain. ANN/NES is a
    #property of the answer KEY and exists only in the list metric, so there is
    #exactly one exact-span number per split, not one per key.
    if gold_spans is not None :
        exact_span_precision , exact_span_recall , exact_span_f1 = score_exact_spans(model_flatten_spans , gold_spans)
        results["span_p"] = exact_span_precision
        results["span_r"] = exact_span_recall
        results["span_f1"] = exact_span_f1

    return results
