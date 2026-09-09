"""T7 run wrapper: seed, device, model, epochs, logging.

Generated scaffolding. The training step, the eval loop and the optimizer /
scheduler construction are hand-written in ``src/models/train_loop.py``; this
file only sequences them and records what happened.

Run:
    python -m src.models.run_train --reason "T7 first end-to-end run"
    python -m src.models.run_train --limit-train 64 --epochs 1 --skip-test   # smoke

Writes:
    results/runs/<run_id>.json      full config, seed, git commit, per-epoch loss,
                                    dev scores per epoch, one test score
    results/test_evaluations.log    one line per htfl evaluation, with its reason
    /runs/<run_id>/                 checkpoint, gitignored, only if save_checkpoint
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, set_seed

from src.data.dataset import (ID2LABEL, LABEL2ID, build_splits, get_tokenizer,
                              load_train_config)
from src.eval.run_eval import _gold_key_path, load_eval_config
from src.eval.scorers import load_gold_list_into_set
from src.models.train_loop import build_optimizer_and_scheduler, evaluate, train_step
from src.stats.loading import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_JSON = _REPO_ROOT / "results" / "runs"
_TEST_LOG = _REPO_ROOT / "results" / "test_evaluations.log"
_CKPT_ROOT = _REPO_ROOT / "runs"


def resolve_device(name: str) -> torch.device:
    """``"auto"`` -> cuda when present. Never hardcoded: T10 may run on Colab,
    and a code edit between runs stops results being comparable."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def git_state() -> dict:
    def run(*args):
        return subprocess.run(args, cwd=_REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        # a dirty tree means the JSON does not fully identify the code that ran
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def build_model(cfg, device: torch.device):
    """id2label / label2id go onto the config so a checkpoint is self-describing
    and a later run cannot silently reorder the classes."""
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.hf_cache_dir,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model.to(device)


def gold_for(domain: str, data_cfg, eval_cfg) -> dict[str, set[str]]:
    """Both tokenised gold keys for one domain, loaded once. The scorers never
    see a path -- that boundary is T3's, and it is why evaluate() takes sets."""
    return {key: load_gold_list_into_set(str(_gold_key_path(data_cfg, domain, key)))
            for key in eval_cfg["keys"]}


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, *, step_counter) -> float:
    """Returns the mean of the micro-batch losses.

    Mean of per-batch means, not a token-weighted mean: batches hold different
    numbers of scored tokens under dynamic padding. Fine as a curve, and it is
    what the JSON records -- not comparable to a per-token loss from elsewhere.
    """
    model.train()
    accumulation = cfg.gradient_accumulation_steps
    n_micro = len(loader)
    total = 0.0

    for i, batch in enumerate(loader, start=1):
        # the epoch's last window may be partial; it still steps, which is what
        # ceil(micro_batches / accumulation) in total_steps accounts for
        is_update = (i % accumulation == 0) or (i == n_micro)
        total += train_step(
            model, batch, optimizer, scheduler,
            device=device,
            gradient_accumulation_steps=accumulation,
            max_grad_norm=cfg.max_grad_norm,
            is_update_step=is_update,
        )
        if is_update:
            step_counter[0] += 1
            if step_counter[0] % cfg.log_every_steps == 0:
                print(f"    step {step_counter[0]:5d}  loss {total / i:.4f}  "
                      f"lr {optimizer.param_groups[0]['lr']:.3e}", flush=True)

    return total / n_micro


def main() -> None:
    parser = argparse.ArgumentParser(description="T7 end-to-end run.")
    parser.add_argument("--reason", default="", help="why this run exists; goes in the test log")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None,
                        help="smoke only: train on the first N sentences")
    parser.add_argument("--skip-test", action="store_true", help="dev only, no htfl")
    parser.add_argument("--tag", default="", help="suffix for the run id")
    args = parser.parse_args()

    cfg = load_train_config()
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, num_epochs=args.epochs)
    seed = args.seed if args.seed is not None else cfg.seed
    data_cfg = load_config()
    eval_cfg = load_eval_config()
    device = resolve_device(cfg.device)

    # one seed for weight init, dropout and data order -- the three sources T8
    # separates. set_seed covers the first two; build_splits takes it for the third.
    set_seed(seed)

    run_id = (f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
              f"_{cfg.model_name.replace('/', '-')}_lr{cfg.learning_rate:g}"
              f"_e{cfg.num_epochs}_seed{seed}{('_' + args.tag) if args.tag else ''}")
    print(f"run {run_id}\ndevice {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    tokenizer = get_tokenizer(cfg)
    splits = build_splits(cfg, data_cfg, tokenizer=tokenizer, seed=seed)

    if args.limit_train is not None:
        from src.data.dataset import ATEDataset, build_dataloader
        small = ATEDataset(splits.datasets["train"].examples[:args.limit_train])
        splits.datasets["train"] = small
        splits.loaders["train"] = build_dataloader(
            small, tokenizer=tokenizer, batch_size=cfg.per_device_train_batch_size,
            shuffle=True, length_grouped=cfg.length_grouped_batching, seed=seed)

    train_loader = splits.loaders["train"]
    model = build_model(cfg, device)
    optimizer, scheduler, total_steps = build_optimizer_and_scheduler(
        model, cfg, micro_batches_per_epoch=len(train_loader))

    print(f"train {len(splits.datasets['train']):,} sentences "
          f"({len(train_loader)} micro-batches x {cfg.num_epochs} epochs "
          f"/ accum {cfg.gradient_accumulation_steps} = {total_steps} optimizer steps)")
    print(f"dev {len(splits.datasets['dev']):,} | test {len(splits.datasets['test']):,}")

    dev_domain, test_domain = data_cfg.dev_domain, data_cfg.test_domain
    dev_gold = gold_for(dev_domain, data_cfg, eval_cfg)

    record = {
        "run_id": run_id,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": seed,
        "config": cfg.raw | {"num_epochs": cfg.num_epochs},
        "data": {
            "train_domains": list(data_cfg.train_domains),
            "dev_domain": dev_domain,
            "test_domain": test_domain,
            "n_train": len(splits.datasets["train"]),
            "n_dev": len(splits.datasets["dev"]),
            "n_test": len(splits.datasets["test"]),
            "micro_batches_per_epoch": len(train_loader),
            "total_optimizer_steps": total_steps,
            "limit_train": args.limit_train,
        },
        "epochs": [],
        "test": None,
        "reason": args.reason,
    }

    step_counter = [0]
    started = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        epoch_started = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device,
                               step_counter=step_counter)
        dev = evaluate(model, splits.loaders["dev"], splits.datasets["dev"],
                       device=device, id2label=ID2LABEL, gold_lists=dev_gold)
        record["epochs"].append({
            "epoch": epoch, "train_loss": loss, "dev": dev,
            "seconds": round(time.time() - epoch_started, 1),
        })
        print(f"  epoch {epoch}/{cfg.num_epochs}  train_loss {loss:.4f}  "
              f"{dev_domain} list_ann_f1 {dev['list_ann_f1']:.4f} "
              f"(P {dev['list_ann_p']:.4f} R {dev['list_ann_r']:.4f}, "
              f"{dev['n_pred_types']} types)  {time.time() - epoch_started:.0f}s", flush=True)

    # the tripwire: the schedule's horizon and the number of updates that
    # actually happened must be the same number, or warmup was not 10% of
    # anything and T12 measures a condition nobody wrote down
    assert step_counter[0] == total_steps, (
        f"optimizer stepped {step_counter[0]} times, schedule was built for {total_steps}")
    record["realised_optimizer_steps"] = step_counter[0]

    if not args.skip_test:
        test_gold = gold_for(test_domain, data_cfg, eval_cfg)
        test = evaluate(model, splits.loaders["test"], splits.datasets["test"],
                        device=device, id2label=ID2LABEL, gold_lists=test_gold)
        record["test"] = test
        print(f"  {test_domain} list_ann_f1 {test['list_ann_f1']:.4f} "
              f"(P {test['list_ann_p']:.4f} R {test['list_ann_r']:.4f}) "
              f"| nes_f1 {test['list_nes_f1']:.4f} | {test['n_pred_types']} types")

        # selection happens on dev, always; every test look is counted, with a reason
        _TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_TEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{record['started']}\t{run_id}\tseed={seed}\t"
                     f"lr={cfg.learning_rate:g}\tepochs={cfg.num_epochs}\t"
                     f"htfl_list_ann_f1={test['list_ann_f1']:.4f}\t"
                     f"htfl_list_nes_f1={test['list_nes_f1']:.4f}\t"
                     f"reason={args.reason or 'UNSTATED'}\n")

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["duration_sec"] = round(time.time() - started, 1)

    _RUNS_JSON.mkdir(parents=True, exist_ok=True)
    out = _RUNS_JSON / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(_REPO_ROOT)}")

    if cfg.save_checkpoint:
        ckpt = _CKPT_ROOT / run_id
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"wrote checkpoint {ckpt} (gitignored)")


if __name__ == "__main__":
    main()
