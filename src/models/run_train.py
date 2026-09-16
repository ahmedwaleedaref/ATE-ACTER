"""T7 run wrapper: seed, device, model, epochs, logging.

Generated scaffolding. The training step, the eval loop and the optimizer /
scheduler construction are hand-written in ``src/models/train_loop.py``; this
file only sequences them and records what happened.

Run one seed:
    python -m src.models.run_train --seed 42 --reason "T8 seed variance"
    python -m src.models.run_train --seed 42 --limit-train 64 --epochs 1 --skip-test

T8's five seeds, then the aggregation:
    for s in 42 43 44 45 46; do python -m src.models.run_train --seed $s --reason "T8 seed variance" || break; done && python -m src.aggregate

Writes:
    results/runs/seed_<seed>.json   full config, seed, git commit, per-epoch loss
                                    and equi scores, best epoch, one htfl score
    results/test_evaluations.log    one line per htfl evaluation, with its reason

No checkpoints are written to disk. The best-epoch weights live in memory for
the single htfl evaluation at the end of the run and are then dropped.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForTokenClassification, set_seed

from src.data.dataset import (ID2LABEL, LABEL2ID, build_splits, get_tokenizer,
                              load_train_config)
from src.eval.nobi import NOBI_ID2LABEL, NOBI_LABEL2ID
from src.eval.run_eval import _gold_key_path, load_eval_config
from src.eval.scorers import load_gold_list_into_set
from src.models.train_loop import build_optimizer_and_scheduler, evaluate, train_step
from src.stats.loading import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_JSON = _REPO_ROOT / "results" / "runs"
_TEST_LOG = _REPO_ROOT / "results" / "test_evaluations.log"
_CKPT_ROOT = _REPO_ROOT / "runs"          # gitignored
_EVAL_CONFIGS = {
    "bio": _REPO_ROOT / "configs" / "eval.yaml",
    "nobi": _REPO_ROOT / "configs" / "nobi_eval.yaml",
}


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


def build_model(cfg, device: torch.device, *, id2label=None, label2id=None):
    """id2label / label2id go onto the config so a checkpoint is self-describing
    and a later run cannot silently reorder the classes.

    ``dtype=torch.float32`` is load-bearing, not tidiness. from_pretrained keeps
    the checkpoint's own dtype, and deberta-v3-base ships fp16 where bert and
    roberta ship fp32. Training fp16 weights with AdamW and no GradScaler is
    silently broken: exp_avg_sq = (1-b2)*g^2 = 1e-3*g^2 underflows to 0 in fp16,
    and AdamW's eps=1e-8 is below fp16's smallest subnormal (~6e-8) so it
    underflows too. denom becomes 0, the update becomes exp_avg/0 = inf, and the
    parameter goes NaN on the FIRST step -- at any learning rate, including the
    lr=0 that warmup starts at. It presents as "deberta diverges", not as an
    error. The assertion below is what stops it coming back.
    """
    id2label = ID2LABEL if id2label is None else id2label
    label2id = LABEL2ID if label2id is None else label2id
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        cache_dir=cfg.hf_cache_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.float32,
        ignore_mismatched_sizes=(len(label2id) != len(LABEL2ID)),
    )
    bad = {n: p.dtype for n, p in model.named_parameters() if p.dtype is not torch.float32}
    assert not bad, f"non-fp32 parameters would break AdamW silently: {bad}"
    return model.to(device)


def encoder_weight_hash(model) -> str:
    """sha256 over the pretrained encoder weights only, classifier head excluded.

    Must be byte-identical across seeds: from_pretrained reads these from the
    checkpoint, and the head (768 x 3 + 3 = 2,307 parameters) is the only thing
    ``set_seed`` touches at construction. If this differs between two runs they
    are not two samples of one config, and T8's std means nothing --
    ``src/aggregate.py`` asserts on it.

    Selection is by ``base_model_prefix`` ("bert" here, "roberta"/"deberta" for
    T10's encoders) rather than by excluding the name "classifier", so the head
    stays excluded when the encoder changes.
    """
    prefix = model.base_model_prefix + "."
    named = [(n, t) for n, t in model.named_parameters() if n.startswith(prefix)]
    assert named, f"no parameters under {prefix!r} -- base_model_prefix is wrong"
    digest = hashlib.sha256()
    for name, tensor in sorted(named, key=lambda kv: kv[0]):
        digest.update(name.encode())
        # .numpy() assumes fp32 weights, which is what from_pretrained gives here;
        # it raises rather than hashing something wrong if that ever changes
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gold_for(domain: str, data_cfg, eval_cfg) -> dict[str, set[str]]:
    """Both tokenised gold keys for one domain, loaded once. The scorers never
    see a path -- that boundary is T3's, and it is why evaluate() takes sets."""
    return {key: load_gold_list_into_set(str(_gold_key_path(data_cfg, domain, key)))
            for key in eval_cfg["keys"]}


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, *,
                    step_counter, first_indices=None) -> float:
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
        # read before train_step, which pops "example_index" off the batch.
        # This is the shuffle order the seed controls -- the third noise source.
        if first_indices is not None and len(first_indices) < 10:
            want = 10 - len(first_indices)
            first_indices.extend(batch["example_index"].tolist()[:want])
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
    parser.add_argument("--seed", type=int, required=True,
                        help="required: T8 fixes 42-46 up front, never chosen as you go")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", default=None,
                        help="override model_name. T10 sweeps encoders at T9's fixed "
                             "hyperparameters, so the encoder is a flag, not a config edit")
    parser.add_argument("--lr", type=float, default=None, help="override learning_rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="override weight_decay")
    parser.add_argument("--scheme", choices=("bio", "nobi"), default="bio",
                        help="labeling scheme; BIO is the compatibility default")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="smoke only: train on the first N sentences")
    parser.add_argument("--limit-eval", type=int, default=None,
                        help="smoke only: evaluate the first N dev/test sentences")
    parser.add_argument("--skip-test", action="store_true", help="dev only, no htfl")
    parser.add_argument("--save-weights", action="store_true",
                        help="write the BEST-EPOCH weights to /runs/<run_id>/ (gitignored). "
                             "Off by default: T8-T10 are 55 runs and none of them needs a "
                             "checkpoint. T11's breakdowns do -- they need per-span "
                             "predictions from a concrete model.")
    parser.add_argument("--tag", default="", help="suffix for the run id")
    parser.add_argument("--group", default="",
                        help="subdirectory under results/runs/ to write into. T9 gives "
                             "each grid cell its own, so cells cannot overwrite each other")
    args = parser.parse_args()

    scheme = args.scheme
    id2label = ID2LABEL if scheme == "bio" else NOBI_ID2LABEL
    label2id = LABEL2ID if scheme == "bio" else NOBI_LABEL2ID

    cfg = load_train_config()
    # CLI overrides exist so a one-number experiment is not a config edit that
    # someone forgets to revert. The effective value is what gets logged.
    overrides = {}
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
        print("overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
    seed = args.seed
    # cfg.seed must reflect what actually ran. record["config"] is built from cfg,
    # and --seed was never folded in, so every run recorded config.seed = whatever
    # train.json said -- a seed-46 run wrote config.seed = 42. Not in `overrides`:
    # that dict is for axes a run deliberately varies, and the seed is already the
    # top-level "seed" field and half the filename.
    cfg = dataclasses.replace(cfg, seed=seed)
    data_cfg = load_config()
    eval_config_path = _EVAL_CONFIGS[scheme]
    eval_cfg = load_eval_config(eval_config_path)
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
    splits = build_splits(cfg, data_cfg, tokenizer=tokenizer, seed=seed, scheme=scheme)

    if args.limit_train is not None:
        from src.data.dataset import ATEDataset, build_dataloader
        small = ATEDataset(splits.datasets["train"].examples[:args.limit_train])
        splits.datasets["train"] = small
        splits.loaders["train"] = build_dataloader(
            small, tokenizer=tokenizer, batch_size=cfg.per_device_train_batch_size,
            shuffle=True, length_grouped=cfg.length_grouped_batching, seed=seed)

    if args.limit_eval is not None:
        from src.data.dataset import ATEDataset, build_dataloader
        for split in ("dev", "test"):
            limited = ATEDataset(splits.datasets[split].examples[:args.limit_eval])
            splits.datasets[split] = limited
            splits.loaders[split] = build_dataloader(
                limited,
                tokenizer=tokenizer,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                length_grouped=False,
            )

    train_loader = splits.loaders["train"]
    # set_seed above ran BEFORE this line: from_pretrained initialises the
    # classifier head randomly, and that head is the dominant noise source.
    model = build_model(cfg, device, id2label=id2label, label2id=label2id)
    enc_hash = encoder_weight_hash(model)   # before any training step
    print(f"encoder sha256 {enc_hash[:16]}... (must match across seeds)")
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
        "scheme": scheme,
        "eval_config": str(eval_config_path.relative_to(_REPO_ROOT)),
        "encoder_weight_hash": enc_hash,
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "config": cfg.raw | {f.name: getattr(cfg, f.name)
                             for f in dataclasses.fields(cfg) if f.name != "raw"},
        "overrides": overrides,
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
            "limit_eval": args.limit_eval,
        },
        "epochs": [],
        "first_batch_indices": [],
        "best_epoch": None,
        "best_equi_f1": None,
        "htfl_f1": None,
        "collapsed": None,
        "test": None,
        "reason": args.reason,
    }

    step_counter = [0]
    # best epoch on equi, held in memory only -- item 7, no checkpoints on disk.
    # .clone() because .cpu() is a no-op returning the live tensor on a CPU run,
    # which would leave best_state aliasing weights that keep training.
    best_epoch, best_equi_f1, best_state = None, -1.0, None
    started = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        epoch_started = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device,
                               step_counter=step_counter,
                               first_indices=record["first_batch_indices"] if epoch == 1 else None)
        dev = evaluate(model, splits.loaders["dev"], splits.datasets["dev"],
                       device=device, id2label=id2label, gold_lists=dev_gold,
                       scheme=scheme)
        # ANN unique-list F1 is the headline metric (configs/eval.yaml), and the
        # per-seed statistic T8 fixes is the best epoch on equi under it
        equi_f1 = dev["list_ann_f1"]
        record["epochs"].append({
            "epoch": epoch, "train_loss": loss, "dev": dev,
            "equi_f1": equi_f1, "equi_p": dev["list_ann_p"], "equi_r": dev["list_ann_r"],
            "seconds": round(time.time() - epoch_started, 1),
        })
        if equi_f1 > best_equi_f1:
            best_epoch, best_equi_f1 = epoch, equi_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch}/{cfg.num_epochs}  train_loss {loss:.4f}  "
              f"{dev_domain} list_ann_f1 {equi_f1:.4f} "
              f"(P {dev['list_ann_p']:.4f} R {dev['list_ann_r']:.4f}, "
              f"{dev['n_pred_types']} types)  {time.time() - epoch_started:.0f}s"
              + ("  <- best" if epoch == best_epoch else ""), flush=True)

    # the tripwire: the schedule's horizon and the number of updates that
    # actually happened must be the same number, or warmup was not 10% of
    # anything and T12 measures a condition nobody wrote down
    assert step_counter[0] == total_steps, (
        f"optimizer stepped {step_counter[0]} times, schedule was built for {total_steps}")
    record["realised_optimizer_steps"] = step_counter[0]

    assert best_state is not None, "no epoch ran, so there is no best epoch"
    record["best_epoch"] = best_epoch
    record["best_equi_f1"] = best_equi_f1
    # flat, majority-class output: every token O, nothing decoded. Recorded and
    # kept -- Tasks_week2.md T8 -- never dropped and never re-rolled.
    record["collapsed"] = (best_equi_f1 == 0.0)
    print(f"  best epoch {best_epoch} ({dev_domain} list_ann_f1 {best_equi_f1:.4f})"
          + ("  COLLAPSED" if record["collapsed"] else ""))

    if not args.skip_test:
        # htfl is evaluated ONCE per run, on the best-equi weights -- never per
        # epoch. Selection already happened above, on equi.
        model.load_state_dict(best_state)
        test_gold = gold_for(test_domain, data_cfg, eval_cfg)
        test = evaluate(model, splits.loaders["test"], splits.datasets["test"],
                        device=device, id2label=id2label, gold_lists=test_gold,
                        scheme=scheme)
        record["test"] = test
        record["htfl_f1"] = test["list_ann_f1"]
        print(f"  {test_domain} list_ann_f1 {test['list_ann_f1']:.4f} "
              f"(P {test['list_ann_p']:.4f} R {test['list_ann_r']:.4f}) "
              f"| nes_f1 {test['list_nes_f1']:.4f} | {test['n_pred_types']} types")

        # selection happens on dev, always; every test look is counted, with a reason
        _TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_TEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{record['started']}\t{run_id}\tseed={seed}\t"
                     f"scheme={scheme}\t"
                     f"lr={cfg.learning_rate:g}\tepochs={cfg.num_epochs}\t"
                     f"best_epoch={best_epoch}\t"
                     f"htfl_list_ann_f1={test['list_ann_f1']:.4f}\t"
                     f"htfl_list_nes_f1={test['list_nes_f1']:.4f}\t"
                     f"reason={args.reason or 'UNSTATED'}\n")

    if args.save_weights:
        # best_state, not the final weights: the run's number is its best epoch,
        # so saving the last epoch would ship a model that scores something else.
        ckpt = _CKPT_ROOT / run_id
        ckpt.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(best_state)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        record["checkpoint"] = str(ckpt.relative_to(_REPO_ROOT))
        print(f"  wrote best-epoch weights to {ckpt.relative_to(_REPO_ROOT)}/ (gitignored)")

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["wall_time_sec"] = round(time.time() - started, 1)

    # seed_<seed>.json, not <run_id>.json: one directory holds the five seeds of
    # ONE config and aggregate.py globs seed_*.json inside it. --group is what
    # keeps T9's cells from overwriting each other, and E02, at the default path.
    assert not Path(args.group).is_absolute() and ".." not in Path(args.group).parts, \
        f"--group must be a relative path under results/runs/: {args.group!r}"
    output_group = args.group or ("nobi" if scheme == "nobi" else "")
    out_dir = _RUNS_JSON / output_group if output_group else _RUNS_JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"seed_{seed}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
