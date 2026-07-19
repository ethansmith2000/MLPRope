#!/usr/bin/env python
"""Causal LM training for position-bias experiments."""
from __future__ import annotations

import argparse
import fcntl
import inspect as _inspect
import json
import logging
import math
import os
import tempfile
import time
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, default_data_collator, get_scheduler

from transformer import Transformer, count_parameters, suggest_matched_baselines

REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE", REPO_DIR.parent))
CACHE_DIR = WORKSPACE_DIR / ".cache"

# Keep HF / compile caches on the workspace volume (shared with noble_research).
os.environ.setdefault("HF_HOME", str(WORKSPACE_DIR / ".hf_home"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(os.environ["HF_HOME"]) / "datasets"),
)
os.environ.setdefault("TRITON_CACHE_DIR", str(CACHE_DIR / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR / "torchinductor"))
os.environ.setdefault("TMPDIR", str(CACHE_DIR / "tmp"))
for cache_path in (
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_DATASETS_CACHE"]),
    Path(os.environ["TRITON_CACHE_DIR"]),
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]),
    Path(os.environ["TMPDIR"]),
):
    cache_path.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = os.environ["TMPDIR"]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _safe_getsourcelines(obj):
    """Work around Triton+Python 'source code not available' under torch.compile."""
    try:
        return _inspect._orig_getsourcelines(obj)  # type: ignore[attr-defined]
    except OSError as exc:
        if "source code not available" in str(exc):
            return ["# source code not available\n"], 0
        raise


if not hasattr(_inspect, "_orig_getsourcelines"):
    _inspect._orig_getsourcelines = _inspect.getsourcelines  # type: ignore[attr-defined]
    _inspect.getsourcelines = _safe_getsourcelines

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

_world_size = int(os.environ.get("WORLD_SIZE", 1))
_threads_per_process = max(2, 12 // max(_world_size, 1))
torch.set_num_threads(_threads_per_process)
torch.set_num_interop_threads(2)

logger = get_logger(__name__)


DEFAULT_CONFIG = {
    "dataset_name": "Skylion007/openwebtext",
    "dataset_config_name": "plain_text",
    "train_file": None,
    "validation_file": None,
    "validation_split_percentage": 5,
    "model_name_or_path": "openai-community/gpt2",
    "tokenizer_name": None,
    "use_slow_tokenizer": False,
    "hf_cache_dir": os.environ["HF_DATASETS_CACHE"],
    "tokenized_dataset_path": None,
    "preprocessing_num_workers": min(8, os.cpu_count() or 1),
    "overwrite_cache": False,
    "block_size": 1024,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "max_train_steps": 10_000,
    "learning_rate": 3.0e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "num_warmup_steps": 200,
    "seed": 123,
    "max_grad_norm": 1.0,
    "checkpointing_steps": None,
    "resume_from_checkpoint": None,
    "output_dir": None,
    "base_output_dir": str(REPO_DIR / "model-output"),
    "run_name": None,
    "with_tracking": False,
    "report_to": "wandb",
    "wandb_project": "mlprope-position-bias",
    "wandb_group": None,
    "mixed_precision": "bf16",
    "num_workers": min(8, os.cpu_count() or 1),
    "num_validation_batches": 25,
    "validate_every": 500,
    "log_every_n_steps": 10,
    "dry_run": False,
    "print_model": False,
    # Model
    "hidden_size": 768,
    "depth": 8,
    "n_head": 8,
    "ff_mult": 4,
    "rope_theta": 10000.0,
    "qk_norm": True,
    # Phase 1: rope | add_rope | linear | low_rank | mlp_rope.
    # Phase 2 stubs: inkling_table | inkling_cosnet.
    "pos_variant": "rope",
    "rel_extent": None,  # None follows block_size.
    "pos_rank": 32,
    "pos_mlp_hidden": 128,
    # Learned logit biases require flex; rope supports sdpa or flex.
    "attn_impl": "sdpa",
    "gradient_checkpointing": False,
    "compile": True,
    "compile_mode": "reduce-overhead",
    "compile_fullgraph": True,
    "optimizer": "adamw",
    "beta1": 0.9,
    "beta2": 0.98,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Position-bias GPT training")
    parser.add_argument("--override_json", type=str, default=None)
    parser.add_argument(
        "--pos_variant",
        choices=(
            "rope",
            "add_rope",
            "linear",
            "low_rank",
            "mlp_rope",
            "inkling_table",
            "inkling_cosnet",
        ),
        default=None,
    )
    parser.add_argument(
        "--attn_impl",
        choices=("sdpa", "flex"),
        default=None,
    )
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--print_model", action="store_true")
    return parser.parse_args()


def load_json_overrides(path: str | Path, seen: set[Path] | None = None) -> dict:
    config_path = Path(path).expanduser().resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        chain_str = " -> ".join(str(item) for item in (*seen, config_path))
        raise ValueError(f"Config inheritance cycle: {chain_str}")
    seen.add(config_path)

    with config_path.open() as config_file:
        overrides = json.load(config_file)
    base_config = overrides.pop("base_config", None)
    if base_config is None:
        return overrides
    if not isinstance(base_config, str):
        raise TypeError("base_config must be a JSON file path")
    base_path = Path(base_config).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    merged = load_json_overrides(base_path, seen)
    merged.update(overrides)
    return merged


def load_config(cli_args):
    cfg = dict(DEFAULT_CONFIG)
    if cli_args.override_json is not None:
        overrides = load_json_overrides(cli_args.override_json)
        unknown = set(overrides) - set(cfg)
        # Allow forward-looking keys used only by the matching helper stub later.
        unknown -= {"_matching_todo"}
        if unknown:
            raise ValueError(f"Unknown override keys: {sorted(unknown)}")
        cfg.update(overrides)
    if cli_args.pos_variant is not None:
        cfg["pos_variant"] = cli_args.pos_variant
    if cli_args.attn_impl is not None:
        cfg["attn_impl"] = cli_args.attn_impl
    if cli_args.max_train_steps is not None:
        cfg["max_train_steps"] = cli_args.max_train_steps
    if cli_args.dry_run:
        cfg["dry_run"] = True
    if cli_args.print_model:
        cfg["print_model"] = True

    model_tag = f"h{cfg['hidden_size']}d{cfg['depth']}"
    variant_tag = cfg["pos_variant"]
    if cfg["pos_variant"] == "low_rank":
        variant_tag = f"{variant_tag}-r{cfg['pos_rank']}"
    elif cfg["pos_variant"] == "mlp_rope":
        variant_tag = f"{variant_tag}-m{cfg['pos_mlp_hidden']}"
    if cfg["pos_variant"] != "rope":
        rel_extent = cfg["rel_extent"] or cfg["block_size"]
        variant_tag = f"{variant_tag}-e{rel_extent}"
    if cfg["attn_impl"] == "flex" and cfg["pos_variant"] == "rope":
        variant_tag = "rope-flex"
    run_name = cfg["run_name"] or f"{variant_tag}-{model_tag}"
    cfg["run_name"] = run_name

    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(Path(cfg["base_output_dir"]) / run_name)
    return SimpleNamespace(**cfg)


def resolve_resume_checkpoint(output_dir, resume_from_checkpoint):
    if not resume_from_checkpoint:
        return None
    if str(resume_from_checkpoint).lower() not in ("auto", "latest"):
        return os.path.abspath(os.path.expanduser(str(resume_from_checkpoint)))
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("step_"):
            try:
                candidates.append((int(name.split("_", 1)[1]), path))
            except ValueError:
                continue
    return max(candidates, default=(None, None))[1]


def checkpoint_step(checkpoint_path):
    name = os.path.basename(os.path.normpath(checkpoint_path))
    if not name.startswith("step_"):
        raise ValueError(f"Checkpoint directory must be named step_N, got {name!r}")
    try:
        return int(name.split("_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Checkpoint directory must be named step_N, got {name!r}") from exc


def make_model(args, vocab_size):
    return Transformer(
        dim=args.hidden_size,
        depth=args.depth,
        heads=args.n_head,
        ff_mult=args.ff_mult,
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        gradient_checkpointing=args.gradient_checkpointing,
        use_rope=True,
        rope_theta=args.rope_theta,
        qk_norm=args.qk_norm,
        pos_variant=args.pos_variant,
        rel_extent=args.rel_extent,
        pos_rank=args.pos_rank,
        pos_mlp_hidden=args.pos_mlp_hidden,
        attn_impl=args.attn_impl,
    )


def make_optimizer(args, model):
    if args.optimizer != "adamw":
        raise ValueError("Only AdamW is supported.")
    no_decay = ("bias", "norm")
    grouped = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else args.weight_decay
        grouped.setdefault(wd, []).append(param)
    param_groups = [{"params": params, "weight_decay": wd} for wd, params in grouped.items()]
    return torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        fused=torch.cuda.is_available(),
    )


def _load_tokenized_datasets(args, tokenizer):
    if args.tokenized_dataset_path and os.path.isdir(args.tokenized_dataset_path):
        logger.info("Loading tokenized dataset from %s", args.tokenized_dataset_path)
        return datasets.load_from_disk(args.tokenized_dataset_path)

    if args.train_file is not None:
        files = {"train": args.train_file}
        if args.validation_file is not None:
            files["validation"] = args.validation_file
        raw = load_dataset("text", data_files=files, cache_dir=args.hf_cache_dir)
        if "validation" not in raw:
            split = raw["train"].train_test_split(
                test_size=args.validation_split_percentage / 100.0,
                seed=args.seed,
            )
            raw = datasets.DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        raw = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split={
                "train": f"train[{args.validation_split_percentage}%:]",
                "validation": f"train[:{args.validation_split_percentage}%]",
            },
            cache_dir=args.hf_cache_dir,
            num_proc=args.preprocessing_num_workers,
        )

    column_names = raw["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]
    block_size = min(args.block_size, tokenizer.model_max_length)

    def tokenize_and_group(examples):
        tokenized = tokenizer(examples[text_column])
        concatenated = {k: list(chain(*tokenized[k])) for k in tokenized}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [values[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, values in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = raw.map(
        tokenize_and_group,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Tokenize and group into {block_size}",
    )
    if args.tokenized_dataset_path:
        lm_datasets.save_to_disk(args.tokenized_dataset_path)
    return lm_datasets


def load_tokenized_datasets(args, tokenizer):
    """Load/build a shared cache without racing independent GPU workers."""
    if not args.tokenized_dataset_path:
        return _load_tokenized_datasets(args, tokenizer)
    lock_path = f"{args.tokenized_dataset_path}.lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _load_tokenized_datasets(args, tokenizer)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@torch.no_grad()
def evaluate(args, model, eval_dataloader, accelerator, step):
    model.eval()
    losses = []
    for idx, batch in enumerate(eval_dataloader):
        input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)[:, :-1]
        targets = batch["labels"].to(accelerator.device, non_blocking=True)[:, 1:]
        loss = model(input_ids=input_ids, targets=targets)
        losses.append(accelerator.gather_for_metrics(loss.detach().float()))
        if args.num_validation_batches is not None and idx + 1 >= args.num_validation_batches:
            break
    eval_loss = torch.cat([loss.reshape(-1) for loss in losses]).mean()
    perplexity = math.exp(eval_loss.item()) if eval_loss.item() < 20 else float("inf")
    logger.info("step %s: eval_loss %.4f perplexity %.4f", step, eval_loss.item(), perplexity)
    if accelerator.is_main_process:
        metrics = {
            "step": int(step),
            "eval_loss": eval_loss.item(),
            "perplexity": perplexity if math.isfinite(perplexity) else None,
            "timestamp": time.time(),
        }
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
    if args.with_tracking:
        accelerator.log({"eval_loss": eval_loss.item(), "perplexity": perplexity}, step=step)
    model.train()


def save_model(args, model, tokenizer, accelerator, completed_steps, max_train_steps):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.save(unwrapped.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        if completed_steps >= max_train_steps:
            with open(os.path.join(args.output_dir, "COMPLETED"), "w") as f:
                json.dump(
                    {"completed_at": time.time(), "completed_steps": completed_steps},
                    f,
                )
                f.write("\n")
        logger.info("saved model to %s", args.output_dir)


def main():
    args = load_config(parse_args())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    if args.seed is not None:
        set_seed(args.seed)

    if args.dry_run:
        vocab_size = math.ceil(50257 / 64) * 64
        model = make_model(args, vocab_size)
        if args.print_model:
            print(model)
        counts = count_parameters(model)
        print(json.dumps({
            "pos_variant": args.pos_variant,
            "attn_impl": args.attn_impl,
            "rel_extent": args.rel_extent or args.block_size,
            **counts,
        }, indent=2))
        # TODO: once implemented, print suggest_matched_baselines(vars(args)) here.
        print(
            "matched baselines: adjust pos_rank / pos_mlp_hidden or ff_mult manually "
            f"(helper stub: {suggest_matched_baselines.__name__})"
        )
        return

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = math.ceil(config.vocab_size / 64) * 64

    model = make_model(args, vocab_size)
    if args.print_model:
        print(model)
    counts = count_parameters(model)
    print("parameters:", json.dumps(counts))

    accelerator_kwargs = {}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = args.report_to
        accelerator_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        **accelerator_kwargs,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        lm_datasets = load_tokenized_datasets(args, tokenizer)

    loader_kwargs = {
        "collate_fn": default_data_collator,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed if args.seed is not None else 0)
    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        generator=train_generator,
        **loader_kwargs,
    )
    eval_dataloader = DataLoader(
        lm_datasets["validation"],
        batch_size=args.per_device_train_batch_size,
        **loader_kwargs,
    )

    model = model.to(accelerator.device)
    if len(tokenizer) > model.token_embedding.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    # Batches are shifted by one token before entering the model.
    model.prepare_flex_masks(args.block_size - 1, accelerator.device)

    optimizer = make_optimizer(args, model)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler,
    )
    if args.compile:
        model = torch.compile(
            model,
            mode=args.compile_mode,
            fullgraph=args.compile_fullgraph,
        )

    update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = max(args.num_train_epochs, math.ceil(max_train_steps / update_steps_per_epoch))

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    completed_steps = 0
    starting_epoch = 0
    resume_batch = None
    if resume_checkpoint is not None:
        if not os.path.isdir(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        accelerator.load_state(resume_checkpoint)
        completed_steps = checkpoint_step(resume_checkpoint)
        consumed_batches = completed_steps * args.gradient_accumulation_steps
        starting_epoch = consumed_batches // len(train_dataloader)
        resume_batch = consumed_batches % len(train_dataloader)
        logger.info("Resumed from %s at optimizer step %s", resume_checkpoint, completed_steps)

    if args.with_tracking:
        wandb_init = {"name": args.run_name}
        if args.wandb_group:
            wandb_init["group"] = args.wandb_group
        accelerator.init_trackers(
            args.wandb_project,
            vars(args),
            init_kwargs={"wandb": wandb_init},
        )
        accelerator.log({
            "num_params": counts["total"],
            "true_num_params": counts["non_embed"],
        })

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(min(completed_steps, max_train_steps))
    train_loss_accum = 0.0
    train_loss_updates = 0

    for epoch in range(starting_epoch, num_train_epochs):
        if completed_steps >= max_train_steps:
            break
        model.train()
        if epoch == starting_epoch and resume_batch:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch)
        else:
            active_dataloader = train_dataloader
        for batch in active_dataloader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)[:, :-1]
                targets = batch["labels"].to(accelerator.device, non_blocking=True)[:, 1:]
                loss = model(input_ids=input_ids, targets=targets)
                train_loss_accum += accelerator.gather_for_metrics(loss.detach().float()).mean().item()
                train_loss_updates += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                logs = {
                    "train_loss": train_loss_accum / max(train_loss_updates, 1),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(loss=f"{logs['train_loss']:.4f}", lr=f"{logs['lr']:.2e}")
                if args.with_tracking and completed_steps % args.log_every_n_steps == 0:
                    accelerator.log(logs, step=completed_steps)
                if args.checkpointing_steps and str(args.checkpointing_steps).isdigit():
                    interval = int(args.checkpointing_steps)
                    if completed_steps % interval == 0:
                        accelerator.save_state(os.path.join(args.output_dir, f"step_{completed_steps}"))
                if completed_steps % args.validate_every == 0:
                    evaluate(args, model, eval_dataloader, accelerator, completed_steps)
                if completed_steps >= max_train_steps:
                    break

    evaluate(args, model, eval_dataloader, accelerator, completed_steps)
    if args.with_tracking:
        accelerator.end_training()
    save_model(
        args, model, tokenizer, accelerator,
        completed_steps=completed_steps,
        max_train_steps=max_train_steps,
    )


if __name__ == "__main__":
    main()
