#!/usr/bin/env python
"""Causal LM training for position-bias experiments."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, default_data_collator, get_scheduler

from position import (
    POSITION_PRESETS,
    POSITION_SCHEMA_VERSION,
    ATTENTION_WRITE_DEFAULTS,
    RESIDUAL_STREAM_DEFAULTS,
    V1_CHANNEL_DEFAULTS,
    deep_merge,
    legacy_position_run_tag,
    normalize_attention_write_config,
    normalize_position_content_config,
    normalize_residual_stream_config,
    resolve_channel_config,
    v2_position_run_tag,
)
from position.config import v2_to_legacy_tag_fields
from transformer import Transformer, count_parameters, suggest_matched_baselines

REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE", REPO_DIR.parent))
CACHE_DIR = WORKSPACE_DIR / ".cache"

# Keep HF / compile / wandb caches on the workspace volume.
os.environ.setdefault("HF_HOME", str(WORKSPACE_DIR / ".hf_home"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(os.environ["HF_HOME"]) / "datasets"),
)
os.environ.setdefault("WANDB_HOME", str(WORKSPACE_DIR / ".wandb_home"))
os.environ.setdefault("WANDB_DIR", str(WORKSPACE_DIR / ".wandb_home"))
os.environ.setdefault("TRITON_CACHE_DIR", str(CACHE_DIR / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR / "torchinductor"))
os.environ.setdefault("TMPDIR", str(CACHE_DIR / "tmp"))
for cache_path in (
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_DATASETS_CACHE"]),
    Path(os.environ["WANDB_HOME"]),
    Path(os.environ["WANDB_DIR"]),
    Path(os.environ["TRITON_CACHE_DIR"]),
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]),
    Path(os.environ["TMPDIR"]),
):
    cache_path.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = os.environ["TMPDIR"]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    "dataset_config_name": None,
    "train_file": None,
    "validation_file": None,
    "validation_split_percentage": 5,
    "model_name_or_path": "openai-community/gpt2",
    "tokenizer_name": None,
    "use_slow_tokenizer": False,
    "hf_cache_dir": os.environ["HF_DATASETS_CACHE"],
    # Shared on-disk cache so parallel gpu-claim jobs don't re-tokenize.
    "tokenized_dataset_path": str(
        WORKSPACE_DIR / ".cache" / "tokenized" / "openwebtext_gpt2_bs1024"
    ),
    "preprocessing_num_workers": min(8, os.cpu_count() or 1),
    "overwrite_cache": False,
    "block_size": 1024,
    # ``block_size`` remains a compatibility alias for training_length.
    "training_length": None,
    "model_position_extent": None,
    "evaluation_lengths": None,
    "scalar_normalization_extent": None,
    "per_device_train_batch_size": 8,
    # Defaults to the train microbatch; long-context evaluation can override.
    "per_device_eval_batch_size": None,
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
    "save_final_model": False,
    "resume_from_checkpoint": None,
    "output_dir": None,
    "base_output_dir": str(REPO_DIR / "model-output"),
    "run_name": None,
    "with_tracking": False,
    "report_to": "wandb",
    "wandb_project": "mlprope-position-bias",
    "wandb_entity": "ethansmith2000",
    "wandb_group": None,
    "mixed_precision": "bf16",
    # Sized for packing many single-GPU jobs on one node (not one job hogging all CPUs).
    "num_workers": 4,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "non_blocking": True,
    "num_validation_batches": 25,
    "validate_every": 1000,
    "log_every_n_steps": 50,
    # CUDA-event stage timings (data / forward / backward / optimizer).
    "profile_every_n_steps": 10,
    "dry_run": False,
    "print_model": False,
    # Model
    "hidden_size": 768,
    "depth": 8,
    "n_head": 8,
    "ff_mult": 4,
    "ff_hidden_dim": None,
    "use_rope": True,
    "rope_theta": 10000.0,
    "qk_norm": True,
    "post_position_qk_norm": False,
    "qk_norm_mode": "legacy_layernorm",
    "position_content_dim": 64,
    "position_content_coupling": "separate",
    # Legacy convenience preset. Nested qk/logit_bias configs are source of truth.
    # Phase 1: rope | add_rope | linear | low_rank | bottleneck_mlp | mlp_rope.
    # Content-routed presets: inkling_table | inkling_cosnet.
    "pos_variant": None,
    "rel_extent": None,  # None follows block_size.
    "pos_rank": 32,
    "pos_mlp_hidden": 128,
    # Raw defaults remain v1-shaped for override merging; load_config upgrades to v2.
    "qk": copy.deepcopy(V1_CHANNEL_DEFAULTS["qk"]),
    "logit_bias": copy.deepcopy(V1_CHANNEL_DEFAULTS["logit_bias"]),
    "residual_stream": copy.deepcopy(RESIDUAL_STREAM_DEFAULTS),
    "attention_write": copy.deepcopy(ATTENTION_WRITE_DEFAULTS),
    "position_schema_version": POSITION_SCHEMA_VERSION,
    "position_source_schema": 1,
    # Only the learned logit-bias channel requires flex.
    "attn_impl": "sdpa",
    "gradient_checkpointing": False,
    "compile": True,
    # reduce-overhead + CUDAGraphs overwrites the CE loss tensor; flex+inductor
    # also OOM'd Triton SMEM under nested compile. Prefer default.
    "compile_mode": "default",
    "compile_fullgraph": False,
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
            "bottleneck_mlp",
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
    return deep_merge(merged, overrides)


def position_run_tag(cfg: dict) -> str:
    """Dispatch to legacy or v2 tags based on ``position_source_schema``."""
    source = int(cfg.get("position_source_schema", 2))
    if source == 1:
        base = legacy_position_run_tag(
            qk_v1=v2_to_legacy_tag_fields("qk", cfg["qk"]),
            logit_v1=v2_to_legacy_tag_fields("logit_bias", cfg["logit_bias"]),
            attn_impl=cfg["attn_impl"],
        )
    else:
        base = v2_position_run_tag(
            qk=cfg["qk"],
            logit_bias=cfg["logit_bias"],
            attn_impl=cfg["attn_impl"],
        )
    extras = []
    residual = cfg.get("residual_stream", {})
    if residual.get("enabled", False):
        extras.append(
            f"res-{residual['source']}-{residual['placement']}"
        )
    write = cfg.get("attention_write", {})
    if write.get("enabled", False):
        extras.append(
            f"write-{write['mode']}-{write['head_coupling']}"
        )
    tag = base if not extras else "+".join((base, *extras))
    if source == 2:
        canonical = {
            "qk": cfg["qk"],
            "logit_bias": cfg["logit_bias"],
            "residual_stream": cfg.get("residual_stream", {}),
            "attention_write": cfg.get("attention_write", {}),
            "model_context": {
                "hidden_size": cfg["hidden_size"],
                "n_head": cfg["n_head"],
                "use_rope": cfg["use_rope"],
                "post_position_qk_norm": cfg["post_position_qk_norm"],
                "qk_norm_mode": cfg["qk_norm_mode"],
                "position_content_dim": cfg["position_content_dim"],
                "position_content_coupling": cfg[
                    "position_content_coupling"
                ],
                "rope_theta": cfg["rope_theta"],
                "extent": cfg.get("rel_extent") or cfg["model_position_extent"],
            },
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:10]
        tag = f"{tag}-c{digest}"
    return tag


def load_config(cli_args):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    overrides = {}
    if cli_args.override_json is not None:
        overrides = load_json_overrides(cli_args.override_json)
        unknown = set(overrides) - set(cfg)
        # Allow forward-looking keys used only by the matching helper stub later.
        unknown -= {"_matching_todo"}
        if unknown:
            raise ValueError(f"Unknown override keys: {sorted(unknown)}")
        cfg = deep_merge(cfg, overrides)
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

    preset = cfg["pos_variant"]
    if preset is not None and preset not in POSITION_PRESETS:
        raise ValueError(f"Unknown position preset: {preset!r}")

    preset_config = POSITION_PRESETS.get(preset, {}) if preset is not None else {}
    # Legacy width knobs remain aliases for preset-generated configs.
    qk_override = copy.deepcopy(overrides.get("qk", {}))
    logit_override = copy.deepcopy(overrides.get("logit_bias", {}))
    if preset is not None and preset not in {"inkling_table", "inkling_cosnet"}:
        if "qk" not in overrides:
            qk_override = deep_merge(
                qk_override,
                {
                    "rank": int(cfg["pos_rank"]),
                    "mlp_hidden": int(cfg["pos_mlp_hidden"]),
                },
            )
        if "logit_bias" not in overrides:
            logit_override = deep_merge(
                logit_override,
                {
                    "rank": int(cfg["pos_rank"]),
                    "mlp_hidden": int(cfg["pos_mlp_hidden"]),
                },
            )

    training_length = int(cfg["training_length"] or cfg["block_size"])
    if training_length < 2:
        raise ValueError("training_length must be at least 2")
    cfg["training_length"] = training_length
    cfg["block_size"] = training_length
    train_batch_size = int(cfg["per_device_train_batch_size"])
    if train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    cfg["per_device_train_batch_size"] = train_batch_size
    eval_batch_size = int(
        train_batch_size
        if cfg["per_device_eval_batch_size"] is None
        else cfg["per_device_eval_batch_size"]
    )
    if eval_batch_size <= 0:
        raise ValueError("per_device_eval_batch_size must be positive")
    cfg["per_device_eval_batch_size"] = eval_batch_size
    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    cfg["gradient_accumulation_steps"] = gradient_accumulation_steps
    if not isinstance(cfg["gradient_checkpointing"], bool):
        raise TypeError("gradient_checkpointing must be a boolean")
    if not isinstance(cfg["compile"], bool):
        raise TypeError("compile must be a boolean")
    if cfg["compile_mode"] not in {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }:
        raise ValueError("unsupported compile_mode")
    if cfg["checkpointing_steps"] is not None:
        checkpointing_steps = int(cfg["checkpointing_steps"])
        if checkpointing_steps <= 0:
            raise ValueError("checkpointing_steps must be positive")
        cfg["checkpointing_steps"] = checkpointing_steps
    if not isinstance(cfg["save_final_model"], bool):
        raise TypeError("save_final_model must be a boolean")

    raw_evaluation_lengths = cfg["evaluation_lengths"]
    if raw_evaluation_lengths is None:
        evaluation_lengths = [training_length]
    elif not isinstance(raw_evaluation_lengths, list):
        raise TypeError("evaluation_lengths must be a list of integers or null")
    else:
        evaluation_lengths = []
        for length in raw_evaluation_lengths:
            if isinstance(length, bool) or not isinstance(length, int):
                raise TypeError("evaluation_lengths must contain only integers")
            if length < 2:
                raise ValueError("evaluation_lengths must be at least 2")
            if length not in evaluation_lengths:
                evaluation_lengths.append(length)
        if training_length not in evaluation_lengths:
            evaluation_lengths.insert(0, training_length)
    cfg["evaluation_lengths"] = evaluation_lengths

    model_position_extent = int(
        cfg["model_position_extent"] or max(evaluation_lengths)
    )
    if model_position_extent < max(training_length, *evaluation_lengths):
        raise ValueError(
            "model_position_extent must cover training_length and every "
            "evaluation length"
        )
    cfg["model_position_extent"] = model_position_extent
    scalar_normalization_extent = int(
        cfg["scalar_normalization_extent"] or training_length
    )
    if scalar_normalization_extent <= 0:
        raise ValueError("scalar_normalization_extent must be positive")
    cfg["scalar_normalization_extent"] = scalar_normalization_extent
    if cfg["rel_extent"] is not None and int(cfg["rel_extent"]) < max(
        evaluation_lengths
    ):
        raise ValueError("rel_extent must cover every evaluation length")

    model_dim = int(cfg["hidden_size"])
    heads = int(cfg["n_head"])
    rope_theta = float(cfg["rope_theta"])
    if not isinstance(cfg["use_rope"], bool):
        raise TypeError("use_rope must be a boolean")
    if not isinstance(cfg["post_position_qk_norm"], bool):
        raise TypeError("post_position_qk_norm must be a boolean")
    if cfg["qk_norm_mode"] not in {
        "legacy_layernorm",
        "method_aware_rms",
    }:
        raise ValueError(
            "qk_norm_mode must be 'legacy_layernorm' or 'method_aware_rms'"
        )
    if (
        cfg["qk_norm_mode"] == "method_aware_rms"
        and cfg["post_position_qk_norm"]
    ):
        raise ValueError(
            "method_aware_rms requires post_position_qk_norm=false"
        )
    content_cfg = normalize_position_content_config(
        cfg["position_content_dim"],
        cfg["position_content_coupling"],
    )
    cfg["position_content_dim"] = content_cfg["dim"]
    cfg["position_content_coupling"] = content_cfg["coupling"]
    qk_config, qk_source = resolve_channel_config(
        "qk",
        preset_fragment=preset_config.get("qk"),
        override=qk_override,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    logit_config, logit_source = resolve_channel_config(
        "logit_bias",
        preset_fragment=preset_config.get("logit_bias"),
        override=logit_override,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    # If a preset set rank aliases onto a v1 fragment, they are already applied
    # above via overrides. For presets with only feature_map, rank aliases need
    # to land on the upgraded mapper — handle when preset set and no channel override.
    if preset is not None:
        if "qk" not in overrides and qk_config["enabled"]:
            qk_config["mapper"]["rank"] = int(cfg["pos_rank"])
            qk_config["mapper"]["hidden_dim"] = int(cfg["pos_mlp_hidden"])
        if "logit_bias" not in overrides and logit_config["enabled"]:
            logit_config["mapper"]["rank"] = int(cfg["pos_rank"])
            logit_config["mapper"]["hidden_dim"] = int(cfg["pos_mlp_hidden"])

    cfg["qk"] = qk_config
    cfg["logit_bias"] = logit_config
    for channel_config in (qk_config, logit_config):
        if channel_config["enabled"] and channel_config["input"]["scalars"]:
            channel_config["input"][
                "normalization_extent"
            ] = scalar_normalization_extent
    cfg["residual_stream"] = normalize_residual_stream_config(
        overrides.get("residual_stream", cfg["residual_stream"]),
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    cfg["attention_write"] = normalize_attention_write_config(
        overrides.get("attention_write", cfg["attention_write"]),
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    cfg["position_schema_version"] = POSITION_SCHEMA_VERSION
    enabled_sources = []
    if qk_config["enabled"]:
        enabled_sources.append(qk_source)
    if logit_config["enabled"]:
        enabled_sources.append(logit_source)
    if cfg["residual_stream"]["enabled"] or cfg["attention_write"]["enabled"]:
        enabled_sources.append(2)
    if not cfg["use_rope"]:
        enabled_sources.append(2)
    # Baseline and wholly legacy active channels retain historical tags.
    cfg["position_source_schema"] = (
        1 if not enabled_sources or all(source == 1 for source in enabled_sources) else 2
    )

    cfg["pos_variant"] = preset or (
        ("rope" if cfg["use_rope"] else "none")
        if (
            not qk_config["enabled"]
            and not logit_config["enabled"]
            and not cfg["residual_stream"]["enabled"]
            and not cfg["attention_write"]["enabled"]
        )
        else "custom"
    )

    # Only learned logit biases require FlexAttention.
    if logit_config["enabled"] and cfg["attn_impl"] != "flex":
        cfg["attn_impl"] = "flex"
    if (
        cfg["attn_impl"] == "flex"
        and cfg["compile"]
        and cfg["compile_fullgraph"]
    ):
        raise ValueError(
            "attn_impl='flex' is incompatible with compile_fullgraph=true: "
            "the FlexAttention wrapper intentionally uses a graph break."
        )

    # Keep the shared tokenized cache keyed by block size when using the default path.
    default_tok_prefix = str(WORKSPACE_DIR / ".cache" / "tokenized" / "openwebtext_gpt2_bs")
    if (
        isinstance(cfg["tokenized_dataset_path"], str)
        and cfg["tokenized_dataset_path"].startswith(default_tok_prefix)
        and cfg["tokenized_dataset_path"].endswith("_ids")
    ):
        # legacy name; prefer the shared nonlinear/calib path without _ids suffix
        cfg["tokenized_dataset_path"] = f"{default_tok_prefix}{cfg['block_size']}"

    model_tag = f"h{cfg['hidden_size']}d{cfg['depth']}"
    variant_tag = position_run_tag(cfg)
    if (
        qk_config["enabled"]
        or logit_config["enabled"]
        or cfg["residual_stream"]["enabled"]
        or cfg["attention_write"]["enabled"]
    ):
        rel_extent = cfg["rel_extent"] or cfg["model_position_extent"]
        variant_tag = f"{variant_tag}-e{rel_extent}"
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


def _interval_due(interval, step: int) -> bool:
    return bool(interval) and step > 0 and step % int(interval) == 0


class CudaStageTimer:
    """Time data-load (host) + forward/backward/optimizer (CUDA events)."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.data_ms = 0.0
        self.forward_ms = 0.0
        self.backward_ms = 0.0
        self.optimizer_ms = 0.0
        self._active = False
        self._data_t0 = 0.0
        if self.enabled:
            self._events = {
                name: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for name in ("forward", "backward", "optimizer")
            }

    def begin_step(self) -> None:
        if not self.enabled:
            self._active = False
            return
        torch.cuda.synchronize()
        self._data_t0 = time.perf_counter()
        self._active = True

    def mark_data_end(self) -> None:
        if self._active:
            self.data_ms = (time.perf_counter() - self._data_t0) * 1e3

    def range_start(self, name: str) -> None:
        if self._active:
            self._events[name][0].record()

    def range_end(self, name: str) -> None:
        if self._active:
            self._events[name][1].record()

    def finish_step(self) -> dict[str, float] | None:
        if not self._active:
            return None
        torch.cuda.synchronize()
        self.forward_ms = self._events["forward"][0].elapsed_time(self._events["forward"][1])
        self.backward_ms = self._events["backward"][0].elapsed_time(self._events["backward"][1])
        self.optimizer_ms = self._events["optimizer"][0].elapsed_time(
            self._events["optimizer"][1]
        )
        self._active = False
        return {
            "time/data_ms": self.data_ms,
            "time/forward_ms": self.forward_ms,
            "time/backward_ms": self.backward_ms,
            "time/optimizer_ms": self.optimizer_ms,
            "time/step_ms": (
                self.data_ms + self.forward_ms + self.backward_ms + self.optimizer_ms
            ),
        }


def make_model(args, vocab_size):
    return Transformer(
        dim=args.hidden_size,
        depth=args.depth,
        heads=args.n_head,
        ff_mult=args.ff_mult,
        ff_hidden_dim=args.ff_hidden_dim,
        vocab_size=vocab_size,
        max_seq_len=args.model_position_extent,
        gradient_checkpointing=args.gradient_checkpointing,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        qk_norm=args.qk_norm,
        post_position_qk_norm=args.post_position_qk_norm,
        qk_norm_mode=args.qk_norm_mode,
        position_content_dim=args.position_content_dim,
        position_content_coupling=args.position_content_coupling,
        rel_extent=args.rel_extent,
        qk_config=args.qk,
        logit_bias_config=args.logit_bias,
        residual_stream_config=args.residual_stream,
        attention_write_config=args.attention_write,
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
        # Keep only input_ids; train loop derives targets via a causal shift.
        # Dropping attention_mask / duplicate labels halves on-disk cache size.
        input_ids = concatenated["input_ids"]
        return {
            "input_ids": [
                input_ids[i:i + block_size] for i in range(0, total_length, block_size)
            ],
        }

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


class RechunkedTokenDataset(Dataset):
    """Expose a fixed-width token dataset at another contiguous width."""

    def __init__(self, source, target_length: int):
        if target_length < 2:
            raise ValueError("target_length must be at least 2")
        if len(source) == 0:
            raise ValueError("Cannot rechunk an empty dataset")
        source_length = len(source[0]["input_ids"])
        if source_length < 1:
            raise ValueError("Source token rows must not be empty")
        self.source = source
        self.source_length = source_length
        self.target_length = int(target_length)
        self._length = (len(source) * source_length) // self.target_length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        start = index * self.target_length
        remaining = self.target_length
        tokens = []
        while remaining:
            row_index, row_offset = divmod(start, self.source_length)
            row = self.source[row_index]["input_ids"]
            take = min(remaining, self.source_length - row_offset)
            tokens.extend(row[row_offset:row_offset + take])
            start += take
            remaining -= take
        return {"input_ids": tokens}


def build_evaluation_datasets(validation_dataset, evaluation_lengths):
    source_length = len(validation_dataset[0]["input_ids"])
    return {
        length: (
            validation_dataset
            if length == source_length
            else RechunkedTokenDataset(validation_dataset, length)
        )
        for length in evaluation_lengths
    }


@torch.no_grad()
def evaluate(
    args,
    model,
    eval_dataloaders,
    accelerator,
    step,
    *,
    include_extrapolation: bool = False,
):
    model.eval()
    evaluation_metrics = {}
    active_dataloaders = (
        eval_dataloaders
        if include_extrapolation
        else {args.training_length: eval_dataloaders[args.training_length]}
    )
    diagnostic_input_ids = None
    for context_length, eval_dataloader in active_dataloaders.items():
        losses = []
        for idx, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"][:, :-1]
            targets = batch["input_ids"][:, 1:]
            if context_length == args.training_length and diagnostic_input_ids is None:
                diagnostic_input_ids = input_ids[:1].detach()
            loss = model(input_ids=input_ids, targets=targets)
            losses.append(accelerator.gather_for_metrics(loss.detach().float()))
            if (
                args.num_validation_batches is not None
                and idx + 1 >= args.num_validation_batches
            ):
                break
        eval_loss = torch.cat([loss.reshape(-1) for loss in losses]).mean()
        eval_loss_value = eval_loss.item()
        perplexity = (
            math.exp(eval_loss_value) if eval_loss_value < 20 else float("inf")
        )
        evaluation_metrics[f"eval_loss/context_{context_length}"] = eval_loss_value
        evaluation_metrics[f"perplexity/context_{context_length}"] = (
            perplexity if math.isfinite(perplexity) else None
        )
        if context_length == args.training_length:
            evaluation_metrics["eval_loss"] = eval_loss_value
            evaluation_metrics["perplexity"] = (
                perplexity if math.isfinite(perplexity) else None
            )
        logger.info(
            "step %s context %s: eval_loss %.4f perplexity %.4f",
            step,
            context_length,
            eval_loss_value,
            perplexity,
        )
    diagnostic_model = accelerator.unwrap_model(model)
    while hasattr(diagnostic_model, "_orig_mod"):
        diagnostic_model = diagnostic_model._orig_mod
    position_metrics, position_profiles = diagnostic_model.position_diagnostics(
        sequence_length=args.training_length,
        input_ids=diagnostic_input_ids,
    )
    if accelerator.is_main_process:
        metrics = {
            "step": int(step),
            "timestamp": time.time(),
            **evaluation_metrics,
            **position_metrics,
        }
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        if position_profiles:
            profile_dir = Path(args.output_dir) / "position_profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": int(step),
                    "profiles": position_profiles,
                },
                profile_dir / f"step_{int(step):08d}.pt",
            )
    if args.with_tracking:
        accelerator.log(
            {
                **evaluation_metrics,
                **position_metrics,
            },
            step=step,
        )
    model.train()


def save_model(args, model, tokenizer, accelerator, completed_steps, max_train_steps):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if completed_steps >= max_train_steps:
        with open(os.path.join(args.output_dir, "COMPLETED"), "w") as f:
            json.dump(
                {"completed_at": time.time(), "completed_steps": completed_steps},
                f,
            )
            f.write("\n")
    if not args.save_final_model:
        logger.info(
            "skipping weight save (save_final_model=false); markers in %s",
            args.output_dir,
        )
        return
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(unwrapped.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)
    logger.info("saved model to %s", args.output_dir)


def pin_cuda_early() -> None:
    """Occupy the visible GPU immediately so unaware jobs see it on nvidia-smi.

    Exclusivity still comes from gpu-claim's lifetime flock; this is visibility only.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(0)
    pin = torch.empty(1, device="cuda", dtype=torch.float32)
    pin.fill_(0)
    del pin


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
            "qk": args.qk,
            "logit_bias": args.logit_bias,
            "residual_stream": args.residual_stream,
            "attention_write": args.attention_write,
            "use_rope": args.use_rope,
            "post_position_qk_norm": args.post_position_qk_norm,
            "qk_norm_mode": args.qk_norm_mode,
            "position_content_dim": args.position_content_dim,
            "position_content_coupling": args.position_content_coupling,
            "attn_impl": args.attn_impl,
            "training_length": args.training_length,
            "model_position_extent": args.model_position_extent,
            "evaluation_lengths": args.evaluation_lengths,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "rel_extent": args.rel_extent or args.model_position_extent,
            "position_schema_version": args.position_schema_version,
            "position_source_schema": args.position_source_schema,
            **counts,
        }, indent=2))
        print(
            "matched baseline:",
            json.dumps(
                suggest_matched_baselines(
                    vars(args),
                    position_params=counts["position_params"],
                )
            ),
        )
        return

    pin_cuda_early()
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
        dataloader_config=DataLoaderConfiguration(non_blocking=bool(args.non_blocking)),
        **accelerator_kwargs,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
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
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed if args.seed is not None else 0)
    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        generator=train_generator,
        **loader_kwargs,
    )
    evaluation_datasets = build_evaluation_datasets(
        lm_datasets["validation"],
        args.evaluation_lengths,
    )
    eval_dataloaders = {
        length: DataLoader(
            evaluation_dataset,
            batch_size=args.per_device_eval_batch_size,
            **loader_kwargs,
        )
        for length, evaluation_dataset in evaluation_datasets.items()
    }

    model = model.to(accelerator.device)
    if len(tokenizer) > model.token_embedding.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    # Batches are shifted by one token before entering the model.
    model.prepare_flex_masks(args.training_length - 1, accelerator.device)

    optimizer = make_optimizer(args, model)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    eval_lengths = list(eval_dataloaders)
    prepared = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        *(eval_dataloaders[length] for length in eval_lengths),
        lr_scheduler,
    )
    model, optimizer, train_dataloader, *prepared_tail = prepared
    lr_scheduler = prepared_tail.pop()
    eval_dataloaders = dict(zip(eval_lengths, prepared_tail, strict=True))
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
        wandb_init = {
            "name": args.run_name,
            "dir": os.environ.get("WANDB_DIR", str(WORKSPACE_DIR / ".wandb_home")),
        }
        if args.wandb_entity:
            wandb_init["entity"] = args.wandb_entity
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
    stage_timer = CudaStageTimer(enabled=bool(args.profile_every_n_steps))
    throughput_warmup_steps = min(20, max_train_steps)
    throughput_start_step = completed_steps
    throughput_started_at = None

    for epoch in range(starting_epoch, num_train_epochs):
        if completed_steps >= max_train_steps:
            break
        model.train()
        if epoch == starting_epoch and resume_batch:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch)
        else:
            active_dataloader = train_dataloader
        train_iter = iter(active_dataloader)
        while True:
            profile_this_step = _interval_due(
                args.profile_every_n_steps, completed_steps + 1
            )
            if profile_this_step:
                # Sync so the following next() wait is true data stall, not GPU overlap.
                stage_timer.begin_step()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if profile_this_step:
                stage_timer.mark_data_end()

            # accelerator.prepare already placed batches on device (non_blocking).
            input_ids = batch["input_ids"][:, :-1]
            targets = batch["input_ids"][:, 1:]
            with accelerator.accumulate(model):
                if args.compile and args.compile_mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                stage_timer.range_start("forward")
                loss = model(input_ids=input_ids, targets=targets)
                stage_timer.range_end("forward")

                stage_timer.range_start("backward")
                accelerator.backward(loss)
                stage_timer.range_end("backward")

                stage_timer.range_start("optimizer")
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                stage_timer.range_end("optimizer")

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1
            progress_bar.update(1)
            if (
                throughput_started_at is None
                and completed_steps >= throughput_warmup_steps
            ):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                throughput_start_step = completed_steps
                throughput_started_at = time.perf_counter()

            timing_logs = stage_timer.finish_step() if profile_this_step else None

            log_loss = _interval_due(args.log_every_n_steps, completed_steps)
            if log_loss or timing_logs is not None:
                logs = {"lr": lr_scheduler.get_last_lr()[0]}
                if log_loss:
                    synced_loss = loss.detach().float()
                    if accelerator.num_processes > 1:
                        synced_loss = accelerator.gather(synced_loss.reshape(1)).mean()
                    logs["train_loss"] = synced_loss.item()
                if timing_logs is not None:
                    logs.update(timing_logs)
                postfix = {}
                if "train_loss" in logs:
                    postfix["loss"] = f"{logs['train_loss']:.4f}"
                if timing_logs is not None:
                    postfix["fwd"] = f"{timing_logs['time/forward_ms']:.1f}"
                    postfix["bwd"] = f"{timing_logs['time/backward_ms']:.1f}"
                    postfix["opt"] = f"{timing_logs['time/optimizer_ms']:.1f}"
                    postfix["data"] = f"{timing_logs['time/data_ms']:.1f}"
                if postfix:
                    progress_bar.set_postfix(**postfix)
                if args.with_tracking:
                    accelerator.log(logs, step=completed_steps)

            if args.checkpointing_steps and str(args.checkpointing_steps).isdigit():
                if _interval_due(int(args.checkpointing_steps), completed_steps):
                    accelerator.save_state(
                        os.path.join(args.output_dir, f"step_{completed_steps}")
                    )
            if _interval_due(args.validate_every, completed_steps):
                evaluate(args, model, eval_dataloaders, accelerator, completed_steps)
            if completed_steps >= max_train_steps:
                break

    if throughput_started_at is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        measured_steps = max(completed_steps - throughput_start_step, 0)
        measured_seconds = max(time.perf_counter() - throughput_started_at, 1e-9)
        measured_tokens = (
            measured_steps
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.training_length
        )
        training_summary = {
            "optimizer_steps": measured_steps,
            "elapsed_seconds": measured_seconds,
            "tokens_per_second": measured_tokens / measured_seconds,
        }
        if torch.cuda.is_available():
            training_summary.update({
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            })
        if accelerator.is_main_process:
            print("training_summary:", json.dumps(training_summary))

    evaluate(
        args,
        model,
        eval_dataloaders,
        accelerator,
        completed_steps,
        include_extrapolation=True,
    )
    if args.with_tracking:
        accelerator.end_training()
    save_model(
        args, model, tokenizer, accelerator,
        completed_steps=completed_steps,
        max_train_steps=max_train_steps,
    )


if __name__ == "__main__":
    main()
