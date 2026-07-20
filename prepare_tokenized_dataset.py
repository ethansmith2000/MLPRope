#!/usr/bin/env python
"""Build the shared tokenized OpenWebText cache once (CPU-only, flocked).

Run this before launching parallel GPU jobs so downloads/tokenization do not
contend across gpu-claim workers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Importing train_gpt sets HF/WANDB cache env defaults.
from train_gpt import load_config, load_tokenized_datasets, parse_args
from transformers import AutoTokenizer


def main() -> int:
    # Force a no-op CLI parse, then reuse training defaults for the cache path.
    sys.argv = [sys.argv[0]]
    args = load_config(parse_args())
    tok_path = args.tokenized_dataset_path
    if not tok_path:
        raise SystemExit("tokenized_dataset_path is unset; nothing to prepare.")
    if os.path.isdir(tok_path):
        print(f"tokenized cache already present: {tok_path}")
        return 0

    print(f"Preparing tokenized cache at {tok_path}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = load_tokenized_datasets(args, tokenizer)
    print(
        "ready:",
        {
            "path": tok_path,
            "train": len(datasets["train"]),
            "validation": len(datasets["validation"]),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
