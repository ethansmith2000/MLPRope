#!/usr/bin/env python
"""Content-mechanism ablations on trained phase-19 carrier models.

Evaluates a trained model on the locked disjoint holdout under deterministic
forward modes that modify only the dedicated positional content signal:

  native       trained model unchanged (must reproduce the recorded holdout loss)
  zero         content set to zero; the carrier falls back to its position path
  prefix_mean  each token receives the running mean of content over positions
               <= t, re-scaled to unit RMS: token-specific detail is replaced by
               a cumulative summary while magnitude is preserved
  lag<k>       position t receives the content of position max(0, t - k), so
               same-token alignment is broken by a known offset

Interpretation (per CONSOLIDATED_RESEARCH_PLAN.md section 7):
  native vs zero         endpoint reliance on the content path after coadaptation
  native vs lag<k>       whether the model uses *same-token* content alignment,
                         graded by how far the misalignment reaches
  native vs prefix_mean  whether a cumulative summary suffices, i.e. whether the
                         useful signal is integrated rather than token-local

Causality: every mode reads content only from positions <= t. An earlier draft
used a within-sequence permutation and a whole-sequence mean; both let content
derived from *future* tokens reach q_t/k_t, and since q_t predicts token t+1
that leaks the target into its own prediction. The leak and the intended
alignment damage push the loss in opposite directions, so those modes were
replaced rather than merely caveated. Ablated content must stay causal.

These are mechanism diagnostics. They do not recover the counterfactual
training trajectory, and a small ablation cost does not by itself refute a
training-time contribution.

Run through gpu-claim; inference only, no retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from train_gpt import (  # noqa: E402
    build_evaluation_datasets,
    evaluate,
    load_config,
    load_tokenized_datasets,
    make_model,
    parse_args,
)
from transformer import PositionContentProjection  # noqa: E402

BASE_MODES = ("native", "zero", "prefix_mean")
DEFAULT_LAGS = (1, 4, 16, 64)


def is_lag_mode(mode: str) -> int | None:
    if mode.startswith("lag") and mode[3:].isdigit():
        return int(mode[3:])
    return None


def valid_mode(mode: str) -> bool:
    return mode in BASE_MODES or is_lag_mode(mode) is not None


def _unit_rms(value: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    return value * scale.to(dtype=value.dtype)


_PRISTINE_FORWARD = PositionContentProjection.forward


def install_ablation(mode: str) -> None:
    """Patch PositionContentProjection.forward for every layer at once.

    Content is shaped [batch, heads, length, dim]; every transform below reads
    only positions <= t so the ablated model stays causal.
    """
    lag = is_lag_mode(mode)

    def patched(self, x: torch.Tensor):
        q_content, k_content = _PRISTINE_FORWARD(self, x)
        if mode == "native":
            return q_content, k_content

        def transform(content: torch.Tensor) -> torch.Tensor:
            if mode == "zero":
                return torch.zeros_like(content)
            if mode == "prefix_mean":
                # Causal running mean over positions <= t, then restore unit RMS
                # so the ablation changes direction/detail rather than scale.
                cumulative = content.float().cumsum(dim=2)
                divisor = torch.arange(
                    1,
                    content.shape[2] + 1,
                    device=content.device,
                    dtype=cumulative.dtype,
                ).view(1, 1, -1, 1)
                pooled = (cumulative / divisor).to(dtype=content.dtype)
                return _unit_rms(pooled)
            if lag is not None:
                length = content.shape[2]
                index = (
                    torch.arange(length, device=content.device) - lag
                ).clamp_(min=0)
                return content.index_select(2, index)
            raise ValueError(f"unknown mode {mode!r}")

        return transform(q_content), transform(k_content)

    PositionContentProjection.forward = patched


def restore() -> None:
    PositionContentProjection.forward = _PRISTINE_FORWARD


def assert_modes_are_causal(modes: tuple[str, ...], length: int = 24) -> None:
    """Fail before any GPU work if a mode lets future content reach position t.

    Perturbs the content input one position at a time and requires every
    strictly earlier output position to be bit-identical. A within-sequence
    permutation fails this; the supported modes pass it.
    """
    torch.manual_seed(0)
    dim, heads = 8, 2
    projection = PositionContentProjection(
        model_dim=dim, content_dim=dim, heads=heads, coupling="shared"
    )
    inputs = torch.randn(1, length, dim)
    for mode in modes:
        install_ablation(mode)
        try:
            baseline, _ = projection(inputs)
            for position in range(1, length):
                perturbed_inputs = inputs.clone()
                perturbed_inputs[:, position] += 5.0
                perturbed, _ = projection(perturbed_inputs)
                if not torch.equal(
                    baseline[:, :, :position], perturbed[:, :, :position]
                ):
                    raise SystemExit(
                        f"mode {mode!r} is not causal: perturbing position "
                        f"{position} changed an earlier output position"
                    )
        finally:
            restore()
    print(f"causality check passed for modes: {', '.join(modes)}", flush=True)


def probe_run(run_dir: Path, modes: tuple[str, ...]) -> dict:
    config_path = run_dir / "training_config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing {config_path}")
    weights_path = run_dir / "pytorch_model.bin"
    if not weights_path.is_file():
        raise SystemExit(f"missing {weights_path}")

    sys.argv = [sys.argv[0], "--override_json", str(config_path)]
    args = load_config(parse_args())
    args.with_tracking = False
    # evaluate() writes evaluation details and position profiles into
    # args.output_dir as a side effect. Redirect them so the locked phase-19
    # run directories are never modified by a diagnostic.
    probe_scratch = REPO_DIR / "results" / "probe_scratch" / run_dir.name
    probe_scratch.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(probe_scratch)
    args.save_evaluation_details = False

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        dataloader_config=DataLoaderConfiguration(
            non_blocking=bool(args.non_blocking)
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = load_tokenized_datasets(args, tokenizer)

    evaluation_datasets = build_evaluation_datasets(
        datasets["validation"], args.evaluation_lengths
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    eval_dataloaders = {
        length: DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            **loader_kwargs,
        )
        for length, dataset in evaluation_datasets.items()
    }

    results: dict[str, float] = {}
    for mode in modes:
        # Rebuild the model per mode so no patched state leaks between modes.
        model = make_model(args, len(tokenizer))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"state dict mismatch for {run_dir.name}: "
                f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}"
            )
        install_ablation(mode)
        try:
            prepared_model, *prepared_loaders = accelerator.prepare(
                model, *(eval_dataloaders[k] for k in eval_dataloaders)
            )
            loaders = dict(zip(eval_dataloaders, prepared_loaders, strict=True))
            metrics = evaluate(
                args,
                prepared_model,
                loaders,
                accelerator,
                step=0,
                final_evaluation=True,
            )
        finally:
            restore()
        key = f"eval_loss/context_{args.training_length}"
        results[mode] = float(metrics[key])
        print(f"  {run_dir.name} {mode:9s} {results[mode]:.6f}", flush=True)
        del model, prepared_model
        torch.cuda.empty_cache()

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(
            REPO_DIR / "model-output" / "position_bias_phase19_confirmation"
        ),
    )
    parser.add_argument("--arm", default="content-position")
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument(
        "--modes",
        default=",".join(BASE_MODES + tuple(f"lag{k}" for k in DEFAULT_LAGS)),
    )
    parser.add_argument(
        "--out", default=str(REPO_DIR / "results" / "content_mechanism_probe.json")
    )
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    for mode in modes:
        if not valid_mode(mode):
            raise SystemExit(
                f"unknown mode {mode!r}; valid: {BASE_MODES} or lag<k>"
            )

    assert_modes_are_causal(modes)

    root = Path(args.output_root)
    payload: dict = {
        "arm": args.arm,
        "modes": list(modes),
        "runs": {},
    }
    for seed in (s.strip() for s in args.seeds.split(",")):
        run_dir = next(
            (
                p
                for p in root.glob(f"phase19-{args.arm}-seed{seed}-*")
                if p.is_dir()
            ),
            None,
        )
        if run_dir is None:
            raise SystemExit(f"no run directory for arm={args.arm} seed={seed}")
        print(f"probing {run_dir.name}", flush=True)
        payload["runs"][seed] = probe_run(run_dir, modes)

    if "native" in modes:
        deltas: dict[str, dict[str, float]] = {}
        for mode in modes:
            if mode == "native":
                continue
            per_seed = {
                seed: values[mode] - values["native"]
                for seed, values in payload["runs"].items()
            }
            per_seed["mean"] = sum(
                v for k, v in per_seed.items() if k != "mean"
            ) / len(per_seed)
            deltas[mode] = per_seed
        payload["deltas_vs_native"] = deltas

        print("\n# Ablation cost (mode minus native; positive = ablation hurts)\n")
        seeds = sorted(payload["runs"])
        header = " | ".join(["Mode"] + [f"Seed {s}" for s in seeds] + ["Mean"])
        print(f"| {header} |")
        print("| --- " * (len(seeds) + 2) + "|")
        for mode, per_seed in deltas.items():
            cells = " | ".join(f"{per_seed[s]:+.6f}" for s in seeds)
            print(f"| {mode} | {cells} | {per_seed['mean']:+.6f} |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
