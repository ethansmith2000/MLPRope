#!/usr/bin/env python
"""Recover and summarize the completed phase-24 screen from durable logs.

The large ``model-output`` directory was intentionally removed after all twelve
runs completed.  The launcher logs retain aggregate final-holdout losses,
throughput summaries, and parameter counts, but not the per-example loss arrays.
This analyzer therefore reports paired *seed-level* deltas and never fabricates
paired-example confidence intervals.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_ROOT = ROOT / "logs" / "phase24_rope_embed_basis"
RESULT_ROOT = ROOT / "results" / "phase24_rope_embed_basis"
SEEDS = (123, 456, 789)
ARMS = ("rope-fixed", "basis16-a03", "basis16-a10", "ropeembed-a10")
CONTEXTS = (1024, 2048, 4096)
CONTRASTS = (
    ("basis16-a03_vs_rope-fixed", "basis16-a03", "rope-fixed"),
    ("basis16-a10_vs_basis16-a03", "basis16-a10", "basis16-a03"),
    ("ropeembed-a10_vs_basis16-a10", "ropeembed-a10", "basis16-a10"),
    ("ropeembed-a10_vs_rope-fixed", "ropeembed-a10", "rope-fixed"),
)

FINAL_EVAL = re.compile(
    r"step 5000 context (?P<context>\d+): eval_loss (?P<loss>[-+0-9.eE]+) "
    r".*\((?P<batches>\d+) batches, start=(?P<start>\d+)\)"
)
TRAINING_SUMMARY = re.compile(r"training_summary: (?P<payload>\{.*\})")
PARAMETERS = re.compile(r"parameters: (?P<payload>\{.*\})")


def log_path(arm: str, seed: int) -> Path:
    return LOG_ROOT / f"phase24-{arm}-seed{seed}-s5000-h768d8.log"


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    evaluations: dict[str, float] = {}
    for match in FINAL_EVAL.finditer(text):
        # The development evaluation has 25 batches at start=0.  Only the
        # disjoint 256-example window beginning at batch 2048 is retained.
        if int(match.group("batches")) == 256 and int(match.group("start")) == 2048:
            evaluations[match.group("context")] = float(match.group("loss"))
    missing_contexts = set(map(str, CONTEXTS)) - set(evaluations)
    if missing_contexts:
        raise RuntimeError(
            f"Missing final-holdout contexts {sorted(missing_contexts)} in {path}"
        )
    summaries = list(TRAINING_SUMMARY.finditer(text))
    parameters = list(PARAMETERS.finditer(text))
    if len(summaries) != 1 or len(parameters) != 1:
        raise RuntimeError(
            f"Expected one training summary and parameter record in {path}"
        )
    return {
        "losses": evaluations,
        "training_summary": json.loads(summaries[0].group("payload")),
        "parameters": json.loads(parameters[0].group("payload")),
    }


def analyze() -> dict:
    missing = [
        str(log_path(arm, seed))
        for arm in ARMS
        for seed in SEEDS
        if not log_path(arm, seed).is_file()
    ]
    if missing:
        raise RuntimeError("Missing phase-24 logs:\n" + "\n".join(missing))

    runs = {
        arm: {seed: parse_log(log_path(arm, seed)) for seed in SEEDS}
        for arm in ARMS
    }
    results: dict = {
        "provenance": {
            "source": "launcher logs after intentional model-output cleanup",
            "per_example_losses_available": False,
            "paired_example_confidence_intervals_available": False,
            "final_holdout_batches": 256,
            "final_holdout_start_batch": 2048,
        },
        "primary_context": 1024,
        "exploratory_contexts": [2048, 4096],
        "arms": {},
        "contrasts": {},
    }
    for arm in ARMS:
        losses_by_context = {
            str(context): [runs[arm][seed]["losses"][str(context)] for seed in SEEDS]
            for context in CONTEXTS
        }
        throughput = [
            float(runs[arm][seed]["training_summary"]["target_tokens_per_second"])
            for seed in SEEDS
        ]
        results["arms"][arm] = {
            "loss_by_context_and_seed": losses_by_context,
            "mean_loss_by_context": {
                context: statistics.fmean(values)
                for context, values in losses_by_context.items()
            },
            "target_tokens_per_second_by_seed": throughput,
            "mean_target_tokens_per_second": statistics.fmean(throughput),
            "std_target_tokens_per_second": statistics.stdev(throughput),
            "position_parameters": int(
                runs[arm][SEEDS[0]]["parameters"]["position_params"]
            ),
        }

    for name, candidate, reference in CONTRASTS:
        context_results = {}
        for context in CONTEXTS:
            deltas = [
                runs[candidate][seed]["losses"][str(context)]
                - runs[reference][seed]["losses"][str(context)]
                for seed in SEEDS
            ]
            context_results[str(context)] = {
                "delta_by_seed": deltas,
                "mean_delta_across_seeds": statistics.fmean(deltas),
                "seed_delta_std": statistics.stdev(deltas),
                "candidate_wins_all_seeds": all(delta < 0 for delta in deltas),
                "clears_5k_screen_gate": (
                    statistics.fmean(deltas) <= -0.01
                    and all(delta < 0 for delta in deltas)
                ),
            }
        results["contrasts"][name] = {
            "candidate": candidate,
            "reference": reference,
            "negative_favors_candidate": True,
            "contexts": context_results,
        }
    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-24 RoPE-embedding basis screen",
        "",
        "All values use the disjoint 256-example holdout beginning at validation",
        "batch 2048. The large run directories were intentionally removed after",
        "completion; durable logs retain aggregate losses but not per-example",
        "arrays, so paired-example confidence intervals cannot be reconstructed.",
        "",
        "## Primary context 1024",
        "",
        "| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Position params |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        result = results["arms"][arm]
        values = result["loss_by_context_and_seed"]["1024"]
        lines.append(
            f"| {arm} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
            f"{result['mean_loss_by_context']['1024']:.6f} | "
            f"{result['position_parameters']:,} |"
        )
    lines.extend(
        [
            "",
            "Deltas are candidate minus reference; negative favors the candidate.",
            "",
            "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, result in results["contrasts"].items():
        primary = result["contexts"]["1024"]
        deltas = primary["delta_by_seed"]
        lines.append(
            f"| {name} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | "
            f"{deltas[2]:+.6f} | {primary['mean_delta_across_seeds']:+.6f} | "
            f"{primary['candidate_wins_all_seeds']} | "
            f"{primary['clears_5k_screen_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory longer contexts",
            "",
            "These contexts exceed the training length and are not primary endpoints.",
            "",
            "| Arm | Mean at 2048 | Mean at 4096 |",
            "| --- | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        result = results["arms"][arm]
        lines.append(
            f"| {arm} | {result['mean_loss_by_context']['2048']:.6f} | "
            f"{result['mean_loss_by_context']['4096']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Raising the amplitude anchor from 0.3 to 1.0 is the decisive change "
            "in this screen. Replacing the compact basis-plus-scalars with the "
            "full native RoPE embedding loses about 0.014 mean loss and loses in "
            "all three seeds. The native basis remains much better than fixed "
            "RoPE, so this is evidence for attention-local additive injection, "
            "not evidence that the full RoPE basis is harmful in absolute terms.",
            "",
            "At the exploratory 4096 context, however, the native-basis arm is "
            "0.356 worse than fixed RoPE while the compact arms remain near it. "
            "The 1024-context selection must therefore not be advertised as a "
            "length-extrapolation result.",
            "",
            "As a 5k screen, this can select controls and parameterizations but is "
            "not a durable positive claim without a longer confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase24_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE24_RESULTS.md").write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
