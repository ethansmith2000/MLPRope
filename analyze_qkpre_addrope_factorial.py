#!/usr/bin/env python
"""Analyze the paired Phase-29 qkpre x AddRoPE factorial screen."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "sweep_configs" / "phase29_qkpre_addrope_factorial"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase29_qkpre_addrope_factorial"
RESULT_ROOT = ROOT / "results" / "phase29_qkpre_addrope_factorial"
SEED = 123
STEPS = 5_000
ARMS = ("rope-fixed", "qkpre-rope", "addrope-a10", "qkpre-addrope-a10")
CONTRASTS = (
    ("qkpre-rope_vs_rope-fixed", "qkpre-rope", "rope-fixed"),
    ("addrope-a10_vs_rope-fixed", "addrope-a10", "rope-fixed"),
    ("qkpre-addrope-a10_vs_rope-fixed", "qkpre-addrope-a10", "rope-fixed"),
    ("combo_vs_qkpre-rope", "qkpre-addrope-a10", "qkpre-rope"),
    ("combo_vs_addrope-a10", "qkpre-addrope-a10", "addrope-a10"),
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase29-{arm}-seed{SEED}-s{STEPS}-h768d8"


def config_path(arm: str) -> Path:
    return CONFIG_ROOT / f"phase29-{arm}-seed{SEED}-s{STEPS}-h768d8.json"


def load_losses(arm: str) -> list[float]:
    path = (
        run_dir(arm)
        / "evaluation_details"
        / "step_00005000_context_001024.json"
    )
    payload = json.loads(path.read_text())
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {path}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout offset in {path}")
    losses = [float(value) for value in payload["losses"]]
    if len(losses) != 256:
        raise ValueError(f"Expected 256 final-holdout losses in {path}")
    return losses


def load_final_metrics(arm: str) -> dict:
    rows = [
        json.loads(line)
        for line in (run_dir(arm) / "metrics.jsonl").read_text().splitlines()
    ]
    final = [row for row in rows if row.get("evaluation_kind") == "final_holdout"]
    if len(final) != 1:
        raise ValueError(f"Expected exactly one final-holdout row for {arm}")
    return final[0]


def summarize_samples(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half_width = 1.96 * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "ci95": [mean - half_width, mean + half_width],
        "std": std,
        "num_paired_examples": len(values),
    }


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    differences = [
        candidate_loss - reference_loss
        for candidate_loss, reference_loss in zip(candidate, reference, strict=True)
    ]
    summary = summarize_samples(differences)
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": summary["mean"],
        "paired_example_ci95": summary["ci95"],
        "paired_example_std": summary["std"],
        "num_paired_examples": summary["num_paired_examples"],
        "negative_favors_candidate": True,
    }


def verify_factorial_isolation() -> dict:
    configs = {arm: json.loads(config_path(arm).read_text()) for arm in ARMS}
    blocks = {}
    common = None
    for arm, cfg in configs.items():
        cfg = dict(cfg)
        cfg.pop("run_name", None)
        blocks[arm] = {
            "qk": cfg.pop("qk"),
            "qk_preprojection": cfg.pop("qk_preprojection"),
        }
        if common is None:
            common = cfg
        elif cfg != common:
            raise RuntimeError(f"Non-factor configuration drift detected for {arm}")

    if blocks["rope-fixed"] != {
        "qk": {"enabled": False},
        "qk_preprojection": {"enabled": False},
    }:
        raise RuntimeError("Unexpected 00 control blocks")
    if blocks["qkpre-rope"]["qk"] != {"enabled": False}:
        raise RuntimeError("qkpre-only arm has an active AddRoPE block")
    if blocks["addrope-a10"]["qk_preprojection"] != {"enabled": False}:
        raise RuntimeError("AddRoPE-only arm has active qk-preprojection")
    if blocks["qkpre-rope"]["qk_preprojection"] != blocks["qkpre-addrope-a10"]["qk_preprojection"]:
        raise RuntimeError("qkpre block differs between its two factorial cells")
    if blocks["addrope-a10"]["qk"] != blocks["qkpre-addrope-a10"]["qk"]:
        raise RuntimeError("AddRoPE block differs between its two factorial cells")
    if blocks["addrope-a10"]["qk"].get("application") != "additive":
        raise RuntimeError("Factorial AddRoPE block is not additive")
    return {"verified": True, "factor_blocks": blocks}


def layer_summary(metrics: dict, marker: str, suffixes: tuple[str, ...]) -> dict:
    result = {}
    for suffix in suffixes:
        values = [
            float(value)
            for key, value in metrics.items()
            if marker in key and key.endswith(suffix)
        ]
        if values:
            result[suffix.lstrip("/")] = {
                "layer_min": min(values),
                "layer_mean": statistics.fmean(values),
                "layer_max": max(values),
            }
    return result


def mechanism_health(metrics: dict) -> dict:
    return {
        "qk_preprojection": layer_summary(
            metrics,
            "/qk_preprojection/",
            ("/gate", "/input_rms", "/projected_q_rms", "/projected_k_rms"),
        ),
        "addrope": layer_summary(
            metrics,
            "/qk/",
            (
                "/addend_q_to_q_rms_ratio",
                "/addend_k_to_k_rms_ratio",
                "/final_q_rms",
                "/final_k_rms",
            ),
        ),
    }


def analyze() -> dict:
    missing = [
        str(run_dir(arm) / "COMPLETED")
        for arm in ARMS
        if not (run_dir(arm) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError("Phase-29 is incomplete; missing markers:\n" + "\n".join(missing))

    isolation = verify_factorial_isolation()
    losses = {arm: load_losses(arm) for arm in ARMS}
    arms = {}
    for arm in ARMS:
        throughput = json.loads(
            (run_dir(arm) / "training_summary.json").read_text()
        )["target_tokens_per_second"]
        arms[arm] = {
            "loss": statistics.fmean(losses[arm]),
            "target_tokens_per_second": throughput,
            "mechanism_health": mechanism_health(load_final_metrics(arm)),
        }

    contrasts = {
        name: paired_summary(losses[candidate], losses[reference])
        for name, candidate, reference in CONTRASTS
    }
    interaction_samples = [
        combo - qkpre - addrope + fixed
        for combo, qkpre, addrope, fixed in zip(
            losses["qkpre-addrope-a10"],
            losses["qkpre-rope"],
            losses["addrope-a10"],
            losses["rope-fixed"],
            strict=True,
        )
    ]
    interaction = summarize_samples(interaction_samples)
    interaction.update(
        {
            "formula": "L_combo - L_qkpre - L_addrope + L_rope",
            "negative_is_super_additive": True,
            "classification": (
                "super_additive"
                if interaction["ci95"][1] < 0
                else "sub_additive_or_redundant"
                if interaction["ci95"][0] > 0
                else "inconclusive"
            ),
        }
    )
    single_best = min(("qkpre-rope", "addrope-a10"), key=lambda arm: arms[arm]["loss"])
    combo_vs_best = paired_summary(losses["qkpre-addrope-a10"], losses[single_best])
    return {
        "scope": "paired_seed123_2x2_qkpre_addrope_screen",
        "seed": SEED,
        "steps": STEPS,
        "primary_context": 1_024,
        "final_holdout_examples": 256,
        "final_holdout_start_batch": 2_048,
        "factorial_isolation": isolation,
        "arms": arms,
        "contrasts": contrasts,
        "interaction": interaction,
        "best_single": single_best,
        "combo_vs_best_single": combo_vs_best,
        "combo_is_best": arms["qkpre-addrope-a10"]["loss"] < arms[single_best]["loss"],
        "caveat": (
            "This is a one-seed architecture screen. Paired-example intervals "
            "measure holdout precision, not across-seed replication."
        ),
    }


def render_markdown(results: dict) -> str:
    labels = {
        "rope-fixed": "Fixed RoPE",
        "qkpre-rope": "qkpre + RoPE",
        "addrope-a10": "AddRoPE a1.0",
        "qkpre-addrope-a10": "qkpre + AddRoPE a1.0",
    }
    lines = [
        "# Phase-29 qkpre x AddRoPE factorial screen",
        "",
        "Paired seed 123, 5k steps, with a disjoint 256-example holdout.",
        "AddRoPE is the additive Q/K replacement for multiplicative RoPE; the",
        "combined cell adds qk-preprojection upstream of that additive channel.",
        "",
        "| Arm | Final loss | Target tok/s |",
        "| --- | ---: | ---: |",
    ]
    for arm in ARMS:
        result = results["arms"][arm]
        lines.append(
            f"| {labels[arm]} | {result['loss']:.6f} | "
            f"{result['target_tokens_per_second']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Delta | Paired 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, result in results["contrasts"].items():
        ci = result["paired_example_ci95"]
        lines.append(
            f"| {name} | {result['delta_candidate_minus_reference']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] |"
        )
    interaction = results["interaction"]
    ci = interaction["ci95"]
    combo = results["combo_vs_best_single"]
    combo_ci = combo["paired_example_ci95"]
    lines.extend(
        [
            "",
            f"Factorial interaction: **{interaction['mean']:+.6f}** "
            f"(95% paired-example CI [{ci[0]:+.6f}, {ci[1]:+.6f}]); "
            f"classification: **{interaction['classification']}**.",
            "",
            f"The best single arm is **{labels[results['best_single']]}**. "
            f"Combo minus best single is {combo['delta_candidate_minus_reference']:+.6f} "
            f"(CI [{combo_ci[0]:+.6f}, {combo_ci[1]:+.6f}]).",
            "",
            "The JSON companion contains mechanism-health diagnostics and the",
            "fully verified factor blocks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase29_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE29_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
