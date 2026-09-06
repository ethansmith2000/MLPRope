#!/usr/bin/env python3
"""
Plot training curves for the SNRAdam paper.

Expects CSV files exported from W&B in paper/data/ with columns:
  Step, <run_name_1> - eval_loss, <run_name_2> - eval_loss, ...

Usage:
  python plot_results.py

Outputs:
  figures/loss_curves.pdf   -- main loss curve figure (Adam family + Muon family)
  figures/soap_curves.pdf   -- SOAP family loss curves (if data present)

Also prints table-ready data to stdout.
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    "adamw": "#1f77b4",
    "snradam": "#ff7f0e",
    "snradam_sma": "#d62728",
    "muon": "#2ca02c",
    "snrmuon_post": "#9467bd",
    "snrmuon_pre": "#8c564b",
    "snrmuon_post_sma": "#e377c2",
    "snrmuon_pre_sma": "#17becf",
    "soap": "#7f7f7f",
    "snrsoap": "#bcbd22",
    "snrsoap_sma": "#ff9896",
}

LABELS = {
    "adamw": "AdamW",
    "snradam": "SNRAdam (p=1.0)",
    "snradam_sma": r"SNRAdam (p=1.0, $\alpha$=1.0)",
    "muon": "Muon",
    "snrmuon_post": "SNRMuon (post-NS)",
    "snrmuon_pre": "SNRMuon (pre-NS)",
    "snrmuon_post_sma": r"SNRMuon (post-NS, $\alpha$=1.0)",
    "snrmuon_pre_sma": r"SNRMuon (pre-NS, $\alpha$=1.0)",
    "soap": "SOAP",
    "snrsoap": "SNRSOAP",
    "snrsoap_sma": r"SNRSOAP ($\alpha$=1.0)",
}


def load_csv(path):
    """Load a W&B-exported CSV, returning (steps, {col_name: values})."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
    
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    steps = data[:, 0]
    runs = {}
    for i, h in enumerate(headers[1:], start=1):
        col = h.strip()
        vals = data[:, i]
        mask = ~np.isnan(vals)
        if mask.any():
            runs[col] = {"steps": steps[mask], "loss": vals[mask]}
    
    return runs


def classify_run(col_name):
    """Map a W&B column name to a canonical run key."""
    col = col_name.lower()
    
    if "snrmuon" in col or "snr_muon" in col:
        has_sma = "sma" in col
        is_pre = "pre_ns" in col or "snrmpre" in col
        if is_pre:
            return "snrmuon_pre_sma" if has_sma else "snrmuon_pre"
        else:
            return "snrmuon_post_sma" if has_sma else "snrmuon_post"
    
    if "snrsoap" in col:
        return "snrsoap_sma" if "sma" in col else "snrsoap"
    
    if "snradam" in col or "power_adam" in col:
        return "snradam_sma" if "sma" in col else "snradam"
    
    if "soap" in col:
        return "soap"
    
    if "muon" in col:
        return "muon"
    
    if "adamw" in col or "adam" in col:
        return "adamw"
    
    return None


def interpolate_step_at_loss(steps, loss, target):
    """Find the step (with linear interpolation) where loss first crosses target."""
    idx = np.where(loss <= target)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(steps[0])
    l0, l1 = loss[i - 1], loss[i]
    s0, s1 = float(steps[i - 1]), float(steps[i])
    frac = (target - l0) / (l1 - l0) if l1 != l0 else 0.0
    return s0 + frac * (s1 - s0)


def plot_family(ax, runs, keys, title):
    """Plot loss curves for a set of runs on the given axes."""
    for key in keys:
        if key not in runs:
            continue
        r = runs[key]
        ax.plot(
            r["steps"] / 1000, r["loss"],
            label=LABELS.get(key, key),
            color=COLORS.get(key, None),
            linewidth=1.5,
            alpha=0.85,
        )
    ax.set_xlabel("Steps (k)")
    ax.set_ylabel("Eval Loss")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}/")
        print("Export data from W&B and place CSV files in paper/data/")
        sys.exit(1)
    
    all_runs = {}
    for csv_path in csv_files:
        print(f"Loading {csv_path.name}...")
        raw = load_csv(csv_path)
        for col_name, data in raw.items():
            key = classify_run(col_name)
            if key is None:
                print(f"  Skipping unrecognized column: {col_name}")
                continue
            if key not in all_runs or len(data["steps"]) > len(all_runs[key]["steps"]):
                all_runs[key] = data
                print(f"  {col_name} -> {key} ({len(data['steps'])} points)")
    
    print(f"\nLoaded {len(all_runs)} runs total.")
    
    # --- Main figure: Adam family + Muon family ---
    adam_keys = ["adamw", "snradam", "snradam_sma"]
    muon_keys = ["muon", "snrmuon_post", "snrmuon_pre", "snrmuon_post_sma", "snrmuon_pre_sma"]
    
    has_adam = any(k in all_runs for k in adam_keys)
    has_muon = any(k in all_runs for k in muon_keys)
    
    if has_adam or has_muon:
        ncols = int(has_adam) + int(has_muon)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)
        col_idx = 0
        if has_adam:
            plot_family(axes[0, col_idx], all_runs, adam_keys, "Adam Family")
            col_idx += 1
        if has_muon:
            plot_family(axes[0, col_idx], all_runs, muon_keys, "Muon Family")
        
        fig.tight_layout()
        out_path = FIG_DIR / "loss_curves.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"\nSaved {out_path} and .png")
        plt.close(fig)
    
    # --- SOAP figure (if data present) ---
    soap_keys = ["soap", "snrsoap", "snrsoap_sma"]
    if any(k in all_runs for k in soap_keys):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plot_family(ax, all_runs, soap_keys, "SOAP Family")
        fig.tight_layout()
        out_path = FIG_DIR / "soap_curves.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {out_path} and .png")
        plt.close(fig)
    
    # --- Print table data ---
    print("\n" + "=" * 90)
    print("TABLE DATA")
    print("=" * 90)
    print(f"{'Key':<22} {'Final Loss':>12} {'Final Step':>12} {'Speedup vs Base':>16}")
    print("-" * 90)
    
    adam_baseline_loss = all_runs["adamw"]["loss"][-1] if "adamw" in all_runs else None
    adam_baseline_steps = float(all_runs["adamw"]["steps"][-1]) if "adamw" in all_runs else None
    muon_baseline_loss = all_runs["muon"]["loss"][-1] if "muon" in all_runs else None
    muon_baseline_steps = float(all_runs["muon"]["steps"][-1]) if "muon" in all_runs else None
    
    for key in adam_keys + muon_keys + soap_keys:
        if key not in all_runs:
            continue
        r = all_runs[key]
        final_loss = r["loss"][-1]
        final_step = r["steps"][-1]
        
        is_muon_family = key.startswith("muon") or key.startswith("snrmuon")
        base_loss = muon_baseline_loss if is_muon_family else adam_baseline_loss
        base_steps = muon_baseline_steps if is_muon_family else adam_baseline_steps
        
        speedup_str = "---"
        if base_loss is not None and base_steps is not None and key not in ("adamw", "muon"):
            sel_step = interpolate_step_at_loss(r["steps"], r["loss"], base_loss)
            if sel_step is not None and sel_step > 0:
                speedup = base_steps / sel_step
                speedup_str = f"{speedup:.2f}x"
        
        print(f"{key:<22} {final_loss:>12.4f} {final_step:>12.0f} {speedup_str:>16}")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
