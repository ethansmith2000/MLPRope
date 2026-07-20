#!/bin/bash
# Wait for the shared tokenized cache, then queue short Phase-1 runs on a
# small GPU slice (default 6,7) so other projects can use 0-5.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source /workspace/.env 2>/dev/null || true
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_HOME="${WANDB_HOME:-/workspace/.wandb_home}"
export WANDB_DIR="${WANDB_DIR:-/workspace/.wandb_home}"
export WANDB_ENTITY="${WANDB_ENTITY:-ethansmith2000}"
export WANDB_PROJECT="${WANDB_PROJECT:-mlprope-position-bias}"

TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/workspace/.cache/mlprope_openwebtext_gpt2_bs1024_ids}"
LOG="${LOG:-logs/queue_short_runs.log}"
mkdir -p logs "$(dirname "${TOKENIZED_DATASET_PATH}")"

echo "SHORT_QUEUE_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "Waiting for tokenized cache: ${TOKENIZED_DATASET_PATH}" | tee -a "${LOG}"

# Do not start a second downloader. prepare_tokenized_dataset.py (or another
# polite holder of the flock) owns Hub/tokenization. We only wait.
while [[ ! -d "${TOKENIZED_DATASET_PATH}" ]]; do
  if ! pgrep -f 'prepare_tokenized_dataset.py' >/dev/null 2>&1; then
    echo "No prepare_tokenized_dataset.py running and cache missing; starting one." | tee -a "${LOG}"
    /venv/main/bin/python -u prepare_tokenized_dataset.py >> logs/prepare_tokenized_dataset.log 2>&1 &
  fi
  echo "  still waiting... $(du -sh "${HF_HOME}" 2>/dev/null | awk '{print $1}') HF_HOME $(date -u +%H:%M:%SZ)" | tee -a "${LOG}"
  sleep 30
done

echo "Cache ready. Ensuring wandb project ${WANDB_ENTITY}/${WANDB_PROJECT}" | tee -a "${LOG}"
/venv/main/bin/python - <<PY
import os
from pathlib import Path
import wandb
os.environ.setdefault("WANDB_HOME", "/workspace/.wandb_home")
os.environ.setdefault("WANDB_DIR", "/workspace/.wandb_home")
Path(os.environ["WANDB_HOME"]).mkdir(parents=True, exist_ok=True)
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "mlprope-position-bias"),
    entity=os.environ.get("WANDB_ENTITY", "ethansmith2000"),
    name="project-bootstrap",
    dir=os.environ["WANDB_DIR"],
    config={"bootstrap": True},
)
print("wandb project:", run.entity, run.project, run.url)
run.finish()
PY

# Phase-1: 10k steps, GPUs 6,7 only (leave 0-5 for siblings), wandb on.
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase1}"
export MODEL_CONFIG="${MODEL_CONFIG:-768 8 8 3.0e-4 8 10000 1024}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-200}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-1000}"
export LOG_EVERY="${LOG_EVERY:-50}"
export PROFILE_EVERY="${PROFILE_EVERY:-10}"
export WITH_TRACKING="${WITH_TRACKING:-true}"
export PARALLEL="${PARALLEL:-true}"
export SUBMIT_JOBS="${SUBMIT_JOBS:-true}"
export GPU_SELECTOR="${GPU_SELECTOR:-6,7}"
export TOKENIZED_DATASET_PATH

echo "Launching short runs on GPUs ${GPU_SELECTOR} (leaving others free)" | tee -a "${LOG}"
gpu-claim status | tee -a "${LOG}" || true
./launch_position_bias.sh 2>&1 | tee -a "${LOG}"
echo "SHORT_QUEUE_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
