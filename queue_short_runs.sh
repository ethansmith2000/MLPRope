#!/bin/bash
# Wait for the shared tokenized cache, then queue Phase-1 runs via gpu-claim.
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

TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/workspace/.cache/tokenized/openwebtext_gpt2_bs1024}"
LOG="${LOG:-logs/queue_short_runs.log}"
mkdir -p logs "$(dirname "${TOKENIZED_DATASET_PATH}")"

cache_ready() {
  [[ -f "${TOKENIZED_DATASET_PATH}/dataset_dict.json" ]] \
    || [[ -f "${TOKENIZED_DATASET_PATH}/.ready" ]]
}

echo "SHORT_QUEUE_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "Waiting for tokenized cache: ${TOKENIZED_DATASET_PATH}" | tee -a "${LOG}"

# Do not start a second downloader while calib/shared prepare holds the flock.
while ! cache_ready; do
  if ! pgrep -f 'prepare_tokenized_dataset.py' >/dev/null 2>&1; then
    echo "No prepare_tokenized_dataset.py running and cache missing; starting one." | tee -a "${LOG}"
    /venv/main/bin/python -u prepare_tokenized_dataset.py >> logs/prepare_tokenized_dataset.log 2>&1 &
  fi
  echo "  still waiting... $(du -sh "${HF_HOME}" 2>/dev/null | awk '{print $1}') HF_HOME $(date -u +%H:%M:%SZ)" | tee -a "${LOG}"
  sleep 30
done

echo "Cache ready at ${TOKENIZED_DATASET_PATH}" | tee -a "${LOG}"
du -sh "${TOKENIZED_DATASET_PATH}" | tee -a "${LOG}" || true

# Phase-1: 10k steps. Claim any free GPU (siblings may hold some).
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase1}"
export MODEL_CONFIG="${MODEL_CONFIG:-768 8 8 3.0e-4 8 10000 1024}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-200}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-1000}"
export LOG_EVERY="${LOG_EVERY:-50}"
export PROFILE_EVERY="${PROFILE_EVERY:-10}"
export WITH_TRACKING="${WITH_TRACKING:-true}"
export PARALLEL="${PARALLEL:-true}"
export SUBMIT_JOBS="${SUBMIT_JOBS:-true}"
export GPU_SELECTOR="${GPU_SELECTOR:-any}"
export TOKENIZED_DATASET_PATH

echo "Launching Phase-1 on GPUs ${GPU_SELECTOR} (gpu-claim --wait)" | tee -a "${LOG}"
gpu-claim status | tee -a "${LOG}" || true
./launch_position_bias.sh 2>&1 | tee -a "${LOG}"
echo "SHORT_QUEUE_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
