#!/bin/bash
# =============================================================================
# MLPRope position-channel sweeps on this 8x RTX 5090 node.
#
# Config-driven like calib_attn/launch_llm.sh, but jobs run through the shared
# lifetime-locking helper (no Slurm):
#   gpu-claim run --owner mlprope --job <name> --wait -- python train_gpt.py ...
#
# Usage:
#   ./launch_position_bias.sh                  # write configs + run Phase 1
#   EXPERIMENT_FAMILY=rope ./launch_position_bias.sh
#   SUBMIT_JOBS=false ./launch_position_bias.sh   # only emit JSON configs
#   DRY_RUN=true ./launch_position_bias.sh        # dry-run each config via gpu-claim
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/logs"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs"
WANDB_PROJECT="${WANDB_PROJECT:-mlprope-position-bias}"
WANDB_ENTITY="${WANDB_ENTITY:-ethansmith2000}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase1}" # phase1 | phase1b | phase1c | individual | all
if [[ "${EXPERIMENT_FAMILY}" == "phase1b" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase1b"
elif [[ "${EXPERIMENT_FAMILY}" == "phase1c" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase1c"
else
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase1"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
SUBMIT_JOBS="${SUBMIT_JOBS:-true}"
DRY_RUN="${DRY_RUN:-false}"
# true: background each job (packs the 8 GPUs). false: run one-after-another.
PARALLEL="${PARALLEL:-true}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  GPU_CLAIM_BIN="/workspace/bin/gpu-claim"
fi
OWNER="${OWNER:-mlprope}"
# Default to a small slice so sibling projects can claim the rest.
GPU_SELECTOR="${GPU_SELECTOR:-6,7}" # e.g. any | 0,1,2 | UUID
PIDS=()
JOB_NAMES=()

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}" "${OUTPUT_ROOT}"

# Common settings (single-GPU jobs; pack up to 8 concurrent via gpu-claim)
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-200}"
WITH_TRACKING="${WITH_TRACKING:-true}"
BASE_WD="0.01"
BASE_BETA1="0.9"
BASE_BETA2="0.98"
POS_RANK="${POS_RANK:-32}"
POS_MLP_HIDDEN="${POS_MLP_HIDDEN:-128}"
POS_SHARING="${POS_SHARING:-per_head}"
REL_EXTENT="${REL_EXTENT:-}" # empty => follow block_size in train_gpt.py

# Model config: "hidden_size depth n_head lr batch_size max_train_steps block_size"
MODEL_CONFIG="${MODEL_CONFIG:-768 8 8 3.0e-4 8 10000 1024}"
read -r HIDDEN_SIZE DEPTH N_HEAD LR BATCH_SIZE MAX_TRAIN_STEPS BLOCK_SIZE <<< "${MODEL_CONFIG}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_HOME="${WANDB_HOME:-/workspace/.wandb_home}"
export WANDB_DIR="${WANDB_DIR:-/workspace/.wandb_home}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${WANDB_HOME}" "${WANDB_DIR}"

# Shared tokenized cache (must match train_gpt default schema: input_ids only).
TOKENIZED_DATASET_PATH="${TOKENIZED_DATASET_PATH:-/workspace/.cache/tokenized/openwebtext_gpt2_bs1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-50}"
PROFILE_EVERY="${PROFILE_EVERY:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1000}"

write_common_config() {
  local cfg_file="$1"
  local extra_json="$2"
  local rel_extent_json="null"
  if [[ -n "${REL_EXTENT}" ]]; then
    rel_extent_json="${REL_EXTENT}"
  fi
  cat > "${cfg_file}" <<JSON
{
  "learning_rate": ${LR},
  "weight_decay": ${BASE_WD},
  "beta1": ${BASE_BETA1},
  "beta2": ${BASE_BETA2},
  "per_device_train_batch_size": ${BATCH_SIZE},
  "max_train_steps": ${MAX_TRAIN_STEPS},
  "block_size": ${BLOCK_SIZE},
  "base_output_dir": "${OUTPUT_ROOT}",
  "hidden_size": ${HIDDEN_SIZE},
  "depth": ${DEPTH},
  "n_head": ${N_HEAD},
  "pos_rank": ${POS_RANK},
  "pos_mlp_hidden": ${POS_MLP_HIDDEN},
  "rel_extent": ${rel_extent_json},
  "tokenized_dataset_path": "${TOKENIZED_DATASET_PATH}",
  "num_workers": ${NUM_WORKERS},
  "prefetch_factor": ${PREFETCH_FACTOR},
  "persistent_workers": true,
  "non_blocking": true,
  "log_every_n_steps": ${LOG_EVERY},
  "profile_every_n_steps": ${PROFILE_EVERY},
  "validate_every": ${VALIDATE_EVERY},
  "checkpointing_steps": null,
  "save_final_model": false,
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_entity": "${WANDB_ENTITY}",
  "with_tracking": ${WITH_TRACKING},
  "num_warmup_steps": ${NUM_WARMUP_STEPS},
  "compile": true,
  "compile_mode": "default",
  "compile_fullgraph": false,
  ${extra_json}
}
JSON
}

run_job() {
  local job_name="$1"
  local cfg_file="$2"
  local log_file="${LOG_DIR}/${job_name}.log"

  if [[ "${SUBMIT_JOBS}" != "true" ]]; then
    echo "Generated: ${cfg_file}"
    return 0
  fi

  if [[ ! -x "${GPU_CLAIM_BIN}" && ! -f "${GPU_CLAIM_BIN}" ]]; then
    echo "gpu-claim not found at ${GPU_CLAIM_BIN}. See /workspace/GPU_QUEUEING.md" >&2
    exit 1
  fi

  local dry_args=()
  if [[ "${DRY_RUN}" == "true" ]]; then
    dry_args+=(--dry_run)
  fi

  echo "QUEUE_START ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ) -> ${log_file}"
  # One exclusive GPU for the whole train process (lifetime flock).
  if [[ "${PARALLEL}" == "true" ]]; then
    (
      "${GPU_CLAIM_BIN}" run \
        --owner "${OWNER}" \
        --job "${job_name}" \
        --gpu "${GPU_SELECTOR}" \
        --wait \
        -- \
        "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" --override_json "${cfg_file}" "${dry_args[@]}"
    ) >"${log_file}" 2>&1 &
    PIDS+=("$!")
    JOB_NAMES+=("${job_name}")
    echo "QUEUE_BG ${job_name} pid=$!"
    return 0
  fi

  "${GPU_CLAIM_BIN}" run \
    --owner "${OWNER}" \
    --job "${job_name}" \
    --gpu "${GPU_SELECTOR}" \
    --wait \
    -- \
    "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" --override_json "${cfg_file}" "${dry_args[@]}" \
    > >(tee -a "${log_file}") 2>&1
  local rc=$?
  echo "QUEUE_DONE ${job_name} rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "${rc}"
}

emit_variant() {
  local variant="$1"
  local attn_impl="$2"
  local job_name="$3"
  local cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"pos_variant\": \"${variant}\", \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\""
  run_job "${job_name}" "${cfg_file}"
}

emit_channel_variant() {
  local job_name="$1"
  local attn_impl="$2"
  local qk_json="$3"
  local logit_json="$4"
  local cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"pos_variant\": null, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\""
  run_job "${job_name}" "${cfg_file}"
}

want_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "phase1" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase1b_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "phase1b" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase1c_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "phase1c" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

echo "MLPROPE_QUEUE_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "family=${EXPERIMENT_FAMILY} model='${MODEL_CONFIG}' submit=${SUBMIT_JOBS} dry_run=${DRY_RUN} parallel=${PARALLEL}"
echo "gpu-claim=${GPU_CLAIM_BIN} selector=${GPU_SELECTOR}"
"${GPU_CLAIM_BIN}" status || true

# Direction 1 / Phase 1 expressiveness sweep.
if want_family "rope"; then
  emit_variant "rope" "sdpa" "rope-h${HIDDEN_SIZE}d${DEPTH}"
fi
if want_family "add_rope"; then
  emit_variant "add_rope" "flex" "add_rope-h${HIDDEN_SIZE}d${DEPTH}"
fi
if want_family "linear"; then
  emit_variant "linear" "flex" "linear-h${HIDDEN_SIZE}d${DEPTH}"
fi
if want_family "low_rank"; then
  emit_variant "low_rank" "flex" "low_rank-r${POS_RANK}-h${HIDDEN_SIZE}d${DEPTH}"
fi
if want_family "mlp_rope"; then
  emit_variant "mlp_rope" "flex" "mlp_rope-m${POS_MLP_HIDDEN}-h${HIDDEN_SIZE}d${DEPTH}"
fi

# Direction 1b. Existing completed RoPE and linear-logit runs are the anchors;
# this family emits only the four new/corrected ablations.
QK_DISABLED="{\"enabled\": false}"
LOGIT_DISABLED="{\"enabled\": false}"
if want_phase1b_family "low_rank_corrected"; then
  emit_channel_variant \
    "logit-low-rank-linear-r${POS_RANK}-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "flex" \
    "${QK_DISABLED}" \
    "{\"enabled\": true, \"feature_map\": \"low_rank\", \"sharing\": \"${POS_SHARING}\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}"
fi
if want_phase1b_family "bottleneck_mlp"; then
  emit_channel_variant \
    "logit-bottleneck-mlp-r${POS_RANK}-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "flex" \
    "${QK_DISABLED}" \
    "{\"enabled\": true, \"feature_map\": \"bottleneck_mlp\", \"sharing\": \"${POS_SHARING}\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}"
fi
if want_phase1b_family "qk_phase_linear"; then
  emit_channel_variant \
    "qk-phase-linear-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"linear\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"phase_residual\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi
if want_phase1b_family "qk_phase_mlp"; then
  emit_channel_variant \
    "qk-phase-mlp-m${POS_MLP_HIDDEN}-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"mlp\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"phase_residual\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi

# Direction 1c: true AddRoPE (replace multiplicative RoPE with q + f(cis)).
# See https://jonathanc.net/blog/additive-rotary-embedding — extended with our
# feature-map taxonomy. RoPE baseline from Phase 1 remains the anchor.
if want_phase1c_family "qk_add_identity"; then
  emit_channel_variant \
    "qk-addrope-identity-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"identity\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"add\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi
if want_phase1c_family "qk_add_add_rope"; then
  emit_channel_variant \
    "qk-addrope-add_rope-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"add_rope\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"add\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi
if want_phase1c_family "qk_add_linear"; then
  emit_channel_variant \
    "qk-addrope-linear-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"linear\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"add\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi
if want_phase1c_family "qk_add_low_rank"; then
  emit_channel_variant \
    "qk-addrope-low_rank-r${POS_RANK}-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"low_rank\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"add\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi
if want_phase1c_family "qk_add_mlp"; then
  emit_channel_variant \
    "qk-addrope-mlp-m${POS_MLP_HIDDEN}-${POS_SHARING}-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "{\"enabled\": true, \"feature_map\": \"mlp\", \"sharing\": \"${POS_SHARING}\", \"apply\": \"add\", \"rank\": ${POS_RANK}, \"mlp_hidden\": ${POS_MLP_HIDDEN}}" \
    "${LOGIT_DISABLED}"
fi

if [[ "${SUBMIT_JOBS}" == "true" && "${PARALLEL}" == "true" && ${#PIDS[@]} -gt 0 ]]; then
  echo "Waiting on ${#PIDS[@]} background jobs..."
  fail=0
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${JOB_NAMES[$i]}"
    if wait "${pid}"; then
      echo "QUEUE_DONE ${name} rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      rc=$?
      echo "QUEUE_DONE ${name} rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    echo "One or more jobs failed." >&2
    exit 1
  fi
fi

echo "MLPROPE_QUEUE_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
