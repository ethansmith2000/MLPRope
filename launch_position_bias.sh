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
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase1}" # phase1 | ... | phase9_hyper_capacity | phase10_hyper_geometry | phase11_spectral | phase12_offset_qknorm | phase13_decomp_qknorm | phase14_angular_rank | phase15_decay_mixing | phase16_cheap_mixing | individual | all
if [[ "${EXPERIMENT_FAMILY}" == "phase1b" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase1b"
elif [[ "${EXPERIMENT_FAMILY}" == "phase1c" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase1c"
elif [[ "${EXPERIMENT_FAMILY}" == "phase2_coupling" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase2_coupling"
elif [[ "${EXPERIMENT_FAMILY}" == "phase2_followup" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase2_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_geometry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_amplitude_followup" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_amplitude_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_promotion" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_promotion"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_basis_screen" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_basis_screen"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry_transfer" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_geometry_transfer"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_frontier_screen" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_frontier_screen"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_conditioning_retry" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_conditioning_retry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_coupling_transfer" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_coupling_transfer"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_structural_followup" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_structural_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_pairwise_logit" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_pairwise_logit"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_addrope_components" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_addrope_components"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_residual_sector" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_residual_sector"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_final_decision" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_final_decision"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_compact_basis" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase3_compact_basis"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_extrapolation" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase4_extrapolation"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_post_qk_norm" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase4_post_qk_norm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_safe_conditioning" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase4_safe_conditioning"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_additive_geometry" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase4_additive_geometry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_phase_conditioning" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase4_phase_conditioning"
elif [[ "${EXPERIMENT_FAMILY}" == "phase5_null_conditioning" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase5_null_conditioning"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_geometry_norm" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase6_geometry_norm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_content_transfer" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase6_content_transfer"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_efficiency" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase6_efficiency"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_scale" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase6_scale"
elif [[ "${EXPERIMENT_FAMILY}" == "phase7_scale_probe" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase7_scale_probe"
elif [[ "${EXPERIMENT_FAMILY}" == "phase7_scale_50k" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase7_scale_50k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase8_hyper_smoke" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase8_hyper_smoke"
elif [[ "${EXPERIMENT_FAMILY}" == "phase8_hyper_5k" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase8_hyper_5k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase8_addrope_clean" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase8_addrope_clean"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_unit_hyper" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase9_unit_hyper"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_carrier_followup" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase9_carrier_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_hyper_30k" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase9_hyper_30k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_qk_independence" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase9_qk_independence"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_hyper_capacity" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase9_hyper_capacity"
elif [[ "${EXPERIMENT_FAMILY}" == "phase10_hyper_geometry" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase10_hyper_geometry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase11_spectral"
elif [[ "${EXPERIMENT_FAMILY}" == "phase12_offset_qknorm" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase12_offset_qknorm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase13_decomp_qknorm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase14_angular_rank" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase14_angular_rank"
elif [[ "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase15_decay_mixing"
elif [[ "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase16_cheap_mixing"
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
# Suites use every GPU that is available through the shared lifetime-locking
# queue; historical families retain the conservative two-GPU default.
if [[ -z "${GPU_SELECTOR:-}" ]]; then
  if [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry" \
    || "${EXPERIMENT_FAMILY}" == "phase3_amplitude_followup" \
    || "${EXPERIMENT_FAMILY}" == "phase3_promotion" \
    || "${EXPERIMENT_FAMILY}" == "phase3_basis_screen" \
    || "${EXPERIMENT_FAMILY}" == "phase3_geometry_transfer" \
    || "${EXPERIMENT_FAMILY}" == "phase3_frontier_screen" \
    || "${EXPERIMENT_FAMILY}" == "phase3_conditioning_retry" \
    || "${EXPERIMENT_FAMILY}" == "phase3_coupling_transfer" \
    || "${EXPERIMENT_FAMILY}" == "phase3_structural_followup" \
    || "${EXPERIMENT_FAMILY}" == "phase3_pairwise_logit" \
    || "${EXPERIMENT_FAMILY}" == "phase3_addrope_components" \
    || "${EXPERIMENT_FAMILY}" == "phase3_residual_sector" \
    || "${EXPERIMENT_FAMILY}" == "phase3_final_decision" \
    || "${EXPERIMENT_FAMILY}" == "phase3_compact_basis" \
    || "${EXPERIMENT_FAMILY}" == "phase4_extrapolation" \
    || "${EXPERIMENT_FAMILY}" == "phase4_post_qk_norm" \
    || "${EXPERIMENT_FAMILY}" == "phase4_safe_conditioning" \
    || "${EXPERIMENT_FAMILY}" == "phase4_additive_geometry" \
    || "${EXPERIMENT_FAMILY}" == "phase4_phase_conditioning" \
    || "${EXPERIMENT_FAMILY}" == "phase5_null_conditioning" \
    || "${EXPERIMENT_FAMILY}" == "phase6_geometry_norm" \
    || "${EXPERIMENT_FAMILY}" == "phase6_content_transfer" \
    || "${EXPERIMENT_FAMILY}" == "phase6_efficiency" \
    || "${EXPERIMENT_FAMILY}" == "phase6_scale" \
    || "${EXPERIMENT_FAMILY}" == "phase7_scale_probe" \
    || "${EXPERIMENT_FAMILY}" == "phase7_scale_50k" \
    || "${EXPERIMENT_FAMILY}" == "phase8_hyper_smoke" \
    || "${EXPERIMENT_FAMILY}" == "phase8_hyper_5k" \
    || "${EXPERIMENT_FAMILY}" == "phase8_addrope_clean" \
    || "${EXPERIMENT_FAMILY}" == "phase9_unit_hyper" \
    || "${EXPERIMENT_FAMILY}" == "phase9_carrier_followup" \
    || "${EXPERIMENT_FAMILY}" == "phase9_hyper_30k" \
    || "${EXPERIMENT_FAMILY}" == "phase9_qk_independence" \
    || "${EXPERIMENT_FAMILY}" == "phase9_hyper_capacity" \
    || "${EXPERIMENT_FAMILY}" == "phase10_hyper_geometry" \
    || "${EXPERIMENT_FAMILY}" == "phase11_spectral" \
    || "${EXPERIMENT_FAMILY}" == "phase12_offset_qknorm" \
    || "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" \
    || "${EXPERIMENT_FAMILY}" == "phase14_angular_rank" \
    || "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" \
    || "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" ]]; then
    GPU_SELECTOR="any"
  else
    GPU_SELECTOR="6,7"
  fi
fi
# Separate config dirs so Phase-2 writes do not touch historical JSON.
if [[ "${EXPERIMENT_FAMILY}" == "phase2_coupling" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase2_coupling"
elif [[ "${EXPERIMENT_FAMILY}" == "phase2_followup" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase2_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_geometry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_amplitude_followup" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_amplitude_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_promotion" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_promotion"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_basis_screen" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_basis_screen"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry_transfer" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_geometry_transfer"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_frontier_screen" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_frontier_screen"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_conditioning_retry" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_conditioning_retry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_coupling_transfer" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_coupling_transfer"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_structural_followup" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_structural_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_pairwise_logit" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_pairwise_logit"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_addrope_components" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_addrope_components"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_residual_sector" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_residual_sector"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_final_decision" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_final_decision"
elif [[ "${EXPERIMENT_FAMILY}" == "phase3_compact_basis" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase3_compact_basis"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_extrapolation" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase4_extrapolation"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_post_qk_norm" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase4_post_qk_norm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_safe_conditioning" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase4_safe_conditioning"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_additive_geometry" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase4_additive_geometry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase4_phase_conditioning" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase4_phase_conditioning"
elif [[ "${EXPERIMENT_FAMILY}" == "phase5_null_conditioning" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase5_null_conditioning"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_geometry_norm" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase6_geometry_norm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_content_transfer" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase6_content_transfer"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_efficiency" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase6_efficiency"
elif [[ "${EXPERIMENT_FAMILY}" == "phase6_scale" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase6_scale"
elif [[ "${EXPERIMENT_FAMILY}" == "phase7_scale_probe" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase7_scale_probe"
elif [[ "${EXPERIMENT_FAMILY}" == "phase7_scale_50k" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase7_scale_50k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase8_hyper_smoke" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase8_hyper_smoke"
elif [[ "${EXPERIMENT_FAMILY}" == "phase8_hyper_5k" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase8_hyper_5k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase8_addrope_clean" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase8_addrope_clean"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_unit_hyper" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase9_unit_hyper"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_carrier_followup" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase9_carrier_followup"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_hyper_30k" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase9_hyper_30k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_qk_independence" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase9_qk_independence"
elif [[ "${EXPERIMENT_FAMILY}" == "phase9_hyper_capacity" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase9_hyper_capacity"
elif [[ "${EXPERIMENT_FAMILY}" == "phase10_hyper_geometry" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase10_hyper_geometry"
elif [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase11_spectral"
elif [[ "${EXPERIMENT_FAMILY}" == "phase12_offset_qknorm" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase12_offset_qknorm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase13_decomp_qknorm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase14_angular_rank" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase14_angular_rank"
elif [[ "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase15_decay_mixing"
elif [[ "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase16_cheap_mixing"
fi
PIDS=()
JOB_NAMES=()

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}" "${OUTPUT_ROOT}"

# Common settings (single-GPU jobs; pack up to 8 concurrent via gpu-claim)
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-200}"
WITH_TRACKING="${WITH_TRACKING:-true}"
WANDB_GROUP="${WANDB_GROUP:-}"
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
  local train_steps="${3:-${MAX_TRAIN_STEPS}}"
  local validate_every="${4:-${VALIDATE_EVERY}}"
  local rel_extent_json="null"
  local wandb_group_json="null"
  if [[ -n "${REL_EXTENT}" ]]; then
    rel_extent_json="${REL_EXTENT}"
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    wandb_group_json="\"${WANDB_GROUP}\""
  fi
  cat > "${cfg_file}" <<JSON
{
  "learning_rate": ${LR},
  "weight_decay": ${BASE_WD},
  "beta1": ${BASE_BETA1},
  "beta2": ${BASE_BETA2},
  "per_device_train_batch_size": ${BATCH_SIZE},
  "max_train_steps": ${train_steps},
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
  "validate_every": ${validate_every},
  "checkpointing_steps": null,
  "save_final_model": false,
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_entity": "${WANDB_ENTITY}",
  "wandb_group": ${wandb_group_json},
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

# Helper for future v2 experiment families. Does not rewrite completed Phase 1/1b/1c
# JSON files. Example qk fragment:
#   {"enabled": true, "application": "rotary", "geometry": "phase",
#    "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []},
#    "mapper": {"kind": "mlp", "residual": true, "rank": 32, "hidden_dim": 128},
#    "qk_coupling": "shared", "head_coupling": "per_head_independent"}
emit_v2_channel_variant() {
  local job_name="$1"
  local attn_impl="$2"
  local qk_json="$3"
  local logit_json="$4"
  local cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\""
  run_job "${job_name}" "${cfg_file}"
}

# Fully general v2 emitter for future playground sweeps. Callers provide complete
# canonical channel fragments; no family invokes this automatically.
emit_v2_playground_variant() {
  local job_name="$1"
  local attn_impl="$2"
  local qk_json="$3"
  local logit_json="$4"
  local residual_json='{"enabled": false}'
  local write_json='{"enabled": false}'
  if [[ $# -ge 5 ]]; then
    residual_json="$5"
  fi
  if [[ $# -ge 6 ]]; then
    write_json="$6"
  fi
  local cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": ${residual_json}, \"attention_write\": ${write_json}, \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\""
  run_job "${job_name}" "${cfg_file}"
}

emit_v2_playground_seed_variant() {
  local job_name="$1"
  local seed="$2"
  local attn_impl="$3"
  local qk_json="$4"
  local logit_json="$5"
  local residual_json='{"enabled": false}'
  local write_json='{"enabled": false}'
  local use_rope=true
  local post_position_qk_norm=false
  if [[ $# -ge 6 ]]; then
    residual_json="$6"
  fi
  if [[ $# -ge 7 ]]; then
    write_json="$7"
  fi
  if [[ $# -ge 8 ]]; then
    use_rope="$8"
  fi
  if [[ $# -ge 9 ]]; then
    post_position_qk_norm="$9"
  fi
  local cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, \"use_rope\": ${use_rope}, \"post_position_qk_norm\": ${post_position_qk_norm}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": ${residual_json}, \"attention_write\": ${write_json}, \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\""
  run_job "${job_name}" "${cfg_file}"
}

# Extended Q/K helper. Geometry:
#   additive/free | additive/pair_normalized | additive/amplitude_phase
#   rotary/phase | rotary/projected_phase | rotary/scaled_phase
# Input kind:
#   frozen_fourier | learned_temperature_fourier | learned_frequency_fourier
# Conditioning:
#   none | local_residual | content_gate
v2_qk_playground_json() {
  local application="$1"
  local geometry="$2"
  local mapper_kind="$3"
  local residual="$4"
  local qk_coupling="$5"
  local input_kind="${6:-frozen_fourier}"
  local conditioning="${7:-none}"
  local head_coupling="${8:-per_head_independent}"
  local amplitude_init="${9:-0.1}"
  local scalars_json="${10:-[]}"
  local conditioning_hidden="${11:-${POS_MLP_HIDDEN}}"
  local basis_dim="${12:-null}"
  local learn_amplitude="${13:-true}"
  local learn_phase="${14:-true}"
  cat <<JSON
{"enabled": true, "application": "${application}", "geometry": "${geometry}", "input": {"kind": "${input_kind}", "basis_dim": ${basis_dim}, "theta": null, "scalars": ${scalars_json}}, "mapper": {"kind": "${mapper_kind}", "residual": ${residual}, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": ${amplitude_init}, "amplitude_parameterization": "signed", "learn_amplitude": ${learn_amplitude}, "learn_phase": ${learn_phase}, "phase_scale": 1.0, "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": {"kind": "${conditioning}", "hidden_dim": ${conditioning_hidden}}, "qk_coupling": "${qk_coupling}", "head_coupling": "${head_coupling}"}
JSON
}

v2_safe_qk_json() {
  local application="$1"
  local geometry="$2"
  local conditioning="${3:-none}"
  local content_source="${4:-residual}"
  local activation="${5:-scaled_sigmoid}"
  local gate_init="${6:-1.0}"
  local additive_normalization="none"
  if [[ "${application}" == "additive" ]]; then
    additive_normalization="rms"
  fi
  cat <<JSON
{"enabled": true, "application": "${application}", "geometry": "${geometry}", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": 0.3, "amplitude_max": 1.0, "amplitude_parameterization": "bounded_sigmoid", "learn_amplitude": true, "learn_phase": true, "phase_scale": 1.0, "additive_normalization": "${additive_normalization}", "additive_gain_init": 0.212132, "additive_gain_max": 0.5, "learn_additive_gain": true, "scale_init": 1.0, "scale_max": 2.0, "scale_parameterization": "bounded_log"}, "conditioning": {"kind": "${conditioning}", "source": "${content_source}", "activation": "${activation}", "hidden_dim": 32, "gate_init": ${gate_init}}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_phase_conditioned_pair_json() {
  local target="$1" # q | k | both
  local conditioner_coupling="$2" # shared | shared_trunk_separate_readouts
  local phase_bound="${3:-0.25}"
  cat <<JSON
{"enabled": true, "application": "additive", "geometry": "pair_normalized", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": 0.3, "amplitude_parameterization": "signed", "learn_amplitude": true, "learn_phase": true, "phase_scale": 1.0, "additive_normalization": "none", "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": {"kind": "phase_rotation", "source": "residual", "activation": "tanh", "hidden_dim": 32, "gate_init": 0.0, "target": "${target}", "coupling": "${conditioner_coupling}", "phase_bound": ${phase_bound}}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_null_conditioned_qk_json() {
  local kind="$1" # adaptive_gain | additive_phase | rope_phase
  local coupling="$2" # shared | shared_trunk_separate_readouts
  local application="rotary"
  local geometry="phase"
  local learn_amplitude=true
  local learn_phase=false
  local amplitude_init=0.3
  if [[ "${kind}" == "additive_phase" ]]; then
    application="additive"
    geometry="amplitude_phase"
    learn_amplitude=false
    learn_phase=false
  fi
  cat <<JSON
{"enabled": true, "application": "${application}", "geometry": "${geometry}", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "identity", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": ${amplitude_init}, "amplitude_parameterization": "signed", "learn_amplitude": ${learn_amplitude}, "learn_phase": ${learn_phase}, "phase_scale": 1.0, "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": {"kind": "${kind}", "source": "dedicated", "activation": "linear", "hidden_dim": 32, "target": "both", "coupling": "${coupling}", "phase_bound": 0.25}, "qk_coupling": "shared", "head_coupling": "per_head_independent"}
JSON
}

v2_carrier_hyper_qk_json() {
  local application="$1" # additive | rotary
  local input_mode="$2" # content | position | content_position
  local network="$3" # linear | silu_mlp | swiglu_mlp
  local components="$4" # phase | log_gain_phase
  local conditioner_coupling="${5:-shared_trunk_separate_readouts}"
  local target="${6:-both}"
  local geometry="phase"
  local mapper_kind="identity"
  local qk_coupling="shared"
  local learn_phase=false
  local amplitude_init=0.3
  local basis_dim=16
  local scalars_json='["normalized_position", "log_position"]'
  if [[ "${application}" == "additive" ]]; then
    geometry="amplitude_phase"
    mapper_kind="linear"
    qk_coupling="shared_trunk_separate_readouts"
    learn_phase=true
  fi
  cat <<JSON
{"enabled": true, "application": "${application}", "geometry": "${geometry}", "input": {"kind": "frozen_fourier", "basis_dim": ${basis_dim}, "theta": null, "scalars": ${scalars_json}}, "mapper": {"kind": "${mapper_kind}", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": ${amplitude_init}, "amplitude_parameterization": "signed", "learn_amplitude": true, "learn_phase": ${learn_phase}, "phase_scale": 1.0, "additive_normalization": "none", "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": {"kind": "carrier_hypernetwork", "source": "dedicated", "hidden_dim": 64, "input_mode": "${input_mode}", "network": "${network}", "components": "${components}", "target": "${target}", "coupling": "${conditioner_coupling}", "head_coupling": "per_head_independent"}, "qk_coupling": "${qk_coupling}", "head_coupling": "per_head_independent"}
JSON
}

v2_clean_addrope_qk_json() {
  local mode="$1" # direct | fixed | dynamic
  local input_mode="${2:-content}"
  local network="${3:-linear}"
  local parameter_source="direct"
  local learn_amplitude=true
  local learn_phase=true
  local conditioning='{"kind": "none"}'
  if [[ "${mode}" == "fixed" ]]; then
    learn_amplitude=false
    learn_phase=false
  elif [[ "${mode}" == "dynamic" ]]; then
    learn_amplitude=false
    learn_phase=false
    conditioning="{\"kind\": \"carrier_hypernetwork\", \"source\": \"dedicated\", \"hidden_dim\": 64, \"input_mode\": \"${input_mode}\", \"network\": \"${network}\", \"components\": \"amplitude_phase\", \"target\": \"both\", \"coupling\": \"shared_trunk_separate_readouts\", \"head_coupling\": \"per_head_independent\"}"
  fi
  cat <<JSON
{"enabled": true, "application": "additive", "geometry": "amplitude_phase", "input": {"kind": "frozen_fourier", "basis_dim": 16, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"parameter_source": "${parameter_source}", "amplitude_init": 0.3, "amplitude_parameterization": "softplus", "learn_amplitude": ${learn_amplitude}, "learn_phase": ${learn_phase}, "phase_scale": 1.0, "additive_normalization": "none"}, "conditioning": ${conditioning}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_unit_hyper_qk_json() {
  local mode="$1" # direct | hyper
  local input_mode="${2:-content}"
  local network="${3:-linear}"
  local target="${4:-both}"
  local static_complement="${5:-false}"
  local conditioning_coupling="${6:-shared_trunk_separate_readouts}"
  local conditioning_head_coupling="${7:-per_head_independent}"
  local conditioning_hidden_dim="${8:-64}"
  local amplitude_parameterization="${9:-signed}"
  local components="${10:-amplitude_phase}"
  local input_normalization="${11:-none}"
  local learnable_input_gains="${12:-false}"
  local learn_static_amplitude="${13:-false}"
  local offset_parameterization="${14:-raw}"
  local angular_rank="${15:-8}"
  local readout_head_mixing="${16:-none}"
  local readout_mix_rank="${17:-32}"
  local learn_amplitude=true
  local learn_phase=true
  local conditioning='{"kind": "none"}'
  if [[ "${mode}" == "hyper" ]]; then
    learn_amplitude="${learn_static_amplitude}"
    learn_phase=false
    conditioning="{\"kind\": \"carrier_hypernetwork\", \"source\": \"dedicated\", \"hidden_dim\": ${conditioning_hidden_dim}, \"input_mode\": \"${input_mode}\", \"input_normalization\": \"${input_normalization}\", \"learnable_input_gains\": ${learnable_input_gains}, \"network\": \"${network}\", \"components\": \"${components}\", \"target\": \"${target}\", \"coupling\": \"${conditioning_coupling}\", \"static_complement\": ${static_complement}, \"head_coupling\": \"${conditioning_head_coupling}\", \"offset_parameterization\": \"${offset_parameterization}\", \"angular_rank\": ${angular_rank}, \"readout_head_mixing\": \"${readout_head_mixing}\", \"readout_mix_rank\": ${readout_mix_rank}}"
  fi
  cat <<JSON
{"enabled": true, "application": "additive", "geometry": "amplitude_phase", "input": {"kind": "frozen_fourier", "basis_dim": 16, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"parameter_source": "direct", "amplitude_init": 1.0, "amplitude_parameterization": "${amplitude_parameterization}", "learn_amplitude": ${learn_amplitude}, "learn_phase": ${learn_phase}, "phase_scale": 1.0, "additive_normalization": "none"}, "conditioning": ${conditioning}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_phase_hyperrope_qk_json() {
  cat <<JSON
{"enabled": true, "application": "rotary", "geometry": "phase", "input": {"kind": "frozen_fourier", "basis_dim": 16, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"parameter_source": "mapped", "learn_amplitude": true, "learn_phase": false, "phase_scale": 1.0}, "conditioning": {"kind": "carrier_hypernetwork", "source": "dedicated", "hidden_dim": 64, "input_mode": "content_position", "network": "silu_mlp", "components": "phase", "target": "both", "coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_promoted_qk_json() {
  local geometry="$1" # amplitude_phase | pair_normalized
  local basis_dim="${2:-null}"
  local conditioning_kind="${3:-none}"
  local conditioning_target="${4:-both}"
  local conditioning_coupling="${5:-shared_trunk_separate_readouts}"
  local additive_normalization="${6:-none}"
  local conditioning_json
  if [[ "${conditioning_kind}" == "none" ]]; then
    conditioning_json='{"kind": "none", "hidden_dim": 32}'
  else
    conditioning_json="{\"kind\": \"${conditioning_kind}\", \"source\": \"dedicated\", \"activation\": \"linear\", \"hidden_dim\": 32, \"target\": \"${conditioning_target}\", \"coupling\": \"${conditioning_coupling}\", \"phase_bound\": 0.25}"
  fi
  cat <<JSON
{"enabled": true, "application": "additive", "geometry": "${geometry}", "input": {"kind": "frozen_fourier", "basis_dim": ${basis_dim}, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": 0.3, "amplitude_parameterization": "signed", "learn_amplitude": true, "learn_phase": true, "phase_scale": 1.0, "additive_normalization": "${additive_normalization}", "additive_gain_init": 0.212132, "additive_gain_max": 1.0, "learn_additive_gain": true, "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": ${conditioning_json}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_inkling_json() {
  local kind="$1" # inkling_table | inkling_cosnet
  local profiles="${2:-8}"
  local router_hidden="${3:-64}"
  local head_coupling="${4:-per_head_independent}"
  cat <<JSON
{"enabled": true, "application": "logit_bias", "geometry": "scalar_curve", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "identity", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "conditioning": {"kind": "${kind}", "num_profiles": ${profiles}, "router_hidden_dim": ${router_hidden}, "profile_init_std": 0.02, "num_frequencies": 16, "gate_init": 0.0}, "head_coupling": "${head_coupling}"}
JSON
}

v2_pairwise_logit_json() {
  local position_mode="$1" # relative_only | query_absolute | full_absolute
  local pair_rank="${2:-8}"
  local head_coupling="${3:-per_head_independent}"
  local content_source="${4:-qk}"
  cat <<JSON
{"enabled": true, "application": "logit_bias", "geometry": "scalar_curve", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "conditioning": {"kind": "pairwise_low_rank", "source": "${content_source}", "pair_rank": ${pair_rank}, "position_mode": "${position_mode}", "gate_init": 0.0}, "head_coupling": "${head_coupling}"}
JSON
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

want_phase2_coupling_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "phase2_coupling" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase2_followup_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "phase2_followup" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_geometry_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_amplitude_followup_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_amplitude_followup" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_promotion_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_promotion" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_basis_screen_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_basis_screen" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_geometry_transfer_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_geometry_transfer" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_frontier_screen_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_frontier_screen" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_conditioning_retry_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_conditioning_retry" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_coupling_transfer_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_coupling_transfer" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_structural_followup_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_structural_followup" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_pairwise_logit_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_pairwise_logit" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_addrope_components_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_addrope_components" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_residual_sector_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_residual_sector" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_final_decision_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_final_decision" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase3_compact_basis_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase3_compact_basis" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase4_extrapolation_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase4_extrapolation" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase4_post_qk_norm_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase4_post_qk_norm" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase4_safe_conditioning_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase4_safe_conditioning" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase4_additive_geometry_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase4_additive_geometry" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase4_phase_conditioning_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase4_phase_conditioning" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase5_null_conditioning_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase5_null_conditioning" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase6_geometry_norm_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase6_geometry_norm" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase6_content_transfer_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase6_content_transfer" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase6_efficiency_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase6_efficiency" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase6_scale_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase6_scale" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase7_scale_probe_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase7_scale_probe" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase7_scale_50k_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase7_scale_50k" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase8_hyper_smoke_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase8_hyper_smoke" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase8_hyper_5k_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase8_hyper_5k" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase8_addrope_clean_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase8_addrope_clean" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase9_unit_hyper_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase9_unit_hyper" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase9_carrier_followup_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase9_carrier_followup" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase9_hyper_30k_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase9_hyper_30k" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase9_qk_independence_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase9_qk_independence" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase9_hyper_capacity_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase9_hyper_capacity" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase10_hyper_geometry_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase10_hyper_geometry" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase16_cheap_mixing_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase15_decay_mixing_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase14_angular_rank_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase14_angular_rank" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase13_decomp_qknorm_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase12_offset_qknorm_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase12_offset_qknorm" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase11_spectral_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

# Compact v2 Q/K channel JSON. head_coupling defaults to per_head_independent.
v2_qk_json() {
  local application="$1"   # additive | rotary
  local geometry="$2"      # free | phase
  local mapper_kind="$3"
  local residual="$4"      # true | false
  local qk_coupling="$5"
  local head_coupling="${6:-per_head_independent}"
  cat <<JSON
{"enabled": true, "application": "${application}", "geometry": "${geometry}", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "${mapper_kind}", "residual": ${residual}, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "qk_coupling": "${qk_coupling}", "head_coupling": "${head_coupling}"}
JSON
}

v2_logit_json() {
  local mapper_kind="$1"
  local residual="$2"
  local head_coupling="${3:-per_head_independent}"
  cat <<JSON
{"enabled": true, "application": "logit_bias", "geometry": "scalar_curve", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "${mapper_kind}", "residual": ${residual}, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "head_coupling": "${head_coupling}"}
JSON
}

coupling_tag() {
  case "$1" in
    shared) echo "shared" ;;
    shared_trunk_separate_readouts) echo "sep_readout" ;;
    separate) echo "separate" ;;
    *) echo "$1" ;;
  esac
}

echo "MLPROPE_QUEUE_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "family=${EXPERIMENT_FAMILY} model='${MODEL_CONFIG}' submit=${SUBMIT_JOBS} dry_run=${DRY_RUN} parallel=${PARALLEL}"
echo "gpu-claim=${GPU_CLAIM_BIN} selector=${GPU_SELECTOR} config_dir=${CONFIG_DIR}"
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

# Direction 1c: additive Fourier Q/K (replace multiplicative RoPE with q + f(cis)).
# This is not canonical amplitude+phase AddRoPE; see POSITION_CONFIG.md.
# Feature-map taxonomy still applies. RoPE baseline from Phase 1 remains the anchor.
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

# Phase 2: Q/K coupling ablation on the strongest Phase-1c additive mapper
# (linear) and the Phase-1b phase mapper (mlp). Shared reproduces the prior
# implicit coupling; sep_readout / separate are the new axes.
if want_phase2_coupling_family "qk_coupling"; then
  for coupling in shared shared_trunk_separate_readouts separate; do
    tag="$(coupling_tag "${coupling}")"
    emit_v2_channel_variant \
      "qk-add-linear-${tag}-per_head-h${HIDDEN_SIZE}d${DEPTH}" \
      "sdpa" \
      "$(v2_qk_json additive free linear false "${coupling}")" \
      "${LOGIT_DISABLED}"
    emit_v2_channel_variant \
      "qk-phase-mlp-m${POS_MLP_HIDDEN}-${tag}-per_head-h${HIDDEN_SIZE}d${DEPTH}" \
      "sdpa" \
      "$(v2_qk_json rotary phase mlp true "${coupling}")" \
      "${LOGIT_DISABLED}"
  done
fi

# Phase 2 follow-up:
# 1) combine best Q/K (add-linear sep_readout) with best logit (linear)
# 2) widen sep_readout across additive mappers (mlp, low_rank)
if want_phase2_followup_family "combined_and_widen"; then
  emit_v2_channel_variant \
    "qk-add-linear-sep_readout+logit-linear-per_head-h${HIDDEN_SIZE}d${DEPTH}" \
    "flex" \
    "$(v2_qk_json additive free linear false shared_trunk_separate_readouts)" \
    "$(v2_logit_json linear false)"
  emit_v2_channel_variant \
    "qk-add-mlp-m${POS_MLP_HIDDEN}-sep_readout-per_head-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "$(v2_qk_json additive free mlp true shared_trunk_separate_readouts)" \
    "${LOGIT_DISABLED}"
  emit_v2_channel_variant \
    "qk-add-low_rank-r${POS_RANK}-sep_readout-per_head-h${HIDDEN_SIZE}d${DEPTH}" \
    "sdpa" \
    "$(v2_qk_json additive free low_rank true shared_trunk_separate_readouts)" \
    "${LOGIT_DISABLED}"
fi

# Phase 3 screening bundle: one coherent geometry/injection-scale story.
# This family is intentionally excluded from EXPERIMENT_FAMILY=all so historical
# sweeps cannot launch it accidentally.
if want_phase3_geometry_family "canonical_geometry_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  free_add_qk="$(v2_qk_json additive free linear false shared_trunk_separate_readouts)"
  linear_logit="$(v2_logit_json linear false)"

  emit_v2_playground_variant \
    "phase3-rope-${phase3_suffix}" \
    "sdpa" \
    "${QK_DISABLED}" \
    "${LOGIT_DISABLED}"
  emit_v2_playground_variant \
    "phase3-add-linear-sep-${phase3_suffix}" \
    "sdpa" \
    "${free_add_qk}" \
    "${LOGIT_DISABLED}"
  emit_v2_playground_variant \
    "phase3-add-linear-sep+logit-linear-${phase3_suffix}" \
    "flex" \
    "${free_add_qk}" \
    "${linear_logit}"

  for amplitude_spec in \
    "a001:0.01" \
    "a003:0.03" \
    "a010:0.1" \
    "a030:0.3"; do
    amplitude_tag="${amplitude_spec%%:*}"
    amplitude_value="${amplitude_spec#*:}"
    canonical_qk="$(
      v2_qk_playground_json \
        additive amplitude_phase linear false \
        shared_trunk_separate_readouts frozen_fourier none \
        per_head_independent "${amplitude_value}"
    )"
    emit_v2_playground_variant \
      "phase3-canonical-${amplitude_tag}-sep-${phase3_suffix}" \
      "sdpa" \
      "${canonical_qk}" \
      "${LOGIT_DISABLED}"
  done

  canonical_a010_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.1
  )"
  emit_v2_playground_variant \
    "phase3-canonical-a010-sep+logit-linear-${phase3_suffix}" \
    "flex" \
    "${canonical_a010_qk}" \
    "${linear_logit}"
fi

# Phase 3 amplitude follow-up: continue the monotonic Q/K-only amplitude trend
# and bracket the current amplitude-0.1 winner when linear logit bias is active.
if want_phase3_amplitude_followup_family "canonical_amplitude_followup"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  linear_logit="$(v2_logit_json linear false)"
  amplitude_followup_scope="${AMPLITUDE_FOLLOWUP_SCOPE:-all}"
  if [[ "${amplitude_followup_scope}" != "all" \
    && "${amplitude_followup_scope}" != "qk" \
    && "${amplitude_followup_scope}" != "logit" ]]; then
    echo "AMPLITUDE_FOLLOWUP_SCOPE must be all, qk, or logit" >&2
    exit 1
  fi

  if [[ "${amplitude_followup_scope}" != "logit" ]]; then
    for amplitude_spec in \
      "a050:0.5" \
      "a070:0.7" \
      "a100:1.0" \
      "a200:2.0"; do
      amplitude_tag="${amplitude_spec%%:*}"
      amplitude_value="${amplitude_spec#*:}"
      canonical_qk="$(
        v2_qk_playground_json \
          additive amplitude_phase linear false \
          shared_trunk_separate_readouts frozen_fourier none \
          per_head_independent "${amplitude_value}"
      )"
      emit_v2_playground_variant \
        "phase3-canonical-${amplitude_tag}-sep-${phase3_suffix}" \
        "sdpa" \
        "${canonical_qk}" \
        "${LOGIT_DISABLED}"
    done
  fi

  if [[ "${amplitude_followup_scope}" != "qk" ]]; then
    for amplitude_spec in \
      "a003:0.03" \
      "a030:0.3" \
      "a100:1.0" \
      "a200:2.0"; do
      amplitude_tag="${amplitude_spec%%:*}"
      amplitude_value="${amplitude_spec#*:}"
      canonical_qk="$(
        v2_qk_playground_json \
          additive amplitude_phase linear false \
          shared_trunk_separate_readouts frozen_fourier none \
          per_head_independent "${amplitude_value}"
      )"
      emit_v2_playground_variant \
        "phase3-canonical-${amplitude_tag}-sep+logit-linear-${phase3_suffix}" \
        "flex" \
        "${canonical_qk}" \
        "${linear_logit}"
    done
  fi
fi

# Phase 3 promotion: two seeds at 10k for the baseline, historical free-additive
# control, and the two leading canonical amplitudes. FlexAttention is used for
# every run so attention implementation is controlled across the comparison.
if want_phase3_promotion_family "canonical_10k_promotion"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  free_add_qk="$(v2_qk_json additive free linear false shared_trunk_separate_readouts)"
  linear_logit="$(v2_logit_json linear false)"
  canonical_a030_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3
  )"
  canonical_a100_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 1.0
  )"

  for seed in 123 456; do
    emit_v2_playground_seed_variant \
      "phase3-promotion-rope-flex-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${QK_DISABLED}" \
      "${LOGIT_DISABLED}"
    emit_v2_playground_seed_variant \
      "phase3-promotion-add-linear-sep+logit-linear-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${free_add_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-promotion-canonical-a030-sep+logit-linear-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${canonical_a030_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-promotion-canonical-a100-sep+logit-linear-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${canonical_a100_qk}" \
      "${linear_logit}"
  done
fi

# Phase 3 basis screen: hold canonical geometry and logit bias fixed while
# changing only how absolute position is represented at the Q/K input.
if want_phase3_basis_screen_family "canonical_basis_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  linear_logit="$(v2_logit_json linear false)"
  frozen_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3
  )"
  learned_temperature_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts learned_temperature_fourier none \
      per_head_independent 0.3
  )"
  learned_frequency_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts learned_frequency_fourier none \
      per_head_independent 0.3
  )"
  scalar_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      '["normalized_position", "log_position"]'
  )"

  for seed in 123 456; do
    emit_v2_playground_seed_variant \
      "phase3-basis-frozen-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${frozen_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-basis-learned-temperature-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${learned_temperature_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-basis-learned-frequency-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${learned_frequency_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-basis-scalars-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${scalar_qk}" \
      "${linear_logit}"
  done
fi

# Phase 3 geometry transfer: carry the winning scalar features while changing
# only the Q/K positional geometry.
if want_phase3_geometry_transfer_family "scalar_geometry_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  linear_logit="$(v2_logit_json linear false)"
  scalar_features='["normalized_position", "log_position"]'
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  phase_residual_qk="$(
    v2_qk_playground_json \
      rotary phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  projected_phase_qk="$(
    v2_qk_playground_json \
      rotary projected_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  scaled_phase_qk="$(
    v2_qk_playground_json \
      rotary scaled_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"

  for seed in 123 456; do
    emit_v2_playground_seed_variant \
      "phase3-geometry-canonical-scalars-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${canonical_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-geometry-phase-residual-scalars-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${phase_residual_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-geometry-projected-phase-scalars-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${projected_phase_qk}" \
      "${linear_logit}"
    emit_v2_playground_seed_variant \
      "phase3-geometry-scaled-phase-scalars-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${scaled_phase_qk}" \
      "${linear_logit}"
  done
fi

# Phase 3 one-seed frontier screen: keep canonical Q/K geometry and scalar
# inputs fixed while scouting the remaining conditioning and write mechanisms.
if want_phase3_frontier_screen_family "canonical_frontier_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  seed=123
  scalar_features='["normalized_position", "log_position"]'
  residual_disabled='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  local_residual_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier local_residual \
      per_head_independent 0.3 \
      "${scalar_features}" 32
  )"
  content_gate_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier content_gate \
      per_head_independent 0.3 \
      "${scalar_features}" 32
  )"
  residual_per_layer='{"enabled": true, "placement": "per_layer", "source": "position_basis", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "identity", "residual": false, "rank": 32, "hidden_dim": 128}, "gate_init": 0.0, "layer_shared": false}'
  key_position_write='{"enabled": true, "mode": "key_position", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "identity", "residual": false, "rank": 32, "hidden_dim": 128}, "head_coupling": "per_head_independent", "gate_init": 0.0}'
  relative_offset_write='{"enabled": true, "mode": "relative_offset", "input": {"kind": "frozen_fourier", "basis_dim": null, "theta": null, "scalars": []}, "mapper": {"kind": "identity", "residual": false, "rank": 32, "hidden_dim": 128}, "head_coupling": "per_head_independent", "gate_init": 0.0}'

  # The exact anchor already completed in the geometry-transfer screen at this
  # seed and horizon, so do not spend another queued GPU run reproducing it.
  emit_v2_playground_seed_variant \
    "phase3-frontier-qk-local-residual-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${local_residual_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-frontier-qk-content-gate-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${content_gate_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-frontier-inkling-table-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${canonical_qk}" "$(v2_inkling_json inkling_table)"
  emit_v2_playground_seed_variant \
    "phase3-frontier-inkling-cosnet-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${canonical_qk}" "$(v2_inkling_json inkling_cosnet)"
  emit_v2_playground_seed_variant \
    "phase3-frontier-residual-per-layer-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${canonical_qk}" "${linear_logit}" \
    "${residual_per_layer}"
  emit_v2_playground_seed_variant \
    "phase3-frontier-write-key-position-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${canonical_qk}" "${linear_logit}" \
    "${residual_disabled}" "${key_position_write}"
  emit_v2_playground_seed_variant \
    "phase3-frontier-write-relative-offset-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${canonical_qk}" "${linear_logit}" \
    "${residual_disabled}" "${relative_offset_write}"
fi

# Retry the two unstable Q/K conditioning variants after bounding their local
# latent corrections. Everything else matches the completed frontier screen.
if want_phase3_conditioning_retry_family "bounded_conditioning_retry"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  seed=123
  scalar_features='["normalized_position", "log_position"]'
  linear_logit="$(v2_logit_json linear false)"
  local_residual_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier local_residual \
      per_head_independent 0.3 \
      "${scalar_features}" 32
  )"
  content_gate_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier content_gate \
      per_head_independent 0.3 \
      "${scalar_features}" 32
  )"

  emit_v2_playground_seed_variant \
    "phase3-conditioning-bounded-local-residual-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${local_residual_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-conditioning-bounded-content-gate-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${content_gate_qk}" "${linear_logit}"
fi

# Transfer the winning canonical scalar configuration across a small structural
# sharing screen. The existing per-head separate-readout run is the anchor.
if want_phase3_coupling_transfer_family "canonical_coupling_transfer"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  seed=123
  scalar_features='["normalized_position", "log_position"]'
  linear_logit="$(v2_logit_json linear false)"
  shared_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  separate_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      separate frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  shared_head_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      shared_head 0.3 \
      "${scalar_features}"
  )"
  joint_head_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_joint 0.3 \
      "${scalar_features}"
  )"

  emit_v2_playground_seed_variant \
    "phase3-coupling-shared-qk-per-head-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${shared_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-coupling-separate-qk-per-head-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${separate_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-coupling-separate-readouts-shared-head-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${shared_head_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-coupling-separate-readouts-joint-head-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${joint_head_qk}" "${linear_logit}"
fi

# Focused follow-up: isolate joint-head structure from basis width and generic
# capacity, confirm the efficient shared-Q/K result, and decompose the scalar
# and mapper choices without a Cartesian sweep.
if want_phase3_structural_followup_family "canonical_structural_followup"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  linear_logit="$(v2_logit_json linear false)"
  both_scalars='["normalized_position", "log_position"]'
  normalized_scalar='["normalized_position"]'
  log_scalar='["log_position"]'
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${both_scalars}"
  )"
  matched_joint_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_joint 0.3 \
      "${both_scalars}" "${POS_MLP_HIDDEN}" 96
  )"
  wide_joint_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_joint 0.3 \
      "${both_scalars}"
  )"
  shared_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared frozen_fourier none \
      per_head_independent 0.3 \
      "${both_scalars}"
  )"
  normalized_only_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${normalized_scalar}"
  )"
  log_only_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${log_scalar}"
  )"
  low_rank_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase low_rank false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${both_scalars}"
  )"
  mlp_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase mlp false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${both_scalars}"
  )"

  emit_v2_playground_seed_variant \
    "phase3-followup-joint-head-basis96-seed123-${phase3_suffix}" \
    123 "flex" "${matched_joint_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-followup-joint-head-wide-seed456-${phase3_suffix}" \
    456 "flex" "${wide_joint_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-followup-shared-qk-seed456-${phase3_suffix}" \
    456 "flex" "${shared_qk}" "${linear_logit}"

  capacity_job="phase3-followup-anchor-ffn3328-seed123-${phase3_suffix}"
  capacity_cfg="${CONFIG_DIR}/${capacity_job}.json"
  write_common_config "${capacity_cfg}" \
    "\"seed\": 123, \"ff_hidden_dim\": 3328, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${canonical_qk}, \"logit_bias\": ${linear_logit}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${capacity_job}\""
  run_job "${capacity_job}" "${capacity_cfg}"

  emit_v2_playground_seed_variant \
    "phase3-followup-normalized-scalar-only-seed123-${phase3_suffix}" \
    123 "flex" "${normalized_only_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-followup-log-scalar-only-seed123-${phase3_suffix}" \
    123 "flex" "${log_only_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-followup-low-rank-mapper-seed123-${phase3_suffix}" \
    123 "flex" "${low_rank_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-followup-mlp-mapper-seed123-${phase3_suffix}" \
    123 "flex" "${mlp_qk}" "${linear_logit}"
fi

# Factorized pair-aware relative-logit screen. The completed canonical scalar
# Q/K + static linear-logit seed-123 run is the exact zero-gate anchor (4.3980).
# These variants test whether content interaction benefits from retaining
# translation invariance or receiving explicit absolute Fourier features.
if want_phase3_pairwise_logit_family "pairwise_logit_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  seed=123
  pair_rank=8
  scalar_features='["normalized_position", "log_position"]'
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"

  for position_mode in relative_only query_absolute full_absolute; do
    emit_v2_playground_seed_variant \
      "phase3-pairwise-logit-r${pair_rank}-${position_mode}-seed${seed}-${phase3_suffix}" \
      "${seed}" \
      "flex" \
      "${canonical_qk}" \
      "$(v2_pairwise_logit_json "${position_mode}" "${pair_rank}")"
  done
fi

# Canonical AddRoPE 2x2 component isolation. The learned-amplitude +
# learned-phase cell already exists as the scalar geometry-transfer anchor.
if want_phase3_addrope_components_family "addrope_component_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  seed=123
  scalar_features='["normalized_position", "log_position"]'
  linear_logit="$(v2_logit_json linear false)"
  fixed_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}" "${POS_MLP_HIDDEN}" null false false
  )"
  amplitude_only_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}" "${POS_MLP_HIDDEN}" null true false
  )"
  phase_only_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}" "${POS_MLP_HIDDEN}" null false true
  )"

  emit_v2_playground_seed_variant \
    "phase3-addrope-fixed-carrier-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${fixed_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-addrope-amplitude-only-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${amplitude_only_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase3-addrope-phase-only-seed${seed}-${phase3_suffix}" \
    "${seed}" "flex" "${phase_only_qk}" "${linear_logit}"
fi

# Residual-stream positional sector with Q/K and logit channels disabled.
# RoPE is retained only for its explicit control; every residual variant uses
# use_rope=false so the residual stream is the sole explicit position source.
if want_phase3_residual_sector_family "residual_stream_story"; then
  steps_tag="s${MAX_TRAIN_STEPS}"
  phase3_suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  seed=123
  residual_basis_dim=$((HIDDEN_SIZE / N_HEAD))
  residual_disabled='{"enabled": false}'
  write_disabled='{"enabled": false}'
  sinusoidal_residual="{\"enabled\": true, \"placement\": \"input\", \"source\": \"position_basis\", \"input\": {\"kind\": \"frozen_fourier\", \"basis_dim\": ${HIDDEN_SIZE}, \"theta\": null, \"scalars\": []}, \"mapper\": {\"kind\": \"identity\", \"residual\": false, \"rank\": ${POS_RANK}, \"hidden_dim\": ${POS_MLP_HIDDEN}}, \"gate_init\": 1.0, \"layer_shared\": false}"
  learned_absolute_residual="{\"enabled\": true, \"placement\": \"input\", \"source\": \"learned_absolute\", \"input\": {\"kind\": \"frozen_fourier\", \"basis_dim\": ${HIDDEN_SIZE}, \"theta\": null, \"scalars\": []}, \"mapper\": {\"kind\": \"identity\", \"residual\": false, \"rank\": ${POS_RANK}, \"hidden_dim\": ${POS_MLP_HIDDEN}}, \"gate_init\": 1.0, \"layer_shared\": false}"
  linear_fourier_residual="{\"enabled\": true, \"placement\": \"input\", \"source\": \"position_basis\", \"input\": {\"kind\": \"frozen_fourier\", \"basis_dim\": ${residual_basis_dim}, \"theta\": null, \"scalars\": []}, \"mapper\": {\"kind\": \"linear\", \"residual\": false, \"rank\": ${POS_RANK}, \"hidden_dim\": ${POS_MLP_HIDDEN}}, \"gate_init\": 1.0, \"layer_shared\": false}"
  mlp_fourier_residual="{\"enabled\": true, \"placement\": \"input\", \"source\": \"position_basis\", \"input\": {\"kind\": \"frozen_fourier\", \"basis_dim\": ${residual_basis_dim}, \"theta\": null, \"scalars\": []}, \"mapper\": {\"kind\": \"mlp\", \"residual\": false, \"rank\": ${POS_RANK}, \"hidden_dim\": ${POS_MLP_HIDDEN}}, \"gate_init\": 1.0, \"layer_shared\": false}"
  per_layer_residual="{\"enabled\": true, \"placement\": \"per_layer\", \"source\": \"position_basis\", \"input\": {\"kind\": \"frozen_fourier\", \"basis_dim\": ${residual_basis_dim}, \"theta\": null, \"scalars\": []}, \"mapper\": {\"kind\": \"linear\", \"residual\": false, \"rank\": ${POS_RANK}, \"hidden_dim\": ${POS_MLP_HIDDEN}}, \"gate_init\": 0.0, \"layer_shared\": false}"

  emit_v2_playground_seed_variant \
    "phase3-residual-rope-control-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${residual_disabled}" "${write_disabled}" true
  emit_v2_playground_seed_variant \
    "phase3-residual-no-explicit-pe-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${residual_disabled}" "${write_disabled}" false
  emit_v2_playground_seed_variant \
    "phase3-residual-sinusoidal-input-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${sinusoidal_residual}" "${write_disabled}" false
  emit_v2_playground_seed_variant \
    "phase3-residual-learned-absolute-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${learned_absolute_residual}" "${write_disabled}" false
  emit_v2_playground_seed_variant \
    "phase3-residual-linear-fourier-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${linear_fourier_residual}" "${write_disabled}" false
  emit_v2_playground_seed_variant \
    "phase3-residual-mlp-fourier-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${mlp_fourier_residual}" "${write_disabled}" false
  emit_v2_playground_seed_variant \
    "phase3-residual-per-layer-reinjection-seed${seed}-${phase3_suffix}" \
    "${seed}" "sdpa" "${QK_DISABLED}" "${LOGIT_DISABLED}" \
    "${per_layer_residual}" "${write_disabled}" false
fi

# Final decision bundle: promote the independently validated scalar augmentation
# onto the 10k canonical stack, and confirm the only promising pairwise-logit
# variant on the held-out seed at its original 5k horizon.
if want_phase3_final_decision_family "final_decision_story"; then
  scalar_features='["normalized_position", "log_position"]'
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  linear_logit="$(v2_logit_json linear false)"

  for seed in 123 456; do
    job_name="phase3-final-canonical-scalars-seed${seed}-s10000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${canonical_qk}, \"logit_bias\": ${linear_logit}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      10000 1000
    run_job "${job_name}" "${cfg_file}"
  done

  seed=456
  job_name="phase3-final-pairwise-query-absolute-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${canonical_qk}, \"logit_bias\": $(v2_pairwise_logit_json query_absolute 8), \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
    5000 500
  run_job "${job_name}" "${cfg_file}"
fi

# Efficiency-only width reduction against the completed native head-width
# (basis_dim=96) scalar canonical seed-123 anchor at 4.3980.
if want_phase3_compact_basis_family "compact_basis_story"; then
  seed=123
  scalar_features='["normalized_position", "log_position"]'
  linear_logit="$(v2_logit_json linear false)"
  for basis_dim in 16 32 64; do
    compact_qk="$(
      v2_qk_playground_json \
        additive amplitude_phase linear false \
        shared_trunk_separate_readouts frozen_fourier none \
        per_head_independent 0.3 \
        "${scalar_features}" "${POS_MLP_HIDDEN}" "${basis_dim}"
    )"
    emit_v2_playground_seed_variant \
      "phase3-compact-basis${basis_dim}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}" \
      "${seed}" "flex" "${compact_qk}" "${linear_logit}"
  done
fi

# Frozen-finalist length extrapolation. Every model trains on 1024-token rows
# with position capacity allocated to 2048, then evaluates the same validation
# stream at 1024, 1536, and 2048.
if want_phase4_extrapolation_family "extrapolation_story"; then
  seed=123
  scalar_features='["normalized_position", "log_position"]'
  free_qk="$(
    v2_qk_playground_json \
      additive free linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.1 '[]'
  )"
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 \
      "${scalar_features}"
  )"
  linear_logit="$(v2_logit_json linear false)"
  length_json="\"training_length\": 1024, \"model_position_extent\": 2048, \"evaluation_lengths\": [1024, 1536, 2048], \"scalar_normalization_extent\": 1024"

  job_name="phase4-extrapolation-rope-seed${seed}-s10000-train1024-extent2048-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, ${length_json}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${QK_DISABLED}, \"logit_bias\": ${LOGIT_DISABLED}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
    10000 1000
  run_job "${job_name}" "${cfg_file}"

  job_name="phase4-extrapolation-free-additive-seed${seed}-s10000-train1024-extent2048-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, ${length_json}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${free_qk}, \"logit_bias\": ${linear_logit}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
    10000 1000
  run_job "${job_name}" "${cfg_file}"

  job_name="phase4-extrapolation-canonical-scalars-seed${seed}-s10000-train1024-norm1024-extent2048-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, ${length_json}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${canonical_qk}, \"logit_bias\": ${linear_logit}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
    10000 1000
  run_job "${job_name}" "${cfg_file}"
fi

# Post-position RMS normalization screen. Existing seed-123 5k anchors are reused
# for canonical AddRoPE, scaled rotary, and the two bounded conditioners. Free
# additive receives both cells because no exact scalar-augmented 5k anchor exists.
if want_phase4_post_qk_norm_family "post_qk_norm_story"; then
  seed=123
  steps_tag="s${MAX_TRAIN_STEPS}"
  suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  scalar_features='["normalized_position", "log_position"]'
  disabled_channel='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  canonical_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 "${scalar_features}"
  )"
  free_qk="$(
    v2_qk_playground_json \
      additive free linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 "${scalar_features}"
  )"
  local_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier local_residual \
      per_head_independent 0.3 "${scalar_features}" 32
  )"
  gated_qk="$(
    v2_qk_playground_json \
      additive amplitude_phase linear false \
      shared_trunk_separate_readouts frozen_fourier content_gate \
      per_head_independent 0.3 "${scalar_features}" 32
  )"
  scaled_qk="$(
    v2_qk_playground_json \
      rotary scaled_phase linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 "${scalar_features}"
  )"

  emit_v2_playground_seed_variant \
    "phase4-postnorm-canonical-seed${seed}-${suffix}" \
    "${seed}" flex "${canonical_qk}" "${linear_logit}" \
    "${disabled_channel}" "${disabled_channel}" true true
  emit_v2_playground_seed_variant \
    "phase4-postnorm-free-control-seed${seed}-${suffix}" \
    "${seed}" flex "${free_qk}" "${linear_logit}"
  emit_v2_playground_seed_variant \
    "phase4-postnorm-free-seed${seed}-${suffix}" \
    "${seed}" flex "${free_qk}" "${linear_logit}" \
    "${disabled_channel}" "${disabled_channel}" true true
  emit_v2_playground_seed_variant \
    "phase4-postnorm-local-residual-seed${seed}-${suffix}" \
    "${seed}" flex "${local_qk}" "${linear_logit}" \
    "${disabled_channel}" "${disabled_channel}" true true
  emit_v2_playground_seed_variant \
    "phase4-postnorm-content-gate-seed${seed}-${suffix}" \
    "${seed}" flex "${gated_qk}" "${linear_logit}" \
    "${disabled_channel}" "${disabled_channel}" true true
  emit_v2_playground_seed_variant \
    "phase4-postnorm-scaled-rotary-seed${seed}-${suffix}" \
    "${seed}" flex "${scaled_qk}" "${linear_logit}" \
    "${disabled_channel}" "${disabled_channel}" true true
fi

# Safe contribution and content-source redesign. All additive variants normalize
# the completed position branch, mix it through a bounded gain, and normalize Q/K
# again after injection. Conditioning source and activation are isolated.
if want_phase4_safe_conditioning_family "safe_conditioning_story"; then
  seed=123
  steps_tag="s${MAX_TRAIN_STEPS}"
  suffix="${steps_tag}-h${HIDDEN_SIZE}d${DEPTH}"
  disabled_channel='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  safe_control="$(v2_safe_qk_json additive amplitude_phase none residual linear 0.0)"
  residual_sigmoid_gate="$(
    v2_safe_qk_json additive amplitude_phase content_gate residual scaled_sigmoid 1.0
  )"
  qk_sigmoid_gate="$(
    v2_safe_qk_json additive amplitude_phase content_gate qk scaled_sigmoid 1.0
  )"
  residual_tanh_gate="$(
    v2_safe_qk_json additive amplitude_phase content_gate residual tanh 0.0
  )"
  residual_gelu_local="$(
    v2_safe_qk_json additive amplitude_phase local_residual residual gelu 0.0
  )"
  residual_tanh_local="$(
    v2_safe_qk_json additive amplitude_phase local_residual residual tanh 0.0
  )"
  bounded_scaled_rotary="$(
    v2_safe_qk_json rotary scaled_phase none qk linear 0.0
  )"
  unit_pair_rotary="$(
    v2_safe_qk_json rotary unit_pair none qk linear 0.0
  )"

  for spec in \
    "safe-control|${safe_control}" \
    "residual-sigmoid-gate|${residual_sigmoid_gate}" \
    "qk-sigmoid-gate|${qk_sigmoid_gate}" \
    "residual-tanh-gate|${residual_tanh_gate}" \
    "residual-gelu-local|${residual_gelu_local}" \
    "residual-tanh-local|${residual_tanh_local}" \
    "bounded-scaled-rotary|${bounded_scaled_rotary}" \
    "unit-pair-rotary|${unit_pair_rotary}"
  do
    label="${spec%%|*}"
    qk_json="${spec#*|}"
    emit_v2_playground_seed_variant \
      "phase4-safe-${label}-seed${seed}-${suffix}" \
      "${seed}" flex "${qk_json}" "${linear_logit}" \
      "${disabled_channel}" "${disabled_channel}" true true
  done
fi

# Close the two remaining additive-geometry cells against the completed
# phase3-promotion seed-123 anchors: free direct + linear logit (4.0615) and
# canonical AddRoPE + linear logit (4.0525/4.0515 at amplitudes 0.3/1.0).
# Every new cell uses the same basis, mapper parameters, Q/K coupling, head
# coupling, and static linear-logit channel as the free-direct anchor.
if want_phase4_additive_geometry_family "additive_geometry_story"; then
  seed=123
  linear_logit="$(v2_logit_json linear false)"
  residual_qk="$(
    v2_qk_playground_json \
      additive free linear true \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.1 '[]'
  )"
  pair03_qk="$(
    v2_qk_playground_json \
      additive pair_normalized linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 0.3 '[]'
  )"
  pair10_qk="$(
    v2_qk_playground_json \
      additive pair_normalized linear false \
      shared_trunk_separate_readouts frozen_fourier none \
      per_head_independent 1.0 '[]'
  )"

  for spec in \
    "free-residual|${residual_qk}" \
    "pair-normalized-a03|${pair03_qk}" \
    "pair-normalized-a10|${pair10_qk}"
  do
    label="${spec%%|*}"
    qk_json="${spec#*|}"
    job_name="phase4-geometry-${label}-seed${seed}-s10000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${linear_logit}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      10000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Content may only rotate the completed fixed-radius additive pair. The
# conditioner receives block-normalized residual content, has no position
# input, starts at exactly zero phase, and is bounded to +/-0.25 radians.
# The completed unconditioned radius-0.3 run (4.0524) is the exact anchor.
if want_phase4_phase_conditioning_family "phase_conditioning_story"; then
  seed=123
  linear_logit="$(v2_logit_json linear false)"
  for spec in \
    "q-only|q|shared_trunk_separate_readouts" \
    "k-only|k|shared_trunk_separate_readouts" \
    "both-shared|both|shared" \
    "both-separate-readouts|both|shared_trunk_separate_readouts"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    target="${remainder%%|*}"
    conditioner_coupling="${remainder#*|}"
    qk_json="$(
      v2_phase_conditioned_pair_json \
        "${target}" "${conditioner_coupling}" 0.25
    )"
    job_name="phase4-phase-conditioning-${label}-seed${seed}-s10000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${linear_logit}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": {\"enabled\": false}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      10000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Anchor-relative null conditioning with a dedicated 64-d residual-content
# stream and one geometry-aware Q/K RMSNorm. No mechanisms are combined.
if want_phase5_null_conditioning_family "null_conditioning_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  additive_anchor="$(
    v2_qk_playground_json \
      additive amplitude_phase identity false \
      shared frozen_fourier none \
      per_head_independent 0.3 '[]' "${POS_MLP_HIDDEN}" null false false
  )"
  pairwise_logit="$(
    v2_pairwise_logit_json query_absolute 8 per_head_independent dedicated
  )"

  # Fresh controls under the current code: conventional RoPE with the legacy
  # Q/K LayerNorm, then RoPE with the new pre-rotation RMSNorm. The latter
  # isolates the contribution of the linear logit bias in rope-rms-anchor.
  job_name="phase5-null-standard-rope-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, \"qk_norm_mode\": \"legacy_layernorm\", \"post_position_qk_norm\": false, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${disabled_channel}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
    5000 1000
  run_job "${job_name}" "${cfg_file}"

  job_name="phase5-null-rope-rms-no-logit-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${disabled_channel}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
    5000 1000
  run_job "${job_name}" "${cfg_file}"

  for spec in \
    "rope-rms-anchor|${disabled_channel}|${linear_logit}|{\"enabled\": false}" \
    "additive-rms-anchor|${additive_anchor}|${linear_logit}|{\"enabled\": false}" \
    "adaptive-gain-shared|$(v2_null_conditioned_qk_json adaptive_gain shared)|${linear_logit}|{\"enabled\": false}" \
    "adaptive-gain-separate|$(v2_null_conditioned_qk_json adaptive_gain shared_trunk_separate_readouts)|${linear_logit}|{\"enabled\": false}" \
    "additive-phase-shared|$(v2_null_conditioned_qk_json additive_phase shared)|${linear_logit}|{\"enabled\": false}" \
    "additive-phase-separate|$(v2_null_conditioned_qk_json additive_phase shared_trunk_separate_readouts)|${linear_logit}|{\"enabled\": false}" \
    "rope-phase-shared|$(v2_null_conditioned_qk_json rope_phase shared)|${linear_logit}|{\"enabled\": false}" \
    "rope-phase-separate|$(v2_null_conditioned_qk_json rope_phase shared_trunk_separate_readouts)|${linear_logit}|{\"enabled\": false}" \
    "pairwise-query-absolute|${disabled_channel}|${pairwise_logit}|{\"enabled\": false}" \
    "query-position-write|${disabled_channel}|${linear_logit}|{\"enabled\": true, \"mode\": \"query_position\"}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_json="${remainder%%|*}"
    remainder="${remainder#*|}"
    logit_json="${remainder%%|*}"
    write_json="${remainder#*|}"
    job_name="phase5-null-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"separate\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": {\"enabled\": false}, \"attention_write\": ${write_json}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Reconcile the promoted scalar+linear-logit stack across additive geometry and
# Q/K normalization before any model/training-scale promotion.
if want_phase6_geometry_norm_family "breadth_geometry_norm_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  canonical_qk="$(v2_promoted_qk_json amplitude_phase)"
  pair_qk="$(v2_promoted_qk_json pair_normalized)"
  canonical_branch_rms_qk="$(
    v2_promoted_qk_json amplitude_phase null none both \
      shared_trunk_separate_readouts rms
  )"

  for spec in \
    "canonical-legacy|legacy_layernorm|${canonical_qk}|null" \
    "pairnorm-legacy|legacy_layernorm|${pair_qk}|null" \
    "canonical-method-rms|method_aware_rms|${canonical_qk}|null" \
    "pairnorm-method-rms|method_aware_rms|${pair_qk}|null" \
    "canonical-method-rms-branch-rms|method_aware_rms|${canonical_branch_rms_qk}|null" \
    "rope-linear-logit-ffn3168|legacy_layernorm|${disabled_channel}|3168"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    remainder="${remainder#*|}"
    qk_json="${remainder%%|*}"
    ff_hidden_dim="${remainder#*|}"
    job_name="phase6-geometry-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"scalar_normalization_extent\": 1024, \"ff_hidden_dim\": ${ff_hidden_dim}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${linear_logit}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Transfer dedicated-content phase actuation onto the reconciled winner:
# canonical scalar AddRoPE + static linear logit + method-aware Q/K RMS.
if want_phase6_content_transfer_family "breadth_content_transfer_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  for spec in \
    "both-shared|both|shared" \
    "both-separate|both|shared_trunk_separate_readouts" \
    "q-only|q|shared" \
    "k-only|k|shared"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    target="${remainder%%|*}"
    coupling="${remainder#*|}"
    qk_json="$(
      v2_promoted_qk_json amplitude_phase null additive_phase \
        "${target}" "${coupling}"
    )"
    job_name="phase6-content-additive-phase-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"separate\", \"scalar_normalization_extent\": 1024, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${linear_logit}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Compact-basis and factorized-logit checks on the reconciled canonical stack.
if want_phase6_efficiency_family "breadth_efficiency_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  linear_logit="$(v2_logit_json linear false)"
  low_rank_logit="$(v2_logit_json low_rank true)"
  for spec in \
    "basis16|$(v2_promoted_qk_json amplitude_phase 16)|${linear_logit}" \
    "basis32|$(v2_promoted_qk_json amplitude_phase 32)|${linear_logit}" \
    "logit-low-rank-r32|$(v2_promoted_qk_json amplitude_phase)|${low_rank_logit}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_json="${remainder%%|*}"
    logit_json="${remainder#*|}"
    job_name="phase6-efficiency-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"scalar_normalization_extent\": 1024, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Final promotion gate: two positional stacks plus standard-RoPE and
# capacity-matched controls, all trained at the requested model dimensions.
if want_phase6_scale_family "breadth_scale_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  full_qk="$(v2_promoted_qk_json amplitude_phase)"
  linear_logit="$(v2_logit_json linear false)"
  low_rank_logit="$(v2_logit_json low_rank true)"
  matched_ff_hidden=null
  if [[ "${HIDDEN_SIZE}" == "768" && "${DEPTH}" == "8" ]]; then
    matched_ff_hidden=3168
  elif [[ "${HIDDEN_SIZE}" == "1024" && "${DEPTH}" == "12" ]]; then
    matched_ff_hidden=4160
  fi

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}|${disabled_channel}|null" \
    "rope-linear-logit-matched-ffn|method_aware_rms|${disabled_channel}|${linear_logit}|${matched_ff_hidden}" \
    "canonical-full-linear|method_aware_rms|${full_qk}|${linear_logit}|null" \
    "canonical-low-rank-logit|method_aware_rms|${full_qk}|${low_rank_logit}|null"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    remainder="${remainder#*|}"
    qk_json="${remainder%%|*}"
    remainder="${remainder#*|}"
    logit_json="${remainder%%|*}"
    ff_hidden_dim="${remainder#*|}"
    job_name="phase6-scale-${label}-seed${seed}-s10000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"training_length\": 1024, \"model_position_extent\": 4096, \"evaluation_lengths\": [1024, 2048, 4096], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"ff_hidden_dim\": ${ff_hidden_dim}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      10000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Short throughput/memory probes for an effective batch of 32. Failed cells are
# expected if a physical microbatch exceeds device memory.
if want_phase7_scale_probe_family "scale_50k_probe_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  full_qk="$(v2_promoted_qk_json amplitude_phase)"
  linear_logit="$(v2_logit_json linear false)"
  for spec in \
    "b32-a1-gc-default|32|1|true|default" \
    "b32-a1-gc-maxauto|32|1|true|max-autotune-no-cudagraphs" \
    "b16-a2-gc-default|16|2|true|default" \
    "b8-a4-nogc-default|8|4|false|default" \
    "b8-a4-nogc-maxauto|8|4|false|max-autotune-no-cudagraphs" \
    "b8-a4-nogc-reduce-overhead|8|4|false|reduce-overhead"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    train_batch="${remainder%%|*}"
    remainder="${remainder#*|}"
    accumulation="${remainder%%|*}"
    remainder="${remainder#*|}"
    gradient_checkpointing="${remainder%%|*}"
    compile_mode="${remainder#*|}"
    job_name="phase7-probe-${label}-seed${seed}-s200-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_train_batch_size\": ${train_batch}, \"per_device_eval_batch_size\": 1, \"gradient_accumulation_steps\": ${accumulation}, \"gradient_checkpointing\": ${gradient_checkpointing}, \"compile_mode\": \"${compile_mode}\", \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"checkpointing_steps\": null, \"save_final_model\": false, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${full_qk}, \"logit_bias\": ${linear_logit}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"flex\", \"run_name\": \"${job_name}\"" \
      200 100000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Resumable h1024/d12 promotion set. The probe-selected microbatch controls are
# explicit environment knobs so every arm retains the same effective batch.
if want_phase7_scale_50k_family "scale_50k_promotion_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  full_qk="$(v2_promoted_qk_json amplitude_phase)"
  compact_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  linear_logit="$(v2_logit_json linear false)"
  low_rank_logit="$(v2_logit_json low_rank true)"
  physical_batch="${PHYSICAL_BATCH:-8}"
  accumulation="${GRADIENT_ACCUMULATION_STEPS:-4}"
  gradient_checkpointing="${GRADIENT_CHECKPOINTING:-false}"
  compile_mode="${COMPILE_MODE:-default}"
  if (( physical_batch * accumulation != 32 )); then
    echo "phase7_scale_50k requires physical_batch * accumulation == 32" >&2
    exit 2
  fi

  for spec in \
    "canonical-full-linear|method_aware_rms|${full_qk}|${linear_logit}|null|flex" \
    "canonical-low-rank-r32|method_aware_rms|${full_qk}|${low_rank_logit}|null|flex" \
    "compact-basis16-full-linear|method_aware_rms|${compact_qk}|${linear_logit}|null|flex" \
    "standard-rope|legacy_layernorm|${disabled_channel}|${disabled_channel}|null|sdpa" \
    "rope-full-linear|method_aware_rms|${disabled_channel}|${linear_logit}|null|flex" \
    "rope-full-linear-matched-ffn4160|method_aware_rms|${disabled_channel}|${linear_logit}|4160|flex"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    remainder="${remainder#*|}"
    qk_json="${remainder%%|*}"
    remainder="${remainder#*|}"
    logit_json="${remainder%%|*}"
    remainder="${remainder#*|}"
    ff_hidden_dim="${remainder%%|*}"
    attn_impl="${remainder#*|}"
    job_name="phase7-scale50k-${label}-seed${seed}-s50000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_train_batch_size\": ${physical_batch}, \"per_device_eval_batch_size\": 1, \"gradient_accumulation_steps\": ${accumulation}, \"gradient_checkpointing\": ${gradient_checkpointing}, \"compile_mode\": \"${compile_mode}\", \"training_length\": 1024, \"model_position_extent\": 4096, \"evaluation_lengths\": [1024, 2048, 4096], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"ff_hidden_dim\": ${ff_hidden_dim}, \"checkpointing_steps\": 5000, \"resume_from_checkpoint\": \"auto\", \"save_final_model\": true, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\"" \
      50000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Short SDPA-only integration checks. These establish runtime/diagnostic health;
# 200 steps are not used to rank positional methods.
if want_phase8_hyper_smoke_family "carrier_hyper_smoke_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  anchor_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  additive_hyper_qk="$(
    v2_carrier_hyper_qk_json \
      additive content_position silu_mlp log_gain_phase \
      shared_trunk_separate_readouts both
  )"
  rotary_hyper_qk="$(
    v2_carrier_hyper_qk_json \
      rotary content linear log_gain_phase \
      shared_trunk_separate_readouts both
  )"

  for spec in \
    "compact-addrope-anchor|${anchor_qk}" \
    "addrope-content-position-silu|${additive_hyper_qk}" \
    "scaled-rope-content-linear|${rotary_hyper_qk}"
  do
    label="${spec%%|*}"
    qk_json="${spec#*|}"
    job_name="phase8-hyper-${label}-seed${seed}-s200-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"separate\", \"scalar_normalization_extent\": ${BLOCK_SIZE}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      200 200
    run_job "${job_name}" "${cfg_file}"
  done
fi

# One-seed, 5k SDPA attribution screen. This compares input interaction and
# nonlinear trunks on AddRoPE, then isolates phase-only versus radial+phase RoPE.
if want_phase8_hyper_5k_family "carrier_hyper_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  anchor_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  add_content_linear="$(
    v2_carrier_hyper_qk_json \
      additive content linear log_gain_phase \
      shared_trunk_separate_readouts both
  )"
  add_content_position_linear="$(
    v2_carrier_hyper_qk_json \
      additive content_position linear log_gain_phase \
      shared_trunk_separate_readouts both
  )"
  add_content_position_silu="$(
    v2_carrier_hyper_qk_json \
      additive content_position silu_mlp log_gain_phase \
      shared_trunk_separate_readouts both
  )"
  add_content_position_swiglu="$(
    v2_carrier_hyper_qk_json \
      additive content_position swiglu_mlp log_gain_phase \
      shared_trunk_separate_readouts both
  )"
  rope_content_phase="$(
    v2_carrier_hyper_qk_json \
      rotary content linear phase \
      shared_trunk_separate_readouts both
  )"
  rope_content_gain_phase="$(
    v2_carrier_hyper_qk_json \
      rotary content linear log_gain_phase \
      shared_trunk_separate_readouts both
  )"

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}" \
    "rope-rms-anchor|method_aware_rms|${disabled_channel}" \
    "compact-addrope-anchor|method_aware_rms|${anchor_qk}" \
    "addrope-content-linear|method_aware_rms|${add_content_linear}" \
    "addrope-content-position-linear|method_aware_rms|${add_content_position_linear}" \
    "addrope-content-position-silu|method_aware_rms|${add_content_position_silu}" \
    "addrope-content-position-swiglu|method_aware_rms|${add_content_position_swiglu}" \
    "rope-content-phase-linear|method_aware_rms|${rope_content_phase}" \
    "rope-content-gain-phase-linear|method_aware_rms|${rope_content_gain_phase}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase8-hyper5k-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"separate\", \"scalar_normalization_extent\": ${BLOCK_SIZE}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Cleanly separate canonical direct AddRoPE, the generalized mapped carrier,
# and a gauge-free dynamic replacement whose hypernetwork is the sole learner.
if want_phase8_addrope_clean_family "addrope_clean_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  mapped_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  direct_qk="$(v2_clean_addrope_qk_json direct)"
  fixed_qk="$(v2_clean_addrope_qk_json fixed)"
  dynamic_content_qk="$(v2_clean_addrope_qk_json dynamic content linear)"
  dynamic_content_position_qk="$(
    v2_clean_addrope_qk_json dynamic content_position linear
  )"

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}" \
    "mapped-addrope-anchor|method_aware_rms|${mapped_qk}" \
    "direct-addrope-softplus|method_aware_rms|${direct_qk}" \
    "fixed-addrope-softplus|method_aware_rms|${fixed_qk}" \
    "dynamic-addrope-content-linear|method_aware_rms|${dynamic_content_qk}" \
    "dynamic-addrope-content-position-linear|method_aware_rms|${dynamic_content_position_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase8-clean-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"separate\", \"scalar_normalization_extent\": ${BLOCK_SIZE}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Unit-anchor raw-polar screen. Input source and trunk nonlinearity vary while
# head/QK/content coupling, SDPA, and add-then-RMS normalization stay fixed.
if want_phase9_unit_hyper_family "unit_hyper_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  mapped_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  direct_unit_qk="$(v2_unit_hyper_qk_json direct)"
  position_linear_qk="$(v2_unit_hyper_qk_json hyper position linear)"
  content_linear_qk="$(v2_unit_hyper_qk_json hyper content linear)"
  both_linear_qk="$(v2_unit_hyper_qk_json hyper content_position linear)"
  position_silu_qk="$(v2_unit_hyper_qk_json hyper position silu_mlp)"
  content_silu_qk="$(v2_unit_hyper_qk_json hyper content silu_mlp)"
  both_silu_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp)"
  both_swiglu_qk="$(v2_unit_hyper_qk_json hyper content_position swiglu_mlp)"

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}" \
    "mapped-addrope-a03|method_aware_rms|${mapped_qk}" \
    "direct-unit-addrope|method_aware_rms|${direct_unit_qk}" \
    "unit-position-linear|method_aware_rms|${position_linear_qk}" \
    "unit-content-linear|method_aware_rms|${content_linear_qk}" \
    "unit-content-position-linear|method_aware_rms|${both_linear_qk}" \
    "unit-position-silu|method_aware_rms|${position_silu_qk}" \
    "unit-content-silu|method_aware_rms|${content_silu_qk}" \
    "unit-content-position-silu|method_aware_rms|${both_silu_qk}" \
    "unit-content-position-swiglu|method_aware_rms|${both_swiglu_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase9-unit-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"shared\", \"scalar_normalization_extent\": ${BLOCK_SIZE}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Follow up the unit-HyperAddRoPE win with branch asymmetry, dedicated-content
# coupling, and the norm-preserving phase-only rotary analogue.
if want_phase9_carrier_followup_family "carrier_followup_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  dynamic_both_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false
  )"
  dynamic_q_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp q true
  )"
  dynamic_k_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp k true
  )"
  phase_hyperrope_qk="$(v2_phase_hyperrope_qk_json)"

  for spec in \
    "addrope-dynamic-both-shared-content|shared|${dynamic_both_qk}" \
    "addrope-dynamic-both-separate-content|separate|${dynamic_both_qk}" \
    "addrope-dynamic-q-static-k|shared|${dynamic_q_qk}" \
    "addrope-static-q-dynamic-k|shared|${dynamic_k_qk}" \
    "hyperrope-phase-shared-content|shared|${phase_hyperrope_qk}" \
    "hyperrope-phase-separate-content|separate|${phase_hyperrope_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    content_coupling="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase9-followup-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"${content_coupling}\", \"scalar_normalization_extent\": ${BLOCK_SIZE}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Longer SDPA gate for the two tied HyperAddRoPE trunks and their established
# RoPE/mapped-AddRoPE controls. Train and evaluate at 1024 every 5k.
if want_phase9_hyper_30k_family "hyper_30k_promotion_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  mapped_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  hyper_silu_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false
  )"
  hyper_linear_qk="$(
    v2_unit_hyper_qk_json hyper content_position linear both false
  )"

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}" \
    "mapped-addrope-a03|method_aware_rms|${mapped_qk}" \
    "hyperaddrope-content-position-silu|method_aware_rms|${hyper_silu_qk}" \
    "hyperaddrope-content-position-linear|method_aware_rms|${hyper_linear_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase9-30k-${label}-seed${seed}-s30000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      30000 5000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Factor the Q/K independence hypothesis into content-projection sharing and
# hypernetwork-trunk sharing while retaining separate Q/K readouts throughout.
if want_phase9_qk_independence_family "qk_independence_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  shared_trunk_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts
  )"
  separate_trunks_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false separate
  )"

  for spec in \
    "shared-content-shared-trunk|shared|${shared_trunk_qk}" \
    "shared-content-separate-trunks|shared|${separate_trunks_qk}" \
    "separate-content-shared-trunk|separate|${shared_trunk_qk}" \
    "separate-content-separate-trunks|separate|${separate_trunks_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    content_coupling="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase9-qkgrid-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 64, \"position_content_coupling\": \"${content_coupling}\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Structural/capacity screen around the shared-content HyperAddRoPE winner.
# Each single-axis arm changes only Q/K readout sharing, head sharing, content
# rank, or SiLU trunk width; two combined arms test the larger-rank interaction.
if want_phase9_hyper_capacity_family "hyper_capacity_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  control_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64
  )"
  shared_qk_readout="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared per_head_independent 64
  )"
  shared_head_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts shared_head 64
  )"
  trunk128_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 128
  )"
  trunk256_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 256
  )"

  for spec in \
    "control-c64-h64|64|${control_qk}" \
    "shared-qk-readout-c64-h64|64|${shared_qk_readout}" \
    "shared-head-c64-h64|64|${shared_head_qk}" \
    "content128-h64|128|${control_qk}" \
    "content64-h128|64|${trunk128_qk}" \
    "content64-h256|64|${trunk256_qk}" \
    "content128-h128|128|${trunk128_qk}" \
    "content128-h256|128|${trunk256_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    content_dim="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase9-capacity-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": ${content_dim}, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Normalization/output-geometry micro-screen around the c128/h64
# HyperAddRoPE structural default. All hypernetwork readouts start at an exact
# unit RoPE carrier; each arm changes only the named input/output geometry axis.
if want_phase10_hyper_geometry_family "hyper_geometry_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  mapped_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  polar_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed amplitude_phase none false false
  )"
  modality_rms_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed amplitude_phase modality_rms false false
  )"
  modality_rms_gains_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed amplitude_phase modality_rms true false
  )"
  softplus_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      softplus amplitude_phase none false false
  )"
  cartesian_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed cartesian none false false
  )"
  full_frequency_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed amplitude_phase_frequency none false false
  )"
  static_amplitude_frequency_phase_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed frequency_phase none false true
  )"
  amplitude_only_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed amplitude none false false
  )"
  phase_only_qk="$(
    v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
      shared_trunk_separate_readouts per_head_independent 64 \
      signed phase none false false
  )"

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}" \
    "mapped-addrope-a03|method_aware_rms|${mapped_qk}" \
    "polar-signed-control|method_aware_rms|${polar_qk}" \
    "polar-modality-rms|method_aware_rms|${modality_rms_qk}" \
    "polar-modality-rms-gains|method_aware_rms|${modality_rms_gains_qk}" \
    "polar-softplus|method_aware_rms|${softplus_qk}" \
    "cartesian-residual|method_aware_rms|${cartesian_qk}" \
    "polar-amplitude-phase-frequency|method_aware_rms|${full_frequency_qk}" \
    "static-amplitude-frequency-phase|method_aware_rms|${static_amplitude_frequency_phase_qk}" \
    "polar-amplitude-only|method_aware_rms|${amplitude_only_qk}" \
    "polar-phase-only|method_aware_rms|${phase_only_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase10-geometry-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# Narrow spectral readouts around the c128/h64 HyperAddRoPE default. phase10
# showed the carrier gain lives in the amplitude branch (amplitude-only 4.3024
# vs full 4.2840, phase-only 4.4348) and that a content-conditioned frequency
# multiplier is destructive (4.5369), because content-dependent omega makes the
# logit depend on absolute position with error growing as m*p.
#
# Both new arms are translation-invariant by construction:
#   amplitude_slope  - 2 scalars/head tilt the amplitude envelope across
#                      log-frequency (a locality / decay-rate control);
#   position_offset  - 1 scalar/head sets phase = omega*m, which is exactly
#                      cis(omega*((p+m_q)-(p+m_k))), i.e. a content-dependent
#                      shift of effective position rather than a free phase.
# Free per-frequency controls are re-run in-family so the delta is measured
# against identical conditions rather than across families.
if want_phase11_spectral_family "spectral_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  hyper_args=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed)
  full_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_phase none false false)"
  amplitude_only_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude none false false)"
  amplitude_slope_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_slope none false false)"
  position_offset_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" position_offset none false false)"
  slope_offset_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_offset none false false)"

  for spec in \
    "standard-rope|legacy_layernorm|${disabled_channel}" \
    "free-amplitude-phase-control|method_aware_rms|${full_qk}" \
    "free-amplitude-only|method_aware_rms|${amplitude_only_qk}" \
    "amplitude-slope|method_aware_rms|${amplitude_slope_qk}" \
    "position-offset|method_aware_rms|${position_offset_qk}" \
    "slope-offset|method_aware_rms|${slope_offset_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase11-spectral-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# phase11 follow-up. Three axes, all against re-run phase11 controls:
#   (a) offset parameterization -- phase11 used bound*tanh(raw); tanh saturates
#       and compresses the integer-token scale the offset actually lives on, so
#       `raw` (unbounded signed) and `softplus(z + log(e-1)) - 1` (zero-anchored,
#       positive-leaning, lower-bounded at -1) are tested instead;
#   (b) compression decomposition -- phase11's slope+offset narrows the
#       amplitude readout (48->2) and the angular readout (48->1) at once, so
#       its 0.0112 gap cannot be attributed. One arm narrows each in isolation;
#   (c) per-head Q/K norm -- [heads, head_dim] gains instead of one shared
#       [head_dim], on plain RoPE and on the free carrier control.
if want_phase12_offset_qknorm_family "offset_qknorm_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  hyper_args=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed)
  full_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_phase none false false raw)"
  offset_raw_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" position_offset none false false raw)"
  offset_softplus_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" position_offset none false false softplus)"
  slope_offset_raw_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_offset none false false raw)"
  slope_offset_softplus_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_offset none false false softplus)"
  warp_qk="$(v2_unit_hyper_qk_json hyper position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 64 signed \
    position_offset none false false raw)"

  # label|qk_norm_mode|per_head_qk_norm|qk_json
  for spec in \
    "offset-raw|method_aware_rms|false|${offset_raw_qk}" \
    "offset-softplus|method_aware_rms|false|${offset_softplus_qk}" \
    "slope-offset-raw|method_aware_rms|false|${slope_offset_raw_qk}" \
    "slope-offset-softplus|method_aware_rms|false|${slope_offset_softplus_qk}" \
    "position-warp-offset|method_aware_rms|false|${warp_qk}" \
    "qknorm-perhead-rope|legacy_layernorm|true|${disabled_channel}" \
    "qknorm-perhead-free-carrier|method_aware_rms|true|${full_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    qk_norm_mode="${remainder%%|*}"
    remainder="${remainder#*|}"
    per_head_qk_norm="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase12-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"qk_norm_per_head\": ${per_head_qk_norm}, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# (1) De-confounded per-head Q/K norm: phase12 compared PerHeadRMSNorm against a
#     legacy_layernorm baseline, mixing per-head gains with a LayerNorm ->
#     RMSNorm change. LayerNorm centers, which is not a typical Q/K choice, so
#     both cells here use RMSNorm and differ only in shared vs per-head gains.
# (2) Compression decomposition: phase11 narrowed the amplitude readout (48->2)
#     and the angular readout (48->1) simultaneously, so its 0.0112 gap could
#     not be attributed. Each mixed arm narrows exactly one branch. Offsets use
#     tanh, which phase12 found mildly best.
if want_phase13_decomp_qknorm_family "decomp_qknorm_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  hyper_args=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed)
  amplitude_offset_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_offset none false false tanh)"
  slope_phase_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_phase none false false tanh)"

  for spec in \
    "rope-shared-rmsnorm|false|${disabled_channel}" \
    "rope-perhead-rmsnorm|true|${disabled_channel}" \
    "freeamp-plus-offset|false|${amplitude_offset_qk}" \
    "slope-plus-freephase|false|${slope_phase_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    per_head_qk_norm="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase13-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"qk_norm_per_head\": ${per_head_qk_norm}, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# phase13 localized the cost to the angular branch: narrowing amplitude 48->2
# costs 0.0035, narrowing angular 48->1 costs 0.0219. This traces the angular
# curve between those endpoints with a factorized phase readout (rank r, then a
# learned [r, pair_dim] basis), to find whether the cost is smooth or has a
# knee. The rank-48 endpoint is `slope_phase` from phase13 (4.28827) and the
# rank-1-like endpoint is `slope_offset` from phase11 (4.29601).
# The rank-8 cell saves final weights so the learned amplitude and phase
# profiles can be inspected offline; no other cell writes weights.
if want_phase14_angular_rank_family "angular_rank_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  hyper_args=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed)

  for rank in 2 4 8 16 32; do
    qk_json="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_phase_lowrank none false false tanh "${rank}")"
    save_weights=false
    if [[ "${rank}" == "8" ]]; then save_weights=true; fi
    job_name="phase14-angular-rank${rank}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"save_final_model\": ${save_weights}, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done

  # Free-phase reference that also saves weights, for the profile comparison.
  free_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_phase none false false tanh)"
  job_name="phase14-angular-free-weights-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"save_final_model\": true, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${free_qk}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
    5000 1000
  run_job "${job_name}" "${cfg_file}"
fi

# Two axes found by inspecting shapes and optimizer groups rather than by sweeping.
#
# (a) Weight decay on zero-anchored position parameters. The no_decay rule
#     matches only "bias"/"norm", so the carrier readouts -- which are
#     zero-initialized precisely so the channel starts at exactly cis(omega*p)
#     -- are decayed toward zero at 0.01. For these, zero is the anchor, so
#     decay is a prior against using the mechanism at all rather than a
#     shrinkage prior on large weights. Every hypernetwork screen so far ran
#     under that force.
# (b) Cross-head readout mixing. The trunk's grouping is free (its input is
#     identical across heads, so grouped == dense-then-split), but the grouped
#     readout confines head h to its own post-nonlinearity features. A dense
#     [heads*hidden -> heads*out] readout mirrors how one W_q feeds all heads.
#     It costs heads-times the readout parameters, so it needs the
#     parameter-matched wide-trunk control alongside it.
if want_phase15_decay_mixing_family "decay_mixing_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  hyper_args=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed)
  full_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_phase none false false tanh 8 none)"
  slope_offset_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_offset none false false tanh 8 none)"
  mapped_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  mixed_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_phase none false false tanh 8 dense)"
  wide_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 256 signed \
    amplitude_phase none false false tanh 8 none)"

  # label|exclude_position_from_decay|qk_norm_mode|qk_json
  for spec in \
    "nodecay-free-control|true|method_aware_rms|${full_qk}" \
    "nodecay-slope-offset|true|method_aware_rms|${slope_offset_qk}" \
    "nodecay-mapped-addrope|true|method_aware_rms|${mapped_qk}" \
    "headmix-readout|false|method_aware_rms|${mixed_qk}" \
    "headmix-readout-nodecay|true|method_aware_rms|${mixed_qk}" \
    "widetrunk256-control|false|method_aware_rms|${wide_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    no_decay="${remainder%%|*}"
    remainder="${remainder#*|}"
    qk_norm_mode="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase15-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"${qk_norm_mode}\", \"post_position_qk_norm\": false, \"exclude_position_from_decay\": ${no_decay}, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# phase15 found dense cross-head readout mixing worth -0.0162 against the free
# control, and -0.0125 against a parameter-matched wide trunk, so the gain is
# feature sharing rather than capacity. It costs 3.5x the control's positional
# parameters. This screen asks how much of that survives a low-rank residual:
# the per-head readout is kept and a rank-r cross-head path is added alongside.
# Init follows LoRA (down random with rank-independent fan-in, up zero) with
# alpha/rank scaling, so ranks differ in capacity rather than in effective
# learning rate -- the confound that made phase14's angular sweep unreadable.
if want_phase16_cheap_mixing_family "cheap_mixing_5k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  base=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed
    amplitude_phase none false false tanh 8)
  none_qk="$(v2_unit_hyper_qk_json "${base[@]}" none)"
  dense_qk="$(v2_unit_hyper_qk_json "${base[@]}" dense)"

  for spec in "free-control|${none_qk}" "dense-mixing|${dense_qk}"; do
    label="${spec%%|*}"; qk_json="${spec#*|}"
    job_name="phase16-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done

  for rank in 8 16 32 64; do
    qk_json="$(v2_unit_hyper_qk_json "${base[@]}" lowrank "${rank}")"
    job_name="phase16-lowrank-r${rank}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
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
