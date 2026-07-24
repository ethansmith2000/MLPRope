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
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase1}" # phase1 | phase1b | phase1c | phase2_coupling | phase2_followup | phase3_geometry | phase3_amplitude_followup | phase3_promotion | phase3_basis_screen | individual | all
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
    || "${EXPERIMENT_FAMILY}" == "phase3_basis_screen" ]]; then
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
  local residual_json="${5:-{\"enabled\": false}}"
  local write_json="${6:-{\"enabled\": false}}"
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
  local residual_json="${6:-{\"enabled\": false}}"
  local write_json="${7:-{\"enabled\": false}}"
  local cfg_file="${CONFIG_DIR}/${job_name}.json"
  write_common_config "${cfg_file}" \
    "\"seed\": ${seed}, \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${logit_json}, \"residual_stream\": ${residual_json}, \"attention_write\": ${write_json}, \"attn_impl\": \"${attn_impl}\", \"run_name\": \"${job_name}\""
  run_job "${job_name}" "${cfg_file}"
}

# Extended Q/K helper. Geometry:
#   additive/free | additive/amplitude_phase
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
  cat <<JSON
{"enabled": true, "application": "${application}", "geometry": "${geometry}", "input": {"kind": "${input_kind}", "basis_dim": null, "theta": null, "scalars": ${scalars_json}}, "mapper": {"kind": "${mapper_kind}", "residual": ${residual}, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": ${amplitude_init}, "amplitude_parameterization": "signed", "phase_scale": 1.0, "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": {"kind": "${conditioning}", "hidden_dim": ${POS_MLP_HIDDEN}}, "qk_coupling": "${qk_coupling}", "head_coupling": "${head_coupling}"}
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
