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
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase16_cheap_mixing}"
# Live families only. Historical families (phase1-phase10, phase12, phase14)
# were pruned 2026-07-31; their generated configs remain under sweep_configs/
# and the generating blocks are recoverable from git history.
# phase11_spectral | phase13_decomp_qknorm | phase15_decay_mixing | phase16_cheap_mixing | phase17_gate_30k | phase18_scale_h1024 | individual | all
if [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase11_spectral"
elif [[ "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase13_decomp_qknorm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase15_decay_mixing"
elif [[ "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase16_cheap_mixing"
elif [[ "${EXPERIMENT_FAMILY}" == "phase17_gate_30k" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase17_gate_30k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase18_scale_h1024" ]]; then
  DEFAULT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase18_scale_h1024"
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
  if [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" \
    || "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" \
    || "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" \
    || "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" \
    || "${EXPERIMENT_FAMILY}" == "phase17_gate_30k" \
    || "${EXPERIMENT_FAMILY}" == "phase18_scale_h1024" ]]; then
    GPU_SELECTOR="any"
  else
    GPU_SELECTOR="6,7"
  fi
fi
# Separate config dirs so Phase-2 writes do not touch historical JSON.
if [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase11_spectral"
elif [[ "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase13_decomp_qknorm"
elif [[ "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase15_decay_mixing"
elif [[ "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase16_cheap_mixing"
elif [[ "${EXPERIMENT_FAMILY}" == "phase17_gate_30k" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase17_gate_30k"
elif [[ "${EXPERIMENT_FAMILY}" == "phase18_scale_h1024" ]]; then
  CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase18_scale_h1024"
fi
PIDS=()
JOB_NAMES=()

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}" "${OUTPUT_ROOT}"

# Common settings (single-GPU jobs; pack up to 8 concurrent via gpu-claim)
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-200}"
WITH_TRACKING="${WITH_TRACKING:-true}"
WANDB_GROUP="${WANDB_GROUP:-}"
BASE_WD="0.01"
# Optimizer betas are family-scoped so historical families keep the settings
# their results were produced under.
if [[ "${EXPERIMENT_FAMILY}" == "phase18_scale_h1024" ]]; then
  BASE_BETA1="${BASE_BETA1:-0.95}"
  BASE_BETA2="${BASE_BETA2:-0.999}"
else
  BASE_BETA1="${BASE_BETA1:-0.9}"
  BASE_BETA2="${BASE_BETA2:-0.98}"
fi
POS_RANK="${POS_RANK:-32}"
POS_MLP_HIDDEN="${POS_MLP_HIDDEN:-128}"
POS_SHARING="${POS_SHARING:-per_head}"
REL_EXTENT="${REL_EXTENT:-}" # empty => follow block_size in train_gpt.py

# Model config: "hidden_size depth n_head lr batch_size max_train_steps block_size"
# h1024/d12 gate. 8 heads (head_dim 128) rather than the phase6 gate's 16, so
# the per-head carrier structure matches h768/d8's 8 heads; lr 4e-4 with
# beta1 0.95 / beta2 0.999 per prior experience at this size.
if [[ "${EXPERIMENT_FAMILY}" == "phase18_scale_h1024" ]]; then
  MODEL_CONFIG="${MODEL_CONFIG:-1024 12 8 4.0e-4 8 30000 1024}"
else
  MODEL_CONFIG="${MODEL_CONFIG:-768 8 8 3.0e-4 8 10000 1024}"
fi
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
#   rotary/phase
# Input kind:
#   frozen_fourier
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
  local components="$4" # phase
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
  local offset_parameterization="${14:-tanh}"
  local readout_head_mixing="${15:-none}"
  local readout_mix_rank="${16:-32}"
  local learn_amplitude=true
  local learn_phase=true
  local conditioning='{"kind": "none"}'
  if [[ "${mode}" == "hyper" ]]; then
    learn_amplitude="${learn_static_amplitude}"
    learn_phase=false
    conditioning="{\"kind\": \"carrier_hypernetwork\", \"source\": \"dedicated\", \"hidden_dim\": ${conditioning_hidden_dim}, \"input_mode\": \"${input_mode}\", \"input_normalization\": \"${input_normalization}\", \"learnable_input_gains\": ${learnable_input_gains}, \"network\": \"${network}\", \"components\": \"${components}\", \"target\": \"${target}\", \"coupling\": \"${conditioning_coupling}\", \"static_complement\": ${static_complement}, \"head_coupling\": \"${conditioning_head_coupling}\", \"offset_parameterization\": \"${offset_parameterization}\", \"readout_head_mixing\": \"${readout_head_mixing}\", \"readout_mix_rank\": ${readout_mix_rank}}"
  fi
  cat <<JSON
{"enabled": true, "application": "additive", "geometry": "amplitude_phase", "input": {"kind": "frozen_fourier", "basis_dim": 16, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"parameter_source": "direct", "amplitude_init": 1.0, "amplitude_parameterization": "${amplitude_parameterization}", "learn_amplitude": ${learn_amplitude}, "learn_phase": ${learn_phase}, "phase_scale": 1.0, "additive_normalization": "none"}, "conditioning": ${conditioning}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

v2_promoted_qk_json() {
  # Canonical active mapped carrier.  Earlier builders above retain 0.3 only
  # so archived experiment families can still be reconstructed faithfully.
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
{"enabled": true, "application": "additive", "geometry": "${geometry}", "input": {"kind": "frozen_fourier", "basis_dim": ${basis_dim}, "theta": null, "scalars": ["normalized_position", "log_position"]}, "mapper": {"kind": "linear", "residual": false, "rank": ${POS_RANK}, "hidden_dim": ${POS_MLP_HIDDEN}}, "output": {"amplitude_init": 1.0, "amplitude_parameterization": "signed", "learn_amplitude": true, "learn_phase": true, "phase_scale": 1.0, "additive_normalization": "${additive_normalization}", "additive_gain_init": 0.212132, "additive_gain_max": 1.0, "learn_additive_gain": true, "scale_init": 1.0, "scale_parameterization": "exp"}, "conditioning": ${conditioning_json}, "qk_coupling": "shared_trunk_separate_readouts", "head_coupling": "per_head_independent"}
JSON
}

# v2_pairwise_logit_json was removed with the relative logit-bias channel
# (see CONCAT_QK_POSITION.md); its archived sweeps live in git history.

want_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase18_scale_h1024_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase18_scale_h1024" \
    || "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase17_gate_30k_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase17_gate_30k" \
    || "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase16_cheap_mixing_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase16_cheap_mixing" \
    || "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase15_decay_mixing_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase15_decay_mixing" \
    || "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase13_decomp_qknorm_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase13_decomp_qknorm" \
    || "${EXPERIMENT_FAMILY}" == "all" \
    || "${EXPERIMENT_FAMILY}" == "${family}" ]]
}

want_phase11_spectral_family() {
  local family="$1"
  [[ "${EXPERIMENT_FAMILY}" == "phase11_spectral" \
    || "${EXPERIMENT_FAMILY}" == "all" \
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

# v2_logit_json was removed with the relative logit-bias channel
# (see CONCAT_QK_POSITION.md).

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

# Direction 1 / Phase 1 expressiveness sweep. The logit-bias preset families
# (add_rope/linear/low_rank/mlp_rope) were removed with the relative logit-bias
# channel; see CONCAT_QK_POSITION.md.
if want_family "rope"; then
  emit_variant "rope" "sdpa" "rope-h${HIDDEN_SIZE}d${DEPTH}"
fi

# Direction 1b. Existing completed RoPE and linear-logit runs are the anchors;
# this family emits only the four new/corrected ablations.
QK_DISABLED="{\"enabled\": false}"
LOGIT_DISABLED="{\"enabled\": false}"

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

# (1) De-confounded per-head Q/K norm: phase12 compared PerHeadRMSNorm against a
#     legacy_layernorm baseline, mixing per-head gains with a LayerNorm ->
#     RMSNorm change. The per-head QK-norm option was removed on 2026-08-19
#     (lost at every tested scale), so only the shared-gain cells remain here.
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
    "rope-shared-rmsnorm|${disabled_channel}" \
    "freeamp-plus-offset|${amplitude_offset_qk}" \
    "slope-plus-freephase|${slope_phase_qk}"
  do
    label="${spec%%|*}"
    qk_json="${spec#*|}"
    job_name="phase13-${label}-seed${seed}-s5000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      5000 1000
    run_job "${job_name}" "${cfg_file}"
  done
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
  full_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_phase none false false tanh none)"
  slope_offset_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" slope_offset none false false tanh none)"
  mapped_qk="$(v2_promoted_qk_json amplitude_phase 16)"
  mixed_qk="$(v2_unit_hyper_qk_json "${hyper_args[@]}" amplitude_phase none false false tanh dense)"
  wide_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 256 signed \
    amplitude_phase none false false tanh none)"

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
    amplitude_phase none false false tanh)
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

# 30k horizon gate. Nothing in phases 11-16 has been run past 5k, and this
# project has already seen 10k -> 50k rank reversals (rank-32 logit, basis
# size). Every cell uses method_aware_rms so RoPE differs from the others only
# in the position channel, and tokens/s is logged for the iso-wallclock question
# (dense mixing costs 3.5x the control's positional parameters).
#
# Cells answer, in order: does the RoPE gap keep halving per 6x tokens; does
# dense cross-head mixing's -0.017 survive; is it still not capacity; does
# content conditioning matter at a horizon worth trusting (position-only vs
# free control); does the compressed frontier hold.
if want_phase17_gate_30k_family "gate_30k_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  base=(hyper content_position silu_mlp both false
    shared_trunk_separate_readouts per_head_independent 64 signed
    amplitude_phase none false false tanh)
  free_qk="$(v2_unit_hyper_qk_json "${base[@]}" none)"
  dense_qk="$(v2_unit_hyper_qk_json "${base[@]}" dense)"
  wide_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 256 signed \
    amplitude_phase none false false tanh none)"
  position_only_qk="$(v2_unit_hyper_qk_json hyper position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 64 signed \
    amplitude_phase none false false tanh none)"
  slope_phase_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 64 signed \
    slope_phase none false false tanh none)"

  # label|save_weights|qk_json
  for spec in \
    "standard-rope|false|${disabled_channel}" \
    "free-control|false|${free_qk}" \
    "headmix-dense|true|${dense_qk}" \
    "widetrunk256|false|${wide_qk}" \
    "position-only|false|${position_only_qk}" \
    "slope-free-phase|false|${slope_phase_qk}"
  do
    label="${spec%%|*}"
    remainder="${spec#*|}"
    save_weights="${remainder%%|*}"
    qk_json="${remainder#*|}"
    job_name="phase17-${label}-seed${seed}-s30000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"save_final_model\": ${save_weights}, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      30000 5000
    run_job "${job_name}" "${cfg_file}"
  done
fi

# h1024/d12 scale gate. Everything in phases 11-17 is h768/d8, so model size is
# the last untested dimension. The phase17 30k gate showed the RoPE gap halving
# per ~6x tokens; the open question is whether it also erodes with width. The
# phase6 scale gate found the logit stack held 0.094 over RoPE at both h768 and
# h1024, so width did not erode that gap the way tokens did -- this tests the
# same for the Q/K carrier line.
#
# This gate uses 8 heads (head_dim 128, pair_dim 64) so the number of carrier
# groups matches h768/d8; head_dim still differs (128 vs 96). It also uses
# lr 4e-4 and betas 0.95/0.999 rather than phase17's 3e-4 and 0.9/0.98, so the
# cross-scale RoPE-gap comparison carries an optimizer confound. Within-family
# comparisons are clean.
if want_phase18_scale_h1024_family "scale_h1024_story"; then
  seed=123
  disabled_channel='{"enabled": false}'
  free_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 64 signed \
    amplitude_phase none false false tanh none)"
  position_only_qk="$(v2_unit_hyper_qk_json hyper position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 64 signed \
    amplitude_phase none false false tanh none)"
  wide_qk="$(v2_unit_hyper_qk_json hyper content_position silu_mlp both false \
    shared_trunk_separate_readouts per_head_independent 256 signed \
    amplitude_phase none false false tanh none)"

  for spec in \
    "standard-rope|${disabled_channel}" \
    "position-only|${position_only_qk}" \
    "free-control|${free_qk}" \
    "widetrunk256|${wide_qk}"
  do
    label="${spec%%|*}"
    qk_json="${spec#*|}"
    job_name="phase18-${label}-seed${seed}-s30000-h${HIDDEN_SIZE}d${DEPTH}"
    cfg_file="${CONFIG_DIR}/${job_name}.json"
    write_common_config "${cfg_file}" \
      "\"seed\": ${seed}, \"per_device_eval_batch_size\": 1, \"training_length\": 1024, \"model_position_extent\": 1024, \"evaluation_lengths\": [1024], \"scalar_normalization_extent\": 1024, \"qk_norm_mode\": \"method_aware_rms\", \"post_position_qk_norm\": false, \"position_content_dim\": 128, \"position_content_coupling\": \"shared\", \"pos_variant\": null, \"position_schema_version\": 2, \"qk\": ${qk_json}, \"logit_bias\": ${disabled_channel}, \"residual_stream\": ${disabled_channel}, \"attention_write\": ${disabled_channel}, \"attn_impl\": \"sdpa\", \"run_name\": \"${job_name}\"" \
      30000 5000
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
