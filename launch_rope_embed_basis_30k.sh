#!/bin/bash
# Queue the phase-25 additive-carrier 30k promotion through gpu-claim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase25_rope_embed_basis_30k"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase25_rope_embed_basis_30k"
LOG_DIR="${SCRIPT_DIR}/logs/phase25_rope_embed_basis_30k"
SNAPSHOT_DIR="${LOG_DIR}/source_snapshot"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-any}"
MAX_WORKERS="${MAX_WORKERS:-3}"
SKIP_CUDA_SMOKE="${SKIP_CUDA_SMOKE:-false}"

if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  echo "gpu-claim is required; see /workspace/GPU_QUEUEING.md" >&2
  exit 1
fi
if ! [[ "${MAX_WORKERS}" =~ ^[1-3]$ ]]; then
  echo "MAX_WORKERS must be 1, 2, or 3" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
echo "$$" >"${LOG_DIR}/launcher.pid"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_rope_embed_basis_30k.py"

# Jobs may spend hours waiting for a claim. Run every job from one immutable
# snapshot so later mechanism development cannot change code between arms.
if [[ ! -f "${SNAPSHOT_DIR}/SNAPSHOT_READY" ]]; then
  snapshot_tmp="${SNAPSHOT_DIR}.tmp.$$"
  mkdir -p "${snapshot_tmp}/scripts" \
    "${snapshot_tmp}/sweep_configs/phase25_rope_embed_basis_30k"
  cp "${SCRIPT_DIR}/train_gpt.py" "${snapshot_tmp}/"
  cp "${SCRIPT_DIR}/transformer.py" "${snapshot_tmp}/"
  cp -a "${SCRIPT_DIR}/position" "${snapshot_tmp}/position"
  cp "${SCRIPT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    "${snapshot_tmp}/scripts/"
  cp "${CONFIG_DIR}"/phase25-*.json \
    "${snapshot_tmp}/sweep_configs/phase25_rope_embed_basis_30k/"
  git -C "${SCRIPT_DIR}" rev-parse HEAD >"${snapshot_tmp}/git-commit.txt"
  git -C "${SCRIPT_DIR}" status --short >"${snapshot_tmp}/git-status.txt"
  git -C "${SCRIPT_DIR}" diff --binary >"${snapshot_tmp}/working-tree.patch"
  (
    cd "${snapshot_tmp}"
    sha256sum train_gpt.py transformer.py position/*.py \
      scripts/position_dynamics_cuda_smoke.py \
      sweep_configs/phase25_rope_embed_basis_30k/*.json \
      >SOURCE_SHA256SUMS
  )
  touch "${snapshot_tmp}/SNAPSHOT_READY"
  mv "${snapshot_tmp}" "${SNAPSHOT_DIR}"
fi
RUN_CONFIG_DIR="${SNAPSHOT_DIR}/sweep_configs/phase25_rope_embed_basis_30k"

# Validate every locked config before occupying a GPU. Seed differences can
# change initialization but must not change parsing, shapes, or parameter sets.
for cfg in "${RUN_CONFIG_DIR}"/phase25-*.json; do
  "${PYTHON_BIN}" "${SNAPSHOT_DIR}/train_gpt.py" \
    --override_json "${cfg}" --dry_run \
    >"${LOG_DIR}/$(basename "${cfg}" .json).dry-run.log"
done

if [[ "${SKIP_CUDA_SMOKE}" != "true" ]]; then
  smoke_log="${LOG_DIR}/position-dynamics-cuda-smoke.log"
  echo "CUDA_SMOKE_QUEUE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${GPU_CLAIM_BIN}" run --owner "${OWNER}" \
    --job "phase25-position-dynamics-smoke" --gpu "${GPU_SELECTOR}" --wait -- \
    "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    >>"${smoke_log}" 2>&1
  echo "CUDA_SMOKE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# Cyclic order prevents a worker from being permanently associated with one
# arm. gpu-claim independently chooses an available physical GPU for each job.
CONFIGS=(
  "${RUN_CONFIG_DIR}/phase25-rope-fixed-seed123-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-basis16-a03-seed123-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-basis16-a10-seed123-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-basis16-a03-seed456-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-basis16-a10-seed456-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-rope-fixed-seed456-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-basis16-a10-seed789-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-rope-fixed-seed789-s30000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase25-basis16-a03-seed789-s30000-h768d8.json"
)

run_worker() {
  local worker_index="$1"
  local failed=0
  local index cfg job_name output_dir log_file rc
  for ((index=worker_index; index<${#CONFIGS[@]}; index+=MAX_WORKERS)); do
    cfg="${CONFIGS[$index]}"
    job_name="$(basename "${cfg}" .json)"
    output_dir="${OUTPUT_ROOT}/${job_name}"
    log_file="${LOG_DIR}/${job_name}.log"
    if [[ -f "${output_dir}/COMPLETED" ]]; then
      echo "SKIP_COMPLETED worker=${worker_index} ${job_name}"
      continue
    fi
    echo "QUEUE_START worker=${worker_index} ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if "${GPU_CLAIM_BIN}" run --owner "${OWNER}" --job "${job_name}" \
      --gpu "${GPU_SELECTOR}" --wait -- \
      "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/train_gpt.py" \
      --override_json "${cfg}" >>"${log_file}" 2>&1
    then
      echo "QUEUE_DONE worker=${worker_index} ${job_name} rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      rc=$?
      echo "QUEUE_DONE worker=${worker_index} ${job_name} rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
      failed=1
    fi
  done
  return "${failed}"
}

echo "MLPROPE_PHASE25_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ) workers=${MAX_WORKERS} gpus=${GPU_SELECTOR}"
"${GPU_CLAIM_BIN}" status || true
PIDS=()
for ((worker=0; worker<MAX_WORKERS; worker+=1)); do
  run_worker "${worker}" &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "One or more phase-25 jobs failed" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_rope_embed_basis_30k.py"
echo "MLPROPE_PHASE25_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
