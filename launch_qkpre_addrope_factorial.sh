#!/bin/bash
# Queue the Phase-29 qkpre x AddRoPE factorial screen through gpu-claim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase29_qkpre_addrope_factorial"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase29_qkpre_addrope_factorial"
LOG_DIR="${SCRIPT_DIR}/logs/phase29_qkpre_addrope_factorial"
SNAPSHOT_DIR="${LOG_DIR}/source_snapshot"
PHASE28_SNAPSHOT="${SCRIPT_DIR}/logs/phase28_qkpre_rope_30k/source_snapshot"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-any}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SKIP_CUDA_SMOKE="${SKIP_CUDA_SMOKE:-false}"

if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  echo "gpu-claim is required; see /workspace/GPU_QUEUEING.md" >&2
  exit 1
fi
if ! [[ "${MAX_WORKERS}" =~ ^[1-4]$ ]]; then
  echo "MAX_WORKERS must be an integer from 1 through 4" >&2
  exit 2
fi
if [[ ! -f "${PHASE28_SNAPSHOT}/SNAPSHOT_READY" ]]; then
  echo "The successful Phase-28 source snapshot is missing" >&2
  exit 3
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
exec 9>"${LOG_DIR}/launcher.lock"
if ! flock -n 9; then
  echo "Another Phase-29 launcher already holds ${LOG_DIR}/launcher.lock" >&2
  exit 4
fi
echo "$$" >"${LOG_DIR}/launcher.pid"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_qkpre_addrope_factorial.py"

# No position primitive changed for this combination. Only the train/model
# isolation guards and CUDA coverage differ from the Phase-28 implementation.
for source in "${SCRIPT_DIR}"/position/*.py; do
  rel="position/$(basename "${source}")"
  if ! cmp -s "${source}" "${PHASE28_SNAPSHOT}/${rel}"; then
    echo "Unexpected position primitive drift from Phase 28: ${rel}" >&2
    exit 5
  fi
done

if [[ ! -f "${SNAPSHOT_DIR}/SNAPSHOT_READY" ]]; then
  snapshot_tmp="${SNAPSHOT_DIR}.tmp.$$"
  mkdir -p "${snapshot_tmp}/scripts" \
    "${snapshot_tmp}/sweep_configs/phase29_qkpre_addrope_factorial"
  cp "${SCRIPT_DIR}/train_gpt.py" "${snapshot_tmp}/"
  cp "${SCRIPT_DIR}/transformer.py" "${snapshot_tmp}/"
  cp -a "${SCRIPT_DIR}/position" "${snapshot_tmp}/position"
  cp "${SCRIPT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    "${snapshot_tmp}/scripts/"
  cp "${CONFIG_DIR}"/phase29-*.json \
    "${snapshot_tmp}/sweep_configs/phase29_qkpre_addrope_factorial/"
  git -C "${SCRIPT_DIR}" rev-parse HEAD >"${snapshot_tmp}/git-commit.txt"
  git -C "${SCRIPT_DIR}" status --short >"${snapshot_tmp}/git-status.txt"
  git -C "${SCRIPT_DIR}" diff --binary >"${snapshot_tmp}/working-tree.patch"
  cp "${PHASE28_SNAPSHOT}/SOURCE_SHA256SUMS" \
    "${snapshot_tmp}/PHASE28_SOURCE_SHA256SUMS"
  (
    cd "${snapshot_tmp}"
    sha256sum train_gpt.py transformer.py position/*.py \
      scripts/position_dynamics_cuda_smoke.py \
      sweep_configs/phase29_qkpre_addrope_factorial/*.json \
      >SOURCE_SHA256SUMS
  )
  touch "${snapshot_tmp}/POSITION_PRIMITIVES_MATCH_PHASE28"
  touch "${snapshot_tmp}/SNAPSHOT_READY"
  mv "${snapshot_tmp}" "${SNAPSHOT_DIR}"
fi
RUN_CONFIG_DIR="${SNAPSHOT_DIR}/sweep_configs/phase29_qkpre_addrope_factorial"

for cfg in "${RUN_CONFIG_DIR}"/phase29-*.json; do
  "${PYTHON_BIN}" "${SNAPSHOT_DIR}/train_gpt.py" \
    --override_json "${cfg}" --dry_run \
    >"${LOG_DIR}/$(basename "${cfg}" .json).dry-run.log"
done

if [[ "${SKIP_CUDA_SMOKE}" != "true" ]]; then
  smoke_log="${LOG_DIR}/position-dynamics-cuda-smoke.log"
  echo "CUDA_SMOKE_QUEUE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${GPU_CLAIM_BIN}" run --owner "${OWNER}" \
    --job "phase29-qkpre-addrope-smoke" --gpu "${GPU_SELECTOR}" --wait -- \
    "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    >>"${smoke_log}" 2>&1
  echo "CUDA_SMOKE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

CONFIGS=(
  "${RUN_CONFIG_DIR}/phase29-rope-fixed-seed123-s5000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase29-qkpre-addrope-a10-seed123-s5000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase29-qkpre-rope-seed123-s5000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase29-addrope-a10-seed123-s5000-h768d8.json"
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

echo "MLPROPE_PHASE29_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ) workers=${MAX_WORKERS} gpus=${GPU_SELECTOR}"
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
  echo "One or more Phase-29 jobs failed; no automatic retry was attempted" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_qkpre_addrope_factorial.py"
echo "MLPROPE_PHASE29_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
