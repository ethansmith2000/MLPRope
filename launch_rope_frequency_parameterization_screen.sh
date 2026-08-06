#!/bin/bash
# Queue phase-21 through two sequential gpu-claim workers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase21_rope_parameterization"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase21_rope_parameterization"
LOG_DIR="${SCRIPT_DIR}/logs/phase21_rope_parameterization"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-6,7}"
MAX_WORKERS="${MAX_WORKERS:-2}"

if [[ "${MAX_WORKERS}" != "1" && "${MAX_WORKERS}" != "2" ]]; then
  echo "MAX_WORKERS must be 1 or 2 for the current allocation" >&2
  exit 2
fi
if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  echo "gpu-claim is required; see /workspace/GPU_QUEUEING.md" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_rope_frequency_parameterization_screen.py"

CONFIGS=()
for seed in 123 456 789; do
  for arm in exp-full-ste softplus additive bounded-log; do
    CONFIGS+=("${CONFIG_DIR}/phase21-${arm}-seed${seed}-s5000-h768d8.json")
  done
done

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
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_gpt.py" \
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

echo "MLPROPE_PHASE21_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ) workers=${MAX_WORKERS} gpus=${GPU_SELECTOR}"
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
  echo "One or more phase-21 jobs failed" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_rope_frequency_parameterization_screen.py"
echo "MLPROPE_PHASE21_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"

