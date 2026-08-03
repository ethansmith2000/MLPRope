#!/bin/bash
# Queue the locked phase-20 learned-RoPE frequency screen through gpu-claim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase20_rope_frequency"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase20_rope_frequency"
LOG_DIR="${SCRIPT_DIR}/logs/phase20_rope_frequency"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-any}"
PARALLEL="${PARALLEL:-true}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_rope_frequency_screen.py"
if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  echo "gpu-claim is required; see /workspace/GPU_QUEUEING.md" >&2
  exit 1
fi

PIDS=()
NAMES=()
for cfg in "${CONFIG_DIR}"/phase20-*.json; do
  job_name="$(basename "${cfg}" .json)"
  output_dir="${OUTPUT_ROOT}/${job_name}"
  log_file="${LOG_DIR}/${job_name}.log"
  if [[ -f "${output_dir}/COMPLETED" ]]; then
    echo "SKIP_COMPLETED ${job_name}"
    continue
  fi
  echo "QUEUE_START ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${PARALLEL}" == "true" ]]; then
    (
      "${GPU_CLAIM_BIN}" run --owner "${OWNER}" --job "${job_name}" \
        --gpu "${GPU_SELECTOR}" --wait -- \
        "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_gpt.py" \
        --override_json "${cfg}" >>"${log_file}" 2>&1
    ) &
    PIDS+=("$!")
    NAMES+=("${job_name}")
  else
    "${GPU_CLAIM_BIN}" run --owner "${OWNER}" --job "${job_name}" \
      --gpu "${GPU_SELECTOR}" --wait -- \
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_gpt.py" \
      --override_json "${cfg}" >>"${log_file}" 2>&1
    echo "QUEUE_DONE ${job_name} rc=0"
  fi
done

if [[ "${PARALLEL}" == "true" && ${#PIDS[@]} -gt 0 ]]; then
  fail=0
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      echo "QUEUE_DONE ${NAMES[$index]} rc=0"
    else
      rc=$?
      echo "QUEUE_DONE ${NAMES[$index]} rc=${rc}" >&2
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    exit "${fail}"
  fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_rope_frequency_screen.py"
