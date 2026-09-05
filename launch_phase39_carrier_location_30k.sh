#!/bin/bash
# Queue the frozen Phase-39A carrier-location screen through gpu-claim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase39_carrier_location_30k"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase39_carrier_location_30k"
LOG_DIR="${SCRIPT_DIR}/logs/phase39_carrier_location_30k"
SNAPSHOT_DIR="${LOG_DIR}/source_snapshot"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-any}"
MAX_WORKERS="${MAX_WORKERS:-4}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/tokenized/openwebtext_gpt2_bs1024}"

if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  echo "gpu-claim is required; see /workspace/GPU_QUEUEING.md" >&2
  exit 1
fi
if ! [[ "${MAX_WORKERS}" =~ ^[1-4]$ ]]; then
  echo "MAX_WORKERS must be an integer from 1 through 4" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
exec 9>"${LOG_DIR}/launcher.lock"
if ! flock -n 9; then
  echo "Another Phase-39 launcher holds ${LOG_DIR}/launcher.lock" >&2
  exit 3
fi
echo "$$" >"${LOG_DIR}/launcher.pid"

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_phase39_carrier_location_30k.py"

# Every queued process must load exactly the tested source, even if the live
# checkout changes while it waits for a shared GPU.
if ! git -C "${SCRIPT_DIR}" diff --quiet -- \
  train_gpt.py transformer.py position; then
  echo "Uncommitted model/training/position drift; refusing to freeze Phase 39" >&2
  exit 4
fi

if [[ ! -f "${SNAPSHOT_DIR}/SNAPSHOT_READY" ]]; then
  snapshot_tmp="${SNAPSHOT_DIR}.tmp.$$"
  mkdir -p "${snapshot_tmp}/scripts" \
    "${snapshot_tmp}/sweep_configs/phase39_carrier_location_30k/preflight"
  cp "${SCRIPT_DIR}/train_gpt.py" "${snapshot_tmp}/"
  cp "${SCRIPT_DIR}/transformer.py" "${snapshot_tmp}/"
  cp -a "${SCRIPT_DIR}/position" "${snapshot_tmp}/position"
  cp "${SCRIPT_DIR}/scripts/analyze_phase39_carrier_location_30k.py" \
    "${snapshot_tmp}/scripts/"
  cp "${CONFIG_DIR}"/*.json \
    "${snapshot_tmp}/sweep_configs/phase39_carrier_location_30k/"
  cp "${CONFIG_DIR}/preflight"/*.json \
    "${snapshot_tmp}/sweep_configs/phase39_carrier_location_30k/preflight/"
  git -C "${SCRIPT_DIR}" rev-parse HEAD >"${snapshot_tmp}/git-commit.txt"
  git -C "${SCRIPT_DIR}" status --short >"${snapshot_tmp}/git-status.txt"
  git -C "${SCRIPT_DIR}" diff --binary >"${snapshot_tmp}/working-tree.patch"
  (
    cd "${snapshot_tmp}"
    sha256sum train_gpt.py transformer.py position/*.py \
      scripts/analyze_phase39_carrier_location_30k.py \
      sweep_configs/phase39_carrier_location_30k/*.json \
      sweep_configs/phase39_carrier_location_30k/preflight/*.json \
      >SOURCE_SHA256SUMS
  )
  touch "${snapshot_tmp}/SNAPSHOT_READY"
  mv "${snapshot_tmp}" "${SNAPSHOT_DIR}"
fi

RUN_CONFIG_DIR="${SNAPSHOT_DIR}/sweep_configs/phase39_carrier_location_30k"
for cfg in "${RUN_CONFIG_DIR}"/*.json "${RUN_CONFIG_DIR}/preflight"/*.json; do
  "${PYTHON_BIN}" "${SNAPSHOT_DIR}/train_gpt.py" \
    --override_json "${cfg}" --dry_run \
    >"${LOG_DIR}/$(basename "$(dirname "${cfg}")")-$(basename "${cfg}" .json).dry-run.log"
done

if [[ ! -f "${DATASET_PATH}/dataset_dict.json" ]]; then
  echo "Dataset is absent or incomplete: ${DATASET_PATH}" >&2
  exit 5
fi
"${PYTHON_BIN}" - "${DATASET_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import datasets

path = Path(sys.argv[1])
dataset = datasets.load_from_disk(str(path))
expected = {"train": 8_372_843, "validation": 443_501}
actual = {name: len(dataset[name]) for name in expected}
if actual != expected:
    raise RuntimeError(f"OpenWebText block-count mismatch: {actual} != {expected}")
for name in expected:
    if len(dataset[name][0]["input_ids"]) != 1_024:
        raise RuntimeError(f"OpenWebText {name} row width is not 1,024")
manifest = json.loads((path / ".tokenized-cache-manifest.json").read_text())
signature = manifest["signature"]
if signature["block_size"] != 1_024:
    raise RuntimeError(f"Unexpected tokenized-cache manifest: {signature}")
print("DATASET_VERIFIED", actual)
PY

PREFLIGHT_CONFIGS=("${RUN_CONFIG_DIR}/preflight"/*.json)
FULL_CONFIGS=("${RUN_CONFIG_DIR}"/*.json)

run_worker() {
  local stage="$1"
  local worker_index="$2"
  local -n stage_configs="$3"
  local failed=0
  local index cfg job_name output_dir log_file rc
  for ((index=worker_index; index<${#stage_configs[@]}; index+=MAX_WORKERS)); do
    cfg="${stage_configs[$index]}"
    job_name="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_name"])' "${cfg}")"
    output_dir="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "${cfg}")"
    log_file="${LOG_DIR}/${stage}-${job_name}.log"
    if [[ -f "${output_dir}/COMPLETED" ]]; then
      echo "SKIP_COMPLETED stage=${stage} worker=${worker_index} ${job_name}"
      continue
    fi
    echo "QUEUE_START stage=${stage} worker=${worker_index} ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if "${GPU_CLAIM_BIN}" run --owner "${OWNER}" --job "${job_name}" \
      --gpu "${GPU_SELECTOR}" --wait -- \
      "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/train_gpt.py" \
      --override_json "${cfg}" >>"${log_file}" 2>&1
    then
      echo "QUEUE_DONE stage=${stage} worker=${worker_index} ${job_name} rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      rc=$?
      echo "QUEUE_DONE stage=${stage} worker=${worker_index} ${job_name} rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
      failed=1
    fi
  done
  return "${failed}"
}

run_stage() {
  local stage="$1"
  local array_name="$2"
  local worker failed=0
  local -a pids=()
  for ((worker=0; worker<MAX_WORKERS; worker+=1)); do
    run_worker "${stage}" "${worker}" "${array_name}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "One or more ${stage} jobs failed; no automatic retry attempted" >&2
    return 1
  fi
}

echo "MLPROPE_PHASE39_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ) workers=${MAX_WORKERS}"
"${GPU_CLAIM_BIN}" status || true
run_stage "preflight" PREFLIGHT_CONFIGS
echo "MLPROPE_PHASE39_PREFLIGHT_PASSED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_stage "full" FULL_CONFIGS

MLPROPE_RESULT_REPO_ROOT="${SCRIPT_DIR}" \
  "${PYTHON_BIN}" "${SNAPSHOT_DIR}/scripts/analyze_phase39_carrier_location_30k.py"
echo "MLPROPE_PHASE39_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
