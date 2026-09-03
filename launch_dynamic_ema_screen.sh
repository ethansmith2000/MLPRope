#!/bin/bash
# Queue the Phase-31 15k causal-EMA screen through lifetime gpu-claim locks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase31_dynamic_ema_screen"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase31_dynamic_ema"
PREFLIGHT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase31_preflight"
LOG_DIR="${SCRIPT_DIR}/logs/phase31_dynamic_ema_screen"
SNAPSHOT_REVISION="${SNAPSHOT_REVISION:-r1}"
SNAPSHOT_DIR="${LOG_DIR}/source_snapshot_${SNAPSHOT_REVISION}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_CLAIM_BIN="${GPU_CLAIM_BIN:-$(command -v gpu-claim || true)}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-any}"
SKIP_CUDA_SMOKE="${SKIP_CUDA_SMOKE:-false}"
DATASET_PATH="${DATASET_PATH:-/workspace/data/tokenized/openwebtext_gpt2_bs1024}"

if [[ -z "${GPU_CLAIM_BIN}" ]]; then
  echo "gpu-claim is required; see /workspace/GPU_QUEUEING.md" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${PREFLIGHT_OUTPUT_ROOT}"
exec 9>"${LOG_DIR}/launcher.lock"
if ! flock -n 9; then
  echo "Another Phase-31 launcher holds ${LOG_DIR}/launcher.lock" >&2
  exit 3
fi
echo "$$" >"${LOG_DIR}/launcher.pid"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_dynamic_ema_screen.py"

# Freeze every training source and config before any queue wait. The working
# tree is intentionally dirty because Phase 31 introduces the EMA mechanism;
# the immutable snapshot, hashes, and patch are the run's provenance boundary.
if [[ ! -f "${SNAPSHOT_DIR}/SNAPSHOT_READY" ]]; then
  snapshot_tmp="${SNAPSHOT_DIR}.tmp.$$"
  mkdir -p "${snapshot_tmp}/scripts" \
    "${snapshot_tmp}/sweep_configs/phase31_dynamic_ema_screen"
  cp "${SCRIPT_DIR}/train_gpt.py" "${snapshot_tmp}/"
  cp "${SCRIPT_DIR}/transformer.py" "${snapshot_tmp}/"
  cp -a "${SCRIPT_DIR}/position" "${snapshot_tmp}/position"
  cp "${SCRIPT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    "${snapshot_tmp}/scripts/"
  cp "${SCRIPT_DIR}/run_dynamic_ema_group.sh" "${snapshot_tmp}/"
  cp -a "${CONFIG_DIR}/." \
    "${snapshot_tmp}/sweep_configs/phase31_dynamic_ema_screen/"
  git -C "${SCRIPT_DIR}" rev-parse HEAD >"${snapshot_tmp}/git-commit.txt"
  git -C "${SCRIPT_DIR}" status --short >"${snapshot_tmp}/git-status.txt"
  git -C "${SCRIPT_DIR}" diff --binary >"${snapshot_tmp}/working-tree.patch"
  (
    cd "${snapshot_tmp}"
    sha256sum train_gpt.py transformer.py position/*.py \
      scripts/position_dynamics_cuda_smoke.py run_dynamic_ema_group.sh \
      sweep_configs/phase31_dynamic_ema_screen/*.json \
      sweep_configs/phase31_dynamic_ema_screen/preflight/*.json \
      >SOURCE_SHA256SUMS
  )
  touch "${snapshot_tmp}/SNAPSHOT_READY"
  mv "${snapshot_tmp}" "${SNAPSHOT_DIR}"
fi
RUN_CONFIG_DIR="${SNAPSHOT_DIR}/sweep_configs/phase31_dynamic_ema_screen"

for cfg in "${RUN_CONFIG_DIR}"/phase31-*.json; do
  "${PYTHON_BIN}" "${SNAPSHOT_DIR}/train_gpt.py" \
    --override_json "${cfg}" --dry_run \
    >"${LOG_DIR}/$(basename "${cfg}" .json).dry-run.log"
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
if signature["block_size"] != 1_024 or signature["tokenizer_name"] != "openai-community/gpt2":
    raise RuntimeError(f"Unexpected tokenized-cache manifest: {signature}")
print("DATASET_VERIFIED", actual)
PY

if [[ "${SKIP_CUDA_SMOKE}" != "true" ]]; then
  smoke_log="${LOG_DIR}/position-dynamics-cuda-smoke.log"
  echo "CUDA_SMOKE_QUEUE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${GPU_CLAIM_BIN}" run --owner "${OWNER}" \
    --job "phase31-${SNAPSHOT_REVISION}-ema-smoke" \
    --gpu "${GPU_SELECTOR}" --wait -- \
    "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    >>"${smoke_log}" 2>&1
  echo "CUDA_SMOKE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

PREFLIGHT_CONFIGS=(
  "${RUN_CONFIG_DIR}/preflight/phase31-preflight-clock-ema-seed123-s20-h768d8.json"
  "${RUN_CONFIG_DIR}/preflight/phase31-preflight-addrope-content-pointwise-seed123-s20-h768d8.json"
  "${RUN_CONFIG_DIR}/preflight/phase31-preflight-addrope-content-ema-seed123-s20-h768d8.json"
)
for cfg in "${PREFLIGHT_CONFIGS[@]}"; do
  job_name="$(basename "${cfg}" .json)"
  output_dir="${PREFLIGHT_OUTPUT_ROOT}/${job_name}"
  if [[ -f "${output_dir}/COMPLETED" ]]; then
    echo "SKIP_COMPLETED preflight ${job_name}"
    continue
  fi
  echo "PREFLIGHT_START ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${GPU_CLAIM_BIN}" run --owner "${OWNER}" --job "${job_name}" \
    --gpu "${GPU_SELECTOR}" --wait -- \
    "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/train_gpt.py" \
    --override_json "${cfg}" \
    >>"${LOG_DIR}/${job_name}.log" 2>&1
  echo "PREFLIGHT_DONE ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

# Each family is one lifetime-claimed job. Its control, pointwise arm, and EMA
# arm therefore execute on exactly the same physical GPU, removing a hardware
# assignment confound from the two primary EMA-minus-pointwise contrasts.
CLOCK_CONFIGS=(
  "${RUN_CONFIG_DIR}/phase31-rope-fixed-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase31-clock-pointwise-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase31-clock-ema-seed123-s15000-h768d8.json"
)
ADDROPE_CONFIGS=(
  "${RUN_CONFIG_DIR}/phase31-addrope-position-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase31-addrope-content-pointwise-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase31-addrope-content-ema-seed123-s15000-h768d8.json"
)

run_family() {
  local family="$1"
  shift
  "${GPU_CLAIM_BIN}" run --owner "${OWNER}" \
    --job "phase31-${family}-family" --gpu "${GPU_SELECTOR}" --wait -- \
    /bin/bash "${SNAPSHOT_DIR}/run_dynamic_ema_group.sh" \
    "${PYTHON_BIN}" "${SNAPSHOT_DIR}/train_gpt.py" "${LOG_DIR}" "$@"
}

echo "MLPROPE_PHASE31_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${GPU_CLAIM_BIN}" status || true
run_family clock "${CLOCK_CONFIGS[@]}" &
clock_pid="$!"
run_family addrope "${ADDROPE_CONFIGS[@]}" &
addrope_pid="$!"

failed=0
if ! wait "${clock_pid}"; then
  failed=1
fi
if ! wait "${addrope_pid}"; then
  failed=1
fi
if [[ "${failed}" -ne 0 ]]; then
  echo "One or more Phase-31 families failed; no automatic retry attempted" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_dynamic_ema_screen.py"
echo "MLPROPE_PHASE31_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
