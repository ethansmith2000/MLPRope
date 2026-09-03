#!/bin/bash
# Queue the Phase-32 AddRoPE EMA coefficient-axis screen via gpu-claim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase32_dynamic_ema_coupling"
OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase32_ema_coupling"
PREFLIGHT_OUTPUT_ROOT="${SCRIPT_DIR}/model-output/position_bias_phase32_ema_coupling_preflight"
LOG_DIR="${SCRIPT_DIR}/logs/phase32_dynamic_ema_coupling"
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
  echo "Another Phase-32 launcher holds ${LOG_DIR}/launcher.lock" >&2
  exit 3
fi
echo "$$" >"${LOG_DIR}/launcher.pid"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_dynamic_ema_coupling.py"

# Freeze training code and locked configs before waiting for a GPU. The copied
# files plus hashes are the executable provenance boundary for every arm.
if [[ ! -f "${SNAPSHOT_DIR}/SNAPSHOT_READY" ]]; then
  snapshot_tmp="${SNAPSHOT_DIR}.tmp.$$"
  mkdir -p "${snapshot_tmp}/scripts" \
    "${snapshot_tmp}/sweep_configs/phase32_dynamic_ema_coupling"
  cp "${SCRIPT_DIR}/train_gpt.py" "${snapshot_tmp}/"
  cp "${SCRIPT_DIR}/transformer.py" "${snapshot_tmp}/"
  cp -a "${SCRIPT_DIR}/position" "${snapshot_tmp}/position"
  cp "${SCRIPT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    "${snapshot_tmp}/scripts/"
  cp "${SCRIPT_DIR}/run_dynamic_ema_coupling_group.sh" "${snapshot_tmp}/"
  cp -a "${CONFIG_DIR}/." \
    "${snapshot_tmp}/sweep_configs/phase32_dynamic_ema_coupling/"
  git -C "${SCRIPT_DIR}" rev-parse HEAD >"${snapshot_tmp}/git-commit.txt"
  git -C "${SCRIPT_DIR}" status --short >"${snapshot_tmp}/git-status.txt"
  git -C "${SCRIPT_DIR}" diff HEAD --binary >"${snapshot_tmp}/working-tree.patch"
  (
    cd "${snapshot_tmp}"
    sha256sum train_gpt.py transformer.py position/*.py \
      scripts/position_dynamics_cuda_smoke.py \
      run_dynamic_ema_coupling_group.sh \
      sweep_configs/phase32_dynamic_ema_coupling/*.json \
      sweep_configs/phase32_dynamic_ema_coupling/preflight/*.json \
      >SOURCE_SHA256SUMS
  )
  touch "${snapshot_tmp}/SNAPSHOT_READY"
  mv "${snapshot_tmp}" "${SNAPSHOT_DIR}"
fi
RUN_CONFIG_DIR="${SNAPSHOT_DIR}/sweep_configs/phase32_dynamic_ema_coupling"

for cfg in "${RUN_CONFIG_DIR}"/phase32-*.json; do
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
    --job "phase32-${SNAPSHOT_REVISION}-ema-coupling-smoke" \
    --gpu "${GPU_SELECTOR}" --wait -- \
    "${PYTHON_BIN}" -u "${SNAPSHOT_DIR}/scripts/position_dynamics_cuda_smoke.py" \
    >>"${smoke_log}" 2>&1
  echo "CUDA_SMOKE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

PREFLIGHT_CONFIGS=(
  "${RUN_CONFIG_DIR}/preflight/phase32-preflight-addrope-content-ema-scalar-seed123-s20-h768d8.json"
  "${RUN_CONFIG_DIR}/preflight/phase32-preflight-addrope-content-ema-per-head-seed123-s20-h768d8.json"
  "${RUN_CONFIG_DIR}/preflight/phase32-preflight-addrope-content-ema-per-dim-seed123-s20-h768d8.json"
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

# All arms execute under one lifetime claim on one physical GPU, in increasing
# coefficient complexity, so hardware assignment cannot confound the contrast.
CONFIGS=(
  "${RUN_CONFIG_DIR}/phase32-addrope-content-pointwise-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase32-addrope-content-ema-scalar-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase32-addrope-content-ema-per-head-seed123-s15000-h768d8.json"
  "${RUN_CONFIG_DIR}/phase32-addrope-content-ema-per-dim-seed123-s15000-h768d8.json"
)

echo "MLPROPE_PHASE32_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${GPU_CLAIM_BIN}" status || true
"${GPU_CLAIM_BIN}" run --owner "${OWNER}" \
  --job "phase32-ema-coupling-family" --gpu "${GPU_SELECTOR}" --wait -- \
  /bin/bash "${SNAPSHOT_DIR}/run_dynamic_ema_coupling_group.sh" \
  "${PYTHON_BIN}" "${SNAPSHOT_DIR}/train_gpt.py" "${LOG_DIR}" \
  "${CONFIGS[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_dynamic_ema_coupling.py"
echo "MLPROPE_PHASE32_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
