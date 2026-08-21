#!/bin/bash
# Launch the phase-24 RoPE-embed-basis screen through the shared GPU queue.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs/phase24_rope_embed_basis"
LOG_DIR="${SCRIPT_DIR}/logs/phase24_rope_embed_basis"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
OWNER="${OWNER:-mlprope}"
GPU_SELECTOR="${GPU_SELECTOR:-any}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

mkdir -p "${LOG_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_rope_embed_basis_screen.py"

pids=()
for cfg in "${CONFIG_DIR}"/phase24-*.json; do
  job="$(basename "${cfg}" .json)"
  out="${SCRIPT_DIR}/model-output/position_bias_phase24_rope_embed_basis/${job}"
  if [[ -f "${out}/COMPLETED" ]]; then echo "SKIP_COMPLETED ${job}"; continue; fi
  echo "QUEUE_START ${job} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ( gpu-claim run --owner "${OWNER}" --job "${job}" --gpu "${GPU_SELECTOR}" --wait -- \
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_gpt.py" --override_json "${cfg}" \
      >"${LOG_DIR}/${job}.log" 2>&1 && echo "QUEUE_DONE ${job} rc=0" \
      || echo "QUEUE_DONE ${job} rc=$?" ) &
  pids+=("$!")
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
done
for p in "${pids[@]}"; do wait "${p}" || true; done
echo "SCREEN_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
