#!/bin/bash
# Run a sequence of Phase-31 configs while one outer gpu-claim lock is held.
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 PYTHON TRAIN_SCRIPT LOG_DIR CONFIG..." >&2
  exit 2
fi

PYTHON_BIN="$1"
TRAIN_SCRIPT="$2"
LOG_DIR="$3"
shift 3

for cfg in "$@"; do
  job_name="$(basename "${cfg}" .json)"
  output_root="$(${PYTHON_BIN} - "${cfg}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["base_output_dir"])
PY
)"
  output_dir="${output_root}/${job_name}"
  log_file="${LOG_DIR}/${job_name}.log"
  if [[ -f "${output_dir}/COMPLETED" ]]; then
    echo "SKIP_COMPLETED ${job_name}"
    continue
  fi
  echo "GROUP_RUN_START ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" --override_json "${cfg}" \
    >>"${log_file}" 2>&1
  echo "GROUP_RUN_DONE ${job_name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
