#!/bin/bash
# Sequential Phase 1 queue (ngpt_dynamic/run_experiment_queue.sh style).
# Prefer launch_position_bias.sh for config emission + gpu-claim details.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-phase1}"
export SUBMIT_JOBS="${SUBMIT_JOBS:-true}"

echo "MLPROPE_PHASE1_QUEUE_STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
./launch_position_bias.sh
echo "MLPROPE_PHASE1_QUEUE_COMPLETED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
