#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

phase50_report=results/phase50_training_seed_replication/REPORT.md
while [ ! -f "$phase50_report" ]; do
  phase50_state=$(supervisorctl status mlprope-phase50-training-seed-replication 2>/dev/null | awk '{print $2}')
  case "$phase50_state" in
    RUNNING|STARTING) ;;
    *)
      echo "Phase 50 ended with state ${phase50_state:-unknown} before producing $phase50_report" >&2
      exit 1
      ;;
  esac
  sleep 60
done

/venv/main/bin/python -u scripts/cleanup_phase50_completed_checkpoints.py
/venv/main/bin/python -u scripts/analyze_phase51_carrier_structure.py \
  --require-seeds 123 456 789
/venv/main/bin/python -u scripts/launch_phase51_carrier_counterfactuals.py \
  --max-concurrent 2 \
  --batch-size 1
/venv/main/bin/python -u scripts/analyze_phase51_carrier_counterfactuals.py
