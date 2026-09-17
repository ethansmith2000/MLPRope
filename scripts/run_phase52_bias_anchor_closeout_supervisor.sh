#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/launch_phase52_bias_anchor_closeout.py \
  --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase52_reference_evaluations.py \
  --max-concurrent 2 \
  --batch-size 1
/venv/main/bin/python -u scripts/analyze_phase52_bias_anchor_closeout.py
