#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase56_lr_boundary.py
/venv/main/bin/python -u scripts/launch_phase56_lr_boundary.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase56_lr_boundary.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase56_lr_boundary.py
/venv/main/bin/python -u scripts/cleanup_phase56_recovery_checkpoints.py
