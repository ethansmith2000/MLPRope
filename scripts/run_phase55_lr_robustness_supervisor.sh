#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase55_lr_robustness.py
/venv/main/bin/python -u scripts/launch_phase55_lr_robustness.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase55_lr_references.py
/venv/main/bin/python -u scripts/launch_phase55_lr_robustness.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase55_lr_robustness.py
/venv/main/bin/python -u scripts/cleanup_phase55_recovery_checkpoints.py
