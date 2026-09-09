#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/launch_phase46_linear_rank128.py --preflight-first
/venv/main/bin/python -u scripts/analyze_phase46_linear_rank128.py
