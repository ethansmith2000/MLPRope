#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/launch_phase43_lowrank_qk_pathways.py --preflight-first
/venv/main/bin/python -u scripts/analyze_phase43_lowrank_qk_pathways.py
