#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/launch_phase45_novel_static_maps.py --preflight-first
/venv/main/bin/python -u scripts/analyze_phase45_novel_static_maps.py
