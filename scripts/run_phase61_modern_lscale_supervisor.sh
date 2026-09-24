#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase61_modern_lscale.py
/venv/main/bin/python -u scripts/launch_phase61_modern_lscale.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase61_modern_lscale.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase61_modern_lscale.py
