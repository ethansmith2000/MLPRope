#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase57_positional_baselines.py
/venv/main/bin/python -u scripts/launch_phase57_positional_baselines.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase57_positional_baselines.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase57_positional_baselines.py

