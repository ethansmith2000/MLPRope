#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase58_gate_initialization.py
/venv/main/bin/python -u scripts/launch_phase58_gate_initialization.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase58_gate_initialization.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase58_gate_initialization.py
