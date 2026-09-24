#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase60_modern_readout.py
/venv/main/bin/python -u scripts/launch_phase60_modern_readout.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase60_modern_readout.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase60_modern_readout.py
