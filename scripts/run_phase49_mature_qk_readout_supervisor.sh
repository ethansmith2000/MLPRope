#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/launch_phase49_mature_qk_readout.py --preflight-first
/venv/main/bin/python -u scripts/analyze_phase49_mature_qk_readout.py
