#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/prepare_phase59_modern_backbone.py
/venv/main/bin/python -u scripts/launch_phase59_modern_backbone.py --preflight --max-concurrent 2
/venv/main/bin/python -u scripts/launch_phase59_modern_backbone.py --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase59_modern_backbone.py
