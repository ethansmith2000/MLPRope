#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/MLPRope

/venv/main/bin/python -u scripts/launch_phase53_paper_mechanism.py \
  --max-concurrent 2
/venv/main/bin/python -u scripts/analyze_phase53_paper_mechanism.py
