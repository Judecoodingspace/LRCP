#!/bin/bash
set -euo pipefail
exec /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/P5_B0_C3_20261031_cpu/controller.py status "$@"
