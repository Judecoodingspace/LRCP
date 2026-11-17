#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/P4_PILOT_RETRY_20261023_234457_384594/launch_control.py launch "$@"
