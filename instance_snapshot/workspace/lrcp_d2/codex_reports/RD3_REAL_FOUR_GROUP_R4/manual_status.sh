#!/usr/bin/env bash
set -euo pipefail
exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/RD3_REAL_FOUR_GROUP_R4/status.py "$@"
