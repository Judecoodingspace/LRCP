#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
if [[ "$#" -ne 1 || -z "${P4_LAUNCH_NONCE:-}" || "$1" != "$P4_LAUNCH_NONCE" ]]; then
  echo "内部worker只接受launch_ppu.sh生成的握手；请使用手动启动入口。" >&2
  exit 64
fi
exec /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/P4_PILOT_RETRY_20261023_234457_384594/launch_control.py worker --nonce "$1"
