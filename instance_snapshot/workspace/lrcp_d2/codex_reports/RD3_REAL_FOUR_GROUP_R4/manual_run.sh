#!/usr/bin/env bash
set -euo pipefail
readonly R4='/workspace/lrcp_d2/codex_reports/RD3_REAL_FOUR_GROUP_R4'
readonly PY='/workspace/venvs/lrcp/bin/python'
case "${1:-}" in
  --help|-h) printf '%s\n' 'Manual foreground real run: manual_run.sh --confirm-real-forwards' 'Safe no-forward check: manual_run.sh --dry-run' 'One launch only. No retry/resume. Failure stops all later tasks.'; exit 0 ;;
  --dry-run) [[ $# -eq 1 ]] || exit 2; exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$PY" -B "$R4/runner.py" --mode CPU_PREFLIGHT --dry-run ;;
  --confirm-real-forwards) [[ $# -eq 1 ]] || { printf '%s\n' 'Unexpected arguments' >&2; exit 2; } ;;
  *) printf '%s\n' 'MANUAL_CONFIRMATION_REQUIRED. No forward started. Use --help.' >&2; exit 2 ;;
esac
exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 RD3_R4_MANUAL_ENTRY=1 "$PY" -B "$R4/runner.py" --mode REAL_FORWARD --confirm-real-forwards
