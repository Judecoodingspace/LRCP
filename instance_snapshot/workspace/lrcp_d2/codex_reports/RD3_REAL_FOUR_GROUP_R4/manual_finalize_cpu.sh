#!/usr/bin/env bash
set -euo pipefail
readonly R4='/workspace/lrcp_d2/codex_reports/RD3_REAL_FOUR_GROUP_R4'
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  printf '%s\n' 'CPU-only: independently validate all 32 real records, then global PR/utility analysis.' 'Incomplete science: NOT_ESTIMABLE, exit 3; no forward or CUDA. No arguments required.'; exit 0
fi
[[ $# -eq 0 ]] || { printf '%s\n' 'No arguments accepted; use --help' >&2; exit 2; }
exec env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /workspace/venvs/lrcp/bin/python -B "$R4/finalize_cpu.py"
