#!/bin/bash
set -euo pipefail
for arg in "$@"; do if [ "$arg" = "--mock" ]; then echo "validate_cpu.sh requires REAL mode" >&2; exit 2; fi; done
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/P5_B0_C3_20261031_cpu/controller.py validate "$@"
