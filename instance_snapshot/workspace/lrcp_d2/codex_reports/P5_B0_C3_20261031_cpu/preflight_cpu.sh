#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /workspace/venvs/lrcp/bin/python -B /workspace/lrcp_d2/codex_reports/P5_B0_C3_20261031_cpu/preflight_cpu.py "$@"
