#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
exec /workspace/venvs/lrcp/bin/python -B "$RUN_DIR/validate_cpu.py"
