#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/workspace/venvs/lrcp/bin/python
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
exec "$PYTHON" -B "$RUN_DIR/scientific_validator.py"
