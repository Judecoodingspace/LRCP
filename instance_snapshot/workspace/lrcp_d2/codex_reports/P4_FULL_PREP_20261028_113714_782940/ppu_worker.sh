#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/workspace/venvs/lrcp/bin/python
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -B "$RUN_DIR/full_launch_control.py" worker --nonce "$1"
