#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /workspace/venvs/lrcp/bin/python -B "$RUN_DIR/p5_control.py" status
