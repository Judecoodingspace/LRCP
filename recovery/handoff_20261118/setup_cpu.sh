#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_env="${LRCP_CPU_VENV:-/workspace/venvs/lrcp_cpu_audit}"
if [ ! -x "$audit_env/bin/python" ]; then python3.10 -m venv "$audit_env"; fi
if [ "$#" -gt 0 ]; then
  "$audit_env/bin/python" -m pip install --no-index --find-links "$1" -r "$script_dir/requirements.cpu-audit.txt"
else
  "$audit_env/bin/python" -m pip install --index-url https://pypi.org/simple -r "$script_dir/requirements.cpu-audit.txt"
fi
CUDA_VISIBLE_DEVICES='' "$audit_env/bin/python" -B -c 'import sys, numpy, shapely, yaml; assert "torch" not in sys.modules; print("CPU_AUDIT_ENV_READY",sys.executable,numpy.__version__,shapely.__version__,yaml.__version__)'
