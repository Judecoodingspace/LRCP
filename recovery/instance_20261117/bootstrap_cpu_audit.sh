#!/usr/bin/env bash
set -euo pipefail
# CPU package installation only. This does not start LRCP inference or any study.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${LRCP_CPU_VENV:-/workspace/venvs/lrcp_cpu_audit}"
python3.10 -m venv "$audit_python"
"$audit_python/bin/python" -m pip install -r "$script_dir/requirements.cpu-audit.txt"
"$audit_python/bin/python" -B -c 'import numpy, shapely, yaml; print("CPU dependencies ready")'
