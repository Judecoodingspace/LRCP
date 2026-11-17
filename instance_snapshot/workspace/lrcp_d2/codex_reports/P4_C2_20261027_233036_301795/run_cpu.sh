#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/workspace/venvs/lrcp/bin/python
if [[ -e "$RUN_DIR/baseline_validation.json" || -e "$RUN_DIR/COMPLETION.json" ]]; then
  echo 'Refusing to overwrite an existing P4-C2 run' >&2
  exit 2
fi
mkdir -p "$RUN_DIR/cases"
"$PYTHON" -B - "$RUN_DIR" <<'VERIFY'
import sys,json,hashlib
from pathlib import Path
r=Path(sys.argv[1]);d=json.loads((r/'inputs.json').read_text())
for key in ['history_hashes','protected_hashes']:
 for name,want in d[key].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,name
VERIFY
"$PYTHON" -B "$RUN_DIR/run_factorial.py" > "$RUN_DIR/baseline.log" 2>&1
"$PYTHON" -B "$RUN_DIR/run_factorial.py" --mixed > "$RUN_DIR/mixed.log" 2>&1
"$PYTHON" -B "$RUN_DIR/analyze_factorial.py" > "$RUN_DIR/analysis.log" 2>&1
"$PYTHON" -B "$RUN_DIR/validate_factorial.py" > "$RUN_DIR/validation.log" 2>&1
"$PYTHON" -B "$RUN_DIR/plan_full.py" > "$RUN_DIR/full_plan_evidence.log" 2>&1
"$PYTHON" -B "$RUN_DIR/write_reports.py" > "$RUN_DIR/report.log" 2>&1
"$PYTHON" -B "$RUN_DIR/seal_results.py"
